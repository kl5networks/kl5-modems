"""A software UPMS1200: the Protel native 1200 modem, over AudioSocket.

A Protel 7000 at 1200 speaks a modulation no modem but Protel's own UPMS
does (``native1200``), wrapped in the UPMS's REL link (``native_rel``).
Asterisk's Softmodem speaks neither. So the dialplan hands a 7000's call-in
to ``AudioSocket()`` instead, and this answers it: line audio in, REL frames
off it, their payloads out to a manager over plain TCP -- the same byte pipe
Softmodem would give a manager -- and the manager's bytes
back to the phone as REL frames.

``Call`` is the modem with no socket, so a whole call can be run at a desk
sample by sample; ``Bridge`` is the asyncio glue.

Half duplex, as the UPMS is: while we transmit, what comes back is our own
echo off the ATA's hybrid on the same two tones, and a receiver left open
would read our reply -- data bit and all -- as the phone's.
"""

from __future__ import annotations

import asyncio
import logging
import struct

from . import audiosocket
from . import _protel_compat as protel
from . import native1200, native_rel

log = logging.getLogger(__name__)

DEFAULT_PORT = 2528             # where Asterisk's AudioSocket() reaches this
GAP_CHARS = 3                   # character times of quiet that end a partial frame
ECHO_GUARD_CELLS = 8            # receiver stays shut this long after we key down
# A 7000 that has dialled waits for the native mark before it sends anything
# ($DC36, tone class $11: 24 of 32 half-period pairs in the 2151 Hz band;
# it accepted a steady mark within 25 ms in the emulator).
# Then it plays 1.45 s of its own handshake tones and sends its first frame,
# so a second of mark is heard well before, and over well before, that frame.
ANSWER_MARK_SECONDS = 1.0
# But a 7000 only starts listening ~3.2 s after its last digit, and the call
# may be answered before that: a single mark at answer is over before the
# phone is listening. So until the phone is first heard we key short bursts
# of mark, as the UPMS's own answerer re-asserts its tone, for at most
# MARK_BURST_LIMIT_SECONDS ($DC36 listens at least 34 s).
MARK_BURST_SECONDS = 0.25
MARK_BURST_EVERY = 1.0          # seconds of quiet between bursts
MARK_BURST_LIMIT_SECONDS = 40.0


class Call:
    """One call's modem: ``audio(pcm)`` in, (bytes for the manager, pcm out)."""

    def __init__(self, amplitude=8000, idle_seconds=native_rel.IDLE_SECONDS,
                 answer_mark_seconds=ANSWER_MARK_SECONDS, mark_bursts=True):
        self.rx = native1200.Demodulator()
        self.tx = native1200.Modulator(amplitude=amplitude, keyed=True)
        self.deframer = native_rel.Deframer()
        self.station = native_rel.Station()
        self.idle_samples = int(idle_seconds * native1200.RATE)
        cell = native1200.RATE / native1200.BAUD
        self.gap_samples = int(GAP_CHARS * 10 * cell)
        self.guard_samples = int(ECHO_GUARD_CELLS * cell)
        self.now = 0
        self.last_byte = 0
        self.quiet_since = 0            # last time either side finished a frame
        self.deaf_until = 0
        self.frames_out = 0
        self.echo_bytes = 0
        self.heard_phone = False
        self.bursts = 0
        self.burst_every = int(MARK_BURST_EVERY * native1200.RATE)
        self.burst_limit = int(MARK_BURST_LIMIT_SECONDS * native1200.RATE) if mark_bursts else 0
        if answer_mark_seconds:
            self.tx.mark(int(answer_mark_seconds * native1200.BAUD))

    def send(self, data):
        """Bytes from the manager, for the phone."""
        self.station.send(data)

    def audio(self, samples):
        """Feed the phone's audio; return (bytes for the manager, our audio)."""
        out_bytes = bytearray()
        heard = self.rx.feed(samples)
        self.now += len(samples)
        listening = not self.transmitting and self.now > self.deaf_until
        if listening and self.rx.carrier:
            self.heard_phone = True
        if not listening:
            self.echo_bytes += len(heard)
            heard = b""
        if heard:
            self.last_byte = self.now
        elif self.deframer.buffer and self.now - self.last_byte >= self.gap_samples:
            self.deframer.gap()
        for wire in self.deframer.feed(heard):
            reply, delivered = self.station.receive(wire)
            out_bytes += delivered
            self._transmit(reply)
        if not self.transmitting and self._bursting():
            if self.now - self.quiet_since >= self.burst_every:
                self.tx.mark(int(MARK_BURST_SECONDS * native1200.BAUD))
                self.bursts += 1
        elif not self.transmitting and self.now - self.quiet_since >= self.idle_samples:
            self._transmit(self.station.idle())
        was = self.transmitting
        pcm = self.tx.pcm(len(samples))
        if was and not self.transmitting:
            self.quiet_since = self.now
            self.deaf_until = self.now + self.guard_samples
        return bytes(out_bytes), pcm

    def _bursting(self):
        """Still calling the phone with mark: not heard yet, within the limit.
        Idle requests wait meanwhile; nothing is listening for them."""
        return not self.heard_phone and self.now < self.burst_limit

    @property
    def transmitting(self):
        return self.tx.on

    def _transmit(self, wire):
        self.tx.queue(wire)
        self.frames_out += 1
        self.quiet_since = self.now

    def snapshot(self):
        return {"seconds": round(self.now / native1200.RATE, 2), "carrier": self.rx.carrier,
                "characters": self.rx.frames, "framing_errors": self.rx.framing_errors,
                "echo_bytes": self.echo_bytes,
                "heard_phone": self.heard_phone, "mark_bursts": self.bursts, "frames_dropped": self.deframer.dropped,
                "link": self.station.snapshot()}


SUMMARY_KEYS = ("mode", "decided_at_seconds", "seconds", "carrier", "decoded_uart_frames",
                "framing_errors", "false_starts", "manager_bytes_accepted", "queued_bytes")


def call_summary(snapshot):
    """The few numbers of a call's snapshot worth a journal line: which modem
    it chose, when, and whether it was still framing the phone's bytes. An
    auto call keeps its chosen modem's figures under `modem`."""
    flat = dict(snapshot.get("modem") or {}, **{k: v for k, v in snapshot.items() if k != "modem"})
    return " ".join("%s=%s" % (k, flat[k]) for k in SUMMARY_KEYS if k in flat) or "no figures"


class Bridge:
    """The listener: one AudioSocket call in, one TCP manager connection out.

    ``target`` is (host, port) of what Softmodem would have reached: the
    manager. ``state()`` follows the rule that a
    service that is not listening always says why.
    """

    def __init__(self, target, call_factory=Call):
        self.target = target
        self.call_factory = call_factory
        self.server = None
        self.why = "not started"
        self.calls = self.active = 0
        self.last = None
        # Each live call by the address its manager connection comes FROM,
        # which is the peer the manager sees: so the manager can ask what
        # modem a call chose (``mode_for``) without a side channel.
        self.live = {}
        self.uuids = {}             # the same key -> the call's AudioSocket id

    def window_for(self, peer):
        """The provisioning window the call reaching the manager from
        ``peer`` was answered for (its AudioSocket id, protel.window_uuid),
        or None."""
        return protel.window_of_uuid(self.uuids.get(tuple(peer[:2]))) if peer else None

    def mode_for(self, peer):
        """What modem the call reaching the manager from ``peer`` (host,
        port) chose: 'bell103', 'v22', 'native' -- or None when that is not
        one of our calls, or its modem cannot choose (a plain native Call)."""
        call = self.live.get(tuple(peer[:2])) if peer else None
        return getattr(call, "mode", None)

    async def start(self, host, port=DEFAULT_PORT):
        try:
            self.server = await asyncio.start_server(self._handle, host, port)
        except OSError as exc:
            self.why = "cannot listen on %s:%s: %s" % (host, port, exc.strerror or exc)
            log.warning("native modem %s", self.why)
            return False
        self.why = ""
        return True

    async def stop(self):
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
            self.why = "stopped"

    def state(self):
        sock = self.server.sockets[0].getsockname() if self.server else None
        return {"listening": bool(sock), "address": "%s:%d" % sock[:2] if sock else "",
                "why": self.why, "calls": self.calls, "active": self.active,
                "target": "%s:%d" % self.target, "last": self.last}

    async def _handle(self, reader, writer):
        self.active += 1
        self.calls += 1
        call = self.call_factory()
        frames = audiosocket.Reader()
        uuid = ""
        try:
            m_reader, m_writer = await asyncio.open_connection(*self.target)
        except OSError as exc:
            log.warning("native modem: manager %s:%d unreachable: %s", *self.target, exc)
            self.last = {"error": "manager unreachable: %s" % exc}
            writer.write(audiosocket.terminate_frame())
            writer.close()
            self.active -= 1
            return
        local = m_writer.get_extra_info("sockname")
        key = tuple(local[:2]) if local else None
        if key:
            self.live[key] = call

        async def from_manager():
            while True:
                data = await m_reader.read(4096)
                if not data:
                    return
                call.send(data)

        pump = asyncio.ensure_future(from_manager())
        try:
            while True:
                # Wait on the manager too, not just the phone's audio.
                # Asterisk sends AudioSocket nothing while the caller's RTP
                # has stopped, so a loop that only noticed the manager
                # between audio frames kept a dead call up after the manager had
                # given up: 2026-10-07 23:50, a 7000 went silent mid-download,
                # the session ended stuck at 335 s and the carrier billed
                # 920 s. Our hang-up must not wait on the far end's audio.
                read = asyncio.ensure_future(reader.read(4096))
                await asyncio.wait({read, pump}, return_when=asyncio.FIRST_COMPLETED)
                if not read.done():
                    read.cancel()
                    writer.write(audiosocket.terminate_frame())
                    break
                chunk = read.result()
                if not chunk:
                    break
                for kind, payload in frames.feed(chunk):
                    if kind == audiosocket.UUID:
                        uuid = audiosocket.call_id(payload)
                        if key:
                            self.uuids[key] = uuid
                    elif kind == audiosocket.AUDIO:
                        samples = struct.unpack("<%dh" % (len(payload) // 2), payload[:len(payload) & ~1])
                        data, pcm = call.audio(samples)
                        if data:
                            m_writer.write(data)
                        writer.write(audiosocket.audio_frame(struct.pack("<%dh" % len(pcm), *pcm)))
                    elif kind == audiosocket.ERROR:
                        log.warning("native modem %s: %s", uuid, audiosocket.error_text(payload))
                    elif kind == audiosocket.TERMINATE:
                        return
                await writer.drain()
                await m_writer.drain()
        except (audiosocket.ProtocolError, ConnectionError) as exc:
            log.warning("native modem %s: %s", uuid, exc)
        finally:
            pump.cancel()
            self.last = dict(call.snapshot(), uuid=uuid)
            # One journal line per call: `last` is only the most recent call
            # and is gone at the next restart, and a stalled session's capture
            # cannot say whether the modem was still hearing the phone. At
            # WARNING because the host program may configure no logging of
            # its own, and below that level nothing reaches the journal.
            log.warning("modem call %s ended: %s", uuid or "?", call_summary(self.last))
            if key:
                self.live.pop(key, None)
                self.uuids.pop(key, None)
            self.active -= 1
            for w in (m_writer, writer):
                try:
                    w.close()
                except Exception:        # noqa: BLE001 -- either side may already be gone
                    pass
