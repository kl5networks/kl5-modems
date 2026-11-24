"""The UPMS1200's REL link, as a station: frames in, replies and host bytes out.

The native 1200 line carries REL frames (``upms1200_rel``): a control byte,
payload, CRC-16 and an ``FF A5 5A`` trailer. What the control bits MEAN was
never written down anywhere; this module is the UPMS ROM's behaviour as the
REL peer bench executed it in an emulator, turned into rules:

- bit 6 is the sender's receive toggle. A station flips its own each time it
  commits a peer's payload, and a peer that flips ITS bit 6 has acknowledged
  our last data, so we may send the next;
- bit 5 marks a frame that carries data to commit. A frame without it (the
  peer's control ``00`` poll) commits nothing and, when it acknowledges
  nothing either, is answered with our last frame byte for byte;
- bit 3 says the sender cannot take data: we hold ours until it clears;
- a reply to a frame is control ``30`` plus our bit 6; a frame sent on our
  own (the idle request, or after a frame that failed its CRC) is just our
  bit 6 and no data, which is the ``40`` the ROM sends from reset and the
  ``00`` it sends after a first exchange;
- a payload the 511-byte receive ring cannot hold is not committed, and the
  toggle stays where it was, so the peer resends it.

Each rule was held against the ROM, scenario for scenario, in that
emulator; test_native_rel replays the turn tables it recorded. Bit 4 is set
in every reply the ROM makes and in no frame the bench's peer made; its meaning is not known, so it is copied, not
interpreted. A 7000's own use of these bits is not traced yet.

A pure state machine: no clock, no sockets. The caller
says when the line has been quiet long enough for an idle request.
"""

from __future__ import annotations

from .upms1200_rel import UPMS1200Error
from .upms1200_rel import decode_rel_frame, encode_rel_frame

ACK = 0x40          # our receive toggle
DATA = 0x20         # this frame carries data to commit
REPLY = 0x30        # every reply the ROM makes to a peer's data frame
BUSY = 0x08         # the sender's receive buffer is full

RECEIVE_RING = 511          # 512 positions, one reserved to tell full from empty
MAX_PAYLOAD = 64            # ours; the ROM's own split is timing, not a rule
IDLE_SECONDS = 2.6          # the ROM's idle request: 2.3M-2.4M cycles at OSC/4


class Deframer:
    """Line bytes in, whole REL wire frames out, split at an unquoted A5 5A.

    The same quoting ``decode_rel_frame`` applies: after A5, a 5B is
    dropped and the byte after it taken literally, so A5 5B 5A inside a
    frame is data and not its end.
    """

    LIMIT = 2 * (RECEIVE_RING + 3) + 4

    def __init__(self):
        self.buffer = bytearray()
        self.after_a5 = False
        self.overruns = self.dropped = 0

    def feed(self, data):
        frames = []
        for value in bytes(data):
            self.buffer.append(value)
            if self.after_a5 and value == 0x5A:
                frames.append(bytes(self.buffer))
                self.buffer.clear()
                self.after_a5 = False
                continue
            if self.after_a5 and value == 0x5B:
                self.after_a5 = False       # the quote: next byte is literal
                continue
            self.after_a5 = value == 0xA5
            if len(self.buffer) > self.LIMIT:
                # Line noise with no terminator: drop it rather than grow.
                self.buffer.clear()
                self.after_a5 = False
                self.overruns += 1
        return frames

    def gap(self):
        """The line went idle mid-frame: what came before was not a frame.

        Ours, not the ROM's (how its receiver finds a frame's first byte is
        not traced): a frame is sent back to back, so a pause of a few
        character times means noise or a lost carrier, and keeping those
        bytes would put them in front of the next real frame and fail its
        CRC.
        """
        if self.buffer:
            self.buffer.clear()
            self.after_a5 = False
            self.dropped += 1


class Station:
    """The UPMS end of the link: the side a 7000 calls."""

    def __init__(self, max_payload=MAX_PAYLOAD):
        self.max_payload = max_payload
        self.rx_toggle = 1          # the ROM's first frames are control 40
        self.peer_ack = 1           # so a peer's first bit 6 of 0 acknowledges
        self.outgoing = bytearray()
        self.in_flight = None       # payload sent and not yet acknowledged
        self.peer_busy = False
        self.last_sent = None
        self.frames_in = self.frames_rejected = self.frames_out = 0
        self.resends = self.uncommitted = 0

    def send(self, data):
        """Queue host bytes for the peer."""
        self.outgoing.extend(data)

    @property
    def unacknowledged(self):
        return bool(self.in_flight) or bool(self.outgoing)

    def receive(self, wire):
        """One wire frame from the line: (reply wire frame, bytes for the host)."""
        try:
            frame = decode_rel_frame(wire)
        except UPMS1200Error:
            self.frames_rejected += 1
            return self._emit(encode_rel_frame(self.rx_toggle * ACK)), b""
        self.frames_in += 1
        control = frame.control
        acknowledged = bool(control & ACK) != bool(self.peer_ack)
        if acknowledged:
            self.peer_ack = 1 if control & ACK else 0
            self.in_flight = None
        self.peer_busy = bool(control & BUSY)
        delivered = b""
        if control & DATA:
            if len(frame.payload) <= RECEIVE_RING:
                delivered = frame.payload
                self.rx_toggle ^= 1
            else:
                self.uncommitted += 1
        elif not acknowledged and self.last_sent is not None:
            self.resends += 1
            return self._emit(self.last_sent), b""
        return self._emit(self._reply()), delivered

    def idle(self):
        """The line has been quiet IDLE_SECONDS: ask again, carrying no data."""
        return self._emit(encode_rel_frame(self.rx_toggle * ACK))

    def _reply(self):
        if self.in_flight is None and self.outgoing and not self.peer_busy:
            self.in_flight = bytes(self.outgoing[:self.max_payload])
            del self.outgoing[:self.max_payload]
        payload = b"" if self.peer_busy or self.in_flight is None else self.in_flight
        return encode_rel_frame(REPLY | self.rx_toggle * ACK, payload)

    def _emit(self, wire):
        self.last_sent = wire
        self.frames_out += 1
        return wire

    def snapshot(self):
        return {"rx_toggle": self.rx_toggle, "peer_ack": self.peer_ack,
                "in_flight": len(self.in_flight or b""), "queued": len(self.outgoing),
                "peer_busy": self.peer_busy, "frames_in": self.frames_in,
                "frames_rejected": self.frames_rejected, "frames_out": self.frames_out,
                "resends": self.resends, "uncommitted": self.uncommitted}
