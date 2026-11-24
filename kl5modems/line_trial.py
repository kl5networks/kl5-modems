"""Each Protel modulation across a ``line_model.Line``: what survives.

The question this answers is the one the project had only reasoned about:
over an HT802, is 300 more reliable than 1200? A trial sends 69-byte
packets (the size of a Protel's call-in packet) each way between a phone
side and our answering modem, and counts the packets that arrive whole.
XNet and REL both resend a damaged packet, so a packet lost is time lost,
not a failed call, and the table says both.

The answering side is our own code under test, exactly as a call-in uses
it: ``protel_pcm_peer``'s Bell 103 and 212A answerers, and the native
receiver ``native_modem`` is built on. The phone side is the matching
originating modem: the 73K212L chip model's Bell 103 receiver, dpsk212's
originate carrier, the native FSK. Bell 103 and 212A run full duplex, both
carriers up the whole time, as a call has them; the native FSK is half
duplex, so each direction runs with the other side silent.

    python3 -m kl5modems.line_trial [seconds]

prints the table. Emulator evidence only: the mute model is inferred
(line_model), and no real HT802 has carried these trials.
"""

from __future__ import annotations

from collections import deque
import math
import random
import struct
import sys

from . import dpsk212, line_model, native1200
from .protel_bell103 import Bell103Receiver
from .protel_devices import SSI73K212L
from .protel_pcm_peer import Bell103AnswerPeer, Bell103UART, DPSKAnswerPeer

RATE = 8000
FRAME = 160
PACKET = 69                     # a Protel call-in packet
GAP_MS = 100                    # idle between packets
MODES = ("bell103", "v22", "native")
BAUD = {"bell103": 300, "v22": 1200, "native": 1200}


def _frame_bits(data):
    for value in data:
        yield 0
        for n in range(8):
            yield (value >> n) & 1
        yield 1


class _Bell103Phone:
    """Originate: sends 1270/1070 Hz, hears 2225/2025 through the chip model."""

    def __init__(self, amplitude=6000):
        self.amplitude = amplitude
        self.bits = deque()
        self.phase = 0.0
        self.remaining = 0.0
        self.level = 1
        self.modem = SSI73K212L(deque(maxlen=64))
        self.modem.write_register(0, 0x31)          # originate: receive the high band
        self.receiver = Bell103Receiver(self.modem)
        self.uart = Bell103UART()

    def queue(self, data):
        self.bits.extend(_frame_bits(data))

    def busy(self):
        return bool(self.bits)

    def tx(self, n):
        out = []
        for _ in range(n):
            if self.remaining <= 0:
                self.level = self.bits.popleft() if self.bits else 1
                self.remaining += RATE / 300
            self.remaining -= 1
            out.append(round(self.amplitude * math.sin(self.phase)))
            self.phase = (self.phase + 2 * math.pi * (1270 if self.level else 1070) / RATE) % (2 * math.pi)
        return out

    def rx(self, samples):
        out = bytearray()
        for v in samples:
            self.receiver.sample(v)
            value = self.uart.sample(self.modem.rxd_level)
            if value is not None:
                out.append(value)
        return bytes(out)

    def losses(self):
        return self.receiver.losses


class _V22Phone:
    """Originate: 212A on 1200 Hz out, 2400 Hz in."""

    def __init__(self, amplitude=6000):
        self.mod = dpsk212.Modulator(dpsk212.ORIGINATE_HZ, amplitude)
        self.demod = dpsk212.Demodulator(dpsk212.ANSWER_HZ)
        self.uart = dpsk212.AsyncDeframer()
        self.bits = deque()

    def queue(self, data):
        self.bits.extend(_frame_bits(data))

    def busy(self):
        return bool(self.bits)

    def tx(self, n):
        while self.mod.ready() < n:
            self.mod.push_bit(self.bits.popleft() if self.bits else 1)
        return self.mod.pcm(n)

    def rx(self, samples):
        out = bytearray()
        for _, bit in self.demod.feed(samples):
            value = self.uart.bit(bit)
            if value is not None:
                out.append(value)
        return bytes(out)

    def losses(self):
        return sum(1 for _, on in self.demod.carrier_events if not on)


class _NativeSide:
    """Either end of the native FSK: keyed per packet, as both stations are,
    so its carrier drops at every packet's end and is not counted."""

    def __init__(self, amplitude=8000):
        self.mod = native1200.Modulator(amplitude=amplitude, keyed=True)
        self.demod = native1200.Demodulator()

    def queue(self, data):
        self.mod.queue(data)

    def busy(self):
        return self.mod.on

    def tx(self, n):
        return self.mod.pcm(n)

    def rx(self, samples):
        return self.demod.feed(samples)

    def losses(self):
        return None


class _Hub:
    """One of ``protel_pcm_peer``'s answerers behind the sample interface."""

    def __init__(self, peer, losses):
        self.peer = peer
        self._losses = losses
        self.sent = 0

    def queue(self, data):
        self.peer.queue(data)
        self.sent += len(data)

    def busy(self):
        return bool(self.peer.bytes)

    def tx(self, n):
        return list(struct.unpack("<%dh" % n, self.peer.render(n)))

    def rx(self, samples):
        return self.peer.receive(struct.pack("<%dh" % len(samples), *samples))

    def losses(self):
        return self._losses(self.peer)


def _pair(mode):
    if mode == "bell103":
        return _Bell103Phone(), _Hub(Bell103AnswerPeer(), lambda p: p.receiver.losses)
    if mode == "v22":
        return _V22Phone(), _Hub(DPSKAnswerPeer(),
                                 lambda p: sum(1 for _, on in p.demod.carrier_events if not on))
    if mode == "native":
        return _NativeSide(), _NativeSide()
    raise ValueError("unknown mode %r" % mode)


class _Stream:
    """Packets out one side, bytes in at the other; a packet counts if it
    arrived whole, in order, somewhere in what was heard."""

    def __init__(self, rng):
        self.rng = rng
        self.packets = []
        self.heard = bytearray()
        self.gap = 0

    def next_packet(self):
        packet = bytes(self.rng.randrange(256) for _ in range(PACKET))
        self.packets.append(packet)
        return packet

    def intact(self):
        heard = bytes(self.heard)
        # The last packet may still be on the wire when the trial stops.
        sent = self.packets[:-1] if self.packets else []
        return sum(1 for p in sent if p in heard), len(sent)


def trial(mode, line, seconds=30.0, seed=0, directions=("up", "down")):
    """Run one modulation across ``line``. ``up`` is phone to us (the
    direction the HT802 mutes), ``down`` us to the phone."""
    phone, hub = _pair(mode)
    rng = random.Random(seed)
    up, down = _Stream(rng), _Stream(rng)
    half_duplex = mode == "native"
    frames = int(seconds * RATE / FRAME)
    gap_frames = max(1, GAP_MS * RATE // 1000 // FRAME)
    turn = "up"                        # half duplex: whose half it is
    for n in range(frames):
        if half_duplex:
            turn = "up" if n < frames // 2 or "down" not in directions else "down"
            if "up" not in directions:
                turn = "down"
        for name, side, stream in (("up", phone, up), ("down", hub, down)):
            if name not in directions or (half_duplex and turn != name):
                continue
            if not side.busy():
                stream.gap += 1
                if stream.gap >= gap_frames:
                    side.queue(stream.next_packet())
                    stream.gap = 0
        ours = hub.tx(FRAME)
        theirs = phone.tx(FRAME)
        up.heard += hub.rx(line.to_hub(theirs))
        down.heard += phone.rx(line.to_phone(ours))
    result = {"mode": mode, "baud": BAUD[mode], "seconds": seconds, "line": line.snapshot()}
    for name, stream, receiver in (("up", up, hub), ("down", down, phone)):
        if name in directions:
            ok, sent = stream.intact()
            result[name] = {"packets": sent, "intact": ok, "carrier_losses": receiver.losses()}
    return result


SCENARIOS = {
    # Codec only: what every ATA call has.
    "clean": lambda seed: line_model.Line(seed=seed),
    # A fair line: -55 dBm0 noise, our echo back 22 dB down after 40 ms,
    # the phone's own 14 dB down at once.
    "echo+noise": lambda seed: line_model.Line(noise_dbm0=-55, hub_echo=(40, 22), phone_echo=(0.5, 14), seed=seed),
    # The HT802's mute as modelled (inferred), on that fair line.
    "ht802": lambda seed: line_model.Line(mute=line_model.DTMFMute(), noise_dbm0=-55, hub_echo=(40, 22),
                                          phone_echo=(0.5, 14), seed=seed),
    # One mute a second whatever the signal: what a mute costs each modem.
    "mute 1/s": lambda seed: line_model.Line(mute=line_model.RateMute(1.0, seed), noise_dbm0=-55,
                                             hub_echo=(40, 22), phone_echo=(0.5, 14), seed=seed),
}


def table(seconds=60.0, seed=1, out=sys.stdout):
    print("%-11s %-8s %5s  %-24s %-24s %s" % ("line", "mode", "baud", "up (phone->us) whole",
                                               "down (us->phone) whole", "mutes"), file=out)
    for name, make in SCENARIOS.items():
        for mode in MODES:
            r = trial(mode, make(seed), seconds, seed)
            cells = []
            for d in ("up", "down"):
                x = r[d]
                drops = x["carrier_losses"]
                cells.append("%3d/%-3d  %s" % (x["intact"], x["packets"], "keyed" if drops is None else
                                               "%d carrier drop%s" % (drops, "" if drops == 1 else "s")))
            print("%-11s %-8s %5d  %-24s %-24s %d" % (name, mode, r["baud"], cells[0], cells[1],
                                                      r["line"]["mutes"]), file=out)


if __name__ == "__main__":
    table(float(sys.argv[1]) if len(sys.argv) > 1 else 60.0)
