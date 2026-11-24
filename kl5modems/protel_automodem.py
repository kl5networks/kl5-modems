"""One answering modem for every Protel: it listens, then speaks what it heard.

An answering modem normally has to match each phone: Bell 103 for a board on 300, Bell
212A for a 4000/8000 on 1200, the native FSK for a 7000 on 1200. Get it wrong
and the call-in is silence. This modem answers without knowing, and decides
from the first thing the phone sends after it hears us:

- a 4000/8000 on 300 sends its Bell 103 originate mark, a steady 1270 Hz;
- a 4000/8000 on 1200 sends scrambled 212A DPSK on 1200 Hz: broadband,
  energy across 700-1700 Hz and no single tone;
- a 7000 on native 1200 opens with its own handshake, 1059 Hz for 1.2 s
  (then 1252 Hz, then its native mark).

What it sends while it listens is the part that has to suit all three. A
4000/8000 on 1200 must hear a 212A answer carrier and NOT Bell's 2225 Hz
tone: a real 1D took 2225 Hz for carrier, went to slave sync with nothing to
clock from, and sent 52 s of bare carrier. So the
answer starts as the 212A answerer, scrambled marks on 2400 Hz, and only
after ``PROBE_DPSK_SECONDS`` with no DPSK heard becomes a steady 2225 Hz:
Bell 103's answer mark, and inside the band a 7000 takes for the native
mark (1.9-2.5 kHz; INFERRED, not run against one yet).

Once it has decided it is exactly one of the three modems already proven
against the firmware in an emulator: ``protel_pcm_peer``'s Bell 103 and 212A answerers,
or ``native_modem.Call``. Same interface as ``native_modem.Call``, so the
AudioSocket bridge carries it unchanged.
"""

from __future__ import annotations

import cmath
import math
import struct

from . import native1200, native_modem
from .protel_pcm_peer import Bell103AnswerPeer, DPSKAnswerPeer

RATE = 8000
WINDOW = 320                    # 40 ms: 25 Hz bins, a Hann main lobe of +-50 Hz
AGREE = 4                       # windows in a row, 160 ms, before deciding
MIN_AMPLITUDE = 200             # PCM units, about -44 dBFS: below it, nothing heard
DOMINANCE = 3.0                 # a tone is a tone when its bin is 3x every probe bin
LOW_HZ = 1065                   # the 7000's 1059 Hz handshake, and Bell 103's space (1070)
MARK_HZ = 1270                  # Bell 103 originate mark
# Probes for 212A's broadband carrier, kept 150+ Hz from both tones so
# neither leaks into them through the window's sidelobes. Inside its band
# (1200 +- 525 Hz at 75% roll-off): probes at 700 and 1700 sat on its edges
# and read near zero in half the windows of the emulated 1D's carrier.
PROBES = (900, 1500)
PROBE_DPSK_SECONDS = 2.0
ANSWER_TONE_HZ = 2225
ANSWER_AMPLITUDE = 6000

MODES = ("bell103", "v22", "native")


def _hann(n):
    return [0.5 - 0.5 * math.cos(2 * math.pi * k / (n - 1)) for k in range(n)]


class Detector:
    """Phone audio in, a decision out: 'bell103', 'v22', 'native' or None."""

    def __init__(self):
        self.window = _hann(WINDOW)
        self.gain = 2 / sum(self.window)
        freqs = (LOW_HZ, MARK_HZ) + PROBES
        self.basis = {f: [cmath.exp(-2j * math.pi * f * n / RATE) for n in range(WINDOW)] for f in freqs}
        self.buffer = []
        self.recent = []
        self.windows = 0

    def amplitudes(self, frame):
        w = [x * h for x, h in zip(frame, self.window)]
        return {f: abs(sum(a * b for a, b in zip(w, basis))) * self.gain for f, basis in self.basis.items()}

    def classify(self, frame):
        """One 40 ms window: 'low', 'fsk', 'v22' or None.

        'low' is the 1059/1070 Hz bin alone. Four of those in a row is the
        7000's handshake: Bell 103 data never holds space for 160 ms (eight
        zero bits and a start bit is 30 ms), so it is never 'low' four times
        running. 'fsk' is FSK with the 1270 Hz mark in it.
        """
        amp = self.amplitudes(frame)
        probes = [amp[f] for f in PROBES]
        low, mark = amp[LOW_HZ], amp[MARK_HZ]
        fsk = max(low, mark)
        if fsk >= MIN_AMPLITUDE and fsk >= DOMINANCE * max(probes):
            return "fsk" if mark >= low / 4 else "low"
        if min(probes) >= MIN_AMPLITUDE / 2:
            return "v22"
        return None

    def feed(self, samples):
        self.buffer.extend(samples)
        while len(self.buffer) >= WINDOW:
            frame, self.buffer = self.buffer[:WINDOW], self.buffer[WINDOW:]
            self.recent = (self.recent + [self.classify(frame)])[-AGREE:]
            self.windows += 1
            decision = self.decide(self.recent)
            if decision:
                return decision
        return None

    @staticmethod
    def decide(recent):
        if len(recent) < AGREE:
            return None
        if all(v == "low" for v in recent):
            return "native"
        # Scrambled data puts a tone-like lump in one window in five or so
        # (the emulated 1D's carrier did), so three of four is enough.
        if recent.count("v22") >= AGREE - 1 and "low" not in recent:
            return "v22"
        if all(v in ("fsk", "low") for v in recent) and recent.count("fsk") >= AGREE // 2:
            return "bell103"
        return None


class _Peer:
    """A byte-PCM answer peer (protel_pcm_peer) behind Call's sample interface."""

    def __init__(self, peer):
        self.peer = peer

    def send(self, data):
        self.peer.queue(data)

    def audio(self, samples):
        heard = self.peer.receive(struct.pack("<%dh" % len(samples), *samples))
        pcm = self.peer.render(len(samples))
        return heard, list(struct.unpack("<%dh" % len(samples), pcm))

    def snapshot(self):
        return self.peer.snapshot()


class AutoCall:
    """``native_modem.Call``'s interface; the modem is chosen by listening."""

    def __init__(self, probe_dpsk_seconds=PROBE_DPSK_SECONDS):
        self.probe_samples = int(probe_dpsk_seconds * RATE)
        self.detector = Detector()
        # Both FSK/PSK answerers run from the first sample, so whichever is
        # chosen has been in sync with the line all along; the 212A one is
        # also what we send until the probe ends.
        self.v22 = _Peer(DPSKAnswerPeer())
        self.bell = _Peer(Bell103AnswerPeer())
        self.held = {"v22": bytearray(), "bell103": bytearray()}
        self.modem = None
        self.mode = None
        self.decided_at = None
        self.pending = bytearray()
        self.now = 0
        self.phase = 0.0

    def send(self, data):
        if self.modem is None:
            self.pending.extend(data)
        else:
            self.modem.send(data)

    def audio(self, samples):
        if self.modem is not None:
            out = self.modem.audio(samples)
            self.now += len(samples)
            return out
        heard_v22, pcm_v22 = self.v22.audio(samples)
        heard_bell, _ = self.bell.audio(samples)
        self.held["v22"] += heard_v22
        self.held["bell103"] += heard_bell
        start = self.now
        self.now += len(samples)
        mode = self.detector.feed(samples)
        if mode:
            return self._decide(mode), self._silence_or(pcm_v22, mode, len(samples))
        if start < self.probe_samples:
            return b"", pcm_v22
        return b"", self._tone(len(samples))

    def _silence_or(self, pcm_v22, mode, count):
        if mode == "v22":
            return pcm_v22
        if mode == "bell103":
            return self._tone(count)
        return [0] * count

    def _decide(self, mode):
        self.mode = mode
        self.decided_at = round(self.now / RATE, 2)
        if mode == "native":
            # Already answered: the 7000 is past waiting for our mark.
            self.modem = native_modem.Call(answer_mark_seconds=0, mark_bursts=False)
            heard = b""
        else:
            self.modem = self.v22 if mode == "v22" else self.bell
            heard = bytes(self.held[mode])
        self.held = None
        if self.pending:
            self.modem.send(bytes(self.pending))
            self.pending.clear()
        return heard

    def _tone(self, count):
        out = []
        step = 2 * math.pi * ANSWER_TONE_HZ / RATE
        for _ in range(count):
            out.append(int(round(ANSWER_AMPLITUDE * math.sin(self.phase))))
            self.phase = (self.phase + step) % (2 * math.pi)
        return out

    def snapshot(self):
        return {"mode": self.mode, "decided_at_seconds": self.decided_at,
                "seconds": round(self.now / RATE, 2), "windows": self.detector.windows,
                "recent_windows": self.detector.recent,
                "modem": self.modem.snapshot() if self.modem is not None else None}


class AutoAnswerPeer:
    """AutoCall behind protel_pcm_peer's byte interface (queue/render/receive),
    for the firmware-line harness, which renders before it receives."""

    def __init__(self, **options):
        self.call = AutoCall(**options)
        self.out = b""

    def queue(self, data):
        self.call.send(data)

    def end_input(self):
        pass

    def render(self, samples=160):
        out, self.out = self.out[:2 * samples], self.out[2 * samples:]
        return out + bytes(2 * samples - len(out))

    def receive(self, pcm):
        samples = struct.unpack("<%dh" % (len(pcm) // 2), pcm)
        heard, ours = self.call.audio(samples)
        self.out += struct.pack("<%dh" % len(ours), *ours)
        return heard

    @property
    def mode(self):
        return self.call.mode

    def snapshot(self):
        return self.call.snapshot()


assert native1200.RATE == RATE
