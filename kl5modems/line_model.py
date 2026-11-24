"""A phone line that is not ideal: G.711, noise, echo and the HT802's mutes.

Every modem test in the emulator ran over a perfect wire: the phone's
samples arrived at our modem exactly as made. A real call-in crosses an
HT802 (an FXS port, its hybrid, its DSP, a G.711 codec) and a network, and
the bench has measured what that does:

- **the HT802 mutes single 20 ms packets of the phone's audio** to digital
  silence, -60 to -76 dBFS, 2 to 13 times in 93 to 243 s of a Millennium's
  212A carrier. Its syslog puts a DSP event 20-35 ms before each one and
  logs ``DTMF Mute`` at call set-up, so the mute is its DTMF detector
  false-triggering. Turning the port's gain down 4 dB did not lower the
  rate, so whatever triggers it is a ratio, not a level;
- the mutes are in the ATA's OWN RTP: the phone-to-us direction only.

``DTMFMute`` is the model of that detector, and the part of this file that
is INFERRED rather than measured: per 20 ms frame, the strongest DTMF row
tone and the strongest column tone must EACH hold a share of the frame's
energy (``MUTE_SHARE``), as a dual-tone detector demands. Calibrated so a
212A originate carrier trips it about once in 25 s, inside the measured
range. With nothing else changed it never trips on a Bell 103 originate
carrier (1070/1270 Hz: no row tone) or on a 7000's native FSK (1187/2151
Hz: none either), which matches the one other bench fact there is: Protel
call-ins at 300 through the same HT802 trained and ran. A detector that
needed only a column tone would fire on Bell 103 more often than on 212A
(1270 Hz leaks into the 1209 and 1336 Hz bins), and the 300-baud calls
would not have worked. ``RateMute`` is the other tool: mutes at a fixed
rate whatever the signal, for asking what one mute costs each modem.

Echo is the far end's own signal coming back off a hybrid: ours off the
ATA's 2-wire port, after a network round trip; the phone's off its own
line interface, at once. Noise is white, set in dBm0 (0 dBm0 is an RMS of
16050 in 16-bit PCM, as dpsk212 has it).

A pure model: samples in, samples out, seeded, no clock. ``Line`` holds
both directions; ``ImpairedPeer`` puts one in front of any byte-PCM answer
peer (``protel_pcm_peer``'s interface). Nothing here is
measured on an HT802 beyond the facts above.
"""

from __future__ import annotations

from collections import deque
import cmath
import math
import random
import struct

RATE = 8000
FRAME = 160                     # 20 ms: the ATA's packet, and the unit it mutes
DBM0_RMS = 16050.0              # 0 dBm0 in 16-bit PCM (3.14 dBm0 full scale)
DTMF_ROWS = (697, 770, 852, 941)
DTMF_COLUMNS = (1209, 1336, 1477, 1633)
# Each of the strongest row and column bins must hold this share of the
# frame's energy. 0.195 gives a 212A originate carrier ~0.04 mutes/s (600 s
# simulated); the bench saw 0.022-0.053/s. Bell 103's and the native FSK's
# largest share in 120 s was 0.027 and 0.021: an order of magnitude short.
MUTE_SHARE = 0.195
MIN_FRAME_DBM0 = -45.0          # a frame quieter than this is not looked at


def dbm0(level):
    """dBm0 to an RMS in PCM units."""
    return DBM0_RMS * 10 ** (level / 20)


def _ulaw_encode(sample):
    # G.711 mu-law, the segment-and-mantissa form (ITU-T G.711, Table 2a).
    sign = 0x80 if sample < 0 else 0
    magnitude = min(-sample if sample < 0 else sample, 32635) + 0x84
    exponent = 7
    for exponent in range(7, -1, -1):
        if magnitude & (0x4000 >> (7 - exponent)):
            break
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def _ulaw_decode(code):
    code = ~code & 0xFF
    exponent, mantissa = (code >> 4) & 7, code & 0x0F
    magnitude = ((mantissa << 3) + 0x84 << exponent) - 0x84
    return -magnitude if code & 0x80 else magnitude


_DECODE = [_ulaw_decode(c) for c in range(256)]


def g711(samples):
    """Through a mu-law codec and back, as every ATA call is."""
    out = []
    for v in samples:
        v = int(round(v))
        v = -32768 if v < -32768 else 32767 if v > 32767 else v
        out.append(_DECODE[_ulaw_encode(v)])
    return out


class DTMFMute:
    """The HT802's DTMF detector, false-triggering: INFERRED, see the module.

    Fed the phone's audio as it reaches the ATA; a 20 ms frame in which a
    row tone and a column tone each hold ``share`` of the energy mutes the
    NEXT frame (the DSP event comes 20-35 ms before the silent packet).
    """

    def __init__(self, share=MUTE_SHARE, min_dbm0=MIN_FRAME_DBM0):
        self.share = share
        self.min_energy = dbm0(min_dbm0) ** 2 * FRAME
        self.basis = {f: [cmath.exp(-2j * math.pi * f * n / RATE) for n in range(FRAME)]
                      for f in DTMF_ROWS + DTMF_COLUMNS}
        self.frame = []
        self.mute_next = False
        self.triggers = 0
        self.frames = 0

    def score(self, frame):
        """(row share, column share) of one frame's energy."""
        energy = sum(v * v for v in frame)
        if energy < self.min_energy:
            return 0.0, 0.0
        # A sine of amplitude A over N samples is |X| = A*N/2 and energy
        # A*A*N/2, so |X|^2 / (energy * N / 2) is 1 for a pure tone.
        norm = energy * FRAME / 2
        power = {f: abs(sum(v * b for v, b in zip(frame, basis))) ** 2 / norm
                 for f, basis in self.basis.items()}
        return max(power[f] for f in DTMF_ROWS), max(power[f] for f in DTMF_COLUMNS)

    def muted(self, value):
        """Feed one sample of the phone's audio; True if it is muted."""
        mute = self.mute_next
        self.frame.append(value)
        if len(self.frame) == FRAME:
            row, column = self.score(self.frame)
            self.frame = []
            self.frames += 1
            self.mute_next = min(row, column) >= self.share
            self.triggers += self.mute_next
        return mute


class RateMute:
    """Mutes at ``per_second`` on average, whatever is on the line: a
    20 ms frame muted with that probability per frame, seeded."""

    def __init__(self, per_second, seed=0):
        self.chance = per_second * FRAME / RATE
        self.rng = random.Random(seed)
        self.count = 0
        self.mute = False
        self.triggers = 0
        self.frames = 0

    def muted(self, value):
        if self.count == 0:
            self.mute = self.rng.random() < self.chance
            self.triggers += self.mute
            self.frames += 1
        self.count = (self.count + 1) % FRAME
        return self.mute


class _Echo:
    def __init__(self, delay_ms, loss_db):
        self.delay = int(round(delay_ms * RATE / 1000))
        self.gain = 10 ** (-loss_db / 20)
        self.history = deque([0] * self.delay)

    def push(self, value):
        self.history.append(value)

    def pop(self):
        return self.gain * self.history.popleft() if self.history else 0.0


class Line:
    """Both directions of a call through an ATA.

    ``to_hub(phone_pcm)`` is what our modem hears; ``to_phone(hub_pcm)``
    what the phone hears. Each direction keeps its own sample count; the
    echo of one direction is the other's history, so call both in step
    (equal lengths, either order).

    - ``mute``: a ``DTMFMute`` or ``RateMute``, phone-to-us only (the
      ATA's own RTP); muted samples are digital zero;
    - ``noise_dbm0``: white noise added to both directions;
    - ``hub_echo``: (delay ms, loss dB) of OUR signal coming back to us;
    - ``phone_echo``: (delay ms, loss dB) of the phone's signal coming back
      to the phone;
    - ``loss_db``: flat loss each way;
    - ``codec``: mu-law both ways (default on).
    """

    def __init__(self, mute=None, noise_dbm0=None, hub_echo=None, phone_echo=None,
                 loss_db=0.0, codec=True, seed=0):
        self.mute = mute
        self.noise = dbm0(noise_dbm0) if noise_dbm0 is not None else 0.0
        self.hub_echo = _Echo(*hub_echo) if hub_echo else None
        self.phone_echo = _Echo(*phone_echo) if phone_echo else None
        self.gain = 10 ** (-loss_db / 20)
        self.codec = codec
        self.rng = random.Random(seed)
        self.muted_samples = 0

    def _noise(self):
        return self.rng.gauss(0, self.noise) if self.noise else 0.0

    def to_hub(self, samples):
        out = []
        for v in samples:
            if self.phone_echo:
                self.phone_echo.push(v)
            # Our own signal comes back off the ATA's hybrid and is in the
            # audio its DSP looks at, so it is in the frame the mute judges.
            heard = v * self.gain + (self.hub_echo.pop() if self.hub_echo else 0.0) + self._noise()
            if self.mute is not None and self.mute.muted(heard):
                self.muted_samples += 1
                heard = 0
            out.append(heard)
        return g711(out) if self.codec else [int(round(v)) for v in out]

    def to_phone(self, samples):
        out = []
        for v in samples:
            if self.hub_echo:
                self.hub_echo.push(v)
            out.append(v * self.gain + (self.phone_echo.pop() if self.phone_echo else 0.0) + self._noise())
        return g711(out) if self.codec else [max(-32768, min(32767, int(round(v)))) for v in out]

    def snapshot(self):
        return {"mutes": getattr(self.mute, "triggers", 0), "muted_ms": self.muted_samples * 1000 // RATE,
                "frames_judged": getattr(self.mute, "frames", 0)}


class ImpairedPeer:
    """A byte-PCM answer peer (queue/render/receive) behind a ``Line``.

    The firmware-line harness renders our audio and then feeds back the
    phone's, a frame at a time; this passes each across the line. Anything
    else (``mode``, ``snapshot``) is the peer's own.
    """

    def __init__(self, peer, line):
        self.peer = peer
        self.line = line

    def __getattr__(self, name):
        return getattr(self.peer, name)

    def render(self, samples=160):
        ours = struct.unpack("<%dh" % samples, self.peer.render(samples))
        return struct.pack("<%dh" % samples, *self.line.to_phone(ours))

    def receive(self, pcm):
        theirs = struct.unpack("<%dh" % (len(pcm) // 2), pcm)
        return self.peer.receive(struct.pack("<%dh" % len(theirs), *self.line.to_hub(theirs)))
