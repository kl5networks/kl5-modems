"""Streaming Bell 212A / V.22 DPSK at 8 kHz: the modem the 1D's board has.

On the 1D's DD8822 a call-in at 1200 selects the 73K212L's DPSK mode (CR0
0x1D/0x1F/0x0F). This is the line side of that chip: a modulator fed one line bit at a time, and a receiver
that finds the carrier, recovers symbol timing itself and hands back bits.

Unlike the reference in upms1200_dpsk_pcm.py, nothing here is told when a
burst starts, how long it is, or where the symbols fall: a live call has
none of that. The reference stays the oracle the tests hold this against.

Standard pieces only: 600 baud, dibits mapped by V.22 Table 1 as phase
steps, the 1+x^-14+x^-17 scrambler, 75% root-raised-cosine shaping. The
originating modem transmits on 1200 Hz and receives 2400 Hz. Plain Python,
no numpy, so it runs under PyPy.
"""

from __future__ import annotations

from collections import deque
import cmath
import math

from .upms1200_dpsk import PHASE_STEPS
from .upms1200_dpsk_pcm import SPAN, rrc_pulse

RATE = 8000
BAUD = 600
BIT_RATE = 1200
ORIGINATE_HZ = 1200
ANSWER_HZ = 2400
# 8000/600 samples per symbol is 40/3: sample n sits 3n/40 symbols in.
_SPS_NUM, _SPS_DEN = 40, 3
SPS = _SPS_NUM / _SPS_DEN
_HALF_SPAN = SPAN // 2                       # symbols each side of a pulse
_TX_TAPS = {m: rrc_pulse(m / _SPS_NUM) for m in range(-_SPS_NUM * _HALF_SPAN, _SPS_NUM * _HALF_SPAN + 1)}
_FRAC = 32                                   # matched-filter table resolution, per sample
_RX_HALF = int(_HALF_SPAN * SPS)             # 53 samples
_RX_TABLE = [rrc_pulse(max(-_HALF_SPAN, min(_HALF_SPAN, (i / _FRAC - _RX_HALF) / SPS)))
             for i in range(2 * _RX_HALF * _FRAC + 1)]
_STEP_PHASE = [cmath.exp(1j * math.pi / 2 * k) for k in range(4)]


def _carrier_table(hz):
    period = RATE // math.gcd(RATE, hz)
    return [cmath.exp(2j * math.pi * hz * n / RATE) for n in range(period)]


class Scrambler:
    """V.22 1+x^-14+x^-17, self-synchronising, with the 64-ones guard.

    Same arithmetic as upms1200_dpsk.DPSKScrambler (the tests hold them
    together) without its batch validation, because this one runs per bit
    on a live line.
    """

    def __init__(self, descramble=False):
        self.descramble = descramble
        self.state = 0
        self.ones = 0

    def bit(self, value):
        invert = self.ones == 64
        result = value ^ ((self.state >> 13) & 1) ^ ((self.state >> 16) & 1) ^ invert
        line = value if self.descramble else result
        self.state = ((self.state << 1) | line) & 0x1FFFF
        self.ones = (self.ones + 1 if line else 0) if not invert else int(bool(line))
        return result


class Modulator:
    """Line bits in (1200 bit/s), shaped DPSK PCM out on one carrier.

    ``push_bit`` takes the bit the chip would see on TXD; scrambling is on
    unless ``scramble=False`` (CR1.D4, and the unscrambled-ones handshake).
    ``pcm(count)`` returns the next ``count`` samples; a sample is only
    finished once every symbol within the pulse span is known, so the
    caller keeps feeding bits ahead of the audio it takes (4 symbols, 6.7 ms).
    """

    def __init__(self, carrier_hz=ORIGINATE_HZ, amplitude=6000):
        self.carrier = _carrier_table(carrier_hz)
        self.amplitude = amplitude
        self.scrambler = Scrambler()
        self.scramble = True
        self.phase = 0
        self.pending = None
        self.symbols = []                    # complex, index = symbol number
        self.first_symbol = 0                # index of self.symbols[0]
        self.sample = 0                      # next sample to produce

    def push_bit(self, bit):
        bit = 1 if bit else 0
        if self.scramble:
            bit = self.scrambler.bit(bit)
        if self.pending is None:
            self.pending = bit
            return
        self.phase = (self.phase + PHASE_STEPS[self.pending * 2 + bit]) & 3
        self.pending = None
        self.symbols.append(_STEP_PHASE[self.phase])

    def ready(self):
        """Samples that can be produced now."""
        known = self.first_symbol + len(self.symbols)
        # sample n needs symbols up to floor(3n/40) + HALF_SPAN.
        last = ((known - 1 - _HALF_SPAN) * _SPS_NUM) // _SPS_DEN
        return max(0, last - self.sample + 1)

    def pcm(self, count):
        out = []
        carrier, period = self.carrier, len(self.carrier)
        for _ in range(count):
            n = self.sample
            centre = (n * _SPS_DEN) // _SPS_NUM
            acc = 0j
            for k in range(centre - _HALF_SPAN, centre + _HALF_SPAN + 2):
                i = k - self.first_symbol
                if 0 <= i < len(self.symbols):
                    m = n * _SPS_DEN - k * _SPS_NUM
                    tap = _TX_TAPS.get(m)
                    if tap:
                        acc += self.symbols[i] * tap
            value = (acc * carrier[n % period]).real * self.amplitude
            out.append(max(-32767, min(32767, int(round(value)))))
            self.sample += 1
        # Symbols wholly behind the pulse span are never needed again.
        drop = (self.sample * _SPS_DEN) // _SPS_NUM - _HALF_SPAN - 2 - self.first_symbol
        if drop > 0:
            del self.symbols[:drop]
            self.first_symbol += drop
        return out


class Demodulator:
    """Streaming DPSK receiver: carrier detect, Gardner timing, differential bits.

    ``feed(samples)`` takes signed 16-bit samples and returns a list of
    ``(sample_time, bit)`` for every bit decided, ``sample_time`` being the
    symbol's centre in samples since the first ``feed``. Bits come out
    descrambled unless ``descramble=False``. ``carrier`` follows the chip's
    DR.D3 rules: on after ``on_ms`` above threshold, off after ``off_ms``
    below it (73K302L: 15-32 ms on, 10-24 ms off for 212A).

    No equaliser: the line is digital end to end (AudioSocket, an ATA
    codec), and nothing measured asks for one.
    """

    # The matched filter's gain is about 9.5 on a DPSK signal, and 0 dBm0 is
    # an RMS of about 16050 in 16-bit PCM (3.14 dBm0 full scale). So -43
    # dBm0 at the input, the chip's 212A turn-on (-49..-42), reads about
    # 1000 here, and -48 dBm0, its turn-off, about 600.
    ON_LEVEL = 1000.0
    OFF_LEVEL = 600.0
    GAIN = 0.02           # Gardner loop, proportional, in samples per unit error
    LEVEL_TC = 0.5        # power average per symbol: a few ms, inside the chip's hold

    def __init__(self, carrier_hz=ANSWER_HZ, descramble=True, on_ms=20, off_ms=15):
        self.carrier = [c.conjugate() for c in _carrier_table(carrier_hz)]
        self.descrambler = Scrambler(descramble=True) if descramble else None
        self.base = deque()                  # mixed-down samples, [base_index...]
        self.base_index = 0
        self.received = 0                    # samples fed so far
        self.next_time = float(_RX_HALF + 1)  # next evaluation instant, samples
        self.half = 0                        # 0 = on-time evaluation, 1 = midpoint
        self.prev_on = None
        self.mid = None
        self.power = 0.0
        self.carrier_on = False
        self._above = self._below = 0
        self.on_symbols = max(1, round(on_ms * BAUD / 1000))
        self.off_symbols = max(1, round(off_ms * BAUD / 1000))
        self.carrier_events = []             # (sample_time, on)
        self.symbols = 0

    def _filtered(self, t):
        start = math.ceil(t - _RX_HALF)
        stop = math.floor(t + _RX_HALF)
        acc = 0j
        base, offset = self.base, self.base_index
        for n in range(start, stop + 1):
            i = n - offset
            if 0 <= i < len(base):
                acc += base[i] * _RX_TABLE[int(round((n - t + _RX_HALF) * _FRAC))]
        return acc

    def feed(self, samples):
        carrier, period = self.carrier, len(self.carrier)
        for value in samples:
            self.base.append(value * carrier[self.received % period])
            self.received += 1
        bits = []
        while self.next_time + _RX_HALF + 1 < self.received:
            y = self._filtered(self.next_time)
            if self.half:
                self.mid = y
            else:
                self._symbol(y, bits)
            self.half ^= 1
            self.next_time += SPS / 2
        # Keep only what the next evaluation's span can reach.
        keep_from = int(self.next_time) - _RX_HALF - 2
        while self.base and self.base_index < keep_from:
            self.base.popleft()
            self.base_index += 1
        return bits

    def _symbol(self, y, bits):
        t = self.next_time
        self.symbols += 1
        level = abs(y) ** 2
        self.power += (level - self.power) * self.LEVEL_TC
        rms = math.sqrt(self.power)
        if rms >= self.ON_LEVEL:
            self._above += 1
            self._below = 0
        elif rms < self.OFF_LEVEL:
            self._below += 1
            self._above = 0
        if not self.carrier_on and self._above >= self.on_symbols:
            self.carrier_on = True
            self.carrier_events.append((t, True))
        elif self.carrier_on and self._below >= self.off_symbols:
            self.carrier_on = False
            self.carrier_events.append((t, False))
        prev = self.prev_on
        self.prev_on = y
        if prev is None or self.mid is None or self.power <= 0:
            return
        # Gardner: the midpoint sits on the zero crossing when timing is right.
        error = ((prev - y) * self.mid.conjugate()).real / self.power
        self.next_time += max(-0.5, min(0.5, self.GAIN * error * SPS))
        if not self.carrier_on:
            return
        step = round(cmath.phase(y * prev.conjugate()) / (math.pi / 2)) & 3
        first, second = divmod(PHASE_STEPS.index(step), 2)
        for bit in (first, second):
            if self.descrambler is not None:
                bit = self.descrambler.bit(bit)
            bits.append((t, bit))


class AsyncDeframer:
    """Line bits in, 8N1 characters out, the way a UART reads them.

    One stop bit is checked and any further marks are idle, so it reads the
    UPMS's 8N1 and the 1D's 11-bit characters alike (upms1200_dpsk.DPSKAsync
    is the batch version the tests hold it against).
    """

    def __init__(self):
        self.frame = None
        self.frames = self.framing_errors = 0

    def bit(self, value):
        if self.frame is None:
            if not value:
                self.frame = []
            return None
        self.frame.append(value)
        if len(self.frame) < 9:
            return None
        frame, self.frame = self.frame, None
        if not frame[8]:
            self.framing_errors += 1
            return None
        self.frames += 1
        return sum(bit << n for n, bit in enumerate(frame[:8]))
