"""The Protel native 1200 FSK: the modulation only a Protel modem speaks.

The Protel 7000 and the UPMS1200 modem both put it on the line from the CPU,
through the G65SC150 CTU's column tone generator:

- a zero cell is column reload $38 (56), a one cell $1E (30); in K=52 tone
  mode that is OSC / (52 * (reload + 2)): 1186.9 Hz and 2151.2 Hz;
- a cell is four timer entries of reload 185 at OSC/4, so 748 CPU cycles:
  1196.4 baud (the 7000's serializer counts 743, and 770 at a byte edge);
- characters are 8N1, least significant bit first; idle is a one.

Both frequencies and the rate follow the manufacturer's stated formula at
the reference 3.579545 MHz oscillator; the frequency table's N+1 reading
(1208/2221 Hz) is an open question.
Nothing here is calibrated against a board. It was checked, in both
directions, against the UPMS and 7000 firmware running in an emulator;
that emulator and the firmware are not part of this repository.

Plain Python, 8 kHz signed 16-bit, streaming: a modulator fed bytes and a
receiver that finds its own start bits, the way a UART does.
"""

from __future__ import annotations

import cmath
from collections import deque
import math

OSC_HZ = 3_579_545
CPU_HZ = OSC_HZ / 4                     # the UPMS/7000 CPU clock: OSC/4
RATE = 8000
ZERO_RELOAD, ONE_RELOAD = 0x38, 0x1E
ZERO_HZ = OSC_HZ / (52 * (ZERO_RELOAD + 2))      # 1186.9
ONE_HZ = OSC_HZ / (52 * (ONE_RELOAD + 2))        # 2151.2
CELL_CYCLES = 748
BAUD = CPU_HZ / CELL_CYCLES                      # 1196.4
CENTRE_HZ = (ZERO_HZ + ONE_HZ) / 2


def frame_bits(data):
    """8N1, LSB first: the bits a UPMS or a 7000 puts on the line."""
    for value in bytes(data):
        yield 0
        for n in range(8):
            yield (value >> n) & 1
        yield 1


def lowpass(count, cutoff_hz):
    """Hamming-windowed sinc, unity gain at DC."""
    taps = []
    for k in range(count):
        x = k - (count - 1) / 2
        s = 2 * cutoff_hz / RATE if x == 0 else math.sin(2 * math.pi * cutoff_hz * x / RATE) / (math.pi * x)
        taps.append(s * (0.54 - 0.46 * math.cos(2 * math.pi * k / (count - 1))))
    total = sum(taps)
    return [t / total for t in taps]


class Modulator:
    """Bits in, phase-continuous native FSK out, on one sample clock.

    The CTU changes frequency at its next carry and never resets phase, so
    a phase-continuous sine is the ideal of its 26-step output. Cells are
    placed on the exact rational grid (CELL_CYCLES at CPU_HZ), so the
    rate never drifts from the firmware's.

    ``keyed`` is how the UPMS sends: silence between frames, and each frame
    led in by ``LEAD_CELLS`` of mark (its reply to a data frame, read off
    its own column output, is 64 marks and then the frame). Unkeyed, the
    idle line is a continuous mark, as the 7000's serializer leaves it.
    """

    LEAD_CELLS = 64
    TAIL_CELLS = 2

    def __init__(self, amplitude=12000, baud=BAUD, keyed=False):
        self.keyed = keyed
        self.on = not keyed
        self.amplitude = amplitude
        self.samples_per_cell = RATE / baud
        self.bits = deque()
        self.phase = 0.0
        self.sample = 0                 # next sample to produce
        self.cell = 0                   # index of the cell sample `sample` is in
        self.level = 1                  # what the current cell sends
        self.cells_sent = 0

    def queue_bits(self, bits):
        self.bits.extend(1 if b else 0 for b in bits)

    def mark(self, cells):
        """Key up and send ``cells`` of mark: an answer, not a frame."""
        self.on = True
        self.queue_bits([1] * cells)

    def queue(self, data):
        if self.keyed and not self.on:
            self.on = True
            self.queue_bits([1] * self.LEAD_CELLS)
        self.queue_bits(frame_bits(data))
        if self.keyed:
            self.queue_bits([1] * self.TAIL_CELLS)

    @property
    def idle(self):
        """Nothing queued: an unkeyed line is sending mark, a keyed one
        has fallen silent or will at the end of this cell."""
        return not self.bits

    def pcm(self, count):
        out = []
        for _ in range(count):
            cell = int(self.sample / self.samples_per_cell)
            while self.cell < cell:
                self.cell += 1
                if self.bits:
                    self.level = self.bits.popleft()
                    self.cells_sent += 1
                else:
                    self.level = 1
                    self.on = not self.keyed
            hz = ONE_HZ if self.level else ZERO_HZ
            out.append(int(round(self.amplitude * math.sin(self.phase))) if self.on else 0)
            self.phase = (self.phase + 2 * math.pi * hz / RATE) % (2 * math.pi)
            self.sample += 1
        return out


class Demodulator:
    """Native FSK PCM in, 8N1 characters out.

    A frequency discriminator at the centre of the two tones (the sign of
    the phase step of the signal mixed down to 1669 Hz, through a 13-tap
    1 kHz lowpass that removes the mixing image) gives a line level per
    sample; a UART on that level finds each start bit itself, to a
    fraction of a sample, and samples the cells at their centres. One stop
    bit is checked and any further marks are idle, so the 7000's longer
    byte-edge cells read the same as the UPMS's.

    ``carrier`` follows signal power with a short hold, for a caller that
    wants to know the line is up; the UART does not depend on it.
    """

    # A 4-sample boxcar left enough of the image at 2856 Hz that the phase
    # step had to be averaged over two samples to read at all, and still lost
    # a fifth of the bytes at 11 dB SNR; this filter loses none there.
    TAPS = 13
    CUTOFF_HZ = 1000
    MIN_RMS = 300           # PCM units: below this the line is silence
    HOLD = 160              # samples of silence before carrier is called off

    def __init__(self, baud=BAUD):
        self.samples_per_cell = RATE / baud
        self.mix = [cmath.exp(-2j * math.pi * CENTRE_HZ * n / RATE) for n in range(RATE)]
        self.taps = lowpass(self.TAPS, self.CUTOFF_HZ)
        self.window = deque([0j] * self.TAPS, maxlen=self.TAPS)
        self.prev = None
        self.disc = 0.0
        self.n = 0
        self.level = 1
        self.last_value = None          # previous discriminator value, for interpolation
        self.power = 0.0
        self.quiet = 0
        self.carrier = False
        self.start = None               # time of the current frame's start edge
        self.bits = []
        self.frames = self.framing_errors = 0

    def feed(self, samples):
        out = bytearray()
        for value in samples:
            self.power += (value * value - self.power) * 0.05
            if self.power >= self.MIN_RMS ** 2:
                self.quiet = 0
                self.carrier = True
            else:
                self.quiet += 1
                if self.quiet >= self.HOLD:
                    self.carrier = False
            self.window.append(value * self.mix[self.n % RATE])
            acc = sum(h * z for h, z in zip(self.taps, self.window))
            if self.prev is not None and acc and self.prev:
                self.disc = cmath.phase(acc * self.prev.conjugate())
            self.prev = acc
            # The tone's own envelope gates the UART: a tone of amplitude A
            # is A/2 here, at once, where the power average above takes a
            # few cells to fall and framed noise as characters meanwhile.
            # Below half the carrier level the line reads as mark.
            self._step(self.n, self.disc if abs(acc) >= self.MIN_RMS / 3 else 1.0, out)
            self.n += 1
        return bytes(out)

    def _step(self, n, d, out):
        level = 1 if d > 0 else 0
        if self.start is None:
            if self.level == 1 and level == 0:
                # Start edge: where the discriminator crossed zero, between
                # the previous sample and this one.
                prev = self.last_value
                frac = prev / (prev - d) if prev is not None and prev > 0 and prev != d else 0.0
                self.start = n - 1 + frac
                self.start_checked = False
                self.bits = []
        elif not self.start_checked:
            # The start bit is checked once, half a cell in, as a UART does:
            # a glitch that is mark again by then was not a character.
            if n >= self.start + 0.5 * self.samples_per_cell:
                if level == 1:
                    self.start = None
                else:
                    self.start_checked = True
        else:
            k = len(self.bits)
            if n >= self.start + (k + 1.5) * self.samples_per_cell:
                self.bits.append(level)
                if k + 1 == 9:          # eight data bits and the stop
                    data, stop = self.bits[:8], self.bits[8]
                    self.start = None
                    if stop:
                        self.frames += 1
                        out.append(sum(b << i for i, b in enumerate(data)))
                    else:
                        self.framing_errors += 1
        self.level = level
        self.last_value = d
