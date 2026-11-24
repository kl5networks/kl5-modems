"""Experimental Bell 103 PCM receive model at the 73K212L chip boundary.

CR0.D5..D2=1100 selects FSK; D0 selects receive band. Only normal Bell 103
is supported. This is a calibrated-in-PCM software approximation, not chip
filter/AGC fidelity or the proprietary Protel 1200 baud modulation.
"""

from collections import deque
import cmath
import math

from .pcm import RATE


class Bell103Receiver:
    WINDOW = 24
    ACQUIRE = 120  # 15 ms of qualifying PCM
    HOLD = 192  # 24 ms of supplied nonqualifying PCM
    MIN_RMS = 100  # PCM units; no dBm/physical calibration is claimed
    MIN_COHERENCE = .8
    DETECT_MASK = 0x28  # carrier D3 and received-data D5 only
    SOURCE = 'experimental Bell 103 PCM demodulator'

    def __init__(self, modem):
        self.modem = modem
        self.mode_key = None
        self.total_samples = self.acquisitions = self.losses = self.transitions = 0
        self.reset_stream()
        self._configure()

    def reset_stream(self):
        if getattr(self, 'carrier', False):
            self.losses += 1
        self.samples = deque()
        self.sum_x = self.sum_x2 = 0
        self.spectra = [0j, 0j]
        self.good = self.bad = 0
        self.carrier = False
        self.level = 1
        self.coherence = self.rms = 0
        self.modem.rxd_level = 1
        self._detect(0)

    def _detect(self, bits):
        if self.modem.detect & self.DETECT_MASK != bits:
            self.modem.set_detect(bits, self.DETECT_MASK, self.SOURCE)

    def _configure(self):
        m = self.modem
        key = (m.cr0 & 0xFD, m.cr1 & 3, m.reset_generation)
        if key == self.mode_key:
            return
        self.mode_key = key
        self.reset_stream()
        self.supported = (m.cr0 & 0xFC) == 0x30 and not m.cr1 & 3
        self.band = 'high' if m.cr0 & 1 else 'low'
        self.frequencies = (2225, 2025) if m.cr0 & 1 else (1270, 1070)
        self.filters = []
        for hz in self.frequencies:
            omega = 2 * math.pi * hz / RATE
            rotate = cmath.exp(-1j * omega)
            cosines = [math.cos(omega * k) for k in range(self.WINDOW)]
            sines = [math.sin(omega * k) for k in range(self.WINDOW)]
            mean_c, mean_s = sum(cosines) / self.WINDOW, sum(sines) / self.WINDOW
            cc = sum((x - mean_c)**2 for x in cosines)
            ss = sum((x - mean_s)**2 for x in sines)
            cs = sum((c - mean_c) * (s - mean_s) for c, s in zip(cosines, sines))
            self.filters.append((rotate, rotate**self.WINDOW, mean_c, mean_s,
                                 cc, ss, cs, cc * ss - cs * cs))

    def sync(self):
        self._configure()

    def sample(self, value):
        self._configure()
        self.total_samples += 1
        if not self.supported:
            return
        old = self.samples.popleft() if len(self.samples) == self.WINDOW else 0
        self.samples.append(value)
        self.sum_x += value - old
        self.sum_x2 += value * value - old * old
        for i, (rotate, tail, *_) in enumerate(self.filters):
            self.spectra[i] = value + rotate * self.spectra[i] - old * tail
        if len(self.samples) < self.WINDOW:
            return
        power = max(0, self.sum_x2 - self.sum_x * self.sum_x / self.WINDOW)
        self.rms = math.sqrt(power / self.WINDOW)
        fits = []
        for spectrum, (_, _, mc, ms, cc, ss, cs, determinant) in zip(self.spectra, self.filters):
            yc = spectrum.real - self.sum_x * mc
            ys = -spectrum.imag - self.sum_x * ms
            fit = (ss * yc * yc - 2 * cs * yc * ys + cc * ys * ys) / determinant
            fits.append(max(0, min(1, fit / power)) if power else 0)
        self.coherence = max(fits)
        valid = self.rms >= self.MIN_RMS and self.coherence >= self.MIN_COHERENCE
        if valid:
            self.good += 1
            self.bad = 0
            if not self.carrier and self.good >= self.ACQUIRE:
                self.carrier = True
                self.acquisitions += 1
            new_level = int(fits[0] >= fits[1])
            if new_level != self.level:
                self.transitions += 1
            self.level = new_level
        else:
            self.good = 0
            self.bad += 1
            if self.carrier and self.bad >= self.HOLD:
                self.carrier = False
                self.losses += 1
                self.level = 1
        self.modem.rxd_level = self.level if self.carrier else 1
        self._detect((0x08 | (self.level << 5)) if self.carrier else 0)

    def snapshot(self):
        return {'mode': 'Bell 103 FSK' if self.supported else 'unsupported or powered down',
                'receive_band': self.band, 'mark_space_hz': list(self.frequencies),
                'carrier': self.carrier, 'rxd_level': self.modem.rxd_level,
                'rxd_tristated': bool(self.modem.tone & 0x80),
                'samples': self.total_samples, 'acquisitions': self.acquisitions,
                'losses': self.losses, 'data_transitions': self.transitions,
                'rms_pcm': round(self.rms, 2), 'coherence': round(self.coherence, 4),
                'window_samples': self.WINDOW, 'minimum_rms_pcm': self.MIN_RMS,
                'minimum_coherence': self.MIN_COHERENCE,
                'acquire_samples': self.ACQUIRE, 'hold_samples': self.HOLD,
                'physical_calibration_verified': False, 'chip_filter_fidelity_verified': False,
                'proprietary_1200_supported': False}
