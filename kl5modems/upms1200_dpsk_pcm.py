"""Finite, shaped DPSK PCM bench with explicitly supplied symbol timing.

This reference has no UPMS pin binding, acquired carrier, training state or
clock recovery. It uses V.22 modes i..iv, 600 symbols/s, 75% root raised
cosine shaping truncated to eight symbols and a matched receive filter.
Only PCM crosses the transmitter/receiver boundary. No line bits or TX
phase are supplied to the receiver. Sample time is one rational grid.
"""

import cmath
from functools import lru_cache
from hashlib import sha256
import math
import struct

from .upms1200_dpsk import (
    MAX_BITS, DPSKAsync, DPSKScrambler, async_bits,
    decode_dpsk_symbols, dpsk_phases,
)

SYMBOL_RATE = 600
ROLL_OFF = 0.75
SPAN = 8
PCM_RATES = (8000, 16000, 48000)
CHANNELS = {'originate': 1200, 'answer': 2400}
MAX_CHUNK = 1600


def rrc_pulse(time_symbols):
    """Unit-energy continuous RRC formula, including both removable poles."""
    if (type(time_symbols) not in (int, float)
            or not math.isfinite(time_symbols) or abs(time_symbols) > SPAN / 2):
        raise ValueError('RRC time must be finite and within the eight-symbol span')
    t, b = time_symbols, ROLL_OFF
    if abs(t) < 1e-12:
        return 1 + b * (4 / math.pi - 1)
    if abs(abs(t) - 1 / (4 * b)) < 1e-12:
        return b / math.sqrt(2) * ((1 + 2 / math.pi) * math.sin(math.pi / (4 * b))
                                 + (1 - 2 / math.pi) * math.cos(math.pi / (4 * b)))
    return (math.sin(math.pi * t * (1 - b)) + 4 * b * t * math.cos(math.pi * t * (1 + b))) / (
        math.pi * t * (1 - (4 * b * t) ** 2))


@lru_cache(maxsize=2048)
def _pulse(numerator, sample_rate):
    return rrc_pulse(numerator / sample_rate)


def _clock(channel, sample_rate):
    if (not isinstance(channel, str) or channel not in CHANNELS
            or type(sample_rate) is not int or sample_rate not in PCM_RATES):
        raise ValueError('DPSK PCM needs originate/answer and 8000/16000/48000 Hz')
    return CHANNELS[channel]


def _chunk(count):
    if type(count) is not int or not 1 <= count <= MAX_CHUNK:
        raise ValueError('DPSK PCM chunks must contain 1..1600 samples')


def _total_samples(symbols, rate):
    # Include the last pulse endpoint and a rounded-up final deadline. The
    # receiver finalizes a symbol only after its entire matched window.
    return ((symbols - 1 + SPAN) * rate + SYMBOL_RATE - 1) // SYMBOL_RATE + 1


def _active_symbols(index, sample_rate, symbols):
    tick = index * SYMBOL_RATE
    lower = max(0, (tick - SPAN * sample_rate + sample_rate - 1) // sample_rate)
    upper = min(symbols - 1, tick // sample_rate)
    return range(lower, upper + 1)


class DPSKPCMTransmitter:
    def __init__(self, phases, *, channel='originate', sample_rate=8000,
                 amplitude=6000, carrier_phase=0):
        self.carrier_hz = _clock(channel, sample_rate)
        if (not isinstance(phases, (tuple, list)) or not 2 <= len(phases) <= MAX_BITS // 2 + 1
                or any(type(p) is not int or not 0 <= p <= 3 for p in phases)
                or type(amplitude) is not int or not 1 <= amplitude <= 12000
                or type(carrier_phase) not in (int, float)
                or not math.isfinite(carrier_phase) or abs(carrier_phase) > 2 * math.pi):
            raise ValueError('DPSK TX needs bounded quarter-turn phases, amplitude 1..12000 and finite phase within +/-2pi')
        self.symbols = tuple((1, 1j, -1, -1j)[p] for p in phases)
        self.channel, self.sample_rate = channel, sample_rate
        self.amplitude, self.carrier_phase = amplitude, carrier_phase
        self.total_samples = _total_samples(len(phases), sample_rate)
        self.samples = self.chunks = self.clipped_samples = 0
        self.digest = sha256()

    def read_pcm(self, max_samples=160):
        _chunk(max_samples)
        count = min(max_samples, self.total_samples - self.samples)
        out = bytearray()
        for n in range(self.samples, self.samples + count):
            envelope = sum((self.symbols[k] * _pulse(
                n * SYMBOL_RATE - (k + SPAN // 2) * self.sample_rate, self.sample_rate)
                for k in _active_symbols(n, self.sample_rate, len(self.symbols))), 0j)
            angle = 2 * math.pi * ((n * self.carrier_hz) % self.sample_rate) / self.sample_rate
            sample = round(self.amplitude * (envelope * cmath.exp(1j * (angle + self.carrier_phase))).real)
            self.clipped_samples += not -32768 <= sample <= 32767
            out.extend(struct.pack('<h', max(-32768, min(32767, sample))))
        self.samples += count
        self.chunks += bool(count)
        self.digest.update(out)
        return bytes(out)

    def report(self):
        return {'channel': self.channel, 'carrier_hz': self.carrier_hz,
                'sample_rate': self.sample_rate, 'symbol_rate': SYMBOL_RATE,
                'roll_off': ROLL_OFF, 'rrc_span_symbols': SPAN,
                'rrc_group_delay_symbols': SPAN // 2,
                'symbols_including_reference': len(self.symbols),
                'samples': self.samples, 'total_samples': self.total_samples,
                'pending_samples': self.total_samples - self.samples,
                'clipped_samples': self.clipped_samples, 'pcm_sha256': self.digest.hexdigest(),
                'chunks': self.chunks, 'physical_spectral_mask_verified': False}


class DPSKPCMReceiver:
    """Bounded matched filter using declared burst timing, not TX metadata.

    Each sample contributes to at most nine future symbol accumulators. The
    caller supplies the selected channel and finite symbol count; it never
    supplies transmitted bits, complex symbols or the unknown carrier phase.
    A completed symbol is returned only after its final future input sample.
    """

    def __init__(self, symbol_count, *, channel='originate', sample_rate=8000):
        self.carrier_hz = _clock(channel, sample_rate)
        if type(symbol_count) is not int or not 2 <= symbol_count <= MAX_BITS // 2 + 1:
            raise ValueError('DPSK RX needs 2..8193 timed symbols including a reference')
        self.symbol_count, self.channel, self.sample_rate = symbol_count, channel, sample_rate
        self.total_samples = _total_samples(symbol_count, sample_rate)
        self.samples = self.completed_symbols = self.high_water = 0
        self.accumulators = {}
        self.digest = sha256()
        self.finished = False

    def feed_pcm(self, pcm):
        if not isinstance(pcm, bytes) or len(pcm) % 2:
            raise ValueError('DPSK RX needs complete s16le PCM bytes')
        _chunk(len(pcm) // 2)
        if self.finished or self.samples + len(pcm) // 2 > self.total_samples:
            raise ValueError('DPSK PCM exceeds the declared burst or is already finished')
        out = []
        for (sample,) in struct.iter_unpack('<h', pcm):
            n = self.samples
            angle = 2 * math.pi * ((n * self.carrier_hz) % self.sample_rate) / self.sample_rate
            baseband = 2 * sample * cmath.exp(-1j * angle)
            for k in _active_symbols(n, self.sample_rate, self.symbol_count):
                weight = _pulse(n * SYMBOL_RATE - (k + SPAN // 2) * self.sample_rate,
                                self.sample_rate)
                self.accumulators[k] = self.accumulators.get(k, 0j) + baseband * weight
            self.high_water = max(self.high_water, len(self.accumulators))
            self.samples += 1
            while (self.completed_symbols < self.symbol_count
                   and n * SYMBOL_RATE >= (self.completed_symbols + SPAN) * self.sample_rate):
                out.append(self.accumulators.pop(self.completed_symbols, 0j))
                self.completed_symbols += 1
        self.digest.update(pcm)
        return tuple(out)

    def finish(self):
        if self.samples != self.total_samples or self.completed_symbols != self.symbol_count:
            raise ValueError('DPSK RX has unfinished samples or matched-filter symbols')
        self.finished = True

    def report(self):
        return {'channel': self.channel, 'carrier_hz': self.carrier_hz,
                'sample_rate': self.sample_rate, 'samples': self.samples,
                'total_samples': self.total_samples, 'completed_symbols': self.completed_symbols,
                'expected_symbols': self.symbol_count,
                'pending_samples': self.total_samples - self.samples,
                'active_symbol_accumulators': len(self.accumulators),
                'accumulator_high_water': self.high_water,
                'pcm_sha256': self.digest.hexdigest(), 'finished': self.finished,
                'symbol_timing': 'explicit reference fixture',
                'carrier_acquisition_implemented': False, 'clock_recovery_implemented': False}


def probe_dpsk_pcm(payload=b'hello', *, channel='originate', sample_rate=8000,
                   chunk_samples=160, carrier_phase=0, invert_polarity=False,
                   stop_bits=2):
    """Async 8N2 (the 1D's 11-bit characters) or 8N1 -> scramble -> shaped PCM -> measured symbols -> host bytes.

    One extra leading idle mark is declared when the framed bits are odd, to retain
    complete UART frames with an even line-bit count. Nothing is appended
    to a half dibit invisibly. There is no ROM or actual chip in this probe.
    """
    _clock(channel, sample_rate)
    _chunk(chunk_samples)
    if type(invert_polarity) is not bool:
        raise ValueError('line polarity must be a boolean')
    frame_bits = 9 + stop_bits if stop_bits in (1, 2) else 0
    idle_bits = 32 + ((len(payload) * frame_bits) % 2 if isinstance(payload, bytes) else 0)
    framed = async_bits(payload, stop_bits=stop_bits, idle_bits=idle_bits)
    scrambler = DPSKScrambler()
    phases = dpsk_phases(scrambler.feed(framed))
    tx = DPSKPCMTransmitter(phases, channel=channel, sample_rate=sample_rate,
                          carrier_phase=carrier_phase)
    rx = DPSKPCMReceiver(len(phases), channel=channel, sample_rate=sample_rate)
    symbols = []
    while True:
        pcm = tx.read_pcm(chunk_samples)
        if not pcm:
            break
        if invert_polarity:
            # The fixed 6000-amplitude reference is below clipping. Treat a
            # future clipped -32768 sample as an error, not a hidden saturation.
            pcm = b''.join(struct.pack('<h', -value) for (value,) in struct.iter_unpack('<h', pcm))
        symbols.extend(rx.feed_pcm(pcm))
    rx.finish()
    descrambler = DPSKScrambler(descramble=True)
    framer = DPSKAsync()  # one stop bit checked, as a UART receives
    recovered = framer.feed(descrambler.feed(decode_dpsk_symbols(symbols)))
    return {'scope': 'standard DPSK PCM reference with declared timing; no UPMS chip or call',
            'payload_bytes': len(payload), 'expected_payload_sha256': sha256(payload).hexdigest(),
            'received_payload_hex': recovered.hex(), 'received_payload_sha256': sha256(recovered).hexdigest(),
            'payload_verified': recovered == payload, 'leading_idle_bits': idle_bits,
            'invert_polarity': invert_polarity, 'transmitter': tx.report(), 'receiver': rx.report(),
            'scrambler': scrambler.report(), 'descrambler': descrambler.report(),
            'framing': framer.report(), 'upms_firmware_binding_implemented': False,
            'training_implemented': False, 'physical_interoperability_verified': False}
