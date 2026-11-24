"""Bounded standard DPSK reference primitives, not an attached UPMS chip.

The phase table is ITU-T V.22 (11/88), Table 1 modes i..iv, and CML
CMX644 D/644/3, TX DATA BYTE. Mode v has a different table and is excluded.
Scrambling follows V.22 section 5. Physical UPMS identity, training, rate
conversion and pin binding remain separate evidence gates.
"""

import cmath
import math

MAX_BITS = 16384
MAX_PAYLOAD = 1024
PHASE_STEPS = (1, 0, 2, 3)  # first bit is the left/MSB bit of the dibit
STEP_DIBITS = ((0, 1), (0, 0), (1, 0), (1, 1))


def _bits(bits, *, even=False):
    if (not isinstance(bits, (list, tuple)) or not 1 <= len(bits) <= MAX_BITS
            or any(type(b) is not int or b not in (0, 1) for b in bits)
            or even and len(bits) % 2):
        raise ValueError('DPSK needs 1..16384 binary integer bits; dibits need an even count')
    return tuple(bits)


class DPSKScrambler:
    """Self-synchronizing 1+x^-14+x^-17, with explicit lockup policy.

    Bit zero of state is the most recent scrambled line bit. At 64 line
    ones, the next input (TX) or output (RX) is inverted. Disable this
    policy during training as required by V.22; this class has no handshake.
    Unknown receiver history converges after 17 correct received bits.
    """

    def __init__(self, *, descramble=False, state=0, prevent_lockup=True):
        if (type(descramble) is not bool or type(prevent_lockup) is not bool
                or type(state) is not int or not 0 <= state < 1 << 17):
            raise ValueError('scrambler needs boolean policies and a 17-bit state')
        self.descramble = descramble
        self.state = state
        self.prevent_lockup = prevent_lockup
        self.ones = self.processed = self.inversions = 0

    def feed(self, bits):
        values = _bits(bits)  # validate the entire batch before changing state
        out = []
        for bit in values:
            invert = self.prevent_lockup and self.ones == 64
            result = bit ^ ((self.state >> 13) & 1) ^ ((self.state >> 16) & 1) ^ invert
            line = bit if self.descramble else result
            self.state = ((self.state << 1) | line) & 0x1FFFF
            self.ones = self.ones + 1 if line else 0
            if invert:
                self.inversions += 1
                self.ones = int(bool(line))
            self.processed += 1
            out.append(result)
        return tuple(out)

    def report(self):
        return {'polynomial_delays': [14, 17], 'descramble': self.descramble,
                'prevent_lockup': self.prevent_lockup, 'bits': self.processed,
                'lockup_inversions': self.inversions, 'consecutive_line_ones': self.ones}


def dpsk_phases(bits, *, initial_phase=0):
    """Quarter-turn symbol phases, including one explicit reference symbol.

    An odd line bit must be retained by the caller; no padding is invented.
    The reference is a test fixture, not an asserted on-wire training word.
    """
    values = _bits(bits, even=True)
    if type(initial_phase) is not int or not 0 <= initial_phase <= 3:
        raise ValueError('initial differential phase must be 0..3')
    phases = [initial_phase]
    for i in range(0, len(values), 2):
        phases.append((phases[-1] + PHASE_STEPS[2 * values[i] + values[i + 1]]) % 4)
    return tuple(phases)


def decode_dpsk_symbols(symbols, *, max_error_degrees=30):
    """Decode measured complex symbols; reject erasures and ambiguous phases.

    Requires explicit symbol timing and a received reference symbol. Absolute
    carrier phase and whole-line polarity cancel in the differential product.
    This is not a carrier/timing acquisition loop or a physical DCD detector.
    """
    if (not isinstance(symbols, (list, tuple)) or not 2 <= len(symbols) <= MAX_BITS // 2 + 1
            or any(type(z) not in (int, float, complex) or not math.isfinite(z.real)
                   or not math.isfinite(z.imag) or abs(z) < 1e-12 for z in symbols)
            or type(max_error_degrees) not in (int, float)
            or not math.isfinite(max_error_degrees) or not 0 < max_error_degrees < 45):
        raise ValueError('DPSK symbols must be finite, nonzero and bounded; phase tolerance must be 0..45 degrees exclusive')
    out = []
    tolerance = math.radians(max_error_degrees)
    for previous, current in zip(symbols, symbols[1:]):
        # Normalize before multiplication so large, finite measured gains do
        # not overflow the differential product and become a false phase.
        angle = cmath.phase((current / abs(current))
                            * complex(previous / abs(previous)).conjugate())
        step = round(angle / (math.pi / 2))
        if abs(angle - step * math.pi / 2) > tolerance:
            raise ValueError('ambiguous differential phase; no bits delivered for this batch')
        out.extend(STEP_DIBITS[step % 4])
    return tuple(out)


def async_bits(payload, *, stop_bits=1, idle_bits=32):
    """Explicit async reference framing, LSB-first bytes at nominal line rate.

    The two ends do not use the same character length. The 1D's DD8822
    selects 73K212L CR0 0x1D at 1200 (DPSK, 11-bit characters) and its
    CPU-clocked data carries two or more stop cells; the UPMS ROM's 0x1A
    request is 10-bit (8N1). Transmit what the sender sends and receive
    with ``DPSKAsync``, which checks one stop bit as a UART does.

    No asynchronous rate conversion, stop-bit deletion or break extension is
    modeled. These are distinct chip behaviors, not covered by this fixture.
    """
    if stop_bits not in (1, 2) or type(stop_bits) is not int:
        raise ValueError('async framing takes 1 or 2 stop bits')
    if (not isinstance(payload, bytes) or not 1 <= len(payload) <= MAX_PAYLOAD
            or type(idle_bits) is not int or not 0 <= idle_bits <= 256):
        raise ValueError('async fixture needs 1..1024 bytes and 0..256 leading idle bits')
    values = [1] * idle_bits
    for byte in payload:
        values.extend((0, *((byte >> i) & 1 for i in range(8)), *(1,) * stop_bits))
    return tuple(values)


def async_8n2_bits(payload, *, idle_bits=32):
    """The 8N2 reference fixture; see ``async_bits``."""
    return async_bits(payload, stop_bits=2, idle_bits=idle_bits)


class DPSKAsync:
    """Incremental bounded async observation; invalid stops are never repaired.

    The default checks one stop bit and then hunts for the next start, as a
    UART does, so it reads 8N1 and 8N2 alike. ``stop_bits=2`` insists on
    both, and so turns back-to-back 8N1 into framing errors: the trap the
    8N2-only receiver this replaced would have set for a UPMS peer.
    """

    def __init__(self, stop_bits=1):
        if stop_bits not in (1, 2) or type(stop_bits) is not int:
            raise ValueError('async framing takes 1 or 2 stop bits')
        self.stop_bits = stop_bits
        self.frame = []
        self.bits = self.frames = self.valid_bytes = self.framing_errors = 0

    def feed(self, bits):
        values = _bits(bits)
        out = bytearray()
        length = 9 + self.stop_bits
        for bit in values:
            self.bits += 1
            if not self.frame:
                if bit == 0:
                    self.frame = [0]
                continue
            self.frame.append(bit)
            if len(self.frame) == length:
                self.frames += 1
                if all(self.frame[9:]):
                    out.append(sum(b << i for i, b in enumerate(self.frame[1:9])))
                    self.valid_bytes += 1
                else:
                    self.framing_errors += 1
                self.frame = []
        return bytes(out)

    def report(self):
        return {'format': '8N%d' % self.stop_bits, 'bits': self.bits, 'frames': self.frames,
                'valid_bytes': self.valid_bytes, 'framing_errors': self.framing_errors,
                'pending_frame_bits': len(self.frame), 'rate_conversion_implemented': False}


class DPSKAsync8N2(DPSKAsync):
    """The 8N2 reference receiver; see ``DPSKAsync``."""

    def __init__(self):
        super().__init__(stop_bits=2)
