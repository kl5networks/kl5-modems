"""Bounded answer-side Bell 103 PCM/DTE adapter for the standalone lab.

The peer decodes PCM at sample-time UART centers and generates continuous
high-band FSK from a manager byte stream. It never reads firmware PCs, UART
observers, configuration or credentials, and never executes a phone CPU.
"""

from collections import deque
import math
import struct

from . import dpsk212
from .protel_bell103 import Bell103Receiver
from .protel_devices import SSI73K212L
from .pcm import RATE, validate_pcm


class Bell103UART:
    """Incremental 300-baud 8N1 decoder; storage is one unfinished frame."""

    def __init__(self):
        self.samples = self.frames = self.framing_errors = self.false_starts = 0
        self.previous = 1
        self.start = None
        self.bit = self.value = 0

    def sample(self, level):
        n = self.samples
        self.samples += 1
        out = None
        if self.start is None:
            if self.previous and not level:
                self.start, self.bit, self.value = n, -1, 0
        elif n - self.start == round((self.bit + 1.5) * RATE / 300):
            if self.bit == -1 and level:
                self.false_starts += 1
                self.start = None
            elif self.bit == 8:
                if level:
                    out = self.value
                    self.frames += 1
                else:
                    self.framing_errors += 1
                self.start = None
            else:
                if self.bit >= 0:
                    self.value |= level << self.bit
                self.bit += 1
        self.previous = level
        return out


class Bell103AnswerPeer:
    QUEUE_BYTES = 4096
    AMPLITUDE = 6000  # lab PCM units, no physical level calibration

    def __init__(self):
        self.modem = SSI73K212L(deque(maxlen=64))
        self.modem.write_register(0, 0x30)  # answer receives low-band FSK
        self.receiver = Bell103Receiver(self.modem)
        self.uart = Bell103UART()
        self.bytes = deque()
        self.word = ()
        self.word_bit = self.preamble = 0
        self.idle = True
        self.phase = self.remaining = 0.0
        self.level = 1
        self.eof = False
        self.tail = None
        self.generated_samples = self.received_samples = self.queued_bytes = 0

    def queue(self, data):
        if self.eof:
            raise ValueError('manager byte stream is closed')
        data = bytes(data)
        if len(data) + len(self.bytes) > self.QUEUE_BYTES:
            raise ValueError('manager byte queue exceeds laboratory limit')
        self.bytes.extend(data)
        self.queued_bytes += len(data)

    def end_input(self):
        # Preserve every accepted byte, including a word currently rendering.
        self.eof = True

    def _next_bit(self):
        if self.preamble:
            self.preamble -= 1
            return 1
        if self.word_bit < len(self.word):
            bit = self.word[self.word_bit]
            self.word_bit += 1
            return bit
        if self.bytes:
            if self.idle:
                self.idle = False
                self.preamble = 23  # 80 ms acquisition before a new burst
                return 1
            value = self.bytes.popleft()
            self.word = (0,) + tuple((value >> i) & 1 for i in range(8)) + (1,)
            self.word_bit = 1
            return self.word[0]
        self.idle = True
        if self.eof:
            if self.tail is None:
                self.tail = 12  # finish accepted data, then 40 ms mark and silence
            if not self.tail:
                return None
            self.tail -= 1
        return 1

    def render(self, samples=160):
        if type(samples) is not int or not 1 <= samples <= 1600:
            raise ValueError('render 1..1600 PCM samples')
        out = []
        for _ in range(samples):
            if self.remaining <= 0:
                self.level = self._next_bit()
                self.remaining += RATE / 300
            if self.level is None:
                out.append(0)
            else:
                out.append(round(self.AMPLITUDE * math.sin(self.phase)))
                hz = 2225 if self.level else 2025
                self.phase = (self.phase + 2 * math.pi * hz / RATE) % (2 * math.pi)
            self.remaining -= 1
        self.generated_samples += samples
        return struct.pack('<%dh' % samples, *out)

    def receive(self, pcm):
        validate_pcm(pcm)
        out = bytearray()
        for sample in struct.unpack('<%dh' % (len(pcm) // 2), pcm):
            self.receiver.sample(sample)
            value = self.uart.sample(self.modem.rxd_level)
            if value is not None:
                out.append(value)
        self.received_samples += len(pcm) // 2
        return bytes(out)

    def snapshot(self):
        return {'scope': 'experimental answer-side Bell 103 PCM/DTE laboratory peer',
                'queued_bytes': len(self.bytes), 'queue_limit_bytes': self.QUEUE_BYTES,
                'manager_bytes_accepted': self.queued_bytes, 'manager_eof': self.eof,
                'generated_samples': self.generated_samples,
                'received_samples': self.received_samples,
                'decoded_uart_frames': self.uart.frames,
                'framing_errors': self.uart.framing_errors,
                'false_starts': self.uart.false_starts,
                'carrier': self.receiver.carrier,
                'physical_fidelity_verified': False, 'real_time_verified': False,
                'proprietary_1200_supported': False}


class DPSKAnswerPeer:
    """The answering side of a 212A call: what Softmodem v(V22) is to a 1D.

    Same interface as Bell103AnswerPeer (queue, end_input, render, receive)
    over dpsk212: answer carrier 2400 Hz, scrambled marks from the first
    sample -- no answer tone, as a Protel calling in at 1200 needs --
    and the manager's bytes as 8N1 characters. What it hears is the
    originate carrier, descrambled and read by a UART.
    """

    QUEUE_BYTES = 4096
    LEAD_BITS = 12

    def __init__(self, amplitude=6000):
        self.mod = dpsk212.Modulator(dpsk212.ANSWER_HZ, amplitude)
        self.demod = dpsk212.Demodulator(dpsk212.ORIGINATE_HZ)
        self.uart = dpsk212.AsyncDeframer()
        self.bytes = deque()
        self.word = deque()
        self.eof = False
        self.tail = None
        self.generated_samples = self.received_samples = self.queued_bytes = 0
        for _ in range(self.LEAD_BITS):
            self.mod.push_bit(1)

    def queue(self, data):
        if self.eof:
            raise ValueError('manager byte stream is closed')
        data = bytes(data)
        if len(data) + len(self.bytes) > self.QUEUE_BYTES:
            raise ValueError('manager byte queue exceeds laboratory limit')
        self.bytes.extend(data)
        self.queued_bytes += len(data)

    def end_input(self):
        self.eof = True

    def _next_bit(self):
        if not self.word and self.bytes:
            value = self.bytes.popleft()
            self.word.extend((0,) + tuple((value >> i) & 1 for i in range(8)) + (1,))
        if self.word:
            return self.word.popleft()
        if self.eof:
            if self.tail is None:
                self.tail = 48          # 40 ms of mark after the last byte
            if not self.tail:
                return None
            self.tail -= 1
        return 1

    def render(self, samples=160):
        if type(samples) is not int or not 1 <= samples <= 1600:
            raise ValueError('render 1..1600 PCM samples')
        while self.mod.ready() < samples:
            bit = self._next_bit()
            if bit is None:
                break
            self.mod.push_bit(bit)
        out = self.mod.pcm(min(samples, self.mod.ready()))
        out += [0] * (samples - len(out))
        self.generated_samples += samples
        return struct.pack('<%dh' % samples, *out)

    def receive(self, pcm):
        validate_pcm(pcm)
        out = bytearray()
        samples = struct.unpack('<%dh' % (len(pcm) // 2), pcm)
        for _, bit in self.demod.feed(samples):
            value = self.uart.bit(bit)
            if value is not None:
                out.append(value)
        self.received_samples += len(samples)
        return bytes(out)

    @property
    def carrier(self):
        return self.demod.carrier_on

    def snapshot(self):
        return {'scope': 'experimental answer-side Bell 212A DPSK PCM/DTE laboratory peer',
                'queued_bytes': len(self.bytes), 'manager_bytes_accepted': self.queued_bytes,
                'manager_eof': self.eof, 'generated_samples': self.generated_samples,
                'received_samples': self.received_samples,
                'decoded_uart_frames': self.uart.frames, 'framing_errors': self.uart.framing_errors,
                'carrier': self.carrier, 'physical_fidelity_verified': False,
                'real_time_verified': False}
