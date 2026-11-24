import math
import random
import struct
import unittest

from kl5modems.protel_bell103 import Bell103Receiver
from kl5modems.protel_devices import SSI73K212L, MODEM_CR0, MODEM_CR1, MODEM_TR


def cpfsk(payload, high=True, phase=.37, noise=0, baud=300, amplitude=6000):
    """Independent continuous-phase 8N1 waveform; no production DSP is used."""
    bits = [1] * 24
    for value in payload:
        bits += [0] + [(value >> i) & 1 for i in range(8)] + [1]
    bits += [1] * 12
    tones = (2025, 2225) if high else (1070, 1270)
    rng = random.Random(42)
    samples = []
    for n in range(math.ceil(len(bits) * 8000 / baud)):
        bit = bits[min(len(bits) - 1, int(n * baud / 8000))]
        samples.append(round(amplitude * math.sin(phase) + rng.uniform(-noise, noise)))
        phase = (phase + 2 * math.pi * tones[bit] / 8000) % (2 * math.pi)
    return samples


def uart_decode(levels, baud=300):
    """Recover UART frames from RXD transitions and bit centers."""
    output, n = [], 1
    while n + math.ceil(10 * 8000 / baud) < len(levels):
        if levels[n - 1] == 1 and levels[n] == 0:
            value = sum(levels[n + round((i + 1.5) * 8000 / baud)] << i for i in range(8))
            stop = n + round(9.5 * 8000 / baud)
            if levels[stop] != 1:
                raise AssertionError('RXD framing error')
            output.append(value)
            n = stop
        else:
            n += 1
    return bytes(output)


class Bell103Test(unittest.TestCase):
    def receiver(self, cr0=0x31):
        modem = SSI73K212L([])
        modem.write_register(MODEM_CR0, cr0)
        return modem, Bell103Receiver(modem)

    def test_all_byte_values_demodulate_from_independent_noisy_cpfsk(self):
        m, rx = self.receiver()
        payload = bytes(range(256))
        levels = []
        for sample in cpfsk(payload, noise=400):
            rx.sample(sample)
            levels.append(m.rxd_level)
        self.assertEqual(uart_decode(levels), payload)
        self.assertTrue(rx.carrier)
        self.assertEqual(rx.acquisitions, 1)
        self.assertFalse(m.rxd_bits, 'sampled RXD must not become per-read queued bits')
        self.assertEqual(m.detect & 0x28, 0x28)

    def test_opposite_receive_band_phase_dc_offset_and_small_baud_error(self):
        for high, cr0 in ((True, 0x33), (False, 0x30)):
            m, rx = self.receiver(cr0)
            payload = b'\x00\xff\xaa\x55\x54\x07\x45\x18'
            levels = []
            for sample in cpfsk(payload, high=high, phase=4.2, noise=100, baud=299, amplitude=1800):
                rx.sample(sample + 400)
                levels.append(m.rxd_level)
            self.assertEqual(uart_decode(levels, baud=299), payload)

    def test_silence_noise_dc_wrong_band_and_weak_tone_never_set_carrier(self):
        rng = random.Random(12)
        controls = [[0] * 1600, [5000] * 1600,
                    [rng.randint(-6000, 6000) for _ in range(1600)]]
        for hz, amplitude in ((1000, 6000), (1270, 6000), (2400, 6000), (2130, 6000), (2225, 40)):
            controls.append([round(amplitude * math.sin(2 * math.pi * hz * n / 8000)) for n in range(1600)])
        for signal in controls:
            m, rx = self.receiver()
            m.set_detect(0x02, 0x02, 'other detector')
            for sample in signal:
                rx.sample(sample)
            self.assertFalse(rx.carrier)
            self.assertEqual(m.detect, 0x02)
            self.assertEqual(m.rxd_level, 1)

    def test_mode_reset_loss_and_tristate_keep_chip_register_semantics(self):
        m, rx = self.receiver()
        for sample in cpfsk(b''):
            rx.sample(sample)
        self.assertTrue(rx.carrier)
        m.write_register(MODEM_TR, 0x80)
        rx.sync()
        self.assertTrue(rx.snapshot()['rxd_tristated'])
        self.assertEqual(m.detect & 0x28, 0x28, 'TR7 must not hide DR receive data')
        for _ in range(240):
            rx.sample(0)
        self.assertFalse(rx.carrier)
        self.assertEqual(m.detect & 8, 0)
        for sample in cpfsk(b''):
            rx.sample(sample)
        self.assertTrue(rx.carrier)
        m.write_register(MODEM_CR1, 4)
        rx.sync()
        self.assertFalse(rx.carrier)
        self.assertEqual(m.cr1, 0)
        for cr0 in (0, 0x19, 0x0D, 0x71, 0xB1):
            m.write_register(MODEM_CR0, cr0)
            for sample in cpfsk(b''):
                rx.sample(sample)
            self.assertFalse(rx.carrier)
            self.assertFalse(rx.supported)
        m.write_register(MODEM_CR0, 0x31)
        for sample in cpfsk(b''):
            rx.sample(sample)
        self.assertTrue(rx.carrier)
        m.write_register(MODEM_CR0, 0x30)
        rx.sync()
        self.assertFalse(rx.carrier, 'old-band acquisition must not survive a band switch')
