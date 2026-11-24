"""The Protel native 1200 FSK modem (kl5modems/native1200.py).

The public tests hold the modem to itself: its own audio round trips, through
noise, at the 7000's longer byte-edge cells. The ROM-gated ones hold it to the
firmware in both directions: every 7000's serializer into our receiver, our
transmitter into both UPMS1200 revisions' receivers (CRC gate and host
delivery), and the UPMS's own acknowledgment back into our receiver.
"""

import os
import random
import struct
import unittest

from kl5modems import native1200 as N
from kl5modems.upms1200_rel import decode_rel_frame, encode_rel_frame



def modulate(data, lead=40, tail=40, amplitude=8000):
    m = N.Modulator(amplitude=amplitude)
    m.queue_bits([1] * lead)
    m.queue(data)
    m.queue_bits([1] * tail)
    return m.pcm(int((lead + 10 * len(data) + tail) * m.samples_per_cell) + 1)


class ToneTest(unittest.TestCase):
    def test_the_formula_at_the_reference_oscillator(self):
        self.assertAlmostEqual(N.ZERO_HZ, 1186.85, places=1)
        self.assertAlmostEqual(N.ONE_HZ, 2151.17, places=1)
        self.assertAlmostEqual(N.BAUD, 1196.37, places=1)

    def test_each_cell_holds_its_tone(self):
        m = N.Modulator()
        m.queue_bits([0] * 200 + [1] * 200)
        pcm = m.pcm(int(400 * m.samples_per_cell))
        half = len(pcm) // 2
        for chunk, hz in ((pcm[100:half - 100], N.ZERO_HZ), (pcm[half + 100:-100], N.ONE_HZ)):
            crossings = sum(1 for a, b in zip(chunk, chunk[1:]) if a < 0 <= b)
            self.assertAlmostEqual(crossings / (len(chunk) / N.RATE), hz, delta=20)

    def test_idle_is_mark(self):
        m = N.Modulator()
        self.assertTrue(m.idle)
        self.assertEqual(N.Demodulator().feed(m.pcm(4000)), b"")


class KeyedTest(unittest.TestCase):
    def test_keyed_is_silence_then_lead_frame_and_silence_again(self):
        m = N.Modulator(keyed=True)
        self.assertEqual(set(m.pcm(800)), {0})
        m.queue(b"\x30")
        cells = m.LEAD_CELLS + 10 + m.TAIL_CELLS
        pcm = m.pcm(int((cells + 4) * m.samples_per_cell))
        on = [n for n, v in enumerate(pcm) if v]
        self.assertAlmostEqual((on[-1] - on[0]) / m.samples_per_cell, cells, delta=1.5)
        self.assertFalse(m.on)
        self.assertEqual(N.Demodulator().feed(pcm), b"\x30")


class RoundTripTest(unittest.TestCase):
    def test_every_byte_value(self):
        data = bytes(range(256)) * 2
        for amplitude in (12000, 3000):
            rx = N.Demodulator()
            self.assertEqual(rx.feed(modulate(data, amplitude=amplitude)), data)
            self.assertEqual((rx.frames, rx.framing_errors), (512, 0))

    def test_through_noise(self):
        # 11 dB SNR: a 4-sample boxcar lost a fifth of these bytes.
        data = bytes(range(256)) * 2
        rng = random.Random(1)
        pcm = [v + int(rng.gauss(0, 1500)) for v in modulate(data)]
        self.assertIn(data, N.Demodulator().feed(pcm))

    def test_in_any_chunking(self):
        data = b"KL5 native 1200"
        pcm = modulate(data)
        rx, got, at, n = N.Demodulator(), b"", 0, 0
        while at < len(pcm):
            size = (1, 7, 160, 3, 999)[n % 5]
            got += rx.feed(pcm[at:at + size])
            at, n = at + size, n + 1
        self.assertEqual(got, data)

    def test_the_7000s_long_byte_edge_reads_as_idle(self):
        # The 7000 counts 743 cycles a cell and 770 at a byte edge: the stop
        # cell runs 3.6% long, which a UART that checks one stop bit ignores.
        bits = []
        for value in b"\x00\xff\x55\xaa":
            bits += [0] + [(value >> n) & 1 for n in range(8)]
            bits += [1, 1] if value & 1 else [1]
        m = N.Modulator(baud=N.CPU_HZ / 743)
        m.queue_bits([1] * 20 + bits + [1] * 20)
        rx = N.Demodulator()
        self.assertEqual(rx.feed(m.pcm(int(60 * 10 * m.samples_per_cell))), b"\x00\xff\x55\xaa")

    def test_a_broken_stop_bit_is_a_framing_error(self):
        m = N.Modulator()
        m.queue_bits([1] * 20 + [0] * 10 + [1] * 20)
        rx = N.Demodulator()
        self.assertEqual(rx.feed(m.pcm(int(50 * m.samples_per_cell))), b"")
        self.assertEqual(rx.framing_errors, 1)

    def test_silence_is_no_carrier_and_no_characters(self):
        rx = N.Demodulator()
        rx.feed(modulate(b"x"))
        self.assertTrue(rx.carrier)
        rng = random.Random(3)
        self.assertEqual(rx.feed([int(rng.gauss(0, 60)) for _ in range(4000)]), b"")
        self.assertFalse(rx.carrier)

    def test_a_rel_frame_survives_the_trip(self):
        wire = encode_rel_frame(0x20, b"hello\xa5")
        self.assertEqual(decode_rel_frame(N.Demodulator().feed(modulate(wire))).payload, b"hello\xa5")






if __name__ == "__main__":
    unittest.main()
