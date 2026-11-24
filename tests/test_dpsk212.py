import random
import struct
import unittest

from kl5modems.dpsk212 import (
    ANSWER_HZ, ORIGINATE_HZ, Demodulator, Modulator, Scrambler,
)
from kl5modems.upms1200_dpsk import (
    DPSKAsync, DPSKScrambler, async_bits, decode_dpsk_symbols, dpsk_phases,
)
from kl5modems.upms1200_dpsk_pcm import DPSKPCMReceiver, DPSKPCMTransmitter


def modulate(bits, carrier=ANSWER_HZ, tail=24, lead=0):
    m = Modulator(carrier)
    for bit in bits:
        m.push_bit(bit)
    for _ in range(tail):
        m.push_bit(1)
    return [0] * lead + m.pcm(m.ready())


def demodulate(pcm, carrier=ANSWER_HZ, chunk=160, **options):
    d = Demodulator(carrier, **options)
    out = []
    for i in range(0, len(pcm), chunk):
        out += d.feed(pcm[i:i + chunk])
    return ''.join(str(b) for _, b in out), d


def resample(pcm, factor):
    """Linear interpolation: the far end's sample clock running ``factor`` fast."""
    out, t = [], 0.0
    while t < len(pcm) - 1:
        i = int(t)
        f = t - i
        out.append(int(pcm[i] * (1 - f) + pcm[i + 1] * f))
        t += factor
    return out


def bits_of(rng, count):
    return [rng.randint(0, 1) for _ in range(count)]


class ScramblerTests(unittest.TestCase):
    def test_matches_the_reference_scrambler_both_ways_including_lockup(self):
        rng = random.Random(7)
        bits = [1] * 200 + bits_of(rng, 3000)
        ours = Scrambler()
        line = [ours.bit(b) for b in bits]
        self.assertEqual(tuple(line), DPSKScrambler().feed(tuple(bits)))
        back = Scrambler(descramble=True)
        self.assertEqual([back.bit(b) for b in line], bits)


class ReferenceInteropTests(unittest.TestCase):
    """Held against the declared-timing reference, which shares no code path."""

    def test_modulator_audio_decodes_in_the_reference_receiver(self):
        rng = random.Random(3)
        bits = bits_of(rng, 400)
        m = Modulator(ORIGINATE_HZ)
        m.scramble = False
        for bit in bits:
            m.push_bit(bit)
        for _ in range(16):
            m.push_bit(0)        # pad the final pulse tails
        pcm = m.pcm(m.ready())
        # Our symbol k peaks at sample 40k/3; the reference's symbol j at
        # 40(j+4)/3, so with no lead its j is our j+4, and its first decoded
        # dibit (j=0 to 1) is our fifth: bits 10 onwards.
        rx = DPSKPCMReceiver(len(bits) // 2 + 1, channel='originate')
        stream = struct.pack('<%dh' % len(pcm), *pcm)[:2 * rx.total_samples]
        stream += bytes(2 * rx.total_samples - len(stream))
        symbols = []
        for i in range(0, len(stream), 320):
            symbols.extend(rx.feed_pcm(stream[i:i + 320]))
        got = decode_dpsk_symbols(symbols)
        self.assertEqual(list(got[:300]), bits[10:310])

    def test_reference_transmitter_audio_decodes_with_recovered_timing(self):
        rng = random.Random(11)
        data = bits_of(rng, 2000)
        line = DPSKScrambler().feed(tuple([1] * 120 + data))
        tx = DPSKPCMTransmitter(dpsk_phases(line), channel='answer', carrier_phase=1.234)
        raw = b''
        while True:
            chunk = tx.read_pcm(1600)
            if not chunk:
                break
            raw += chunk
        pcm = [v for (v,) in struct.iter_unpack('<h', raw)]
        got, _ = demodulate([0] * 17 + pcm)
        self.assertIn(''.join(map(str, data)), got)


class DemodulatorTests(unittest.TestCase):
    def test_recovers_timing_from_any_start_and_any_chunking(self):
        rng = random.Random(1)
        data = bits_of(rng, 1500)
        want = ''.join(map(str, data))
        for lead in (0, 3, 7, 11):
            for chunk in (1, 37, 160):
                with self.subTest(lead=lead, chunk=chunk):
                    got, d = demodulate(modulate([1] * 200 + data, lead=lead), chunk=chunk)
                    self.assertIn(want, got)
                    self.assertTrue(d.carrier_events and d.carrier_events[0][1])

    def test_thirty_seconds_through_clock_offset_and_noise_without_a_bit_error(self):
        # The far end's clock is never ours; a slip would show as errors or
        # a missing/extra dibit after a few seconds.
        rng = random.Random(5)
        data = bits_of(rng, 36000)
        want = ''.join(map(str, data))
        pcm = modulate([1] * 200 + data, lead=5)
        for factor, noise in ((1.003, 0), (0.999, 300)):
            with self.subTest(factor=factor, noise=noise):
                line = [int(v + rng.gauss(0, noise)) for v in resample(pcm, factor)]
                got, _ = demodulate(line)
                start = got.find(want[:64])
                self.assertGreaterEqual(start, 0)
                self.assertEqual(got[start:start + len(want)], want)

    def test_originate_carrier_is_rejected_by_the_answer_receiver_and_vice_versa(self):
        rng = random.Random(9)
        data = bits_of(rng, 600)
        got, d = demodulate(modulate(data, carrier=ORIGINATE_HZ), carrier=ANSWER_HZ)
        self.assertNotIn(''.join(map(str, data[100:300])), got)
        got, d = demodulate(modulate(data, carrier=ORIGINATE_HZ), carrier=ORIGINATE_HZ)
        self.assertIn(''.join(map(str, data[100:300])), got)

    def test_carrier_detect_follows_the_chip_timing(self):
        # 73K302L: 212A carrier on after 15-32 ms, off after 10-24 ms.
        pcm = [0] * 800 + modulate([1] * 600, tail=0) + [0] * 2400
        _, d = demodulate(pcm)
        (on_t, on), (off_t, off) = d.carrier_events
        self.assertEqual((on, off), (True, False))
        burst_end = 800 + 600 / 1200 * 8000
        self.assertTrue(15 <= (on_t - 800) / 8 <= 40, on_t)
        self.assertTrue(10 <= (off_t - burst_end) / 8 <= 40, off_t)
        _, quiet = demodulate([0] * 4000)
        self.assertEqual(quiet.carrier_events, [])

    def test_async_characters_survive_the_line(self):
        # 8N1 from the UPMS, 11-bit characters from the 1D's board: one UART
        # receiver reads both.
        payload = bytes(range(256))
        for stop_bits in (1, 2):
            with self.subTest(stop_bits=stop_bits):
                bits = list(async_bits(payload, stop_bits=stop_bits, idle_bits=200))
                got, _ = demodulate(modulate(bits))
                # Bits decided before the descrambler has 17 bits of history
                # are junk, as on any self-synchronising line: a UART may make
                # a byte of them. The payload itself arrives whole.
                self.assertIn(payload, DPSKAsync().feed(tuple(int(b) for b in got)))


if __name__ == '__main__':
    unittest.main()
