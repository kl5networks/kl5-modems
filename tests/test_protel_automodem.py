"""One answering modem for every Protel (kl5modems/protel_automodem.py).

The public tests hold the detector to the signals each phone opens with,
made by our own modems. The ROM-gated tests let the emulated 1D's firmware
call in at 300 and at 1200 against the auto modem, with nothing told which.
"""

import math
import random
import struct
import unittest

from kl5modems import dpsk212, native1200
from kl5modems.protel_automodem import AutoAnswerPeer, AutoCall, Detector
from kl5modems.upms1200_rel import encode_rel_frame


def tone(hz, n, amplitude=6000):
    return [int(amplitude * math.sin(2 * math.pi * hz * k / 8000)) for k in range(n)]


def dpsk(n, carrier=dpsk212.ORIGINATE_HZ, amplitude=6000):
    m = dpsk212.Modulator(carrier, amplitude)
    rng = random.Random(5)
    while m.ready() < n:
        m.push_bit(rng.randrange(2))
    return m.pcm(n)


def bell103(n):
    rng = random.Random(1)
    bits = [1] * 30 + [rng.randrange(2) for _ in range(600)]
    out, phase = [], 0.0
    for k in range(n):
        phase += 2 * math.pi * (1270 if bits[int(k * 300 / 8000)] else 1070) / 8000
        out.append(int(6000 * math.sin(phase)))
    return out


def native_handshake(n):
    return (tone(1059, 9600) + tone(1252, 1600))[:n]


def decide(samples, noise=0):
    rng = random.Random(2)
    d = Detector()
    return d.feed([v + int(rng.gauss(0, noise)) for v in samples]) if noise else d.feed(samples)


class DetectorTest(unittest.TestCase):
    def test_each_opening_is_named(self):
        for noise in (0, 300):
            with self.subTest(noise=noise):
                self.assertEqual(decide(dpsk(8000), noise), "v22")
                self.assertEqual(decide(bell103(8000), noise), "bell103")
                self.assertEqual(decide(native_handshake(8000), noise), "native")

    def test_our_own_echo_and_silence_decide_nothing(self):
        for samples in ([0] * 8000, tone(2225, 8000, 3000), dpsk(8000, dpsk212.ANSWER_HZ, 3000),
                        [int(random.Random(9).gauss(0, 100)) for _ in range(8000)]):
            self.assertIsNone(decide(samples))

    def test_bell_103_data_never_reads_as_the_7000(self):
        # Eight zero bits and a start bit hold space for 30 ms; the 7000's
        # 1059 Hz runs 1.2 s. A stream of NULs is the worst case.
        bits = [1] * 30 + ([0] * 9 + [1]) * 100
        out, phase = [], 0.0
        for k in range(16000):
            phase += 2 * math.pi * (1270 if bits[int(k * 300 / 8000)] else 1070) / 8000
            out.append(int(6000 * math.sin(phase)))
        self.assertEqual(decide(out), "bell103")

    def test_decisions_are_fast(self):
        d = Detector()
        d.feed(native_handshake(1280))
        self.assertEqual(d.windows, 4)


class AutoCallTest(unittest.TestCase):
    def run_call(self, phone, seconds=3.0):
        call, ours, out = AutoCall(), [0] * 160, []
        for n in range(int(seconds * 50)):
            heard, ours = call.audio(phone(n))
            out += ours
        return call, out

    def test_it_answers_as_212a_then_with_bells_tone(self):
        call, out = self.run_call(lambda n: [0] * 160)
        self.assertIsNone(call.mode)
        early = dpsk212.Demodulator(dpsk212.ANSWER_HZ)
        early.feed(out[:8000])
        self.assertTrue(early.carrier_on)
        late = out[-1600:]
        crossings = sum(1 for a, b in zip(late, late[1:]) if a < 0 <= b)
        self.assertAlmostEqual(crossings / 0.2, 2225, delta=15)

    def test_a_7000_gets_the_native_modem_and_its_frames_reach_the_manager(self):
        mod = native1200.Modulator(amplitude=6000, keyed=True)
        mod.queue(encode_rel_frame(0x70))
        handshake = native_handshake(11200)

        def phone(n):
            at = n * 160
            if at < len(handshake):
                return handshake[at:at + 160] + [0] * max(0, at + 160 - len(handshake))
            return mod.pcm(160)

        call, _ = self.run_call(phone)
        self.assertEqual(call.mode, "native")
        self.assertEqual(call.modem.station.frames_in, 1)

    def test_manager_bytes_wait_for_the_decision(self):
        call = AutoCall()
        call.send(b"Q")
        src = bell103(8000)
        for n in range(0, 8000, 160):
            call.audio(src[n:n + 160])
        self.assertEqual(call.mode, "bell103")
        self.assertEqual(call.modem.peer.queued_bytes, 1)




if __name__ == "__main__":
    unittest.main()
