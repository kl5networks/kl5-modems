"""A phone line that is not ideal (kl5modems/line_model.py, kl5modems/line_trial.py).

The public tests hold the model to what it claims (codec, echo, noise, the
mute's trigger and timing) and pin the findings it was built to reach:
with the HT802's mute modelled as a DTMF false trigger, only 212A trips
it; at the same mute rate, the longer 300-baud packet is the one more
often hit. The ROM-gated class runs the emulated 1D's own firmware across
the modelled line at both speeds. Emulator evidence: the mute's trigger
is inferred, never measured on an HT802 with these signals.
"""

import math
import random
import struct
import unittest

from kl5modems import dpsk212, line_model, line_trial
from kl5modems.line_model import DTMFMute, ImpairedPeer, Line, RateMute, g711
from kl5modems.protel_automodem import AutoCall
from kl5modems.protel_pcm_peer import Bell103AnswerPeer, DPSKAnswerPeer
from tests.test_protel_automodem import bell103, dpsk, native_handshake


def tone(*hz, n=1600, amplitude=4000):
    return [int(sum(amplitude * math.sin(2 * math.pi * f * k / 8000) for f in hz)) for k in range(n)]


def rms(samples):
    return math.sqrt(sum(v * v for v in samples) / len(samples))


def mutes(samples, detector=None):
    d = detector or DTMFMute()
    for v in samples:
        d.muted(v)
    return d.triggers


class CodecTest(unittest.TestCase):
    def test_mu_law_keeps_a_modem_tone_clean(self):
        x = tone(1000, n=8000)
        y = g711(x)
        err = sum((a - b) ** 2 for a, b in zip(x, y)) / len(x)
        self.assertGreater(10 * math.log10(rms(x) ** 2 / err), 30)

    def test_silence_stays_silent_and_full_scale_clips(self):
        self.assertEqual(g711([0] * 10), [0] * 10)
        self.assertEqual(g711([40000, -40000]), [32124, -32124])


class MuteTest(unittest.TestCase):
    def test_a_dtmf_pair_trips_it_and_one_tone_does_not(self):
        self.assertEqual(mutes(tone(941, 1209)), 10)          # '*', every frame
        self.assertEqual(mutes(tone(1209)), 0)                # a column alone
        self.assertEqual(mutes(tone(2225)), 0)                # Bell 103's answer mark
        self.assertEqual(mutes([0] * 1600), 0)

    def test_the_frame_after_the_trigger_is_the_one_muted(self):
        line = Line(mute=DTMFMute(), codec=False)
        out = line.to_hub(tone(941, 1209, n=160) + tone(2225, n=320))
        self.assertTrue(any(out[:160]))
        self.assertEqual(out[160:320], [0] * 160)             # one packet of digital zero
        self.assertTrue(any(out[320:]))
        self.assertEqual(line.snapshot()["muted_ms"], 20)

    def test_rate_mutes_land_at_their_rate_whatever_is_sent(self):
        r = RateMute(1.0, seed=4)
        for _ in range(60 * 8000):
            r.muted(0)
        self.assertAlmostEqual(r.triggers / 60, 1.0, delta=0.3)

    def test_the_model_trips_on_212a_and_never_on_bell_103_or_the_native_fsk(self):
        # Calibrated to the Millennium's 212A (0.022-0.053 mutes/s measured).
        # Bell 103's 1070/1270 Hz and the 7000's 1187/2151 Hz have no DTMF
        # row tone at all; the bench's 300-baud call-ins through the same
        # HT802 trained and ran.
        rate = mutes(dpsk(60 * 8000)) / 60
        self.assertGreater(rate, 0.01)
        self.assertLess(rate, 0.1)
        rng = random.Random(3)
        fsk, phase, bit = [], 0.0, 1
        for k in range(30 * 8000):
            if k % 27 == 0:
                bit = rng.randrange(2)
            phase += 2 * math.pi * (1270 if bit else 1070) / 8000
            fsk.append(int(6000 * math.sin(phase)))
        self.assertEqual(mutes(fsk), 0)
        from kl5modems import native1200
        m = native1200.Modulator(amplitude=8000)
        m.queue(bytes(rng.randrange(256) for _ in range(3600)))
        self.assertEqual(mutes(m.pcm(30 * 8000)), 0)


class LineTest(unittest.TestCase):
    def test_our_echo_comes_back_late_and_quieter(self):
        line = Line(hub_echo=(40, 20), codec=False)
        burst = [10000] * 10 + [0] * 470
        line.to_phone(burst)
        back = line.to_hub([0] * 480)
        self.assertEqual(back.index(1000), 320)               # 40 ms, 20 dB down
        self.assertEqual(sum(1 for v in back if v), 10)

    def test_noise_is_at_its_level_both_ways(self):
        line = Line(noise_dbm0=-40, codec=False, seed=1)
        for heard in (line.to_hub([0] * 16000), line.to_phone([0] * 16000)):
            self.assertAlmostEqual(20 * math.log10(rms(heard) / line_model.DBM0_RMS), -40, delta=0.5)

    def test_the_mute_is_only_on_the_phones_side(self):
        line = Line(mute=RateMute(50.0, seed=1), codec=False)     # every frame
        self.assertEqual(line.to_hub([1000] * 320), [0] * 320)
        self.assertEqual(line.to_phone([1000] * 320), [1000] * 320)

    def test_an_impaired_peer_is_the_peer_across_the_line(self):
        peer = ImpairedPeer(DPSKAnswerPeer(), Line(hub_echo=(10, 6), seed=2))
        out = peer.render(160)
        self.assertEqual(len(out), 320)
        self.assertEqual(peer.receive(struct.pack("<160h", *([0] * 160))), b"")
        self.assertEqual(peer.snapshot()["received_samples"], 160)


class TrialTest(unittest.TestCase):
    def test_a_fair_line_loses_nothing_at_either_speed(self):
        for mode in line_trial.MODES:
            with self.subTest(mode=mode):
                r = line_trial.trial(mode, line_trial.SCENARIOS["echo+noise"](1), seconds=12, seed=1)
                for d in ("up", "down"):
                    self.assertGreater(r[d]["packets"], 1)
                    self.assertEqual(r[d]["intact"], r[d]["packets"], r)

    def test_at_one_mute_rate_the_longer_300_packet_is_hit_more(self):
        # Every mute costs the packet it lands in, whichever modem: so with
        # the trigger taken away, 300 is the WORSE speed, by packet length
        # (2.3 s against 0.58 s). 300 wins only because its carrier does not
        # trip the detector.
        kept = {}
        for mode in ("bell103", "v22"):
            r = line_trial.trial(mode, line_trial.SCENARIOS["mute 1/s"](1), seconds=30, seed=1,
                                 directions=("up",))
            kept[mode] = r["up"]["intact"] / r["up"]["packets"]
        self.assertLess(kept["bell103"], 0.35)
        self.assertGreater(kept["v22"], 0.35)

    def test_the_downstream_direction_is_never_muted(self):
        r = line_trial.trial("v22", line_trial.SCENARIOS["mute 1/s"](2), seconds=10, seed=2)
        self.assertGreater(r["line"]["mutes"], 3)
        self.assertEqual(r["down"]["intact"], r["down"]["packets"])


class AutoAcrossTheLineTest(unittest.TestCase):
    def test_auto_still_names_each_phone_on_a_bad_line(self):
        # Mutes once a second, -40 dBm0 noise, our own 212A and 2225 Hz back
        # 10 dB down: the detector hears all of it and still decides right.
        for want, src in (("v22", dpsk(16000)), ("bell103", bell103(16000)),
                          ("native", native_handshake(11200))):
            with self.subTest(want=want):
                line = Line(mute=RateMute(1.0, 1), noise_dbm0=-40, hub_echo=(40, 10), seed=1)
                call, ours = AutoCall(), [0] * 160
                for n in range(0, len(src), 160):
                    line.to_phone(ours)
                    _, ours = call.audio(line.to_hub(src[n:n + 160]))
                self.assertEqual(call.mode, want)




if __name__ == "__main__":
    unittest.main()
