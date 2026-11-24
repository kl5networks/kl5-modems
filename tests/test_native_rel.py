"""The UPMS's REL link as a station (kl5modems/native_rel.py).

These tests replay the turn tables recorded from the UPMS ROMs running in
an emulator. The emulator and the ROMs are not part of this repository.
"""

import json
import os
import unittest

from kl5modems.native_rel import Deframer, Station
from kl5modems.upms1200_rel import decode_rel_frame, encode_rel_frame


def run(frames, host=b"world"):
    st = Station()
    st.send(host)
    replies, delivered = [], b""
    for wire in frames:
        reply, got = st.receive(wire)
        frame = decode_rel_frame(reply)
        replies.append("%02X:%s" % (frame.control, frame.payload.decode("latin1")))
        delivered += got
    return replies, delivered, st


F = encode_rel_frame


class TurnTableTest(unittest.TestCase):
    def test_acknowledged(self):
        self.assertEqual(run([F(0x20, b"hello"), F(0x60, b"again"), F(0x20, b"done")])[:2],
                         (["30:world", "70:", "30:"], b"helloagaindone"))

    def test_a_missed_ack_resends_our_data(self):
        self.assertEqual(run([F(0x20, b"hello"), F(0x20, b"again"), F(0x60, b"done")])[:2],
                         (["30:world", "70:world", "30:"], b"helloagaindone"))

    def test_a_replayed_data_frame_is_delivered_twice(self):
        # The ROM does not dedupe: a peer must not replay data on ACK loss.
        hello = F(0x20, b"hello")
        self.assertEqual(run([hello, hello, F(0x60, b"done")])[1], b"hellohellodone")

    def test_a_bad_crc_is_answered_with_our_toggle_alone(self):
        bad = bytearray(F(0x20, b"hello"))
        bad[-5] ^= 1
        replies, delivered, st = run([bytes(bad), F(0x20, b"hello"), F(0x60, b"done")])
        self.assertEqual((replies, delivered), (["40:", "30:world", "70:"], b"hellodone"))
        self.assertEqual(st.frames_rejected, 1)

    def test_a_poll_gets_our_last_frame_byte_for_byte(self):
        replies, delivered, st = run([F(0x20, b"hello"), F(0x00), F(0x60, b"done")])
        self.assertEqual((replies, delivered), (["30:world", "30:world", "70:"], b"hellodone"))
        self.assertEqual(st.resends, 1)

    def test_a_busy_peer_gets_no_data_until_it_clears(self):
        frames = [F(0x28, b"hello"), F(0x68, b"again"), F(0x20, b"done"), F(0x60, b"exit")]
        self.assertEqual(run(frames)[:2], (["30:", "70:", "30:world", "70:"], b"helloagaindoneexit"))

    def test_a_payload_the_ring_cannot_hold_is_not_committed(self):
        replies, delivered, _ = run([F(0x20, bytes(512)), F(0x60, b"after-wrap")], host=b"")
        self.assertEqual((replies, delivered), (["70:", "30:"], b"after-wrap"))
        self.assertEqual(run([F(0x20, bytes(511))], host=b"")[0], ["30:"])

    def test_idle_requests_carry_our_toggle_and_no_data(self):
        st = Station()
        st.send(b"world")
        self.assertEqual(decode_rel_frame(st.idle()).control, 0x40)        # from reset
        st.receive(F(0x20, b"hello"))
        idle = decode_rel_frame(st.idle())
        self.assertEqual((idle.control, idle.payload), (0x00, b""))
        # ...and an ack after the pause still releases the data in flight.
        reply, _ = st.receive(F(0x60, b"again"))
        self.assertEqual(decode_rel_frame(reply).payload, b"")
        self.assertFalse(st.unacknowledged)

    def test_long_output_is_split_and_each_piece_waits_for_its_ack(self):
        st = Station(max_payload=10)
        st.send(bytes(range(25)))
        got, control = b"", 0x20
        for _ in range(5):
            reply, _ = st.receive(F(control))
            got += decode_rel_frame(reply).payload
            control ^= 0x40
        self.assertEqual(got, bytes(range(25)))


class DeframerTest(unittest.TestCase):
    def test_splits_frames_and_keeps_quoted_terminators(self):
        frames = [F(0x20, b"\xa5\x5a\xa5\x5b\xa5"), F(0x60), F(0x20, bytes(range(256)))]
        d = Deframer()
        stream = b"".join(frames)
        out = []
        for n in range(0, len(stream), 7):
            out += d.feed(stream[n:n + 7])
        self.assertEqual(out, frames)
        self.assertEqual(decode_rel_frame(out[0]).payload, b"\xa5\x5a\xa5\x5b\xa5")

    def test_noise_without_a_terminator_is_dropped(self):
        d = Deframer()
        self.assertEqual(d.feed(b"\x7e" * 5000), [])
        self.assertEqual(d.overruns, 4)
        d.gap()
        self.assertEqual(d.feed(F(0x20, b"x")), [F(0x20, b"x")])
        self.assertEqual(d.dropped, 1)




if __name__ == "__main__":
    unittest.main()
