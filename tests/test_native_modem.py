"""The software UPMS1200 (kl5modems/native_modem.py): a whole call, sample by sample.

The phone here is a scripted 7000-role peer on our own modem: it sends data
frames with the alternating bit 6 the ROM bench used and reads our replies.
That proves the bridge's plumbing (half duplex, echo, idle requests, gaps);
the ROM-gated tests in test_native1200/test_native_rel prove each layer
against the firmware.
"""

import asyncio
import struct
import unittest

from kl5modems import audiosocket
from kl5modems import native1200, native_modem, native_rel
from kl5modems.native_modem import Bridge, Call
from kl5modems.upms1200_rel import decode_rel_frame, encode_rel_frame


class Phone:
    """A 7000-role peer: data frames 20/60 in turn, each sent after our reply.

    Data comes back only in a reply (the ROM's idle requests carry none), so
    once its own chunks are sent it keeps the turn going with empty frames,
    ``polls`` of them.
    """

    def __init__(self, chunks, polls=4):
        self.chunks = list(chunks)
        self.polls = polls
        self.tx = native1200.Modulator(amplitude=8000, keyed=True)
        self.rx = native1200.Demodulator()
        self.deframer = native_rel.Deframer()
        self.control = 0x20
        self.heard = []
        self.received = b""
        self.waiting = False
        self.answered = False       # heard our answer mark and its end

    def step(self, ours):
        for wire in self.deframer.feed(self.rx.feed(ours)):
            frame = decode_rel_frame(wire)
            self.heard.append(frame.control)
            self.received += frame.payload
            self.waiting = False
        # A 7000 sends nothing until the far end's mark answers its call;
        # this one also waits it out rather than talk over it.
        if self.rx.carrier and not self.answered:
            self.answered = None
        elif self.answered is None and not self.rx.carrier:
            self.answered = True
        if self.answered and not self.waiting and not self.tx.on and (self.chunks or self.polls):
            if self.chunks:
                payload = self.chunks.pop(0)
            else:
                payload, self.polls = b"", self.polls - 1
            self.tx.queue(encode_rel_frame(self.control, payload))
            self.control ^= 0x40
            self.waiting = True
        return self.tx.pcm(160)


def converse(phone, call, manager_reply=b"", seconds=12, echo=0.0):
    to_manager, ours = b"", [0] * 160
    sent_reply = False
    for _ in range(int(seconds * 50)):
        theirs = phone.step(ours)
        line = [int(a + echo * b) for a, b in zip(theirs, ours)]
        data, ours = call.audio(line)
        to_manager += data
        if data and not sent_reply:
            call.send(manager_reply)
            sent_reply = True
    return to_manager


class CallTest(unittest.TestCase):
    def test_the_call_is_answered_with_a_second_of_mark(self):
        call = Call(mark_bursts=False)
        pcm = []
        for _ in range(100):
            pcm += call.audio([0] * 160)[1]
        on = [n for n, v in enumerate(pcm) if v]
        self.assertLessEqual(on[0], 1)          # sin(0) is the first sample
        self.assertAlmostEqual(on[-1] / native1200.RATE, 1.0, delta=0.01)
        self.assertEqual(native1200.Demodulator().feed(pcm), b"")

    def test_the_phones_frames_reach_the_manager_and_its_reply_reaches_the_phone(self):
        phone, call = Phone([b"!TC", b"Q\x01\x02", b"D"]), Call()
        self.assertEqual(converse(phone, call, b"O\x55"), b"!TCQ\x01\x02D")
        self.assertEqual(phone.received, b"O\x55")
        self.assertEqual(call.station.frames_rejected, 0)
        self.assertEqual(phone.heard[:4], [0x30, 0x70, 0x30, 0x70])

    def test_our_own_echo_is_not_read_as_the_phone(self):
        phone, call = Phone([b"hello", b"again"]), Call()
        self.assertEqual(converse(phone, call, b"world", echo=0.5), b"helloagain")
        self.assertGreater(call.echo_bytes, 0)
        self.assertEqual((call.station.frames_in, call.station.frames_rejected), (6, 0))

    def test_until_the_phone_is_heard_it_gets_bursts_of_mark(self):
        # A 7000 listens from ~3.2 s after its last digit: the answer mark
        # alone would be over by then.
        call = Call()
        pcm = []
        for _ in range(6 * 50):
            pcm += call.audio([0] * 160)[1]
        starts, last = [], -1000
        for n, v in enumerate(pcm):         # a sine has zero samples: gaps of 50+
            if v:
                if n - last > 50:
                    starts.append(round(n / native1200.RATE, 2))
                last = n
        self.assertEqual(len(starts), 5)    # the answer, then a burst every ~1.25 s
        for got, want in zip(starts, [0.0, 2.0, 3.25, 4.5, 5.75]):
            self.assertAlmostEqual(got, want, delta=0.04)
        self.assertEqual(call.bursts, 4)
        self.assertEqual(native1200.Demodulator().feed(pcm), b"")

    def test_bursts_stop_once_the_phone_is_heard(self):
        call = Call()
        tone = native1200.Modulator(amplitude=8000)
        for _ in range(3 * 50):                 # the phone's own mark from 1.5 s
            silent = call.now < 1.5 * native1200.RATE or call.transmitting
            call.audio([0] * 160 if silent else tone.pcm(160))
        self.assertTrue(call.heard_phone)
        bursts = call.bursts
        for _ in range(5 * 50):
            call.audio([0] * 160)
        self.assertEqual(call.bursts, bursts)

    def test_bursts_give_up_at_the_limit(self):
        call = Call()
        for _ in range(int((native_modem.MARK_BURST_LIMIT_SECONDS + 4) * 50)):
            call.audio([0] * 160)
        self.assertLess(call.bursts, native_modem.MARK_BURST_LIMIT_SECONDS)
        self.assertGreater(call.frames_out, 0)              # idle requests resume

    def test_a_silent_line_gets_idle_requests(self):
        call = Call(mark_bursts=False)
        heard = native1200.Demodulator()
        out = b""
        for _ in range(int(7 * 50)):           # 1 s of answer mark, then 2.6 s apart
            _, pcm = call.audio([0] * 160)
            out += heard.feed(pcm)
        frames = native_rel.Deframer().feed(out)
        self.assertEqual([decode_rel_frame(f).control for f in frames], [0x40, 0x40])

    def test_a_frame_cut_short_does_not_poison_the_next(self):
        call = Call(answer_mark_seconds=0)
        m = native1200.Modulator(amplitude=8000, keyed=True)
        m.queue(encode_rel_frame(0x20, b"lost")[:5])
        m.queue(b"")
        pcm = m.pcm(4000)
        m2 = native1200.Modulator(amplitude=8000, keyed=True)
        m2.queue(encode_rel_frame(0x20, b"kept"))
        pcm += m2.pcm(3000)
        got = b""
        for n in range(0, len(pcm), 160):
            got += call.audio(pcm[n:n + 160])[0]
        self.assertEqual(got, b"kept")
        self.assertEqual(call.deframer.dropped, 1)


class BridgeTest(unittest.TestCase):
    def test_the_manager_can_ask_which_modem_a_call_chose(self):
        # How a provisioning download keeps the phone at the speed it called
        # in at: the manager asks by the peer it sees.
        class Chosen:
            mode = "v22"

            def send(self, data):
                pass

            def audio(self, samples):
                return b"x", list(samples)

            def snapshot(self):
                return {}

        async def scenario():
            seen = []
            bridge = Bridge(None, call_factory=Chosen)

            async def manager(reader, writer):
                await reader.read(1)
                seen.append(bridge.mode_for(writer.get_extra_info("peername")))
                writer.close()

            mgr = await asyncio.start_server(manager, "127.0.0.1", 0)
            bridge.target = mgr.sockets[0].getsockname()[:2]
            self.assertTrue(await bridge.start("127.0.0.1", 0))
            reader, writer = await asyncio.open_connection(*bridge.server.sockets[0].getsockname()[:2])
            writer.write(audiosocket.audio_frame(bytes(320)))
            await writer.drain()
            for _ in range(100):
                if seen:
                    break
                await asyncio.sleep(0.01)
            writer.close()
            await bridge.stop()
            mgr.close()
            await mgr.wait_closed()
            return seen, bridge.live, bridge.mode_for(("127.0.0.1", 1))

        seen, live, stranger = asyncio.run(scenario())
        self.assertEqual(seen, ["v22"])
        self.assertIsNone(stranger)

    def test_audiosocket_in_tcp_out(self):
        async def scenario():
            got = []

            async def manager(reader, writer):
                got.append(await reader.read(100))
                writer.write(b"reply")
                await writer.drain()
                await reader.read(1)

            mgr = await asyncio.start_server(manager, "127.0.0.1", 0)
            bridge = Bridge(mgr.sockets[0].getsockname()[:2])
            self.assertTrue(await bridge.start("127.0.0.1", 0))
            host, port = bridge.server.sockets[0].getsockname()[:2]
            reader, writer = await asyncio.open_connection(host, port)
            writer.write(audiosocket.build(audiosocket.UUID, bytes(range(16))))
            phone = Phone([b"!TC"])
            frames, ours = audiosocket.Reader(), [0] * 160
            for _ in range(200):
                pcm = phone.step(ours)
                writer.write(audiosocket.audio_frame(struct.pack("<160h", *pcm)))
                await writer.drain()
                ours = []
                while len(ours) < 160:
                    for kind, payload in frames.feed(await reader.read(4096)):
                        ours += struct.unpack("<%dh" % (len(payload) // 2), payload)
                ours = ours[:160]
                if phone.received:
                    break
            writer.write(audiosocket.terminate_frame())
            await writer.drain()
            writer.close()
            await asyncio.sleep(0.05)
            state = bridge.state()
            await bridge.stop()
            mgr.close()
            return got, phone.received, state

        got, received, state = asyncio.run(scenario())
        self.assertEqual(got, [b"!TC"])
        self.assertEqual(received, b"reply")
        self.assertEqual((state["calls"], state["active"]), (1, 0))
        self.assertEqual(state["last"]["uuid"], "00010203-0405-0607-0809-0a0b0c0d0e0f")

    def test_the_manager_giving_up_hangs_up_a_call_whose_audio_stopped(self):
        # 2026-10-07 23:50: the caller's audio stopped, the manager gave up, and
        # the bridge sat waiting for a frame that never came -- 585 s more
        # on the carrier. With no audio at all, the hang-up must still go.
        async def scenario():
            async def manager(reader, writer):
                writer.close()

            mgr = await asyncio.start_server(manager, "127.0.0.1", 0)
            bridge = Bridge(mgr.sockets[0].getsockname()[:2])
            await bridge.start("127.0.0.1", 0)
            reader, writer = await asyncio.open_connection(*bridge.server.sockets[0].getsockname()[:2])
            writer.write(audiosocket.build(audiosocket.UUID, bytes(range(16))))
            await writer.drain()
            data = await asyncio.wait_for(reader.read(4096), timeout=2)
            rest = await asyncio.wait_for(reader.read(4096), timeout=2)
            state = bridge.state()
            await bridge.stop()
            mgr.close()
            return data + rest, state

        data, state = asyncio.run(scenario())
        self.assertTrue(data.endswith(audiosocket.terminate_frame()))
        self.assertEqual(state["active"], 0)

    def test_an_unreachable_manager_hangs_up_and_says_so(self):
        async def scenario():
            bridge = Bridge(("127.0.0.1", 1))
            await bridge.start("127.0.0.1", 0)
            reader, writer = await asyncio.open_connection(*bridge.server.sockets[0].getsockname()[:2])
            data = await reader.read(10)
            await bridge.stop()
            return data, bridge.last

        data, last = asyncio.run(scenario())
        self.assertEqual(data, audiosocket.terminate_frame())
        self.assertIn("unreachable", last["error"])




class CallSummaryTest(unittest.TestCase):
    def test_an_auto_call_reports_its_chosen_modems_figures(self):
        snap = {"mode": "bell103", "decided_at_seconds": 2.3, "seconds": 24.1, "windows": 600,
                "modem": {"carrier": False, "decoded_uart_frames": 101, "framing_errors": 3,
                          "false_starts": 7, "manager_bytes_accepted": 29, "queued_bytes": 0}}
        self.assertEqual(native_modem.call_summary(snap),
                         "mode=bell103 decided_at_seconds=2.3 seconds=24.1 carrier=False "
                         "decoded_uart_frames=101 framing_errors=3 false_starts=7 "
                         "manager_bytes_accepted=29 queued_bytes=0")

    def test_a_call_with_nothing_to_say_says_so(self):
        self.assertEqual(native_modem.call_summary({}), "no figures")



if __name__ == "__main__":
    unittest.main()


