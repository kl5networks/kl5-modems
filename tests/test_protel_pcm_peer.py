import asyncio
import struct
import threading
from types import SimpleNamespace
import unittest

from kl5modems.protel_bell103 import Bell103Receiver
from kl5modems.protel_devices import SSI73K212L
from kl5modems.protel_pcm_manager import PCMManagerBridge
from kl5modems.protel_pcm_peer import Bell103AnswerPeer, Bell103UART
from tests.test_protel_bell103 import cpfsk, uart_decode


class PCMPeerTest(unittest.TestCase):
    def test_fragmented_low_band_pcm_recovers_all_bytes_incrementally(self):
        peer = Bell103AnswerPeer()
        samples = cpfsk(bytes(range(256)), high=False, noise=200)
        raw = struct.pack('<%dh' % len(samples), *samples)
        out = bytearray()
        for offset in range(0, len(raw), 74):
            out.extend(peer.receive(raw[offset:offset + 74]))
        self.assertEqual(out, bytes(range(256)))
        self.assertEqual(peer.uart.framing_errors, 0)
        self.assertEqual(peer.uart.false_starts, 0)
        self.assertLessEqual(len(peer.modem.trace), 64)
        self.assertEqual(peer.snapshot()['received_samples'], len(samples))
        self.assertFalse(peer.snapshot()['physical_fidelity_verified'])

    def test_fragmented_manager_bytes_and_eof_preserve_full_high_band_transmission(self):
        peer = Bell103AnswerPeer()
        payload = bytes(range(256))
        peer.queue(payload[:3])
        modem = SSI73K212L([])
        modem.write_register(0, 0x31)
        rx = Bell103Receiver(modem)
        levels = []
        for n in range(450):
            if n == 2:
                peer.queue(payload[3:71])
            if n == 3:
                peer.queue(payload[71:])
                peer.end_input()  # EOF before most bytes have become PCM
            for value in struct.unpack('<160h', peer.render()):
                rx.sample(value)
                levels.append(modem.rxd_level)
        self.assertEqual(uart_decode(levels), payload)
        self.assertEqual(len(peer.bytes), 0)
        self.assertEqual(peer.render(), bytes(320), 'EOF must eventually release carrier')
        self.assertFalse(rx.carrier)

    def test_bounds_reject_without_mutating_existing_data_or_leaking_payloads(self):
        peer = Bell103AnswerPeer()
        peer.queue(b'X' * peer.QUEUE_BYTES)
        with self.assertRaises(ValueError):
            peer.queue(b'private-code')
        self.assertEqual(len(peer.bytes), peer.QUEUE_BYTES)
        before = peer.generated_samples
        for value in (True, 0, 1601):
            with self.assertRaises(ValueError):
                peer.render(value)
        for raw in (b'', b'x', bytes(3202)):
            with self.assertRaises(ValueError):
                peer.receive(raw)
        self.assertEqual(peer.generated_samples, before)
        self.assertNotIn('private-code', str(peer.snapshot()))
        peer.end_input()
        with self.assertRaises(ValueError):
            peer.queue(b'x')

    def test_uart_false_start_and_bad_stop_do_not_emit_bytes(self):
        uart = Bell103UART()
        for bit in [1] * 30 + [0] * 4 + [1] * 30:
            self.assertIsNone(uart.sample(bit))
        self.assertEqual(uart.false_starts, 1)
        bits = [0] + [0] * 8 + [0] + [1] * 3
        out = []
        for n in range(round(len(bits) * 8000 / 300)):
            value = uart.sample(bits[min(len(bits) - 1, n * 300 // 8000)])
            if value is not None:
                out.append(value)
        self.assertEqual(out, [])
        self.assertEqual(uart.framing_errors, 1)


class PCMBridgeTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_runner_joins_cpu_batch_before_teardown(self):
        started, release = threading.Event(), threading.Event()
        source = SimpleNamespace(bind_pc0=True, queued=0)
        source.snapshot = lambda: {'enabled': True, 'queued_samples': source.queued}
        source.feed = lambda pcm: setattr(source, 'queued', source.queued + len(pcm) // 2)
        tx = SimpleNamespace(snapshot=lambda: {'enabled': True}, take_audio=lambda: None)
        machine = SimpleNamespace(modem_pcm=source, modem_tx=tx, instructions=0,
                                  cpu=SimpleNamespace(pc=0xAB0F))
        def step(count):
            started.set()
            if not release.wait(3):
                raise RuntimeError('test worker was not released')
            machine.instructions += count
            return 'step limit'
        machine.step = step
        finished = asyncio.get_running_loop().create_future()
        async def handler(reader, writer):
            await reader.read()
            writer.close()
            await writer.wait_closed()
            finished.set_result(True)
        server = await asyncio.start_server(handler, '127.0.0.1', 0)
        bridge = PCMManagerBridge()
        runner = None
        try:
            await bridge.connect(server.sockets[0].getsockname()[1])
            for args in ({'budget': True}, {'budget': 0}, {'batch': 10001}):
                with self.assertRaises(ValueError):
                    await bridge.run(machine, **args)
            self.assertEqual(machine.instructions, 0)
            runner = asyncio.create_task(bridge.run(machine, budget=7, batch=3))
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(.001)
            self.assertTrue(started.is_set())
            runner.cancel()
            await asyncio.sleep(.01)
            self.assertFalse(runner.done(), 'CPU work must be joined before caller cleanup')
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await runner
            self.assertEqual(machine.instructions, 3)
            await bridge.close()
            await asyncio.wait_for(finished, 1)
            self.assertEqual(machine.instructions, 3)
        finally:
            release.set()
            if runner is not None:
                await asyncio.gather(runner, return_exceptions=True)
            await bridge.close()
            server.close()
            await server.wait_closed()

    async def test_tcp_fragmentation_eof_and_close_have_no_cpu_side_effects(self):
        seen = asyncio.get_running_loop().create_future()
        async def handler(reader, writer):
            writer.write(b'T\x02')
            await writer.drain()
            await asyncio.sleep(.001)
            writer.write(b'O\x5a')
            await writer.drain()
            data = await reader.readexactly(1)
            seen.set_result(data)
            writer.close()
            await writer.wait_closed()
        server = await asyncio.start_server(handler, '127.0.0.1', 0)
        bridge = PCMManagerBridge()
        try:
            for port in (True, 0, 65536):
                with self.assertRaises(ValueError):
                    await bridge.connect(port)
            await bridge.connect(server.sockets[0].getsockname()[1])
            samples = cpfsk(b'A', high=False)
            raw = struct.pack('<%dh' % len(samples), *samples)
            for n in range(0, len(raw), 320):
                await bridge.forward(raw[n:n + 320])
            self.assertEqual(await asyncio.wait_for(seen, 1), b'A')
            for _ in range(100):
                if bridge.peer.eof:
                    break
                await asyncio.sleep(.001)
            self.assertTrue(bridge.peer.eof)
            self.assertEqual(bytes(bridge.peer.bytes), b'T\x02O\x5a')
            self.assertEqual(bridge.snapshot()['board_bytes_to_manager'], 1)
            self.assertEqual(bridge.snapshot()['manager_bytes_from_tcp'], 4)
            with self.assertRaises(ValueError):
                await bridge.run(SimpleNamespace(instructions=0))
            await bridge.close()
            await bridge.close()
            self.assertFalse(bridge.snapshot()['connected'])
            self.assertEqual(len(bridge.peer.bytes), 0)
        finally:
            await bridge.close()
            server.close()
            await server.wait_closed()

    async def test_a_manager_hang_up_is_reported_as_one(self):
        async def handler(reader, writer):
            writer.close()                  # the manager giving up on idle
            await writer.wait_closed()
        server = await asyncio.start_server(handler, '127.0.0.1', 0)
        bridge = PCMManagerBridge()
        try:
            await bridge.connect(server.sockets[0].getsockname()[1])
            await asyncio.wait_for(bridge.read_task, 1)
            bridge.peer.receive = lambda pcm: b'x' * 4096   # board bytes to send
            with self.assertRaisesRegex(ValueError, 'manager closed the connection'):
                for _ in range(200):
                    await bridge.forward(bytes(320))
                    await asyncio.sleep(.001)
            self.assertIn(bridge.error, ('BrokenPipeError', 'ConnectionResetError'))
            await bridge.close()            # and close stays quiet after it
        finally:
            await bridge.close()
            server.close()
            await server.wait_closed()

    async def test_input_overflow_is_bounded_and_reported_without_data(self):
        async def handler(reader, writer):
            writer.write(b'Z' * (Bell103AnswerPeer.QUEUE_BYTES + 1))
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        server = await asyncio.start_server(handler, '127.0.0.1', 0)
        bridge = PCMManagerBridge()
        try:
            await bridge.connect(server.sockets[0].getsockname()[1])
            await asyncio.wait_for(bridge.read_task, 1)
            self.assertEqual(bridge.error, 'ValueError')
            self.assertEqual(len(bridge.peer.bytes), bridge.peer.QUEUE_BYTES)
            self.assertNotIn('ZZZ', str(bridge.snapshot()))
            with self.assertRaises(ValueError):
                await bridge.forward(bytes(320))
        finally:
            await bridge.close()
            server.close()
            await server.wait_closed()
