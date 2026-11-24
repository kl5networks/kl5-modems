"""A laboratory PCM connection to a manager listening on loopback.

Bytes from the manager go out as Bell 103 answer-side audio, and the
phone's audio comes back as bytes. ``run`` drives an emulated board the
caller supplies (none is included here), in virtual time, for tests and
trials; it is not a wall-clock Softmodem/Asterisk service.
"""

import asyncio

from .protel_pcm_peer import Bell103AnswerPeer


class PCMManagerBridge:
    def __init__(self):
        self.peer = Bell103AnswerPeer()
        self.reader = self.writer = self.read_task = None
        self.error = None
        self.connected = self.closed = False
        self.tx_bytes = self.rx_bytes = 0

    async def connect(self, port):
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('manager port must be an integer from 1 to 65535')
        if self.connected or self.closed:
            raise ValueError('use one manager connection per bridge')
        self.reader, self.writer = await asyncio.wait_for(
            asyncio.open_connection('127.0.0.1', port), 2)
        self.connected = True
        self.read_task = asyncio.create_task(self._read())

    async def _read(self):
        try:
            while True:
                data = await self.reader.read(4096)
                if not data:
                    self.peer.end_input()
                    return
                self.peer.queue(data)
                self.rx_bytes += len(data)
        except (OSError, ValueError) as exc:
            self.error = type(exc).__name__  # no manager bytes or credentials in reports

    async def forward(self, pcm):
        if not self.connected or self.closed:
            raise ValueError('connect the manager before forwarding PCM')
        if self.error:
            raise ValueError('manager input failed: ' + self.error)
        data = self.peer.receive(pcm)
        if data:
            try:
                self.writer.write(data)
                await asyncio.wait_for(self.writer.drain(), 2)
            except ConnectionError as exc:
                # The manager hung up (a manager gives up after its idle time).
                # Say so: a bare BrokenPipeError surfaces on CPython with the
                # traceback of an earlier raise inside close(), which reads as
                # a teardown bug and is not one (2026-10-04).
                self.error = type(exc).__name__
                raise ValueError('the manager closed the connection') from None
            self.tx_bytes += len(data)

    async def run(self, machine, budget=32000000, batch=5000):
        if (type(budget) is not int or not 1 <= budget <= 128000000
                or type(batch) is not int or not 1 <= batch <= 10000):
            raise ValueError('use a 1..128000000 instruction budget and 1..10000 batch')
        if not self.connected or self.closed:
            raise ValueError('connect the manager before running firmware')
        source, tx = getattr(machine, 'modem_pcm', None), getattr(machine, 'modem_tx', None)
        if source is None or tx is None or not source.bind_pc0:
            raise ValueError('select modem PCM, PC0 RXD and PB3 transmit explicitly')
        start = machine.instructions
        reason = 'step limit'
        while machine.instructions - start < budget:
            if (machine.modem_pcm is not source or machine.modem_tx is not tx
                    or not source.snapshot()['enabled'] or not tx.snapshot()['enabled']):
                raise ValueError('firmware PCM sources replaced, reset or completed')
            if self.error:
                raise ValueError('manager input failed: ' + self.error)
            # Stream long commands through the two-second phone FIFO. Never
            # enqueue a whole packet's waveform, truncate it or clear input.
            while source.snapshot()['queued_samples'] < 320:
                source.feed(self.peer.render())
            worker = asyncio.create_task(asyncio.to_thread(
                machine.step, min(batch, budget - (machine.instructions - start))))
            try:
                reason = await asyncio.shield(worker)
            except asyncio.CancelledError:
                # Cancellation cannot stop a running Python worker. Join this
                # bounded batch before the caller releases hardware/socket state.
                await worker
                raise
            while (pcm := tx.take_audio()) is not None:
                await self.forward(pcm)
            # TCP read and manager tasks get their own event-loop progress;
            # the CPU worker has no dependency on a UART sample address.
            await asyncio.sleep(0)
            if reason != 'step limit':
                break
        return {'pc': '%04X' % machine.cpu.pc, 'reason': reason,
                'instructions_executed': machine.instructions - start,
                'budget_exhausted': reason == 'step limit', **self.snapshot()}

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self.read_task is not None:
            self.read_task.cancel()
            await asyncio.gather(self.read_task, return_exceptions=True)
        if self.writer is not None:
            self.writer.close()
            try:
                await asyncio.wait_for(self.writer.wait_closed(), 2)
            except (OSError, asyncio.TimeoutError):
                pass
        self.peer.bytes.clear()
        self.peer.word = ()
        self.peer.word_bit = self.peer.preamble = 0
        self.peer.eof, self.peer.tail, self.peer.remaining = True, 0, 0

    def snapshot(self):
        return {'scope': 'loopback TCP manager with virtual-time Bell 103 PCM',
                'connected': self.connected and not self.closed,
                'closed': self.closed, 'manager_input_error': self.error,
                'board_bytes_to_manager': self.tx_bytes,
                'manager_bytes_from_tcp': self.rx_bytes,
                'peer': self.peer.snapshot(), 'real_time_verified': False,
                'external_softmodem_verified': False, 'physical_wiring_verified': False}
