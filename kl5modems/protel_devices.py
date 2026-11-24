"""External devices fitted to the Protel 4000/8000 board.

These models intentionally stop at the chip boundary.  Public board evidence
identifies the parts, and their manufacturers document their interfaces, but
it is not yet known which G65SC151 pins / address-decode strobes select them
on a Protel board.  Keeping the devices useful-but-UNWIRED lets DD8822 traces
prove the wiring instead of making the emulator's first guess self-fulfilling.

No Protel firmware bytes or EEPROM contents are included here.
"""

from __future__ import annotations

from collections import deque


class ProtelDeviceError(ValueError):
    pass


# ---------------------------------------------------------------------------
# SSI 73K212L Bell 212A / Bell 103 modem
# ---------------------------------------------------------------------------

MODEM_CR0 = 0
MODEM_CR1 = 1
MODEM_DR = 2
MODEM_TR = 3
MODEM_REGISTER_NAMES = {
    MODEM_CR0: "CR0",
    MODEM_CR1: "CR1",
    MODEM_DR: "DR",
    MODEM_TR: "TR",
}


class SSI73K212L:
    """Register/data boundary for the SSI 73K212L modem.

    The original 73K212/K212L documentation exposes four primary 8-bit
    registers at addresses 0..3: CR0, CR1, read-only Detect Register (DR), and
    Tone Register (TR).  RESET clears CR0/CR1/TR.  DR reflects modem-detected
    conditions and is changed only by the modem side of the model.

    The microprocessor *control* interface and the DTE *data* interface are
    kept separate here on purpose.  A later board-wiring adapter can implement
    either the 22-pin LSB-first serial control bus or the 28-pin parallel bus
    without changing modem state.  TXD/RXD are modeled as raw serial bits so
    we do not accidentally turn the control DATA pin into telephone payload.
    """

    def __init__(self, trace=None):
        self.trace = trace if trace is not None else []
        self.txd_bits = deque()
        self.rxd_bits = deque()
        self.reset()

    def reset(self):
        self.reset_generation = getattr(self, 'reset_generation', 0) + 1
        self.cr0 = 0
        self.cr1 = 0
        self.detect = 0
        self.rxd_level = 1
        self.tone = 0
        self.txd_bits.clear()
        self.rxd_bits.clear()
        self.trace.append({"kind": "73k212l-reset"})

    def _check_address(self, address):
        address = int(address)
        if address not in MODEM_REGISTER_NAMES:
            raise ProtelDeviceError("73K212L register address must be 0..3")
        return address

    def read_register(self, address):
        address = self._check_address(address)
        value = (self.cr0, self.cr1, self.detect, self.tone)[address]
        self.trace.append({
            "kind": "73k212l-register-read",
            "address": address,
            "register": MODEM_REGISTER_NAMES[address],
            "value": value,
        })
        return value

    def write_register(self, address, value):
        address = self._check_address(address)
        value = int(value) & 0xFF
        if address == MODEM_DR:
            raise ProtelDeviceError("73K212L Detect Register is read-only")
        if address == MODEM_CR0:
            old = self.cr0
            self.cr0 = value
        elif address == MODEM_CR1:
            old = self.cr1
            self.cr1 = value
        else:
            old = self.tone
            self.tone = value
        self.trace.append({
            "kind": "73k212l-register-write",
            "address": address,
            "register": MODEM_REGISTER_NAMES[address],
            "old": old,
            "value": value,
        })
        if address == MODEM_CR1 and value & 0x04:
            # CR1.D2 is a command: the manufacturer's reset clears all
            # control bits, including this bit. It is not a held-reset latch.
            self.reset()

    def set_detect(self, value, mask=0xFF, source="modem"):
        """Update modem-generated status bits exposed through DR."""
        value = int(value) & 0xFF
        mask = int(mask) & 0xFF
        old = self.detect
        self.detect = (old & ~mask) | (value & mask)
        self.trace.append({
            "kind": "73k212l-detect",
            "old": old,
            "value": self.detect,
            "mask": mask,
            "source": str(source),
        })
        return self.detect

    # DTE serial data plane -------------------------------------------------
    def txd(self, bit):
        bit = 1 if int(bit) else 0
        self.txd_bits.append(bit)
        self.trace.append({"kind": "73k212l-txd", "bit": bit})

    def drain_txd(self, limit=None):
        n = len(self.txd_bits) if limit is None else min(len(self.txd_bits), int(limit))
        out = [self.txd_bits.popleft() for _ in range(max(0, n))]
        return out

    def feed_rxd(self, bits):
        for bit in bits:
            self.rxd_bits.append(1 if int(bit) else 0)
        self.trace.append({"kind": "73k212l-rxd-feed", "bits": len(bits) if hasattr(bits, "__len__") else None})

    def rxd(self, default=1):
        bit = self.rxd_bits.popleft() if self.rxd_bits else (1 if default else 0)
        self.trace.append({"kind": "73k212l-rxd", "bit": bit})
        return bit

    # Serial control helper ------------------------------------------------
    def serial_control_write(self, address, bits_lsb_first):
        """Apply one documented 8-bit LSB-first control-port write.

        This models the chip-side transaction only.  It does not claim that
        the Protel board uses the 22-pin serial package/interface.
        """
        bits = list(bits_lsb_first)
        if len(bits) != 8:
            raise ProtelDeviceError("73K212L serial control write needs exactly 8 bits")
        value = sum((1 if int(bit) else 0) << i for i, bit in enumerate(bits))
        self.write_register(address, value)
        return value

    def serial_control_read(self, address):
        value = self.read_register(address)
        return [(value >> i) & 1 for i in range(8)]

    def snapshot(self):
        return {
            "part": "SSI 73K212L",
            "wired": False,
            "wiring": "unknown Protel board control/data connection",
            "registers": {
                "CR0": self.cr0,
                "CR1": self.cr1,
                "DR": self.detect,
                "TR": self.tone,
            },
            "txd_queued_bits": len(self.txd_bits),
            "rxd_queued_bits": len(self.rxd_bits),
            "rxd_level": self.rxd_level,
            "control_interfaces_supported": ["chip-side serial transaction", "future parallel adapter"],
            "data_interface": "TXD/RXD raw bits",
        }


# ---------------------------------------------------------------------------
# Xicor X28C256 32K x 8 EEPROM
# ---------------------------------------------------------------------------

X28C256_SIZE = 32768
X28C256_PAGE = 64
X28C256_PAGE_LOAD_US = 100
X28C256_WRITE_US = 5000


class X28C256:
    """Deterministic X28C256 byte/page-write model.

    The Xicor part is a 32K x 8 parallel EEPROM.  Consecutive writes to one
    64-byte page within the documented 100 us byte-load interval accumulate in
    a page buffer.  After that interval the self-timed programming cycle begins
    and completes within 5 ms.  While programming, reads expose the documented
    status behavior: I/O7 is the complement of the last byte's I/O7 at the
    last-written address, and I/O6 toggles on successive reads.

    Software-data-protection command sequences are deliberately not guessed
    yet; this device starts unprotected and reports that limitation.
    """

    def __init__(self, data=None, trace=None, write_us=X28C256_WRITE_US):
        self.trace = trace if trace is not None else []
        if data is None:
            self.data = bytearray([0xFF]) * X28C256_SIZE
        else:
            raw = bytes(data)
            if len(raw) != X28C256_SIZE:
                raise ProtelDeviceError("X28C256 initial image must be 32768 bytes")
            self.data = bytearray(raw)
        self.write_us = max(1, int(write_us))
        self.now_us = 0
        self.page_buffer = {}
        self.page_base = None
        self.load_deadline = None
        self.busy_until = None
        self.last_address = None
        self.last_value = None
        self.toggle = False

    @staticmethod
    def _address(address):
        address = int(address)
        if not 0 <= address < X28C256_SIZE:
            raise ProtelDeviceError("X28C256 address outside 0x0000..0x7FFF")
        return address

    @staticmethod
    def _page_base(address):
        return int(address) & ~(X28C256_PAGE - 1)

    @property
    def busy(self):
        return self.busy_until is not None

    @property
    def loading_page(self):
        return bool(self.page_buffer) and not self.busy

    def _begin_program(self):
        if not self.page_buffer or self.busy:
            return
        self.busy_until = self.now_us + self.write_us
        self.load_deadline = None
        self.toggle = False
        self.trace.append({
            "kind": "x28c256-program-start",
            "page": self.page_base,
            "bytes": len(self.page_buffer),
            "busy_until_us": self.busy_until,
        })

    def _finish_program(self):
        if not self.busy:
            return
        for address, value in self.page_buffer.items():
            self.data[address] = value
        count = len(self.page_buffer)
        page = self.page_base
        self.page_buffer = {}
        self.page_base = None
        self.busy_until = None
        self.toggle = False
        self.trace.append({
            "kind": "x28c256-program-complete",
            "page": page,
            "bytes": count,
            "at_us": self.now_us,
        })

    def advance_us(self, delta):
        target = self.now_us + max(0, int(delta))
        # If the page-load window expires before target, programming starts at
        # that exact deadline rather than at the end of the caller's timeslice.
        if self.loading_page and self.load_deadline is not None and target >= self.load_deadline:
            self.now_us = self.load_deadline
            self._begin_program()
        if self.busy and target >= self.busy_until:
            self.now_us = self.busy_until
            self._finish_program()
        self.now_us = target
        return self.now_us

    def write(self, address, value):
        address = self._address(address)
        value = int(value) & 0xFF
        if self.busy:
            raise ProtelDeviceError("X28C256 write attempted during internal programming cycle")
        page = self._page_base(address)
        if self.page_buffer and page != self.page_base:
            # The datasheet requires all writes in one page-load sequence to
            # keep A6..A14 identical. Surfacing a cross-page attempt makes a
            # future Protel bus adapter prove its CE/WE timing instead of
            # silently inventing what happened electrically.
            raise ProtelDeviceError("X28C256 page-load write crossed a 64-byte page")
        if not self.page_buffer:
            self.page_base = page
        self.page_buffer[address] = value
        self.last_address = address
        self.last_value = value
        self.load_deadline = self.now_us + X28C256_PAGE_LOAD_US
        self.trace.append({
            "kind": "x28c256-page-load",
            "address": address,
            "value": value,
            "page": page,
            "buffered": len(self.page_buffer),
            "deadline_us": self.load_deadline,
        })

    def force_program(self):
        """End the page-load window now; useful for a future CE/WE adapter."""
        self._begin_program()

    def read(self, address):
        address = self._address(address)
        if self.busy:
            self.toggle = not self.toggle
            value = self.data[address]
            # I/O6 toggles for reads during the programming cycle.
            if self.toggle:
                value ^= 0x40
            # I/O7 DATA polling applies to the last byte written.
            if address == self.last_address and self.last_value is not None:
                value = (value & 0x7F) | ((~self.last_value) & 0x80)
            self.trace.append({
                "kind": "x28c256-status-read",
                "address": address,
                "value": value,
                "toggle": self.toggle,
            })
            return value
        value = self.data[address]
        self.trace.append({"kind": "x28c256-read", "address": address, "value": value})
        return value

    def image(self):
        """Return persistent EEPROM contents after any completed writes."""
        return bytes(self.data)

    def snapshot(self):
        return {
            "part": "Xicor X28C256",
            "wired": False,
            "wiring": "unknown Protel board address/chip-select decode",
            "size": X28C256_SIZE,
            "page_size": X28C256_PAGE,
            "now_us": self.now_us,
            "loading_page": self.loading_page,
            "buffered_bytes": len(self.page_buffer),
            "page_base": self.page_base,
            "busy": self.busy,
            "busy_until_us": self.busy_until,
            "last_address": self.last_address,
            "software_data_protection": "not modeled",
        }

