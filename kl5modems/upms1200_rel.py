"""Byte-level REL framing recovered from exact UPMS1200 ROM helpers.

This diagnostic codec does not select tones, bit timing, packet sequence
semantics or host delivery. The helper probe explicitly supplies a caller
and consumes raw transmit requests; it is separate from reset execution.
"""

from dataclasses import dataclass

class UPMS1200Error(ValueError):
    pass

MAX_REL_PAYLOAD = 512  # Diagnostic bound, not a proven firmware capacity.
MAX_REL_WIRE = 2 * (1 + MAX_REL_PAYLOAD + 2) + 3 + 1


def _bytes(value, name, maximum):
    if not isinstance(value, (bytes, bytearray)) or len(value) > maximum:
        raise UPMS1200Error("%s must be at most %d bytes" % (name, maximum))
    return bytes(value)


def rel_crc16(data):
    """ROM $E400: reflected 0x8408, initial zero, no final XOR."""
    data = _bytes(data, "CRC input", MAX_REL_PAYLOAD + 1)
    crc = 0
    for value in data:
        crc ^= value
        for _ in range(8):
            crc = (crc >> 1) ^ (0x8408 if crc & 1 else 0)
    return crc


@dataclass(frozen=True)
class RELFrame:
    control: int
    payload: bytes
    checksum: int
    leading_ack: bool = False


def encode_rel_frame(control, payload=b"", *, leading_ack=False):
    """Encode the ROM transmitter's control mask, escaping and trailer.

    Control bit meanings are not assigned here. ROM $E6C6 emits mask 0x78.
    A single leading 0x06 is optionally skipped by the receive entry.
    """
    payload = _bytes(payload, "REL payload", MAX_REL_PAYLOAD)
    if type(control) is not int or not 0 <= control <= 0x78 or control & ~0x78:
        raise UPMS1200Error("REL transmit control must use only bits in 0x78")
    if type(leading_ack) is not bool:
        raise UPMS1200Error("leading_ack must be boolean")
    body = bytes([control]) + payload
    checksum = rel_crc16(body)
    encoded = bytearray(b"\x06" if leading_ack else b"")
    previous = None
    for value in body + checksum.to_bytes(2, "little"):
        if previous == 0xA5 and value in (0x5A, 0x5B):
            encoded.append(0x5B)
        encoded.append(value)
        previous = value
    encoded.extend(b"\xff\xa5\x5a")
    return bytes(encoded)


def decode_rel_frame(wire):
    """Decode one complete frame, including the receiver's quote behavior.

    A5 5B clears the quote flag and drops 5B; the following byte is taken
    literally. A5 5A ends the frame. The receiver delays four data bytes,
    retaining CRC-low, CRC-high, FF and A5 when 5A arrives.
    """
    wire = _bytes(wire, "REL wire frame", MAX_REL_WIRE)
    leading_ack = wire.startswith(b"\x06")
    source = wire[1:] if leading_ack else wire
    decoded = bytearray()
    after_a5 = False
    ended = False
    for index, value in enumerate(source):
        if after_a5 and value == 0x5B:
            after_a5 = False
            continue
        if after_a5 and value == 0x5A:
            if index != len(source) - 1:
                raise UPMS1200Error("bytes follow the REL terminator")
            ended = True
            break
        decoded.append(value)
        after_a5 = value == 0xA5
    if not ended:
        raise UPMS1200Error("REL terminator missing")
    if len(decoded) < 5 or decoded[-2:] != b"\xff\xa5":
        raise UPMS1200Error("REL trailer must retain CRC-low, CRC-high, FF, A5")
    body = bytes(decoded[:-4])
    if len(body) > MAX_REL_PAYLOAD + 1:
        raise UPMS1200Error("REL payload exceeds diagnostic bound")
    checksum = int.from_bytes(decoded[-4:-2], "little")
    if rel_crc16(body) != checksum:
        raise UPMS1200Error("REL checksum mismatch")
    return RELFrame(body[0], body[1:], checksum, leading_ack)


class _ROMHelpers:
    """Synthetic caller for private helpers; never an ordinary modem run."""
    def __init__(self, raw):
        from .upms1200 import UPMS1200Machine
        self.machine = UPMS1200Machine(raw)
        # $E80A dispatches its CRC call through this initialized RAM vector.
        self.machine.bus[0x7AA4] = 0x00
        self.machine.bus[0x7AA5] = 0xE4
        self.raw_transmit = []

    def call(self, entry, value, *, service_transmit=False):
        m = self.machine
        m.cpu.pc, m.cpu.a = entry, value
        m.cpu.stPushWord(0x7EFF)
        for _ in range(512):
            if m.cpu.pc == 0x7F00:
                if m.cpu.sp != 0xFF:
                    raise UPMS1200Error("REL helper unbalanced synthetic stack")
                return m.cpu.a, int(bool(m.cpu.p & m.cpu.CARRY))
            if service_transmit and m.cpu.pc == 0xE3FC:
                self.raw_transmit.append(m.bus[0xD7])
                # Explicit synthetic consumer for the pending-byte handshake.
                # No timer ISR, UART bit, tone or physical transmit is claimed.
                m.bus[0xD2] = m.bus[0xD2] & ~4
            if m.run(1) != "instruction-budget":
                raise UPMS1200Error("REL helper stopped before synthetic return")
        raise UPMS1200Error("REL helper exceeded 512-instruction budget")

    def crc(self):
        return self.machine.bus[0xDA] | self.machine.bus[0xDB] << 8

    def receive(self, wire):
        m = self.machine
        source = wire[1:] if wire.startswith(b"\x06") else wire
        warmup = 4
        body = bytearray()
        terminated = False
        for value in source:
            result, carry = self.call(0xE7CE, value)
            if carry:
                if result == 0:  # Quote marker; no FIFO shift or CRC update.
                    continue
                terminated = True
                break
            if warmup:
                warmup -= 1
                continue
            body.append(result)
            self.call(0xE400, result)
        if not terminated or not body:
            return {"accepted": False, "body": bytes(body), "gate_reached": False}
        # Enter the executed terminator/CRC gate with the filter's flags.
        # CRC helper calls have finished before the filter's final return.
        m.cpu.pc = 0xE563
        for _ in range(64):
            if m.cpu.pc in (0xE578, 0xE5DC):
                return {"accepted": m.cpu.pc == 0xE578, "body": bytes(body), "gate_reached": True}
            if m.run(1) != "instruction-budget":
                break
        raise UPMS1200Error("REL validation gate did not reach a bounded outcome")


def probe_rel_frame_helpers(raw):
    """Compare independent codec with bounded ROM CRC/TX/RX helper calls.

    The fixed corpus covers all byte values and A5-following values, long
    frames, checksum-byte escaping, and CRC/trailer corruption. It checks
    the receive CRC gate only, not downstream sequence/session acceptance.
    """
    from .upms1200 import UPMS1200Image
    image = UPMS1200Image(raw)
    if image.revision not in ("3.7", "4.1"):
        raise UPMS1200Error("REL helper probe requires an exact UPMS 3.7 or 4.1 image")
    payloads = [b"", b"123456789", bytes(range(256)), bytes(range(256)) * 2,
                b"\xa5\x5a\xa5\x5b\xa5\xa5"]
    payloads += [bytes([value]) for value in range(256)]
    payloads += [bytes([0xA5, value]) for value in range(256)]
    # Fixed vectors whose checksum bytes require escaping, also covered by
    # explicit CRC constants in ordinary tests.
    crc_escape_payloads = (bytes.fromhex("d9b6"), bytes.fromhex("51e7"))
    payloads += list(crc_escape_payloads)
    valid = 0
    for index, payload in enumerate(payloads):
        control = (index % 16) << 3
        if index >= len(payloads) - 2:
            control = 0x20
        wire = encode_rel_frame(control, payload)
        decoded = decode_rel_frame(wire)
        if (decoded.control, decoded.payload) != (control, payload):
            raise UPMS1200Error("REL codec round-trip failed at case %d" % index)
        tx = _ROMHelpers(raw)
        body = bytes([control]) + payload
        for value in body:
            tx.call(0xE80A, value, service_transmit=True)
        expected_crc = rel_crc16(body)
        if tx.crc() != expected_crc:
            raise UPMS1200Error("ROM CRC differs from codec at case %d" % index)
        for value in expected_crc.to_bytes(2, "little"):
            tx.call(0xE80A, value, service_transmit=True)
        if bytes(tx.raw_transmit) + b"\xff\xa5\x5a" != wire:
            raise UPMS1200Error("ROM escaping differs from codec at case %d" % index)
        rx = _ROMHelpers(raw).receive(wire)
        if not rx["accepted"] or rx["body"] != body:
            raise UPMS1200Error("ROM receive CRC gate rejected case %d" % index)
        valid += 1
    receive_only = 0
    for wire in (encode_rel_frame(0x20, b"\xa5\x33", leading_ack=True),
                 encode_rel_frame(0x20, b"\xa5\x33").replace(b"\xa5\x33", b"\xa5\x5b\x33")):
        decoded = decode_rel_frame(wire)
        rx = _ROMHelpers(raw).receive(wire)
        if not rx["accepted"] or rx["body"] != bytes([decoded.control]) + decoded.payload:
            raise UPMS1200Error("ROM receive filter rejected quote/leading-ACK fixture")
        receive_only += 1
    valid_wire = encode_rel_frame(0x20, b"check")
    corrupt = [valid_wire[:2] + bytes([valid_wire[2] ^ 1]) + valid_wire[3:],
               valid_wire[:-3] + b"\x00\xa5\x5a",
               valid_wire[:-1]]
    rejected = 0
    for wire in corrupt:
        try:
            decode_rel_frame(wire)
        except UPMS1200Error:
            pass
        else:
            raise UPMS1200Error("REL codec accepted a corrupted fixture")
        if _ROMHelpers(raw).receive(wire)["accepted"]:
            raise UPMS1200Error("ROM receive CRC gate accepted a corrupted fixture")
        rejected += 1
    return {"revision": image.revision, "valid_cases_passed": valid,
            "receive_only_cases_passed": receive_only,
            "corrupt_cases_rejected": rejected, "crc_escape_cases": len(crc_escape_payloads),
            "crc_check_123456789": "%04X" % rel_crc16(b"123456789"),
            "entries": {"crc": "E400", "escape_tx": "E80A", "filter_rx": "E7CE", "crc_gate": "E563"},
            "scope": "isolated byte-level helpers and CRC gate; downstream sequence/session acceptance untested",
            "synthetic_caller": True, "synthetic_raw_transmit_consumer": True,
            "reset_path_exercised": False, "native_modulation_verified": False,
            "host_delivery_verified": False, "physical_interoperability_verified": False}


def probe_rel_transmit_helpers(raw):
    """Drive the actual timer ISR with synthetic entries and pending bytes.

    This verifies the serializer's column-register requests, not analog
    output. Dispatch and pending-byte inputs are explicit helper fixtures.
    Timer entries are ordinal events, not scheduled hardware interrupts.
    """
    from .upms1200 import UPMS1200Image, UPMS1200Machine
    image = UPMS1200Image(raw)
    if image.revision not in ("3.7", "4.1"):
        raise UPMS1200Error("REL transmit probe requires an exact UPMS 3.7 or 4.1 image")
    cases = [bytes([value]) for value in range(256)]
    cases += [b"\x55\xaa", encode_rel_frame(0x20, bytes.fromhex("a55aa55b"))]
    cell_total = 0
    for case, values in enumerate(cases):
        m = UPMS1200Machine(raw)
        m.bus[0x7AEA], m.bus[0x7AEB] = 0xCB, 0xE1
        m.bus[0xD2], m.bus[0xE4] = 0x80, 3
        m.bus.ctu.write(0xF7, 0x64)
        submitted = 0
        requests = []
        for tick in range(len(values) * 40 + 4):
            if submitted < len(values) and not m.bus[0xD2] & 4:
                m.bus[0xD7] = values[submitted]
                m.bus[0xD2] = m.bus[0xD2] | 4
                submitted += 1
            before = m.bus.io_writes[0xF6]
            m.cpu.pc = 0xE1BB
            m.cpu.stPushWord(0x7F00)
            m.cpu.stPush(m.cpu.p)
            if m.run(256, stop_pc=0x7F00) != "helper-return" or m.cpu.sp != 0xFF:
                raise UPMS1200Error("REL timer ISR did not return to synthetic caller")
            writes = m.bus.io_writes[0xF6] - before
            if writes > 1:
                raise UPMS1200Error("REL timer ISR wrote multiple column requests")
            if writes:
                requests.append((tick, m.bus.ctu.column_reload))
        bits = []
        for value in values:
            bits += [0] + [(value >> bit) & 1 for bit in range(8)] + [1]
        expected = [(index * 4, 0x1E if bit else 0x38) for index, bit in enumerate(bits)]
        if requests != expected:
            raise UPMS1200Error("REL timer serializer differs from 8N1 at case %d" % case)
        cell_total += len(requests)
    return {"revision": image.revision, "cases_passed": len(cases),
            "column_requests_checked": cell_total,
            "timer_entry": "E1BB", "synthetic_dispatch": "7AEA -> E1CB",
            "column_store": "E239", "column_reload_by_bit": {"0": 0x38, "1": 0x1E},
            "timer_entries_per_cell": 4,
            "serializer": "start 0, eight data bits LSB first, one stop 1; no parity",
            "scope": "isolated timer ISR serializer; no tone synthesis or physical timing verified",
            "synthetic_caller": True, "synthetic_timer_entries": True,
            "reset_path_exercised": False, "physical_interoperability_verified": False}
