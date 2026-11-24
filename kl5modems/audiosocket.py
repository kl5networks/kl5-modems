"""Asterisk's AudioSocket protocol: frames in, frames out, no sockets.

`app_audiosocket` gives us a plain TCP stream of length-prefixed frames,
which is why it was chosen over ARI external media (a whole RTP stack
and a port allocator) and over `chan_websocket` (not in either of our
Asterisk builds).

    +--------+------------------+------------------+
    | type   | length (2, big)  | payload          |
    | 1 byte | network order    | `length` bytes   |
    +--------+------------------+------------------+

Asterisk sends the UUID first, then audio for the life of the call. We
send audio back and may send TERMINATE to hang up.

## Why this file holds no socket

Everything here is a pure function over bytes. A streaming relay is the
kind of code where the framing bug and the concurrency bug look
identical from the outside -- a call that goes quiet -- and only one of
them can be reproduced at a desk. So the framing is separated out and
tested against the cases that actually break it: a frame split across
two reads, several frames in one read, a length that has arrived but a
payload that has not, and a header that is only two bytes long so far.

TCP will do all four to us, and none of them are rare. A payphone
reaching this over a VPN from a domestic line will find them all in
the first minute.
"""
import uuid as _uuid

# Frame types, as app_audiosocket defines them.
TERMINATE = 0x00     # hang up. Length 0. Either direction.
UUID = 0x01          # 16 raw bytes, Asterisk's first frame.
ERROR = 0xFF         # one byte of error code, Asterisk -> us.
AUDIO = 0x10         # signed linear, 16-bit, 8 kHz, mono, little-endian.

HEADER = 3           # type + 2-byte length
MAX_PAYLOAD = 65535  # what two bytes can say

# What Asterisk reports in an ERROR frame. Worth naming: without them a
# relay logs "error 4" and the next person has to read Asterisk source.
ERRORS = {
    0x00: "none",
    0x01: "out of memory",
    0x02: "write to the channel failed",
    0x03: "read from the channel failed",
    0x04: "frame was not the expected format",
}


class ProtocolError(Exception):
    """A frame that cannot be trusted. The call is over."""


def build(kind, payload=b""):
    """One frame, ready for the wire."""
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError("payload of %d exceeds the 2-byte length field"
                            % len(payload))
    return bytes([kind]) + len(payload).to_bytes(2, "big") + payload


def audio_frame(slin):
    """Audio back to Asterisk. Always type AUDIO and always slin."""
    return build(AUDIO, slin)


def terminate_frame():
    return build(TERMINATE)


class Reader:
    """Incremental frame reader. Feed it whatever TCP hands you.

    `feed(chunk)` returns the frames that are now complete, as
    `(kind, payload)`, and keeps the remainder for next time. That is
    the whole contract, and it is deliberately not a generator over a
    socket: the thing worth testing is what happens at the seams, and a
    seam is just a call boundary here.
    """

    def __init__(self, max_pending=1 << 20):
        self._buf = bytearray()
        # A guard, not a tuning knob. Nothing legitimate leaves a
        # megabyte of half-frame outstanding at 16 kB/s; something that
        # does is a desynchronised stream, and growing a buffer forever
        # is the wrong way to find out.
        self._max_pending = max_pending

    def feed(self, chunk):
        self._buf += chunk
        if len(self._buf) > self._max_pending:
            raise ProtocolError(
                "%d bytes of incomplete frame: the stream is not framed "
                "the way we think it is" % len(self._buf))
        out = []
        while True:
            if len(self._buf) < HEADER:
                return out
            length = int.from_bytes(self._buf[1:3], "big")
            if len(self._buf) < HEADER + length:
                return out          # the header arrived, the payload has not
            kind = self._buf[0]
            payload = bytes(self._buf[HEADER:HEADER + length])
            del self._buf[:HEADER + length]
            out.append((kind, payload))

    @property
    def pending(self):
        """Bytes held back waiting for the rest of their frame."""
        return len(self._buf)


def call_id(payload):
    """The UUID frame's payload as the string Asterisk means by it.

    Sixteen raw bytes, not text. We pass it to KAI as the session id, so
    one call is one session on both sides and a transcript can be found
    from either end without a correlation table.
    """
    if len(payload) != 16:
        raise ProtocolError("UUID frame carried %d bytes, not 16"
                            % len(payload))
    return str(_uuid.UUID(bytes=payload))


def error_text(payload):
    if not payload:
        return "error frame with no code"
    code = payload[0]
    return ERRORS.get(code, "unknown error code %d" % code)
