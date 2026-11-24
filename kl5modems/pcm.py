"""The audio every modem here speaks: 8 kHz mono, signed 16-bit little-endian."""

RATE = 8000


def validate_pcm(data):
    if not data or len(data) % 2 or len(data) > 3200:
        raise ValueError('PCM frames must contain 1..1600 signed 16-bit samples')
