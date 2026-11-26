# kl5modems

**By John Walker, for [KL5 Networks](https://kl5networks.com)** (White Clay Technologies LLC).

Software modems for payphones that call home to a manager. Pure Python,
standard library only, 8 kHz 16-bit audio in and out.

Payphones such as the Protel 4000/8000 and 7000 and the Nortel Millennium
call their manager by modem. Put the phone on a VoIP line and there is no
modem at the other end, only audio. These modules are that other end:
audio from Asterisk in, the phone's bytes out to whatever manager program
you run, and the manager's bytes back to the phone as audio.

## Why KL5 exists

KL5 is a hobby project, made for fun. Payphones like the Protel and the
Nortel Millennium are great machines, and most of the ones that survive sit
in garages collecting dust because the network they were built for is gone.
KL5 is a way to get them working again: making calls, taking coins, calling
in to a manager and being programmed, on an ordinary VoIP line.

The hope is to bring more people in to set up and use their own phones, and
to make that easy. A phone that calls in needs something to answer it, so
the project grew a management console, and the console grew a fictional
telephone company to run it: KL5 Networks, at
[kl5networks.com](https://kl5networks.com). The company is make-believe.
The phones, the call-ins and this code are real.

## What is here

| Module | What it is |
|---|---|
| `protel_bell103` | Bell 103 (300 bps FSK) receiver |
| `protel_pcm_peer` | Bell 103 and Bell 212A answer-side modems, bytes to and from PCM |
| `dpsk212` | Streaming Bell 212A / V.22 DPSK (1200 bps): modulator and a receiver that finds its own timing |
| `upms1200_dpsk`, `upms1200_dpsk_pcm` | Reference 212A DPSK pieces the streaming modem is tested against |
| `native1200` | The Protel native 1200 FSK spoken by the Protel 7000 and the UPMS1200 modem |
| `upms1200_rel`, `native_rel` | The UPMS1200 REL link: frames, CRC, and a station that answers like the real modem |
| `native_modem` | A software UPMS1200: native 1200 plus REL over Asterisk AudioSocket |
| `protel_automodem` | One answering modem for every Protel: listens first, then becomes Bell 103, 212A or native 1200 |
| `audiosocket` | Asterisk AudioSocket framing, no sockets |
| `line_model`, `line_trial` | A phone line that is not ideal: G.711, noise, echo, and an ATA's DTMF-detector mutes |
| `protel_devices` | Register-level model of the 73K212L modem chip |
| `protel_pcm_manager` | Laboratory glue between a modem and a manager on loopback |

Most modules are pure: samples in, samples or bytes out, no clock and no
socket, so a whole call can be run at a desk. `native_modem.Bridge` and
`protel_automodem` add the asyncio glue for a live call.

## Using it with Asterisk

Asterisk's `Softmodem()` already speaks Bell 103 and V.22. These modules
are for what it cannot do: the Protel native 1200, and one answering modem
that works out which of the three a Protel is using. In the dialplan, hand
the call to `AudioSocket()` and point it at a process running
`native_modem.Bridge` (or the auto modem), which connects onward to your
manager over plain TCP:

    same => n,Answer()
    same => n,AudioSocket(<uuid>,127.0.0.1:2528)

Bind it to loopback or a private address. It is a byte pipe into your
manager and has no authentication of its own.

## Tests

    python3 -m unittest discover -s tests -t .

No network, no Asterisk, no third-party packages.

## What this is not

Not affiliated with Protel, Nortel or Grandstream. No firmware, ROM image or
manufacturer software is included or needed. Some behaviour here was
worked out by running a phone's own firmware in an emulator; that emulator
and that firmware are not part of this repository, and claims marked
INFERRED in the code have not been measured on a real board. For phones you
own.

## Licence and credits

MIT, Copyright (c) 2026 White Clay Technologies LLC. See `LICENSE`.
Written by John Walker for White Clay Technologies LLC, part of KL5 Networks.

With thanks to the US Payphones wiki
([payphones.telesfor.org](https://payphones.telesfor.org/), Piotr Czerniak,
with PhilTel and danc256), to PhilTel, and to the participants in the
Millennium and Protel channels on the PhilTel Discord.
