"""Read text metadata embedded in a PNG (ComfyUI PNGInfo).

ComfyUI stamps the API graph and UI workflow into every PNG it saves as text
chunks (keys ``prompt`` and ``workflow``). Reading them back lets proof record
what *actually* rendered instead of trusting the request, so params can't drift
from the image. Pure stdlib — no Pillow dependency for a job this small.
"""

from __future__ import annotations

import struct
import zlib

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def read_dimensions(data: bytes) -> tuple[int, int] | None:
    """``(width, height)`` from a PNG's IHDR, or ``None`` if not a valid PNG.

    The real rendered size read from the image, so a manifest records what came
    out rather than what was asked for.
    """
    if not data.startswith(_PNG_SIGNATURE) or len(data) < 24:
        return None
    # IHDR is always the first chunk: width/height are the first two big-endian
    # u32 of its data, which begins 16 bytes in (8 sig + 4 length + 4 type).
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def read_text_chunks(data: bytes) -> dict[str, str]:
    """Parse the tEXt / zTXt / iTXt chunks of a PNG into ``{keyword: text}``.

    Returns ``{}`` for anything that isn't a PNG or carries no text chunks;
    malformed individual chunks are skipped rather than raising, so a readable
    image never fails a run over metadata.
    """
    if not data.startswith(_PNG_SIGNATURE):
        return {}

    out: dict[str, str] = {}
    pos = len(_PNG_SIGNATURE)
    n = len(data)
    while pos + 8 <= n:
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        ctype = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if len(body) < length:
            break  # truncated file
        pos += 12 + length  # length + type + data + CRC
        if ctype == b"IEND":
            break
        try:
            if ctype == b"tEXt":
                keyword, text = body.split(b"\x00", 1)
                out[keyword.decode("latin-1")] = text.decode("latin-1")
            elif ctype == b"zTXt":
                keyword, rest = body.split(b"\x00", 1)
                # rest = 1-byte compression method + zlib-compressed text
                out[keyword.decode("latin-1")] = zlib.decompress(rest[1:]).decode("latin-1")
            elif ctype == b"iTXt":
                keyword, rest = body.split(b"\x00", 1)
                comp_flag = rest[0]
                # skip compression method, language tag, translated keyword
                after_method = rest[2:]
                _lang, after_lang = after_method.split(b"\x00", 1)
                _trans, text_bytes = after_lang.split(b"\x00", 1)
                if comp_flag:
                    text_bytes = zlib.decompress(text_bytes)
                out[keyword.decode("latin-1")] = text_bytes.decode("utf-8")
        except (ValueError, IndexError, zlib.error, UnicodeDecodeError):
            continue  # skip a malformed chunk, keep the rest
    return out
