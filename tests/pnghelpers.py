"""Minimal in-memory PNG builders shared across tests (no Pillow needed)."""

from __future__ import annotations

import struct
import zlib

_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(ctype: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)


def make_png(width: int = 64, height: int = 48, text: dict[str, str] | None = None) -> bytes:
    """A structurally valid PNG (IHDR + optional tEXt chunks + IEND)."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    out = _SIG + _chunk(b"IHDR", ihdr)
    for key, value in (text or {}).items():
        out += _chunk(b"tEXt", key.encode("latin-1") + b"\x00" + value.encode("latin-1"))
    return out + _chunk(b"IEND", b"")


def make_png_ztxt(key: str, value: str) -> bytes:
    """A PNG carrying a single zlib-compressed zTXt chunk."""
    ihdr = struct.pack(">IIBBBBB", 8, 8, 8, 6, 0, 0, 0)
    body = key.encode("latin-1") + b"\x00" + b"\x00" + zlib.compress(value.encode("latin-1"))
    return _SIG + _chunk(b"IHDR", ihdr) + _chunk(b"zTXt", body) + _chunk(b"IEND", b"")
