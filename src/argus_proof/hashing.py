"""SHA256 hashing of model files — the identity a RunManifest pins.

A manifest records a checkpoint/LoRA by content hash, not just a filename, so a
run can't be silently invalidated by the file behind a name changing. Hashing a
multi-GB checkpoint is expensive, so :func:`sha256_file` streams in chunks and
:func:`sha256_cached` memoizes by ``(path, size, mtime)`` for the process.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB

# Process-lifetime cache: a checkpoint reused across seeds/runs is hashed once.
# Keyed by (resolved path, size, mtime_ns) so an edited file is re-hashed.
_cache: dict[tuple[str, int, int], str] = {}


def sha256_file(path: Path) -> str:
    """Streaming lowercase-hex SHA256 of *path* (no caching)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_cached(path: Path) -> str:
    """SHA256 of *path*, memoized by ``(path, size, mtime)`` for the process."""
    resolved = path.resolve()
    st = resolved.stat()
    key = (str(resolved), st.st_size, st.st_mtime_ns)
    if key not in _cache:
        _cache[key] = sha256_file(resolved)
    return _cache[key]
