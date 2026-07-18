"""Shared helpers for the concrete scorers.

Kept in one place so the import-availability probe, the ``[0, 1]`` clamp, and the
perceptual-hash cache are consistent across every scorer instead of copy-pasted.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imagehash import ImageHash


def module_available(*names: str) -> bool:
    """Whether every named module imports cleanly.

    Catches *any* import-time failure, not just :class:`ImportError` — a
    half-installed torch/onnxruntime/insightface stack raises ``OSError`` /
    ``RuntimeError`` (missing libGL, CUDA/driver mismatch, numpy ABI break), and
    a scorer's ``is_available()`` must report ``False`` for those rather than let
    the exception crash the whole scoring run.
    """
    for name in names:
        try:
            importlib.import_module(name)
        except Exception:
            return False
    return True


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile *q* in ``[0, 1]`` of an ascending list.

    Shared by the tail-aggregate views (safety, policy moderation) that care about
    the extremes of a distribution rather than its mean. Empty list → ``0.0``.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    if lo + 1 >= len(sorted_values):
        return sorted_values[-1]
    return sorted_values[lo] + (pos - lo) * (sorted_values[lo + 1] - sorted_values[lo])


def clamp01(value: float) -> float:
    """Clamp a finite value to ``[0, 1]``; pass a non-finite value through.

    ``max(0.0, min(1.0, nan))`` is ``1.0`` in CPython (``nan < 1.0`` is False), so
    a naive clamp turns a NaN score into a *perfect* 1.0 and slips past the
    orchestrator's ``[0, 1]`` guard. Returning the non-finite value unchanged lets
    that guard reject it loudly instead of silently auto-passing garbage.
    """
    if not math.isfinite(value):
        return value
    return max(0.0, min(1.0, value))


# Process-lifetime pHash cache, keyed by (path, size, mtime, hash_size) so an
# edited file re-hashes. Shared across the deduper and diversity scorer (and
# repeated runs) so each image is opened and hashed once, not once per scorer.
_phash_cache: dict[tuple[str, int, int, int], ImageHash] = {}


def phash_of(path: str, hash_size: int) -> ImageHash:
    """Perceptual hash of *path*, memoized by ``(path, size, mtime, hash_size)``."""
    import imagehash
    from PIL import Image

    resolved = Path(path)
    st = resolved.stat()
    key = (str(resolved), st.st_size, st.st_mtime_ns, hash_size)
    cached = _phash_cache.get(key)
    if cached is None:
        with Image.open(resolved) as im:
            cached = imagehash.phash(im, hash_size=hash_size)
        _phash_cache[key] = cached
    return cached
