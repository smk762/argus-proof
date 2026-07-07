"""Generation backends — one module per engine, selected by name (config).

Swapping the engine that generates the eval grid is a config change, not a code
change: a caller names a backend and constructs it through :func:`get_backend`.
ComfyUI ships first; cloud / diffusers / A1111 adapters land in Phase 7.
"""

from __future__ import annotations

from argus_proof.backends.base import (
    BackendError,
    GenBackend,
    GenResult,
    ModelResolver,
    ProgressSink,
)
from argus_proof.backends.comfyui import ComfyUIBackend

# The backends this build knows how to construct, by name.
KNOWN_BACKENDS: tuple[str, ...] = ("comfyui",)


def get_backend(name: str, **kwargs: object) -> GenBackend:
    """Construct the backend called *name*, forwarding ``kwargs`` to it.

    Raises :class:`BackendError` for an unknown name so a bad config fails
    loudly instead of silently doing nothing.
    """
    if name == "comfyui":
        return ComfyUIBackend(**kwargs)  # type: ignore[arg-type]
    known = ", ".join(KNOWN_BACKENDS)
    raise BackendError(f"unknown generation backend {name!r} (known: {known})")


__all__ = [
    "KNOWN_BACKENDS",
    "BackendError",
    "ComfyUIBackend",
    "GenBackend",
    "GenResult",
    "ModelResolver",
    "ProgressSink",
    "get_backend",
]
