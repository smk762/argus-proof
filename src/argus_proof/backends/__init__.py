"""Generation backends — one module per engine, selected by name (config).

Swapping the engine that generates the eval grid is a config change, not a code
change: a caller names a backend and constructs it through :func:`get_backend`,
and scoring/report code is unchanged regardless of which one produced the run.

* ``comfyui`` — a running ComfyUI instance (the first adapter).
* ``diffusers`` — in-process diffusers SDXL pipeline (deterministic, no service).
* ``a1111`` — an AUTOMATIC1111 / SD.Next ``/sdapi`` server.
* ``remote`` — a hosted/cloud endpoint that speaks the proof wire.
"""

from __future__ import annotations

from argus_proof.backends.a1111 import A1111Backend
from argus_proof.backends.base import (
    BackendError,
    GenBackend,
    GenResult,
    ModelResolver,
    ProgressSink,
    build_local_manifest,
    hash_model,
)
from argus_proof.backends.comfyui import ComfyUIBackend
from argus_proof.backends.diffusers import DiffusersBackend
from argus_proof.backends.remote import RemoteBackend

# The backends this build knows how to construct, by name.
_BACKENDS: dict[str, type] = {
    "comfyui": ComfyUIBackend,
    "diffusers": DiffusersBackend,
    "a1111": A1111Backend,
    "remote": RemoteBackend,
}
KNOWN_BACKENDS: tuple[str, ...] = tuple(_BACKENDS)


def get_backend(name: str, **kwargs: object) -> GenBackend:
    """Construct the backend called *name*, forwarding ``kwargs`` to it.

    Raises :class:`BackendError` for an unknown name so a bad config fails
    loudly instead of silently doing nothing.
    """
    try:
        backend_cls = _BACKENDS[name]
    except KeyError:
        known = ", ".join(KNOWN_BACKENDS)
        raise BackendError(f"unknown generation backend {name!r} (known: {known})") from None
    return backend_cls(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "KNOWN_BACKENDS",
    "A1111Backend",
    "BackendError",
    "ComfyUIBackend",
    "DiffusersBackend",
    "GenBackend",
    "GenResult",
    "ModelResolver",
    "ProgressSink",
    "RemoteBackend",
    "build_local_manifest",
    "get_backend",
    "hash_model",
]
