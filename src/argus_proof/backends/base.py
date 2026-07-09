"""The pluggable generation backend interface.

A backend turns a :class:`~argus_proof.models.RunSpec` (a LoRA + base checkpoint
+ prompt set) into scored-ready images plus a fully reproducible
:class:`~argus_proof.models.RunManifest`. The interface is a
:class:`~typing.Protocol` so ComfyUI (shipped first), and later cloud /
diffusers / A1111 adapters, are swappable by config, not code.
"""

from __future__ import annotations

import glob
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from argus_proof.hashing import sha256_cached
from argus_proof.models import (
    BackendCapabilities,
    GeneratedImage,
    LoRARef,
    ModelRef,
    ProgressEvent,
    ProofError,
    RunManifest,
    RunSpec,
)


class BackendError(ProofError):
    """A generation failure: engine unreachable, bad workflow, missing model."""


# Resolve a model filename (as a RunSpec names it) to a local file, so the
# backend can hash it into the manifest. Raises if the file can't be found.
ModelResolver = Callable[[str], Path]


def make_dir_resolver(root: Path) -> ModelResolver:
    """A :data:`ModelResolver` that finds model files under *root*.

    Engines name a model relatively (``"lora.safetensors"`` or
    ``"sdxl/base.safetensors"``); this tries that path under *root* first, then
    falls back to a recursive search by basename so a checkpoint in
    ``models/checkpoints`` and a LoRA in ``models/loras`` both resolve from one
    root. Raises :class:`FileNotFoundError` (which the backend turns into a
    :class:`BackendError`) if nothing matches.
    """
    root = root.expanduser()

    def resolve(name: str) -> Path:
        direct = root / name
        if direct.is_file():
            return direct
        # glob.escape so metacharacters in a real filename (e.g. "subject[v2].safetensors")
        # are matched literally instead of parsed as a glob pattern (which matches nothing).
        basename = glob.escape(Path(name).name)
        for candidate in sorted(root.rglob(basename)):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"no model named {name!r} found under {root}")

    return resolve


# Sink for streamed progress (NDJSON/SSE). Backends call it as work proceeds;
# a None callback means "don't stream".
ProgressSink = Callable[[ProgressEvent], None]


@dataclass
class GenResult:
    """A completed generation run: the reproducible manifest + the images.

    Pairs the two acceptance artifacts of a run so callers get them together —
    the ``manifest`` reconstructs the run exactly; ``images`` are what to score.
    """

    manifest: RunManifest
    images: list[GeneratedImage] = field(default_factory=list)


@runtime_checkable
class GenBackend(Protocol):
    """A generation backend: capability descriptor + ``generate``.

    Implementations write the produced images into ``out_dir`` and return a
    :class:`GenResult`. They should emit :class:`ProgressEvent`s through
    ``progress`` (when given) rather than blocking silently, and raise
    :class:`BackendError` on any failure that stops the run.
    """

    def capabilities(self) -> BackendCapabilities:
        """What this backend supports (name, seed-set, LoRA count, PNGInfo)."""
        ...

    def generate(self, spec: RunSpec, out_dir: Path, progress: ProgressSink | None = None) -> GenResult:
        """Run *spec*, writing images under *out_dir*; return manifest + images."""
        ...


def hash_model(resolve_model: ModelResolver, name: str) -> str:
    """Resolve *name* to a local file and return its SHA256, or raise
    :class:`BackendError` if it can't be resolved/hashed."""
    try:
        path = resolve_model(name)
    except Exception as exc:  # resolver signals "not found" however it likes
        raise BackendError(f"cannot hash {name!r} for the manifest: {exc}") from exc
    if not path.is_file():
        raise BackendError(f"cannot hash {name!r}: resolved path {path} is not a file")
    return sha256_cached(path)


def build_local_manifest(
    spec: RunSpec,
    *,
    resolve_model: ModelResolver,
    engine: str,
    engine_version: str,
) -> RunManifest:
    """A reproducible :class:`RunManifest` for a backend whose weights are on disk.

    Resolves and SHA256-pins the base checkpoint, VAE, and every LoRA (via
    *resolve_model*) so the run reconstructs exactly — the shared path for the
    local backends (ComfyUI, diffusers, A1111). A missing/unresolvable model
    raises :class:`BackendError` here, before any generation. (A purely remote
    backend, whose weights aren't local, builds its manifest from the service's
    own response instead — see :mod:`argus_proof.backends.remote`.)
    """
    return RunManifest(
        run_id=spec.run_id,
        base_checkpoint=ModelRef(name=spec.base_checkpoint, sha256=hash_model(resolve_model, spec.base_checkpoint)),
        vae=ModelRef(name=spec.vae, sha256=hash_model(resolve_model, spec.vae)) if spec.vae else None,
        loras=[LoRARef(name=lo.name, sha256=hash_model(resolve_model, lo.name), weight=lo.weight) for lo in spec.loras],
        sampling=spec.sampling,
        prompt=spec.prompt,
        negative_prompt=spec.negative_prompt,
        seeds=list(spec.seeds),
        engine=engine,
        engine_version=engine_version,
        source_manifest=spec.source_manifest,
        source_manifest_version=spec.source_manifest_version,
        training_run_id=spec.training_run_id,
    )
