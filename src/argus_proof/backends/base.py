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

from argus_proof.models import (
    BackendCapabilities,
    GeneratedImage,
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
