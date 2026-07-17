"""A deterministic in-memory generation backend shared across tests.

Writes real (Pillow-rendered) PNGs so downstream phash/scoring code can read
pixel data, and a fully valid manifest — everything the run/score paths need,
with no engine or network.
"""

from __future__ import annotations

from pathlib import Path

from argus_proof.backends.base import GenResult, ProgressSink, write_manifest
from argus_proof.models import (
    BackendCapabilities,
    GeneratedImage,
    LoRARef,
    ModelRef,
    ProgressEvent,
    RunManifest,
    RunSpec,
)


def save_png(path: Path, *, size: tuple[int, int] = (16, 16), color: tuple[int, int, int] = (200, 30, 30)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


class FakeBackend:
    """GenBackend double: one PNG per seed, colour varied per seed for phash."""

    def __init__(self) -> None:
        self.generated: list[RunSpec] = []

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name="fake", supports_seed_set=True, reads_pnginfo=False, streams_progress=True)

    def generate(self, spec: RunSpec, out_dir: Path, progress: ProgressSink | None = None) -> GenResult:
        self.generated.append(spec)
        out_dir.mkdir(parents=True, exist_ok=True)

        def emit(event: ProgressEvent) -> None:
            if progress is not None:
                progress(event)

        emit(ProgressEvent(run_id=spec.run_id, type="start", total=len(spec.seeds)))
        manifest = RunManifest(
            run_id=spec.run_id,
            base_checkpoint=ModelRef(name=spec.base_checkpoint, sha256="0" * 64),
            loras=[LoRARef(name=lo.name, sha256="1" * 64, weight=lo.weight) for lo in spec.loras],
            sampling=spec.sampling,
            prompt=spec.prompt,
            negative_prompt=spec.negative_prompt,
            seeds=list(spec.seeds),
            engine="fake",
            engine_version="0.0",
            source_manifest=spec.source_manifest,
        )
        images: list[GeneratedImage] = []
        for done, seed in enumerate(spec.seeds, start=1):
            image_id = f"{spec.run_id}-{seed}"
            path = out_dir / f"{image_id}.png"
            # Vary the colour per seed so phash sees distinct images.
            save_png(path, color=(seed * 37 % 256, seed * 101 % 256, seed * 197 % 256))
            images.append(
                GeneratedImage(image_id=image_id, run_id=spec.run_id, seed=seed, path=str(path), width=16, height=16)
            )
            emit(ProgressEvent(run_id=spec.run_id, type="image", seed=seed, image_id=image_id))
            emit(ProgressEvent(run_id=spec.run_id, type="progress", completed=done, total=len(spec.seeds)))
        write_manifest(out_dir, manifest)
        emit(ProgressEvent(run_id=spec.run_id, type="done", completed=len(spec.seeds), total=len(spec.seeds)))
        return GenResult(manifest=manifest, images=images)
