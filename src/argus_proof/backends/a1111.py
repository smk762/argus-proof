"""AUTOMATIC1111 / SD.Next generation backend — drives their ``/sdapi`` HTTP API.

Submits a ``txt2img`` request per seed, decodes the base64 PNG(s) it returns, and
emits a reproducible :class:`~argus_proof.models.RunManifest`. The base checkpoint
is selected per-request via ``override_settings`` and LoRA(s) via the A1111
``<lora:name:weight>`` prompt syntax. Model files are SHA256-pinned from disk (via
``resolve_model``), so this targets an A1111/SD.Next whose models are locally
resolvable — the manifest reconstructs the run exactly.

The HTTP layer is the shared injectable :class:`~argus_proof.backends.http.Transport`,
so the adapter is unit-testable without a live A1111.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path

import structlog

from argus_proof.backends.base import BackendError, GenResult, ModelResolver, ProgressSink, build_local_manifest
from argus_proof.backends.http import Transport, UrllibTransport
from argus_proof.backends.pnginfo import read_dimensions
from argus_proof.models import BackendCapabilities, GeneratedImage, ProgressEvent, RunSpec

logger = structlog.get_logger()

BACKEND_NAME = "a1111"
DEFAULT_BASE_URL = "http://127.0.0.1:7860"


class A1111Backend:
    """Generate images through an AUTOMATIC1111 / SD.Next ``/sdapi`` server.

    ``resolve_model`` hashes the base checkpoint + LoRA files into the manifest;
    ``transport`` defaults to :class:`UrllibTransport` against ``base_url`` and is
    injectable for testing. ``engine_version`` is recorded on the manifest (A1111
    exposes no portable version endpoint); pass it to pin the server build.
    """

    def __init__(
        self,
        resolve_model: ModelResolver,
        base_url: str = DEFAULT_BASE_URL,
        transport: Transport | None = None,
        engine_version: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.resolve_model = resolve_model
        self.transport = transport or UrllibTransport(base_url, timeout=timeout, label="A1111")
        self._engine_version = engine_version or "unknown"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=BACKEND_NAME,
            supports_seed_set=True,
            max_loras=None,
            reads_pnginfo=False,
            streams_progress=True,
        )

    def generate(self, spec: RunSpec, out_dir: Path, progress: ProgressSink | None = None) -> GenResult:
        """Run *spec*, writing one image per seed (+ manifest.json) under *out_dir*."""
        logger.debug("a1111.generate", run_id=spec.run_id, seeds=len(spec.seeds), out_dir=str(out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)

        def emit(event: ProgressEvent) -> None:
            if progress is not None:
                progress(event)

        total = len(spec.seeds)
        emit(ProgressEvent(run_id=spec.run_id, type="start", total=total))

        images: list[GeneratedImage] = []
        try:
            manifest = build_local_manifest(
                spec, resolve_model=self.resolve_model, engine=BACKEND_NAME, engine_version=self._engine_version
            )
            for done, seed in enumerate(spec.seeds, start=1):
                images.extend(self._generate_seed(spec, seed, out_dir, emit))
                emit(ProgressEvent(run_id=spec.run_id, type="progress", completed=done, total=total))
        except BackendError as exc:
            emit(ProgressEvent(run_id=spec.run_id, type="error", message=str(exc)))
            raise

        (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        emit(ProgressEvent(run_id=spec.run_id, type="done", completed=total, total=total))
        return GenResult(manifest=manifest, images=images)

    def _generate_seed(self, spec: RunSpec, seed: int, out_dir: Path, emit: ProgressSink) -> list[GeneratedImage]:
        resp = self.transport.post_json("/sdapi/v1/txt2img", self._payload(spec, seed))
        encoded = resp.get("images") or []
        if not encoded:
            raise BackendError(f"A1111 produced no images for seed {seed}")
        produced: list[GeneratedImage] = []
        for index, b64 in enumerate(encoded):
            data = _decode_image(b64, seed)
            suffix = "" if index == 0 else f"-{index}"
            image_id = f"{spec.run_id}-{seed}{suffix}"
            path = out_dir / f"{image_id}.png"
            path.write_bytes(data)
            dims = read_dimensions(data) or (spec.sampling.width, spec.sampling.height)
            produced.append(
                GeneratedImage(
                    image_id=image_id,
                    run_id=spec.run_id,
                    seed=seed,
                    path=str(path),
                    width=dims[0],
                    height=dims[1],
                )
            )
            emit(ProgressEvent(run_id=spec.run_id, type="image", seed=seed, image_id=image_id))
        return produced

    def _payload(self, spec: RunSpec, seed: int) -> dict:
        # A1111 applies LoRAs via the prompt: "<lora:filename-stem:weight>".
        lora_tags = "".join(f" <lora:{Path(lo.name).stem}:{lo.weight}>" for lo in spec.loras)
        return {
            "prompt": spec.prompt + lora_tags,
            "negative_prompt": spec.negative_prompt,
            "steps": spec.sampling.steps,
            "cfg_scale": spec.sampling.cfg,
            "width": spec.sampling.width,
            "height": spec.sampling.height,
            "seed": seed,
            "sampler_name": spec.sampling.sampler,
            "scheduler": spec.sampling.scheduler,
            "batch_size": 1,
            "n_iter": 1,
            "override_settings": {"sd_model_checkpoint": spec.base_checkpoint},
            "override_settings_restore_afterwards": True,
        }


def _decode_image(b64: str, seed: int) -> bytes:
    # A1111 returns raw base64; SD.Next may prefix a "data:image/png;base64," header.
    if "," in b64 and b64.lstrip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        return base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BackendError(f"A1111 returned an undecodable image for seed {seed}: {exc}") from exc
