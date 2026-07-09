"""In-process diffusers generation backend — deterministic, no external service.

Renders each seed with a local `diffusers <https://huggingface.co/docs/diffusers>`_
SDXL pipeline (base checkpoint + LoRA(s) loaded from disk), so a run is fully
self-contained and reproducible: the same weights + seed give the same image, and
the :class:`~argus_proof.models.RunManifest` SHA256-pins every model file.

diffusers + torch are heavy, so they live behind the ``[diffusers]`` extra and are
imported lazily by the default renderer. The pixel-generating step is an injectable
:class:`Renderer` (default: :class:`_DiffusersRenderer`), so the backend's
orchestration — manifest, seed loop, progress, saving — is unit-testable with a
fake renderer, no torch required.

Limitation: the requested ``sampler``/``scheduler`` is mapped to a diffusers
scheduler on a best-effort basis (unknown names fall back to the pipeline default),
and ``clip_skip`` is not applied; the manifest still records what was requested.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import structlog

from argus_proof.backends.base import (
    BackendError,
    GenResult,
    ModelResolver,
    ProgressSink,
    build_local_manifest,
    write_manifest,
)
from argus_proof.models import BackendCapabilities, GeneratedImage, ProgressEvent, RunSpec

if TYPE_CHECKING:
    from PIL.Image import Image

logger = structlog.get_logger()

BACKEND_NAME = "diffusers"


class Renderer(Protocol):
    """The pixel-generating step: render one image for *spec* at *seed*."""

    def render(self, spec: RunSpec, seed: int) -> Image: ...
    def engine_version(self) -> str: ...


class DiffusersBackend:
    """Generate images in-process with a diffusers SDXL pipeline.

    ``resolve_model`` maps a checkpoint/LoRA filename to a local path (hashed into
    the manifest); ``renderer`` does the actual generation and defaults to a lazy
    :class:`_DiffusersRenderer` (needs the ``[diffusers]`` extra), injectable for
    testing. ``device``/``dtype`` override the default renderer's device selection.
    """

    def __init__(
        self,
        resolve_model: ModelResolver,
        renderer: Renderer | None = None,
        engine_version: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
    ) -> None:
        self.resolve_model = resolve_model
        self._renderer = renderer
        self._device = device
        self._dtype = dtype
        self._engine_version = engine_version

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=BACKEND_NAME,
            supports_seed_set=True,
            max_loras=None,
            reads_pnginfo=False,
            streams_progress=True,
        )

    def _renderer_or_default(self) -> Renderer:
        if self._renderer is None:
            self._renderer = _DiffusersRenderer(self.resolve_model, device=self._device, dtype=self._dtype)
        return self._renderer

    def generate(self, spec: RunSpec, out_dir: Path, progress: ProgressSink | None = None) -> GenResult:
        """Run *spec*, writing one image per seed (+ manifest.json) under *out_dir*."""
        logger.debug("diffusers.generate", run_id=spec.run_id, seeds=len(spec.seeds), out_dir=str(out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)

        def emit(event: ProgressEvent) -> None:
            if progress is not None:
                progress(event)

        total = len(spec.seeds)
        emit(ProgressEvent(run_id=spec.run_id, type="start", total=total))

        images: list[GeneratedImage] = []
        try:
            # Manifest first: resolves + hashes every model, so a missing checkpoint
            # fails fast (before loading the pipeline) and via the error-event path.
            manifest = build_local_manifest(
                spec, resolve_model=self.resolve_model, engine=BACKEND_NAME, engine_version=self.engine_version()
            )
            renderer = self._renderer_or_default()
            for done, seed in enumerate(spec.seeds, start=1):
                image = renderer.render(spec, seed)
                image_id = f"{spec.run_id}-{seed}"
                path = out_dir / f"{image_id}.png"
                image.save(path)
                images.append(
                    GeneratedImage(
                        image_id=image_id,
                        run_id=spec.run_id,
                        seed=seed,
                        path=str(path),
                        width=image.width,
                        height=image.height,
                    )
                )
                emit(ProgressEvent(run_id=spec.run_id, type="image", seed=seed, image_id=image_id))
                emit(ProgressEvent(run_id=spec.run_id, type="progress", completed=done, total=total))
        except BackendError as exc:
            emit(ProgressEvent(run_id=spec.run_id, type="error", message=str(exc)))
            raise
        except Exception as exc:  # torch OOM, disk-full on save, … -> uniform BackendError (GenBackend contract)
            error = f"diffusers generation failed: {exc}"
            emit(ProgressEvent(run_id=spec.run_id, type="error", message=error))
            raise BackendError(error) from exc

        write_manifest(out_dir, manifest)
        emit(ProgressEvent(run_id=spec.run_id, type="done", completed=total, total=total))
        return GenResult(manifest=manifest, images=images)

    def engine_version(self) -> str:
        """The diffusers version (from the renderer); cached after the first call."""
        if self._engine_version is None:
            self._engine_version = self._renderer_or_default().engine_version()
        return self._engine_version


# Common sampler names -> diffusers scheduler class names (best-effort). Only names
# whose diffusers default config faithfully matches are listed — e.g. the SDE
# variant (dpmpp_2m_sde) is intentionally omitted rather than silently rendered as
# its non-SDE cousin; an unlisted sampler falls back to the pipeline default.
_SCHEDULERS: dict[str, str] = {
    "euler": "EulerDiscreteScheduler",
    "euler_a": "EulerAncestralDiscreteScheduler",
    "euler_ancestral": "EulerAncestralDiscreteScheduler",
    "ddim": "DDIMScheduler",
    "dpmpp_2m": "DPMSolverMultistepScheduler",
    "unipc": "UniPCMultistepScheduler",
}


class _DiffusersRenderer:
    """The real renderer: a lazily-built, cached diffusers SDXL pipeline.

    Loads the base checkpoint from a single ``.safetensors`` file, applies the
    spec's LoRA(s) at their weights, and renders with a seeded generator so a seed
    is reproducible. The pipeline is rebuilt only when the checkpoint/LoRA set
    changes, so a grid over one LoRA at many prompts loads the weights once.
    """

    def __init__(self, resolve_model: ModelResolver, *, device: str | None = None, dtype: str | None = None) -> None:
        self.resolve_model = resolve_model
        self._device = device
        self._dtype = dtype
        self._pipe = None
        self._pipe_key: tuple | None = None
        self._built_device: str | None = None

    def engine_version(self) -> str:
        try:
            import diffusers
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise BackendError("diffusers backend requires: pip install 'argus-proof[diffusers]'") from exc
        return diffusers.__version__

    def _build(self, spec: RunSpec):  # noqa: ANN202 - returns a diffusers pipeline
        try:
            import diffusers
            import torch
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise BackendError("diffusers backend requires: pip install 'argus-proof[diffusers]'") from exc

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self._dtype is not None:
            dtype = getattr(torch, self._dtype, None)
            if dtype is None:
                raise BackendError(f"unknown torch dtype {self._dtype!r} (e.g. 'float16', 'float32', 'bfloat16')")
        else:
            dtype = torch.float16 if device == "cuda" else torch.float32

        base_path = self.resolve_model(spec.base_checkpoint)
        # Load the requested VAE explicitly so it's actually applied — otherwise the
        # manifest would SHA256-pin a VAE the checkpoint's baked-in one silently
        # replaced (build_local_manifest records spec.vae).
        extra = {}
        if spec.vae:
            extra["vae"] = diffusers.AutoencoderKL.from_single_file(
                str(self.resolve_model(spec.vae)), torch_dtype=dtype
            )
        pipe = diffusers.StableDiffusionXLPipeline.from_single_file(str(base_path), torch_dtype=dtype, **extra).to(
            device
        )

        scheduler = _SCHEDULERS.get(spec.sampling.sampler)
        if scheduler is not None:
            use_karras = spec.sampling.scheduler.lower() == "karras"
            pipe.scheduler = getattr(diffusers, scheduler).from_config(
                pipe.scheduler.config, use_karras_sigmas=use_karras
            )

        for i, lora in enumerate(spec.loras):
            pipe.load_lora_weights(str(self.resolve_model(lora.name)), adapter_name=f"lora{i}")
        if spec.loras:
            pipe.set_adapters(
                [f"lora{i}" for i in range(len(spec.loras))], adapter_weights=[lo.weight for lo in spec.loras]
            )
        self._built_device = device
        return pipe

    def _key(self, spec: RunSpec) -> tuple:
        # Everything baked into the pipeline at build time: base, VAE, the
        # scheduler (sampler + karras flag), and the LoRA set + weights. Omitting
        # any of these would serve a stale pipeline on renderer reuse.
        return (
            spec.base_checkpoint,
            spec.vae,
            spec.sampling.sampler,
            spec.sampling.scheduler,
            tuple((lo.name, lo.weight) for lo in spec.loras),
        )

    def render(self, spec: RunSpec, seed: int) -> Image:
        import torch

        key = self._key(spec)
        if self._pipe is None or self._pipe_key != key:
            self._pipe = None  # free the previous pipeline before building the next (avoid 2x VRAM)
            self._pipe = self._build(spec)
            self._pipe_key = key
        generator = torch.Generator(device=self._built_device).manual_seed(seed)
        result = self._pipe(
            prompt=spec.prompt,
            negative_prompt=spec.negative_prompt or None,
            num_inference_steps=spec.sampling.steps,
            guidance_scale=spec.sampling.cfg,
            width=spec.sampling.width,
            height=spec.sampling.height,
            generator=generator,
        )
        return result.images[0]
