"""ComfyUI generation backend — the first (public-friendly) adapter.

Drives the ComfyUI HTTP API from a parametric workflow template
(:mod:`argus_proof.backends.workflow`): submit a graph per seed, poll for
completion in short loops, download each image, read back its PNGInfo, and emit
a fully populated :class:`~argus_proof.models.RunManifest` with the SHA256 of
every checkpoint/LoRA and the engine version — so the run reconstructs exactly.

The HTTP layer is a small injectable :class:`Transport` (default: stdlib
``urllib``), so the adapter is unit-testable without a live ComfyUI.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Protocol

import structlog

from argus_proof.backends.base import BackendError, GenResult, ModelResolver, ProgressSink
from argus_proof.backends.pnginfo import read_dimensions, read_text_chunks
from argus_proof.backends.workflow import render_workflow
from argus_proof.hashing import sha256_cached
from argus_proof.models import (
    BackendCapabilities,
    GeneratedImage,
    LoRARef,
    ModelRef,
    ProgressEvent,
    RunManifest,
    RunSpec,
)

logger = structlog.get_logger()

BACKEND_NAME = "comfyui"
DEFAULT_BASE_URL = "http://127.0.0.1:8188"


class Transport(Protocol):
    """The tiny HTTP surface the ComfyUI adapter needs, so tests can fake it."""

    def post_json(self, path: str, payload: dict) -> dict: ...
    def get_json(self, path: str) -> dict: ...
    def get_bytes(self, path: str) -> bytes: ...


class UrllibTransport:
    """Default :class:`Transport` over stdlib ``urllib`` — no runtime deps.

    Wraps transport-level failures as :class:`BackendError` so callers get one
    error type whether the engine is down or a request was malformed.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0) -> None:
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _open(self, req: urllib.request.Request | str) -> bytes:
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError) as exc:
            url = req.full_url if isinstance(req, urllib.request.Request) else req
            raise BackendError(f"ComfyUI request to {url} failed: {exc}") from exc

    def post_json(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return json.loads(self._open(req))

    def get_json(self, path: str) -> dict:
        return json.loads(self._open(self.base + path))

    def get_bytes(self, path: str) -> bytes:
        return self._open(self.base + path)


class ComfyUIBackend:
    """Generate images through a running ComfyUI instance.

    ``workflow_template`` is a ComfyUI API-format graph with ``$placeholder``
    values (see :mod:`argus_proof.backends.workflow`); ``resolve_model`` maps a
    checkpoint/LoRA filename to a local path so it can be hashed into the
    manifest. ``transport`` defaults to :class:`UrllibTransport` against
    ``base_url`` and is injectable for testing.
    """

    def __init__(
        self,
        workflow_template: dict,
        resolve_model: ModelResolver,
        base_url: str = DEFAULT_BASE_URL,
        transport: Transport | None = None,
        client_id: str = "argus-proof",
        poll_interval: float = 1.0,
        timeout: float = 600.0,
        engine_version: str | None = None,
    ) -> None:
        self.template = workflow_template
        self.resolve_model = resolve_model
        self.transport = transport or UrllibTransport(base_url)
        self.client_id = client_id
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._engine_version = engine_version

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=BACKEND_NAME,
            supports_seed_set=True,
            max_loras=None,
            reads_pnginfo=True,
            streams_progress=True,
        )

    # -- generation --------------------------------------------------------

    def generate(self, spec: RunSpec, out_dir: Path, progress: ProgressSink | None = None) -> GenResult:
        """Run *spec*, writing images (+ manifest.json) under *out_dir*."""
        logger.debug("comfyui.generate", run_id=spec.run_id, seeds=len(spec.seeds), out_dir=str(out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)

        def emit(event: ProgressEvent) -> None:
            if progress is not None:
                progress(event)

        total = len(spec.seeds)
        emit(ProgressEvent(run_id=spec.run_id, type="start", total=total))

        images: list[GeneratedImage] = []
        try:
            # Build the manifest FIRST: this resolves + hashes every model, so a
            # missing/unresolvable checkpoint fails fast (before any GPU spend)
            # and inside the error-event path, instead of after the whole grid ran.
            manifest = self._build_manifest(spec)
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
        graph = render_workflow(self.template, spec, seed)
        resp = self.transport.post_json("/prompt", {"prompt": graph, "client_id": self.client_id})

        node_errors = resp.get("node_errors")
        if node_errors:
            raise BackendError(f"ComfyUI rejected the workflow for seed {seed}: {node_errors}")
        prompt_id = resp.get("prompt_id")
        if not prompt_id:
            raise BackendError(f"ComfyUI returned no prompt_id for seed {seed}: {resp}")

        entry = self._await_history(prompt_id, seed)
        produced = self._collect_images(spec, seed, entry, out_dir)
        if not produced:
            raise BackendError(f"ComfyUI produced no images for seed {seed} (prompt_id {prompt_id})")
        for img in produced:
            emit(ProgressEvent(run_id=spec.run_id, type="image", seed=seed, image_id=img.image_id))
        return produced

    def _await_history(self, prompt_id: str, seed: int) -> dict:
        """Poll /history/{prompt_id} until the run finishes, errors, or times out."""
        deadline = time.monotonic() + self.timeout
        while True:
            history = self.transport.get_json(f"/history/{urllib.parse.quote(str(prompt_id))}")
            entry = history.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise BackendError(f"ComfyUI run failed for seed {seed}: {status}")
                # Return on completion, not merely on outputs being present — a run
                # that finished with no SaveImage output is a "no images" error, not
                # a timeout. Fall back to outputs for engines without a status flag.
                if status.get("completed") is True or status.get("status_str") == "success" or entry.get("outputs"):
                    return entry
            if time.monotonic() >= deadline:
                raise BackendError(f"ComfyUI run for seed {seed} did not finish within {self.timeout}s")
            time.sleep(self.poll_interval)

    def _collect_images(self, spec: RunSpec, seed: int, entry: dict, out_dir: Path) -> list[GeneratedImage]:
        produced: list[GeneratedImage] = []
        index = 0
        for node_output in entry.get("outputs", {}).values():
            for ref in node_output.get("images", []):
                if ref.get("type") == "temp":
                    continue  # skip preview/temp images, keep saved outputs
                data = self._download(ref)
                # One image per seed is the norm; suffix only when a seed batches.
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
                        pnginfo=read_text_chunks(data),
                    )
                )
                index += 1
        return produced

    def _download(self, ref: dict) -> bytes:
        query = urllib.parse.urlencode(
            {
                "filename": ref.get("filename", ""),
                "subfolder": ref.get("subfolder", ""),
                "type": ref.get("type", "output"),
            }
        )
        return self.transport.get_bytes(f"/view?{query}")

    # -- manifest ----------------------------------------------------------

    def _build_manifest(self, spec: RunSpec) -> RunManifest:
        base = ModelRef(name=spec.base_checkpoint, sha256=self._hash(spec.base_checkpoint))
        vae = ModelRef(name=spec.vae, sha256=self._hash(spec.vae)) if spec.vae else None
        loras = [LoRARef(name=lo.name, sha256=self._hash(lo.name), weight=lo.weight) for lo in spec.loras]
        return RunManifest(
            run_id=spec.run_id,
            base_checkpoint=base,
            vae=vae,
            loras=loras,
            sampling=spec.sampling,
            prompt=spec.prompt,
            negative_prompt=spec.negative_prompt,
            seeds=list(spec.seeds),
            engine=BACKEND_NAME,
            engine_version=self.engine_version(),
            source_manifest=spec.source_manifest,
            source_manifest_version=spec.source_manifest_version,
            training_run_id=spec.training_run_id,
        )

    def _hash(self, name: str) -> str:
        try:
            path = self.resolve_model(name)
        except Exception as exc:  # resolver signals "not found" however it likes
            raise BackendError(f"cannot hash {name!r} for the manifest: {exc}") from exc
        if not path.is_file():
            raise BackendError(f"cannot hash {name!r}: resolved path {path} is not a file")
        return sha256_cached(path)

    def engine_version(self) -> str:
        """The ComfyUI version, queried once from /system_stats (best-effort)."""
        if self._engine_version is None:
            self._engine_version = self._query_engine_version()
        return self._engine_version

    def _query_engine_version(self) -> str:
        try:
            stats = self.transport.get_json("/system_stats")
        except BackendError:
            return "unknown"
        system = stats.get("system", {}) if isinstance(stats, dict) else {}
        return system.get("comfyui_version") or "unknown"
