"""ComfyUI generation backend — the first (public-friendly) adapter.

Drives the ComfyUI HTTP API from a parametric workflow template
(:mod:`argus_proof.backends.workflow`): submit a graph per seed, poll for
completion in short loops, download each image, read back its PNGInfo, and emit
a fully populated :class:`~argus_proof.models.RunManifest` with the SHA256 of
every checkpoint/LoRA and the engine version — so the run reconstructs exactly.

The HTTP layer is a small injectable :class:`~argus_proof.backends.http.Transport`
(default: stdlib ``urllib``), so the adapter is unit-testable without a live ComfyUI.
"""

from __future__ import annotations

import time
import urllib.parse
from pathlib import Path

import structlog
from argus_cortex.gpu import GpuCoordinator

from argus_proof import gpu
from argus_proof.backends.base import (
    BackendError,
    BackendTimeout,
    GenResult,
    ModelResolver,
    ProgressSink,
    build_local_manifest,
    write_manifest,
)
from argus_proof.backends.http import Transport, UrllibTransport
from argus_proof.backends.pnginfo import read_dimensions, read_text_chunks
from argus_proof.backends.workflow import checkpoint_family, render_workflow
from argus_proof.models import (
    BackendCapabilities,
    GeneratedImage,
    ProgressEvent,
    RunManifest,
    RunSpec,
)

logger = structlog.get_logger()

BACKEND_NAME = "comfyui"
DEFAULT_BASE_URL = "http://127.0.0.1:8188"


class ComfyUIBackend:
    """Generate images through a running ComfyUI instance.

    ``workflow_template`` is a ComfyUI API-format graph with ``$placeholder``
    values (see :mod:`argus_proof.backends.workflow`). ``family_templates``
    (family -> graph) overrides it per checkpoint family, so one server can
    evaluate SDXL and Flux LoRAs; a family not listed uses
    ``workflow_template``. ``resolve_model`` maps a
    checkpoint/LoRA filename to a local path so it can be hashed into the
    manifest. ``transport`` defaults to :class:`UrllibTransport` against
    ``base_url`` and is injectable for testing.

    Each run holds a GPU lease on ``comfyui`` from ``gpu_coordinator`` (default: the ``ARGUS_GPU_*`` environment, see
    :mod:`argus_proof.gpu`). When the grant names a ComfyUI ``base_url``, the run goes there instead of ``base_url``.
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
        request_timeout: float = 180.0,
        engine_version: str | None = None,
        family_templates: dict[str, dict] | None = None,
        gpu_coordinator: GpuCoordinator | None = None,
        priority: gpu.PriorityClass = gpu.DEFAULT_PRIORITY,
        seed_retries: int = 1,
    ) -> None:
        # request_timeout is per HTTP call, distinct from the overall per-seed
        # `timeout`: ComfyUI's event loop can stall for minutes while a
        # checkpoint first loads, and a /history poll mustn't die under it.
        self.template = workflow_template
        self.family_templates = {k.lower(): v for k, v in (family_templates or {}).items()}
        self.resolve_model = resolve_model
        self.transport = transport or UrllibTransport(base_url, timeout=request_timeout, label="ComfyUI")
        self.request_timeout = request_timeout
        # Transports for the ComfyUIs the GPU broker sends runs to, by base URL.
        self._transports: dict[str, Transport] = {base_url.rstrip("/"): self.transport}
        self.gpu = gpu_coordinator or gpu.default_coordinator()
        self.priority = priority
        self.client_id = client_id
        self.poll_interval = poll_interval
        self.timeout = timeout
        # A seed whose job outlives `timeout` is cancelled and resubmitted this many
        # times before the run fails (#100): ComfyUI stalls are transient, and the
        # same prompt + seed usually finishes in seconds on a second submission.
        # Only timeouts retry; a rejected workflow or missing image fails at once.
        if seed_retries < 0:
            raise ValueError(f"seed_retries must be >= 0, got {seed_retries}")
        self.seed_retries = seed_retries
        self._engine_version = engine_version

    def template_for(self, checkpoint: str) -> dict:
        """The workflow template for *checkpoint*'s family (see
        :func:`~argus_proof.backends.workflow.checkpoint_family`)."""
        family = checkpoint_family(checkpoint)
        return self.family_templates.get(family, self.template) if family else self.template

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
            # Render once up front too: a value the template has no slot for
            # (guidance on SDXL, a negative on Flux) fails before any GPU spend.
            render_workflow(self.template_for(spec.base_checkpoint), spec, spec.seeds[0])
            try:
                placed = self.transport
                # The version of every ComfyUI that rendered a seed, in first-use order:
                # the broker can move seeds between hosts, and a run split across two
                # versions must say so rather than credit only the last host.
                versions: list[str] = []
                # Each host is asked once per run: a later failed /system_stats on a host that already answered
                # (it may be busy loading a checkpoint) must not add a spurious "unknown".
                host_versions: dict[int, str] = {id(self.transport): manifest.engine_version}
                with gpu.SeedLeases(self.gpu, gpu.GENERATE, caller=spec.run_id, priority=self.priority) as leases:
                    for done, seed in enumerate(spec.seeds, start=1):
                        transport = self._transport_for(leases.current().base_url)
                        if transport is not placed or not versions:
                            placed = transport
                            if id(transport) not in host_versions:
                                host_versions[id(transport)] = self._version_of(transport)
                            version = host_versions[id(transport)]
                            if version not in versions:
                                versions.append(version)
                        images.extend(self._generate_seed(spec, seed, out_dir, emit, transport))
                        emit(ProgressEvent(run_id=spec.run_id, type="progress", completed=done, total=total))
                manifest = manifest.model_copy(update={"engine_version": " + ".join(versions)})
            except gpu.LEASE_ERRORS as exc:
                raise BackendError(gpu.lease_failure(exc)) from exc
            write_manifest(out_dir, manifest)
        except BackendError as exc:
            emit(ProgressEvent(run_id=spec.run_id, type="error", message=str(exc)))
            raise

        emit(ProgressEvent(run_id=spec.run_id, type="done", completed=total, total=total))
        return GenResult(manifest=manifest, images=images)

    def _transport_for(self, base_url: str | None) -> Transport:
        """The transport for the ComfyUI at *base_url*; ``None`` is the configured one."""
        if base_url is None:
            return self.transport
        key = base_url.rstrip("/")
        if key not in self._transports:
            if not key.startswith(("http://", "https://")):
                raise BackendError(f"the GPU broker named an unusable ComfyUI URL: {base_url!r}")
            self._transports[key] = UrllibTransport(key, timeout=self.request_timeout, label="ComfyUI")
        return self._transports[key]

    def _generate_seed(
        self, spec: RunSpec, seed: int, out_dir: Path, emit: ProgressSink, transport: Transport | None = None
    ) -> list[GeneratedImage]:
        transport = transport or self.transport
        graph = render_workflow(self.template_for(spec.base_checkpoint), spec, seed)
        for attempt in range(self.seed_retries + 1):
            resp = transport.post_json("/prompt", {"prompt": graph, "client_id": self.client_id})

            node_errors = resp.get("node_errors")
            if node_errors:
                raise BackendError(f"ComfyUI rejected the workflow for seed {seed}: {node_errors}")
            prompt_id = resp.get("prompt_id")
            if not prompt_id:
                raise BackendError(f"ComfyUI returned no prompt_id for seed {seed}: {resp}")

            try:
                entry = self._await_history(prompt_id, seed, transport)
                break
            except BackendTimeout as exc:
                self._cancel(prompt_id, transport)
                if attempt >= self.seed_retries:
                    raise
                logger.warning("comfyui.seed_retry", run_id=spec.run_id, seed=seed, attempt=attempt + 1, error=str(exc))
        produced = self._collect_images(spec, seed, entry, out_dir, transport)
        if not produced:
            raise BackendError(f"ComfyUI produced no images for seed {seed} (prompt_id {prompt_id})")
        for img in produced:
            emit(ProgressEvent(run_id=spec.run_id, type="image", seed=seed, image_id=img.image_id))
        return produced

    def _await_history(self, prompt_id: str, seed: int, transport: Transport) -> dict:
        """Poll /history/{prompt_id} until the run finishes, errors, or times out.

        A single failed poll (e.g. a request timing out while the engine's event
        loop is busy loading a checkpoint) is retried until the overall deadline —
        only the run-level timeout aborts, not one slow HTTP round-trip.
        """
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                history = transport.get_json(f"/history/{urllib.parse.quote(str(prompt_id))}")
            except BackendError as exc:
                if time.monotonic() >= deadline:
                    raise BackendTimeout(
                        f"ComfyUI run for seed {seed} did not finish within {self.timeout}s: {exc}"
                    ) from exc
                logger.debug("comfyui.poll_retry", prompt_id=prompt_id, error=str(exc))
                time.sleep(self.poll_interval)
                continue
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
                raise BackendTimeout(f"ComfyUI run for seed {seed} did not finish within {self.timeout}s")
            time.sleep(self.poll_interval)

    def _cancel(self, prompt_id: str, transport: Transport) -> None:
        """Best-effort removal of a timed-out job, so a resubmission doesn't queue behind it.

        Drops it from the pending queue and interrupts it if it is the running job.
        The interrupt names the prompt: a ComfyUI that honours ``prompt_id`` stops
        only this job; an older one interrupts whatever is running, which on a
        leased GPU is this run's own work. A failure here is logged, not raised:
        the timeout is the error worth reporting.
        """
        for path, payload in (("/queue", {"delete": [prompt_id]}), ("/interrupt", {"prompt_id": prompt_id})):
            try:
                transport.post_json(path, payload)
            except BackendError as exc:
                logger.warning("comfyui.cancel_failed", prompt_id=prompt_id, path=path, error=str(exc))

    def _collect_images(
        self, spec: RunSpec, seed: int, entry: dict, out_dir: Path, transport: Transport
    ) -> list[GeneratedImage]:
        produced: list[GeneratedImage] = []
        index = 0
        for node_output in entry.get("outputs", {}).values():
            for ref in node_output.get("images", []):
                if ref.get("type") == "temp":
                    continue  # skip preview/temp images, keep saved outputs
                data = self._download(ref, transport)
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

    def _download(self, ref: dict, transport: Transport) -> bytes:
        query = urllib.parse.urlencode(
            {
                "filename": ref.get("filename", ""),
                "subfolder": ref.get("subfolder", ""),
                "type": ref.get("type", "output"),
            }
        )
        return transport.get_bytes(f"/view?{query}")

    # -- manifest ----------------------------------------------------------

    def _build_manifest(self, spec: RunSpec) -> RunManifest:
        return build_local_manifest(
            spec, resolve_model=self.resolve_model, engine=BACKEND_NAME, engine_version=self.engine_version()
        )

    def engine_version(self) -> str:
        """The ComfyUI version, queried once from /system_stats (best-effort)."""
        if self._engine_version is None:
            self._engine_version = self._version_of(self.transport)
        return self._engine_version

    @staticmethod
    def _version_of(transport: Transport) -> str:
        try:
            stats = transport.get_json("/system_stats")
        except BackendError:
            return "unknown"
        system = stats.get("system", {}) if isinstance(stats, dict) else {}
        return system.get("comfyui_version") or "unknown"
