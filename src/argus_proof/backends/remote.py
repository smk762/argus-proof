"""Remote / cloud generation backend — offload generation to a hosted service.

Posts a :class:`~argus_proof.models.RunSpec` to an HTTP endpoint that speaks the
proof wire and returns a :class:`~argus_proof.models.RunManifest` plus the images
it produced (base64-encoded). This is the "hosted SDXL+LoRA endpoint" backend: it
runs generation on someone else's GPU (a self-hosted proof-gen service, or a thin
wrapper in front of Replicate / fal / another provider), selectable by config like
any other backend.

Unlike the local backends, the weights aren't on this machine, so the manifest is
**built by the service** (which hashes the weights it actually used) and validated
here as a :class:`RunManifest` — proof's contract (SHA256-pinned models,
``proof_version``) is enforced at the boundary; the service is trusted for the
hashes it reports. Credentials are an ``Authorization: Bearer`` header via
``api_key``. The HTTP layer is the shared injectable
:class:`~argus_proof.backends.http.Transport`, so this is unit-testable offline.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path

import structlog
from pydantic import ValidationError

from argus_proof.backends.base import BackendError, GenResult, ProgressSink
from argus_proof.backends.http import Transport, UrllibTransport
from argus_proof.backends.pnginfo import read_dimensions
from argus_proof.models import BackendCapabilities, GeneratedImage, ProgressEvent, ProofError, RunManifest, RunSpec

logger = structlog.get_logger()

BACKEND_NAME = "remote"


class RemoteBackend:
    """Generate via a hosted endpoint that speaks the proof wire.

    ``base_url`` is the service root; ``api_key`` (if given) is sent as a bearer
    token. ``transport`` defaults to :class:`UrllibTransport` and is injectable for
    testing. The endpoint must accept ``POST /generate`` with ``{"spec": <RunSpec>}``
    and return ``{"manifest": <RunManifest>, "images": [{"seed", "image_id"?,
    "content_base64"}]}``.
    """

    def __init__(
        self,
        base_url: str,
        transport: Transport | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.transport = transport or UrllibTransport(base_url, timeout=timeout, headers=headers, label="remote")

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=BACKEND_NAME,
            supports_seed_set=True,
            max_loras=None,
            reads_pnginfo=False,
            streams_progress=False,
        )

    def generate(self, spec: RunSpec, out_dir: Path, progress: ProgressSink | None = None) -> GenResult:
        """Run *spec* on the remote service, writing the returned images under *out_dir*."""
        logger.debug("remote.generate", run_id=spec.run_id, seeds=len(spec.seeds), out_dir=str(out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)

        def emit(event: ProgressEvent) -> None:
            if progress is not None:
                progress(event)

        total = len(spec.seeds)
        emit(ProgressEvent(run_id=spec.run_id, type="start", total=total))

        try:
            resp = self.transport.post_json("/generate", {"spec": spec.model_dump(mode="json")})
            manifest = self._parse_manifest(resp, spec)
            images = self._collect_images(resp, spec, out_dir, emit)
        except BackendError as exc:
            emit(ProgressEvent(run_id=spec.run_id, type="error", message=str(exc)))
            raise

        if not images:
            error = f"remote service returned no images for run {spec.run_id!r}"
            emit(ProgressEvent(run_id=spec.run_id, type="error", message=error))
            raise BackendError(error)

        (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        emit(ProgressEvent(run_id=spec.run_id, type="done", completed=total, total=total))
        return GenResult(manifest=manifest, images=images)

    def _parse_manifest(self, resp: dict, spec: RunSpec) -> RunManifest:
        raw = resp.get("manifest")
        if not isinstance(raw, dict):
            raise BackendError(f"remote service returned no manifest for run {spec.run_id!r}")
        try:
            manifest = RunManifest.model_validate(raw)  # enforces proof_version + SHA256-pinned models
        except (ValidationError, ProofError) as exc:
            raise BackendError(f"remote service returned an invalid manifest: {exc}") from exc
        if manifest.run_id != spec.run_id:
            raise BackendError(f"remote manifest run_id {manifest.run_id!r} != requested {spec.run_id!r}")
        return manifest

    def _collect_images(self, resp: dict, spec: RunSpec, out_dir: Path, emit: ProgressSink) -> list[GeneratedImage]:
        produced: list[GeneratedImage] = []
        for index, item in enumerate(resp.get("images") or []):
            if "seed" not in item or "content_base64" not in item:
                raise BackendError(f"remote image {index} is missing 'seed' or 'content_base64'")
            data = _decode_image(item["content_base64"], index)
            seed = item["seed"]
            image_id = item.get("image_id") or f"{spec.run_id}-{seed}"
            path = out_dir / f"{image_id}.png"
            path.write_bytes(data)
            dims = read_dimensions(data) or (spec.sampling.width, spec.sampling.height)
            produced.append(
                GeneratedImage(
                    image_id=image_id, run_id=spec.run_id, seed=seed, path=str(path), width=dims[0], height=dims[1]
                )
            )
            emit(ProgressEvent(run_id=spec.run_id, type="image", seed=seed, image_id=image_id))
        return produced


def _decode_image(b64: str, index: int) -> bytes:
    if "," in b64 and b64.lstrip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        return base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BackendError(f"remote image {index} is not valid base64: {exc}") from exc
