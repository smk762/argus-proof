"""FastAPI micro-server for argus-proof (peer to argus-forge on :8104).

Routes:

    GET  /health                          -> {status, service, version}
    GET  /exports                         -> {exports: [...]}  (curator export dirs to evaluate against)
    GET  /models                          -> {checkpoints: [...], loras: [...]}  (engine-loadable names)
    GET  /scorers                         -> {scorers: [{metric, name, available}]}  (installed-scorer probe)
    POST /run/stream                      -> NDJSON: generate a seed-set, score it, store the report
    GET  /reports                         -> {reports: [ReportSummary, ...]}  (run browser)
    GET  /report/{run_id}                 -> EvalReport
    PUT  /report/{run_id}                 -> EvalReport   (store/overwrite a scored report)
    GET  /report/{run_id}/refined         -> {images: [...]}  (passing subset, refined order first)
    GET  /report/{run_id}/image/{image_id} -> image bytes (from the run dir; ids, never paths)
    GET  /report/{run_id}/image_at/{index} -> image bytes (by report position; seed-free blind-review URL)
    POST /report/{run_id}/hitl            -> EvalReport   (apply a HITL review, recompute)
    POST /report/{run_id}/refine          -> EvalReport   (second-pass re-rank of the passing subset)

Reports are served from a directory of ``<run_id>.json`` files
(``$ARGUS_PROOF_REPORTS_DIR``, default ``reports/``); generated runs live under
``$ARGUS_PROOF_RUNS_DIR`` (default ``runs/``). Images are addressed by
``(run_id, image_id)`` — both validated against a strict charset — so no
client-supplied filesystem path is ever resolved (no traversal surface).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import time
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, StreamingResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError("Server requires: pip install argus-proof[server]") from exc

from pydantic import BaseModel, Field

from argus_proof import __version__
from argus_proof.models import EvalReport, LoRASpec, ProofError, RunSpec, SamplingParams
from argus_proof.refinement import RefinementRequest, refined_ranking
from argus_proof.reports import HitlRequest, ReportStore, ReportSummary, summarise_report

# Where the run trigger reads curator exports from (compose mounts /data/out here).
ENV_EXPORTS_DIR = "ARGUS_PROOF_EXPORTS_DIR"

# run_id / image_id / export names become path segments under a fixed root —
# allow only a strict filename charset so traversal is impossible by construction.
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_IMAGE_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

_MODEL_SUFFIXES = (".safetensors", ".ckpt", ".pt")


def _require_safe(value: str, label: str) -> str:
    if not _SAFE_ID.match(value or ""):
        raise HTTPException(status_code=400, detail=f"invalid {label} {value!r}")
    return value


class RunRequest(BaseModel):
    """One evaluation request: generate a control seed-set for a LoRA, score it, store the report.

    The prompt comes from ``prompt`` when given, otherwise from the named
    curator ``export``'s captions (first base prompt). ``export`` is a
    directory *name* under ``$ARGUS_PROOF_EXPORTS_DIR``, never a path. When the
    export contains a ``references/`` subdir its images drive identity scoring
    (held out — they must not overlap the training set).
    """

    lora: str
    base_checkpoint: str
    lora_weight: float = 1.0
    export: str | None = None
    prompt: str | None = None
    negative_prompt: str = ""
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3], min_length=1)
    steps: int = Field(default=25, gt=0)
    cfg: float = 7.0
    sampler: str = "dpmpp_2m"
    scheduler: str = "karras"
    width: int = Field(default=1024, gt=0)
    height: int = Field(default=1024, gt=0)
    clip_skip: int = 1
    run_id: str | None = None


def create_app(
    cors: bool = False,
    cors_origins: list[str] | None = None,
    reports_dir: str | None = None,
    runs_dir: str | None = None,
    exports_dir: str | None = None,
) -> FastAPI:
    """Create the proof FastAPI application.

    ``reports_dir`` / ``runs_dir`` / ``exports_dir`` override where reports,
    generated runs, and curator exports live; omitted they fall back to
    ``$ARGUS_PROOF_REPORTS_DIR`` / ``$ARGUS_PROOF_RUNS_DIR`` /
    ``$ARGUS_PROOF_EXPORTS_DIR``.
    """
    from argus_proof.evaluate import runs_root

    app = FastAPI(
        title="Argus Proof",
        description="Post-training LoRA evaluation: generated samples in, scored verdicts out.",
        version=__version__,
    )

    if cors:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    store = ReportStore(reports_dir)

    def _runs_root() -> Path:
        return runs_root(runs_dir)

    def _serve_run_image(run_id: str, image_id: str) -> FileResponse:
        """Serve <runs_root>/<run_id>/<image_id>.<ext> (ids only, never paths).

        ``image_id`` is validated here — not just at the route — so every caller,
        including ones that resolve it from a stored report (``image_at``), is
        held to the strict filename charset and can't join a traversal path under
        the runs root.
        """
        _require_safe(image_id, "image_id")
        run_dir = _runs_root() / run_id
        for suffix, media_type in _IMAGE_MEDIA.items():
            path = run_dir / f"{image_id}{suffix}"
            if path.is_file():
                return FileResponse(path, media_type=media_type)
        # Don't echo image_id: for the by-index (blind-review) route it is
        # ``<run_id>-<seed>``, and leaking the seed defeats that URL's purpose.
        raise HTTPException(status_code=404, detail=f"no such image in run {run_id!r}")

    def _load_report(run_id: str) -> EvalReport:
        """Load one stored report, mapping storage errors to HTTP consistently:
        missing -> 404; unreadable/invalid (``ProofError`` or a pydantic
        ``ValidationError``, a ``ValueError``) -> 400 rather than a bare 500."""
        try:
            return store.get(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"no report for run {run_id!r}") from exc
        except (ProofError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _exports_root() -> Path | None:
        value = exports_dir or os.environ.get(ENV_EXPORTS_DIR)
        return Path(value) if value else None

    def _resolve_export(name: str) -> Path:
        """The export dir called *name* under the exports root — names, not paths."""
        _require_safe(name, "export name")
        root = _exports_root()
        if root is None:
            raise HTTPException(status_code=503, detail=f"no exports directory configured — set {ENV_EXPORTS_DIR}")
        path = root / name
        if not path.is_dir():
            raise HTTPException(status_code=404, detail=f"no export named {name!r}")
        return path

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "argus-proof", "version": __version__}

    @app.get("/exports")
    async def list_exports() -> dict[str, list[dict]]:
        """Curator export dirs (containing manifest.jsonl) available to evaluate against."""
        root = _exports_root()
        if root is None or not root.is_dir():
            return {"exports": []}
        exports: list[dict] = []
        for child in sorted(root.iterdir()):
            manifest = child / "manifest.jsonl"
            if not child.is_dir() or not manifest.is_file() or not _SAFE_ID.match(child.name):
                continue
            try:
                n_rows = sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
            except (OSError, UnicodeDecodeError):
                continue
            exports.append(
                {
                    "name": child.name,
                    "n_rows": n_rows,
                    "has_references": (child / "references").is_dir(),
                }
            )
        return {"exports": exports}

    @app.get("/models")
    async def list_models() -> dict[str, list[str]]:
        """Checkpoints + LoRAs under $PROOF_MODELS_DIR, named as the engine loads them
        (relative to each root's checkpoints/ / loras/ subdir — the ComfyUI layout;
        PROOF_MODELS_DIR may list several roots separated by os.pathsep)."""
        from argus_proof.evaluate import models_roots

        result: dict[str, set[str]] = {"checkpoints": set(), "loras": set()}
        for root in models_roots():
            for key in result:
                base = root / key
                if not base.is_dir():
                    continue
                result[key].update(
                    str(p.relative_to(base))
                    for p in base.rglob("*")
                    if p.is_file() and p.suffix.lower() in _MODEL_SUFFIXES
                )
        return {key: sorted(names) for key, names in result.items()}

    @app.get("/scorers")
    def list_scorers() -> dict[str, list[dict]]:
        """Which scorers a run applies and whether this image can run each, so the
        UI can warn up-front when the learned metrics aren't installed (a run then
        falls back to all-HITL). The set mirrors ``evaluate.score_images`` exactly
        (via :func:`all_reporting_scorers`). Probing ``available`` *imports* each
        scorer's extra (no model weights load, but a real import), so this is a
        sync ``def`` — FastAPI runs it in a threadpool, keeping those imports off
        the event loop."""
        from argus_proof.evaluate import all_reporting_scorers

        rows: list[dict] = []
        for scorer in all_reporting_scorers():
            prov = scorer.provenance()  # carries the metric for every scorer (incl. phash dedup/diversity)
            rows.append({"metric": prov.metric, "name": prov.name, "available": scorer.is_available()})
        return {"scorers": rows}

    @app.post("/run/stream")
    async def run_stream(request: RunRequest) -> StreamingResponse:
        """Generate + score + store one evaluation run, streaming NDJSON progress.

        Frames: ``start`` (total images) / ``image`` (one landed) / ``progress``
        / ``scoring`` / ``complete`` (the stored report's summary) / ``error``.
        Generation runs in a worker thread; frames bridge back via a queue.
        """
        from argus_proof.evaluate import EvaluateError, backend_from_env, reference_images, score_images
        from argus_proof.grid import read_export_prompts

        # Resolve everything that can fail on bad input BEFORE streaming starts,
        # so the client gets a clean 4xx instead of an error frame mid-stream.
        references: list[Path] = []
        source_manifest: str | None = None
        prompt = request.prompt
        if request.export is not None:
            export_dir = _resolve_export(request.export)
            source_manifest = str(export_dir / "manifest.jsonl")
            refs_dir = export_dir / "references"
            if refs_dir.is_dir():
                references = reference_images(refs_dir)
            if prompt is None:
                prompts = read_export_prompts(export_dir)
                if not prompts:
                    raise HTTPException(
                        status_code=400,
                        detail=f"export {request.export!r} has no captions/.txt sidecars — supply a prompt",
                    )
                prompt = prompts[0]
        if prompt is None:
            raise HTTPException(status_code=400, detail="supply a prompt or an export to source one from")

        run_id = request.run_id or f"proof-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
        _require_safe(run_id, "run_id")

        spec = RunSpec(
            run_id=run_id,
            base_checkpoint=request.base_checkpoint,
            loras=[LoRASpec(name=request.lora, weight=request.lora_weight)],
            sampling=SamplingParams(
                sampler=request.sampler,
                scheduler=request.scheduler,
                steps=request.steps,
                cfg=request.cfg,
                clip_skip=request.clip_skip,
                width=request.width,
                height=request.height,
            ),
            prompt=prompt,
            negative_prompt=request.negative_prompt,
            seeds=request.seeds,
            source_manifest=source_manifest,
        )
        run_dir = _runs_root() / run_id

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        def emit(frame: dict | None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, frame)

        def work() -> None:
            try:
                backend = backend_from_env()

                def on_progress(event) -> None:  # noqa: ANN001 - ProgressEvent
                    emit(event.model_dump(exclude_none=True))

                result = backend.generate(spec, run_dir, progress=on_progress)
                emit({"type": "scoring", "run_id": run_id, "n_images": len(result.images)})
                report = score_images(result.manifest, result.images, references=references)
                store.save(report)
                emit({"type": "complete", "run_id": run_id, "report": summarise_report(report).model_dump()})
            except (EvaluateError, ProofError, ValueError) as exc:
                emit({"type": "error", "run_id": run_id, "message": str(exc)})
            finally:
                emit(None)  # sentinel: stream is over

        async def stream():
            worker = loop.run_in_executor(None, work)
            try:
                while True:
                    frame = await queue.get()
                    if frame is None:
                        break
                    yield json.dumps(frame) + "\n"
            finally:
                await worker

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    @app.get("/reports")
    async def list_reports() -> dict[str, list[ReportSummary]]:
        """Digest of every stored report — the /proof run browser's index."""
        return {"reports": store.list()}

    @app.get("/report/{run_id}")
    async def get_report(run_id: str) -> EvalReport:
        """The full scored report for one run."""
        return _load_report(run_id)

    @app.put("/report/{run_id}")
    async def put_report(run_id: str, report: EvalReport) -> EvalReport:
        """Store (or overwrite) a scored report. ``run_id`` in the path must match
        the report body, so the stored filename can't drift from its contents."""
        if report.run_id != run_id:
            raise HTTPException(
                status_code=400,
                detail=f"run_id in path ({run_id!r}) != report.run_id ({report.run_id!r})",
            )
        try:
            store.save(report)
        except ProofError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return report

    @app.get("/report/{run_id}/refined")
    async def get_refined(run_id: str) -> dict:
        """The passing subset in refined order (refined ranks first, best first)."""
        report = _load_report(run_id)
        return {"run_id": run_id, "images": [img.model_dump() for img in refined_ranking(report)]}

    @app.get("/report/{run_id}/image/{image_id}")
    async def get_image(run_id: str, image_id: str) -> FileResponse:
        """One generated image, addressed by ids resolved under the runs root.

        Both ids are validated against a strict filename charset and joined
        under ``$ARGUS_PROOF_RUNS_DIR/<run_id>/`` — no client-supplied path is
        ever resolved, so there is no traversal surface (``image_id`` is checked
        inside ``_serve_run_image``).
        """
        _require_safe(run_id, "run_id")
        return _serve_run_image(run_id, image_id)

    @app.get("/report/{run_id}/image_at/{index}")
    async def get_image_at(run_id: str, index: int) -> FileResponse:
        """One generated image addressed by its position in the report's image
        list — a seed-free URL for blind review (the by-id route embeds the seed
        as ``<run_id>-<seed>``). Resolves the id via the stored report so the
        same strict runs-root join applies.
        """
        _require_safe(run_id, "run_id")
        report = _load_report(run_id)
        if not 0 <= index < len(report.images):
            raise HTTPException(status_code=404, detail=f"no image at index {index} in run {run_id!r}")
        return _serve_run_image(run_id, report.images[index].image_id)

    @app.post("/report/{run_id}/hitl")
    async def review_report(run_id: str, request: HitlRequest) -> EvalReport:
        """Apply a reviewer's ratings + reject reasons and return the recomputed
        report (aggregate pass-rate and verdict fold the human decisions in)."""
        try:
            return store.review(run_id, request)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"no report for run {run_id!r}") from exc
        except ProofError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/report/{run_id}/refine")
    async def refine_report(run_id: str, request: RefinementRequest) -> EvalReport:
        """Apply a second-pass re-rank of the passing subset (1–5 + notes) and
        return the report with the refinement layer added — first-pass ratings,
        aggregate, and verdict are left unchanged. ``rank: null`` retracts."""
        try:
            return store.refine(run_id, request)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"no report for run {run_id!r}") from exc
        except ProofError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
