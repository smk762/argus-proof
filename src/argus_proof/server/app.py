"""FastAPI micro-server for argus-proof (peer to argus-forge on :8104).

Routes:

    GET  /health                          -> {status, service, version, read_only, capabilities}
    GET  /exports                         -> {exports: [...]}  (curator export dirs to evaluate against)
    GET  /models                          -> {checkpoints: [...], loras: [...]}  (engine-loadable names)
    GET  /scorers                         -> {scorers: [{metric, name, available}]}  (installed-scorer probe)
    POST /run/stream                      -> NDJSON: generate a seed-set, score it, store the report
    GET  /reports                         -> {reports: [ReportSummary, ...]}  (run browser)
    GET  /report/{run_id}                 -> EvalReport
    PUT  /report/{run_id}                 -> EvalReport   (store/overwrite a scored report)
    GET  /report/{run_id}/manifest        -> RunManifest  (the validated setup; 410 if the run dir is gone)
    GET  /report/{run_id}/refined         -> {images: [...]}  (passing subset, refined order first)
    GET  /report/{run_id}/image/{image_id} -> image bytes (from the run dir; ids, never paths)
    GET  /report/{run_id}/image_at/{index} -> image bytes (by report position; seed-free blind-review URL)
    POST /report/{run_id}/hitl            -> EvalReport   (apply a HITL review, recompute)
    POST /report/{run_id}/refine          -> EvalReport   (second-pass re-rank of the passing subset)
    POST /purge/{run_id}                  -> {run_id, report, run_dir}  (devmode only: delete one run)
    POST /purge                           -> {reports, run_dirs}        (devmode only: delete every run)

Reports are served from a directory of ``<run_id>.json`` files
(``$ARGUS_PROOF_REPORTS_DIR``, default ``reports/``); generated runs live under
``$ARGUS_PROOF_RUNS_DIR`` (default ``runs/``). Images are addressed by
``(run_id, image_id)`` — both validated against a strict charset — so no
client-supplied filesystem path is ever resolved (no traversal surface).

Set ``$ARGUS_PROOF_READ_ONLY`` (or ``create_app(read_only=True)``) for
replay/demo mode (issue #45): stored reports keep serving, but every mutating
request — live GPU eval (``POST /run/stream``) above all — is refused with
``403``, so the GPU-less public demo host can't kick off compute. ``GET /health``
echoes the resolved flag so the frontend can hide the "run evaluation" action.

Set ``$ARGUS_PROOF_DEVMODE`` (or ``create_app(devmode=True)``) to register the
``/purge`` routes, which delete a run's report *and* its run dir (issue #59).
Reports are the evaluation record, so the routes don't exist otherwise; they
need a ``{"confirm": "purge"}`` body, refuse a run that is still generating
(``409``), and read-only mode refuses them like every other write.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import threading
import time
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import FileResponse, StreamingResponse
    from starlette.background import BackgroundTask
except ImportError as exc:  # pragma: no cover
    raise ImportError("Server requires: pip install argus-proof[server]") from exc

from argus_cortex.server import WriteGuard, constant_refuse, env_flag
from pydantic import BaseModel, Field, model_validator

from argus_proof import __version__
from argus_proof.models import (
    CompleteFrame,
    EvalReport,
    LoRASpec,
    ProgressEvent,
    ProofError,
    RunManifest,
    RunSpec,
    SamplingParams,
    ScoringFrame,
)
from argus_proof.refinement import RefinementRequest, refined_ranking
from argus_proof.reports import HitlRequest, ReportStore, ReportSummary, summarise_report

# Where the run trigger reads curator exports from (compose mounts /data/out here).
ENV_EXPORTS_DIR = "ARGUS_PROOF_EXPORTS_DIR"

# Replay/demo mode (issue #45): serve stored reports but refuse every mutating
# request — live GPU eval above all — so the GPU-less public demo host can't
# kick off compute. Truthy values: 1/true/yes/on.
ENV_READ_ONLY = "ARGUS_PROOF_READ_ONLY"

# Dev mode (issue #59): registers the /purge routes that delete runs. Off by
# default — reports are the evaluation record. Compose sets it from DEVMODE.
ENV_DEVMODE = "ARGUS_PROOF_DEVMODE"

# The body a /purge request must carry (same as argus-darkroom's).
_PURGE_CONFIRM = {"confirm": "purge"}

# Claimed in the active-run set while ``POST /purge`` runs, so no run starts
# mid-purge without holding the lock (and blocking the event loop) throughout.
# Not a valid run id (fails _SAFE_ID), so it can't collide with one.
_PURGING_ALL = "*purge-all*"

# run_id / image_id / export names become path segments under a fixed root —
# allow only a strict filename charset so traversal is impossible by construction.
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_IMAGE_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

_MODEL_SUFFIXES = (".safetensors", ".ckpt", ".pt")


def _require_safe(value: str, label: str) -> str:
    if not _SAFE_ID.match(value or ""):
        raise HTTPException(status_code=400, detail=f"invalid {label} {value!r}")
    return value


# Optional routes a client can't infer from ``version`` alone, advertised on
# ``GET /health`` so a consumer (argus-studio) can tell an older proof that
# lacks a route from a run that lacks the data. Append-only.
CAPABILITIES = ("report_manifest",)
# Advertised on top of CAPABILITIES only when devmode registers /purge.
DEVMODE_CAPABILITIES = ("purge",)

# The message a refused write carries in read-only mode.
_READ_ONLY_REFUSAL = "argus-proof is in read-only (replay) mode; writes and live evaluation are disabled"


class RunRequest(BaseModel):
    """One evaluation request: generate a control seed-set for a LoRA, score it, store the report.

    The prompt comes from ``prompt`` when given, otherwise from the named
    curator ``export``'s captions (first base prompt). ``export`` is a
    directory *name* under ``$ARGUS_PROOF_EXPORTS_DIR``, never a path. When the
    export contains a ``references/`` subdir its images drive identity scoring
    (held out — they must not overlap the training set).

    ``cfg`` / ``sampler`` / ``scheduler`` / ``guidance`` left out take the
    checkpoint family's defaults (``workflow.sampling_defaults``): SDXL's
    cfg 7 / dpmpp_2m / karras, or Flux's cfg 1 / euler / simple / guidance 3.5.
    ``vae`` / ``text_encoders`` left out take its model defaults
    (``workflow.model_defaults``): none for SDXL, Flux's ``ae`` VAE and T5 +
    CLIP-L encoders. An explicit ``null`` VAE or ``[]`` keeps them off.
    """

    lora: str
    base_checkpoint: str
    lora_weight: float = 1.0
    export: str | None = None
    prompt: str | None = None
    negative_prompt: str = ""
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3], min_length=1)
    steps: int = Field(default=25, gt=0)
    cfg: float | None = None
    sampler: str | None = None
    scheduler: str | None = None
    guidance: float | None = Field(default=None, ge=0)
    width: int = Field(default=1024, gt=0)
    height: int = Field(default=1024, gt=0)
    clip_skip: int = 1
    vae: str | None = None
    text_encoders: list[str] | None = None
    run_id: str | None = None

    @model_validator(mode="after")
    def _family_defaults(self) -> RunRequest:
        from argus_proof.backends.workflow import model_defaults, sampling_defaults

        for key, value in model_defaults(self.base_checkpoint).items():
            if key not in self.model_fields_set:
                setattr(self, key, value)
        if self.text_encoders is None:  # an explicit null is no encoders
            self.text_encoders = []

        for key, value in sampling_defaults(self.base_checkpoint).items():
            # guidance is only defaulted when the client sent nothing at all for
            # it: an explicit null keeps it off (e.g. a custom Flux template).
            if getattr(self, key) is None and (key != "guidance" or key not in self.model_fields_set):
                setattr(self, key, value)
        return self


def create_app(
    cors: bool = False,
    cors_origins: list[str] | None = None,
    reports_dir: str | None = None,
    runs_dir: str | None = None,
    exports_dir: str | None = None,
    read_only: bool | None = None,
    devmode: bool | None = None,
) -> FastAPI:
    """Create the proof FastAPI application.

    ``reports_dir`` / ``runs_dir`` / ``exports_dir`` override where reports,
    generated runs, and curator exports live; omitted they fall back to
    ``$ARGUS_PROOF_REPORTS_DIR`` / ``$ARGUS_PROOF_RUNS_DIR`` /
    ``$ARGUS_PROOF_EXPORTS_DIR``.

    ``read_only`` enables replay/demo mode (issue #45): stored reports are still
    served, but every mutating request — live GPU eval (``POST /run/stream``)
    above all — is refused with ``403`` so the GPU-less demo host can't kick off
    compute. Omitted (``None``) it falls back to ``$ARGUS_PROOF_READ_ONLY``;
    ``GET /health`` reports the resolved flag.

    ``devmode`` registers the ``/purge`` routes (issue #59); omitted it falls
    back to ``$ARGUS_PROOF_DEVMODE``. ``GET /health`` reports it, and lists
    ``purge`` in ``capabilities`` when the routes will actually work (devmode
    on and not read-only).
    """
    from argus_proof.backends.base import MANIFEST_NAME
    from argus_proof.evaluate import EvaluateError, load_manifest, runs_root

    read_only = env_flag(ENV_READ_ONLY) if read_only is None else read_only
    devmode = env_flag(ENV_DEVMODE) if devmode is None else devmode

    app = FastAPI(
        title="Argus Proof",
        description="Post-training LoRA evaluation: generated samples in, scored verdicts out.",
        version=__version__,
    )

    if read_only:
        # Read-only means NO mutation at all: a public demo must not let visitors
        # overwrite the shared report tape (PUT / HITL / refine) or spend GPU (run).
        # constant_refuse gates every unsafe method, so it covers every write route
        # — current and future — without per-route guards to forget. The shared
        # WriteGuard is pure ASGI (not BaseHTTPMiddleware), so it never wraps or
        # buffers the NDJSON stream POST /run/stream returns. Registered before CORS
        # so that, when CORS is enabled, CORSMiddleware stays outermost and a
        # cross-origin 403 still carries its CORS headers.
        app.add_middleware(WriteGuard, refuse=constant_refuse(_READ_ONLY_REFUSAL))

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

    # run_ids a live /run/stream is generating into (or a purge is deleting),
    # so a purge can't pull a run dir out from under its generation and two
    # streams can't write into the same run dir.
    active_runs: set[str] = set()
    active_lock = threading.Lock()

    def _claim_run(run_id: str) -> None:
        with active_lock:
            if run_id in active_runs or _PURGING_ALL in active_runs:
                raise HTTPException(status_code=409, detail=f"run {run_id!r} is in progress")
            active_runs.add(run_id)

    def _release_run(run_id: str) -> None:
        with active_lock:
            active_runs.discard(run_id)

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
    async def health() -> dict[str, str | bool | list[str]]:
        """Liveness + capability probe. ``read_only`` tells the UI whether live
        evaluation is available — a replay/demo host reports ``True`` and the
        frontend hides the "run evaluation" action (argus-studio#59).
        ``capabilities`` lists optional routes (see :data:`CAPABILITIES`);
        ``purge`` appears only when devmode is on and the host isn't read-only."""
        return {
            "status": "ok",
            "service": "argus-proof",
            "version": __version__,
            "read_only": read_only,
            "devmode": devmode,
            "capabilities": list(CAPABILITIES) + (list(DEVMODE_CAPABILITIES) if devmode and not read_only else []),
        }

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
        / ``scoring`` / ``complete`` (the stored report's summary) / ``error``,
        each a ``RunStreamFrame`` member, built from those models so the stream
        can't drift from the wire schema. Generation runs in a worker thread;
        frames bridge back via a queue.
        """
        from argus_proof.evaluate import (
            EvaluateError,
            backend_from_env,
            check_held_out,
            reference_images,
            score_images,
        )
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
                try:
                    check_held_out(references, export_dir)  # before any GPU spend
                except EvaluateError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
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
            vae=request.vae,
            text_encoders=list(request.text_encoders or []),
            loras=[LoRASpec(name=request.lora, weight=request.lora_weight)],
            sampling=SamplingParams(
                sampler=request.sampler,
                scheduler=request.scheduler,
                steps=request.steps,
                cfg=request.cfg,
                guidance=request.guidance,
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
        _claim_run(run_id)  # 409 if this run_id is already generating or being purged

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        # Exactly one side releases the claim: work() once the body has started,
        # else the response's background task (the body never ran). A release
        # that isn't tied to this claim could free a later claim on the same id
        # (a purge, or a new stream) or free this one while work() still runs.
        started = False

        def emit(frame: dict | None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, frame)

        def work() -> None:
            errored = False  # the backend already streamed an error frame
            try:
                backend = backend_from_env()

                def on_progress(event) -> None:  # noqa: ANN001 - ProgressEvent
                    nonlocal errored
                    errored = errored or event.type == "error"
                    emit(event.model_dump(exclude_none=True))

                result = backend.generate(spec, run_dir, progress=on_progress)
                emit(ScoringFrame(run_id=run_id, n_images=len(result.images)).model_dump(mode="json"))
                report = score_images(result.manifest, result.images, references=references)
                store.save(report)
                emit(CompleteFrame(run_id=run_id, report=summarise_report(report)).model_dump(mode="json"))
            except (EvaluateError, ProofError, ValueError) as exc:
                if not errored:
                    emit(ProgressEvent(run_id=run_id, type="error", message=str(exc)).model_dump(exclude_none=True))
            finally:
                _release_run(run_id)  # before the sentinel: a client that saw the end can purge
                emit(None)  # sentinel: stream is over

        async def stream():
            nonlocal started
            started = True
            worker = loop.run_in_executor(None, work)
            try:
                while True:
                    frame = await queue.get()
                    if frame is None:
                        break
                    yield json.dumps(frame) + "\n"
            finally:
                await worker

        def release_if_unstarted() -> None:
            # The body never started (a client gone before the first frame), so
            # work() never ran and won't release.
            if not started:
                _release_run(run_id)

        return StreamingResponse(
            stream(), media_type="application/x-ndjson", background=BackgroundTask(release_if_unstarted)
        )

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

    @app.get("/report/{run_id}/manifest")
    def get_manifest(run_id: str) -> RunManifest:
        """The :class:`RunManifest` behind a report — checkpoint, VAE, LoRAs (each
        SHA256-pinned), sampling, prompt and seeds — so a client can carry a
        validated setup forward (e.g. studio's proof -> darkroom handoff).

        The report is the anchor: no report -> ``404``. The manifest lives in the
        run dir, which can be pruned while the report is kept; a report whose
        manifest is gone -> ``410 Gone``, so a client can tell "never existed"
        from "existed, setup no longer recoverable". Sync ``def``: file I/O runs
        in the threadpool.
        """
        _require_safe(run_id, "run_id")
        _load_report(run_id)
        run_dir = _runs_root() / run_id
        if not (run_dir / MANIFEST_NAME).is_file():
            raise HTTPException(
                status_code=410,
                detail=f"run {run_id!r} has a report but its run directory (and manifest) is no longer available",
            )
        try:
            manifest = load_manifest(run_dir)
        except EvaluateError as exc:
            raise HTTPException(status_code=400, detail=f"unreadable manifest for run {run_id!r}") from exc
        if manifest.run_id != run_id:
            # Never hand a client another run's setup under this run's name.
            raise HTTPException(
                status_code=400,
                detail=f"manifest in run dir {run_id!r} is for run {manifest.run_id!r}",
            )
        return manifest

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

    if devmode:

        async def _require_confirm(request: Request) -> None:
            try:
                body = await request.json()
            except ValueError:
                body = None
            if body != _PURGE_CONFIRM:
                raise HTTPException(status_code=400, detail='the body must be {"confirm": "purge"}')

        def _remove_run_dir(run_id: str) -> bool:
            """Delete <runs_root>/<run_id> (a validated id, never a path). A symlink
            is unlinked, not followed, so a purge can't reach outside the runs root."""
            path = _runs_root() / run_id
            if path.is_symlink():
                path.unlink()
                return True
            if path.is_dir():
                shutil.rmtree(path)
                return True
            return False

        @app.post("/purge/{run_id}")
        async def purge_run(run_id: str, request: Request) -> dict:
            """Delete one run: its report (with the HITL review and refinement
            stored in it) and its run dir (manifest + images). ``404`` when neither
            exists; a run dir without a report (a failed generation) is purgeable."""
            _require_safe(run_id, "run_id")
            await _require_confirm(request)
            _claim_run(run_id)  # 409 while generating; also blocks a new run on this id meanwhile

            def purge_one() -> dict:
                report = store.delete(run_id)
                run_dir = _remove_run_dir(run_id)
                if not (report or run_dir):
                    raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
                return {"run_id": run_id, "report": report, "run_dir": run_dir}

            try:
                return await run_in_threadpool(purge_one)
            finally:
                _release_run(run_id)

        @app.post("/purge")
        async def purge_all(request: Request) -> dict[str, int]:
            """Delete every report and run dir. Refused (``409``) while any run is
            generating. Only entries with a valid run id are touched."""
            await _require_confirm(request)

            def purge_everything() -> dict[str, int]:
                removed = {"reports": 0, "run_dirs": 0}
                for run_id in store.run_ids():
                    if _SAFE_ID.match(run_id) and store.delete(run_id):
                        removed["reports"] += 1
                root = _runs_root()
                if root.is_dir():
                    for path in sorted(root.iterdir()):
                        if _SAFE_ID.match(path.name) and _remove_run_dir(path.name):
                            removed["run_dirs"] += 1
                return removed

            # Claim the whole store rather than hold active_lock across the
            # deletes: _claim_run/_release_run take that lock on the event loop,
            # which would stall every request until the purge finished.
            with active_lock:
                if active_runs:
                    raise HTTPException(status_code=409, detail="a run is in progress")
                active_runs.add(_PURGING_ALL)
            try:
                return await run_in_threadpool(purge_everything)
            finally:
                _release_run(_PURGING_ALL)

    return app
