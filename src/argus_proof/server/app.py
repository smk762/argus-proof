"""FastAPI micro-server for argus-proof (peer to argus-forge on :8104).

Routes:

    GET  /health                 -> {status, service, version}
    GET  /reports                -> {reports: [ReportSummary, ...]}  (run browser)
    GET  /report/{run_id}        -> EvalReport
    PUT  /report/{run_id}        -> EvalReport   (store/overwrite a scored report)
    POST /report/{run_id}/hitl   -> EvalReport   (apply a HITL review, recompute)

Reports are served from a directory of ``<run_id>.json`` files
(``$ARGUS_PROOF_REPORTS_DIR``, default ``reports/``). The generate/score verbs
that produce them land with the epic phases:
https://github.com/smk762/argus-studio/issues/6
"""

from __future__ import annotations

try:
    from fastapi import FastAPI, HTTPException
except ImportError as exc:  # pragma: no cover
    raise ImportError("Server requires: pip install argus-proof[server]") from exc

from argus_proof import __version__
from argus_proof.models import EvalReport, ProofError
from argus_proof.reports import HitlRequest, ReportStore, ReportSummary


def create_app(
    cors: bool = False,
    cors_origins: list[str] | None = None,
    reports_dir: str | None = None,
) -> FastAPI:
    """Create the proof FastAPI application.

    ``reports_dir`` overrides where reports are read/written; omit it to use
    ``$ARGUS_PROOF_REPORTS_DIR`` (default ``reports/``).
    """
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

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "argus-proof", "version": __version__}

    @app.get("/reports")
    async def list_reports() -> dict[str, list[ReportSummary]]:
        """Digest of every stored report — the /proof run browser's index."""
        return {"reports": store.list()}

    @app.get("/report/{run_id}")
    async def get_report(run_id: str) -> EvalReport:
        """The full scored report for one run."""
        try:
            return store.get(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"no report for run {run_id!r}") from exc
        except ProofError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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

    return app
