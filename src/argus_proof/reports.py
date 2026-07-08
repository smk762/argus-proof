"""Persistence + human-in-the-loop review for :class:`EvalReport`s.

Where a report lives between "scored" and "reviewed": a flat directory of
``<run_id>.json`` files, a compact summary listing for the /proof run browser,
and the apply-HITL flow that folds a reviewer's 5-star ratings and structured
reject reasons back into a report — then recomputes its group-collapsed pass-rate
and verdict through the same :func:`~argus_proof.scoring.summary.summarise` the
automated pass uses, so a human decision and an auto decision roll up identically.

The store is deliberately a directory of JSON files, not a database: a proof run
already owns a directory, reports are write-once-then-annotate, and a plain file
is trivially inspectable and portable (the CLI ``gate`` verb reads the same JSON).
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from argus_proof.models import EvalReport, GateConfig, ProofError, RejectReason
from argus_proof.scoring.summary import summarise

# Where reports live when the server/CLI isn't told otherwise. Override per
# deployment (compose mounts a volume here) with ARGUS_PROOF_REPORTS_DIR.
ENV_REPORTS_DIR = "ARGUS_PROOF_REPORTS_DIR"
DEFAULT_REPORTS_DIR = "reports"


class ReportSummary(BaseModel):
    """A one-line digest of a stored report for the /proof run browser.

    Everything the list view needs without shipping every per-image row: the
    verdict, the group-collapsed pass-rate, and how many groups still await
    human review.
    """

    run_id: str
    passed: bool
    pending: bool
    pass_rate: float
    n_images: int
    n_groups: int | None = None
    n_needs_hitl: int = 0
    created_at: str | None = None


class HitlImageUpdate(BaseModel):
    """A reviewer's verdict on one image: a star rating and/or reject reasons.

    Addressed by ``image_id`` (the report's opaque handle). An update *fully
    specifies* the reviewer's decision for that image: ``hitl_rating`` and
    ``reject_reasons`` replace whatever the image had, so send the existing
    values to keep them and send ``null`` / ``[]`` to clear them (letting a
    reviewer retract a rating). A non-empty ``reject_reasons`` marks the image
    rejected regardless of stars.
    """

    image_id: str
    hitl_rating: int | None = Field(default=None, ge=1, le=5)
    reject_reasons: list[RejectReason] = Field(default_factory=list)


class HitlRequest(BaseModel):
    """A batch of HITL updates from one rater against a run's images.

    ``rater`` is stamped onto every image the batch touches (for later
    inter-rater reliability). ``gate`` lets a caller re-roll the verdict under
    the same thresholds the run was scored with; omit it for the defaults.
    """

    rater: str | None = None
    gate: GateConfig | None = None
    updates: list[HitlImageUpdate] = Field(default_factory=list)


def reports_root(root: str | os.PathLike[str] | None = None) -> Path:
    """The reports directory: explicit *root* > ``$ARGUS_PROOF_REPORTS_DIR`` > default."""
    return Path(root or os.environ.get(ENV_REPORTS_DIR, DEFAULT_REPORTS_DIR))


def summarise_report(report: EvalReport) -> ReportSummary:
    """Compress a full report down to its list-view digest."""
    agg = report.aggregate
    return ReportSummary(
        run_id=report.run_id,
        passed=report.verdict.passed,
        pending=report.verdict.pending,
        pass_rate=agg.pass_rate,
        n_images=agg.n_images,
        n_groups=agg.n_groups,
        n_needs_hitl=agg.n_needs_hitl,
        created_at=report.created_at,
    )


def _hitl_decision(rating: int | None, reject_reasons: list[RejectReason]) -> bool | None:
    """A human's pass/fail from their rating + reasons; ``None`` = still undecided.

    Any reject reason fails the image outright. Otherwise a rating of 3+ passes
    (borderline-but-acceptable) and 1–2 fails; no rating leaves it undecided.
    """
    if reject_reasons:
        return False
    if rating is None:
        return None
    return rating >= 3


def apply_hitl(report: EvalReport, request: HitlRequest) -> EvalReport:
    """Fold *request*'s reviewer decisions into a **copy** of *report*.

    Each touched image gets the rating, reasons, and rater recorded, and its
    ``passed`` flag set from the human decision (a rated/rejected image is no
    longer "needs review"). An update fully specifies the reviewer's decision:
    its ``hitl_rating`` and ``reject_reasons`` replace the image's values, so a
    reviewer can *retract* a rating (send ``null`` / ``[]``) — a cleared image
    with no human verdict falls back to the gate's auto ``passed``. The aggregate
    and verdict are then recomputed over the updated rows, so the run's pass-rate
    and pending/passed status reflect the review. Untouched images (not in the
    batch) are carried through unchanged.
    """
    gate = request.gate or GateConfig()
    by_id = {u.image_id: u for u in request.updates}
    rows = list(report.images)
    for i, row in enumerate(rows):
        upd = by_id.get(row.image_id)
        if upd is None:
            continue
        # Authoritative: the update's values are the reviewer's final decision
        # for this image (the client sends the full intended state), so they
        # replace the row's — otherwise a cleared rating/reason can't be honored.
        decision = _hitl_decision(upd.hitl_rating, upd.reject_reasons)
        rows[i] = row.model_copy(
            update={
                "hitl_rating": upd.hitl_rating,
                "hitl_rater": request.rater if request.rater is not None else row.hitl_rater,
                "reject_reasons": upd.reject_reasons,
                "passed": decision if decision is not None else row.passed,
            }
        )
    aggregate, verdict = summarise(
        rows, gate=gate, diversity=report.aggregate.diversity, n_images=report.aggregate.n_images
    )
    return report.model_copy(update={"images": rows, "aggregate": aggregate, "verdict": verdict})


class ReportStore:
    """A directory of ``<run_id>.json`` :class:`EvalReport`s."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = reports_root(root)

    def _path(self, run_id: str) -> Path:
        # run_id becomes a filename — reject anything that could escape the dir.
        if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id or "\x00" in run_id:
            raise ProofError(f"invalid run_id {run_id!r}")
        return self.root / f"{run_id}.json"

    def list(self) -> list[ReportSummary]:
        """Every readable report's summary, newest run_id first isn't meaningful —
        sorted by run_id for a stable order. Unreadable/foreign files are skipped
        rather than failing the whole listing."""
        if not self.root.is_dir():
            return []
        summaries: list[ReportSummary] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                report = EvalReport.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, ProofError):
                continue
            summaries.append(summarise_report(report))
        return summaries

    def get(self, run_id: str) -> EvalReport:
        """Load one report; raise :class:`FileNotFoundError` if it isn't stored."""
        path = self._path(run_id)
        if not path.is_file():
            raise FileNotFoundError(run_id)
        return EvalReport.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, report: EvalReport) -> Path:
        """Write (or overwrite) a report, keyed by its ``run_id``."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(report.run_id)
        path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    def review(self, run_id: str, request: HitlRequest) -> EvalReport:
        """Apply a HITL batch to a stored report and persist the result."""
        updated = apply_hitl(self.get(run_id), request)
        self.save(updated)
        return updated
