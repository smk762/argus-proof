"""Optional refinement stage (#7): re-rank the passing subset, finer than pass/fail.

After the quality stage (automated gate + HITL) has decided which images *pass*,
this second, optional pass lets a reviewer re-rank just that passing subset 1–5
with free-text notes — for finer discrimination between "good" configs than a
binary verdict gives. It's a **separate layer**: the refined rating lands in
:attr:`~argus_proof.models.ImageScores.refinement` and never touches the
first-pass ``hitl_rating``/``reject_reasons``/``passed`` or the run's aggregate
and verdict, so both the original decision and the refined ordering are retained
(the acceptance for smk762/argus-proof#7).

Regeneration of variants (same seed-set) is just another harness run via the
generation backend; this module owns the refinement *layer* those re-ranks land
in and the reranked view over it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from argus_proof.models import EvalReport, ImageScores, ProofError, Refinement


class RefinementError(ProofError):
    """A refinement targeted an image that isn't in the run's passing subset."""


class RefinementImageUpdate(BaseModel):
    """A reviewer's refined rank (+ notes) for one passing image, by ``image_id``."""

    image_id: str
    rank: int = Field(ge=1, le=5)
    notes: str | None = None


class RefinementRequest(BaseModel):
    """A batch of refinement re-ranks from one rater against a run's passing images.

    ``rater`` is stamped onto every image the batch refines (for provenance /
    later inter-rater analysis on the refined layer).
    """

    rater: str | None = None
    updates: list[RefinementImageUpdate] = Field(default_factory=list)


def passing_subset(report: EvalReport) -> list[ImageScores]:
    """The images that survived the quality stage (``passed`` is True) — the only
    images the refinement stage may re-rank."""
    return [img for img in report.images if img.passed]


def apply_refinement(report: EvalReport, request: RefinementRequest) -> EvalReport:
    """Fold *request*'s refined re-ranks into a **copy** of *report*.

    Each targeted image gets a :class:`~argus_proof.models.Refinement` set; the
    first-pass ``hitl_rating``/``reject_reasons``/``passed`` and the run's
    ``aggregate``/``verdict`` are left untouched (refinement is additive). Every
    update must target an image in the passing subset — a non-passing or unknown
    ``image_id`` raises :class:`RefinementError` rather than silently no-op, since
    refinement is a deliberate second pass over a known set.

    An update's ``rank`` always replaces the prior rank; an omitted ``notes`` or
    ``rater`` (``None``) keeps the existing value rather than clearing it, so a
    bare rank correction can't silently drop a reviewer's note or authorship
    (matching how :func:`~argus_proof.reports.apply_hitl` carries the rater).
    """
    passing_ids = {img.image_id for img in passing_subset(report)}
    stray = [u.image_id for u in request.updates if u.image_id not in passing_ids]
    if stray:
        raise RefinementError(f"cannot refine images not in the passing subset: {stray}")

    by_id = {u.image_id: u for u in request.updates}
    rows = [
        row.model_copy(update={"refinement": _merge_refinement(row.refinement, upd, request.rater)})
        if (upd := by_id.get(row.image_id)) is not None
        else row
        for row in report.images
    ]
    return report.model_copy(update={"images": rows})


def _merge_refinement(prior: Refinement | None, upd: RefinementImageUpdate, rater: str | None) -> Refinement:
    """Build the new refinement for an image: ``rank`` replaces, but ``notes``/
    ``rater`` fall back to the prior refinement when this update omits them."""
    return Refinement(
        rank=upd.rank,
        notes=upd.notes if upd.notes is not None else (prior.notes if prior else None),
        rater=rater if rater is not None else (prior.rater if prior else None),
    )


def refined_ranking(report: EvalReport) -> list[ImageScores]:
    """The passing subset ordered best-first by refined rank.

    Refined images sort above not-yet-refined ones; within each, higher refined
    rank (then first-pass ``hitl_rating``) wins, with ``image_id`` as a stable
    tie-break. So a caller can surface "the best of the good" once some of the
    passing subset has been re-ranked, without losing the un-refined remainder.
    """

    def key(img: ImageScores) -> tuple:
        # Ascending sort: negate the "higher is better" fields so they land first,
        # then image_id ascending as a stable, deterministic tie-break.
        rank = img.refinement.rank if img.refinement else 0
        return (0 if img.refinement else 1, -rank, -(img.hitl_rating or 0), img.image_id)

    return sorted(passing_subset(report), key=key)
