"""Turn eval findings into routed recommendations (#12, findings half).

The CI gate (:mod:`argus_proof.acceptance`) answers *did it pass?*; this answers
*what should I change, and where?* — mapping weak metrics to the suite stage that
owns the fix, so a failing eval loops back concretely instead of vaguely:

* identity / aesthetic didn't transfer → **forge** (training config / data)
* prompt adherence low → **lens** (captioning) and the prompt **grid**
* low diversity → the **grid** (token axes) or forge (overfit)
* unsafe outputs → **lens** (data/caption filtering)
* which checkpoint / LoRA weight to prefer → the cross-run **store**
* borderline band → **refine** (run HITL)

Pure rules over an :class:`~argus_proof.models.EvalReport` (+ an optional
:class:`~argus_proof.crossrun.CrossRunStore` for cross-run picks); no heavy deps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel

from argus_proof.models import EvalReport

if TYPE_CHECKING:
    from argus_proof.crossrun import SliceStats

RecommendStage = Literal["forge", "lens", "grid", "weight", "checkpoint", "refine"]


class Recommendation(BaseModel):
    """One routed suggestion: which stage to act on, why, and what to do."""

    stage: RecommendStage
    issue: str
    action: str
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None


class RecommendConfig(BaseModel):
    """Floors below which a metric triggers a recommendation (``None`` = skip)."""

    min_identity: float | None = 0.6
    min_clip_score: float | None = 0.5
    min_aesthetic: float | None = 0.5
    min_preference: float | None = None
    min_diversity: float | None = 0.3
    max_unsafe_rate: float | None = 0.0
    unsafe_safety_floor: float = 0.5


class _Sliceable(Protocol):
    def slice_pass_rate(self, dimension: str) -> list[SliceStats]: ...


def _unsafe_rate(report: EvalReport, floor: float) -> float:
    n = len(report.images)
    if n == 0:
        return 0.0
    unsafe = sum(
        1
        for img in report.images
        if (img.metrics.safety is not None and img.metrics.safety < floor)
        or any(r.code == "unsafe" for r in img.reject_reasons)
    )
    return unsafe / n


def recommend(
    report: EvalReport,
    *,
    config: RecommendConfig | None = None,
    store: _Sliceable | None = None,
) -> list[Recommendation]:
    """Routed recommendations for improving a run's outcome.

    Each below-floor metric maps to the stage that owns the fix; if a *store* is
    given, the best checkpoint / LoRA weight across runs (by evidence-adjusted
    pass-rate) is surfaced too. Returns ``[]`` when nothing is actionable.
    """
    cfg = config or RecommendConfig()
    means = report.aggregate.means
    recs: list[Recommendation] = []

    def below(value: float | None, floor: float | None) -> bool:
        return floor is not None and value is not None and value < floor

    if below(means.identity, cfg.min_identity):
        recs.append(
            Recommendation(
                stage="forge",
                issue="identity didn't transfer",
                action="add/curate more identity-representative training images, or adjust training steps/rank",
                metric="identity",
                value=means.identity,
                threshold=cfg.min_identity,
            )
        )
    if below(means.clip_score, cfg.min_clip_score):
        recs.append(
            Recommendation(
                stage="lens",
                issue="prompt adherence low",
                action="outputs drift from the prompt — revisit the captioning strategy (more descriptive captions)",
                metric="clip_score",
                value=means.clip_score,
                threshold=cfg.min_clip_score,
            )
        )
        recs.append(
            Recommendation(
                stage="grid",
                issue="prompt adherence low",
                action="try different prompt / token combinations in the grid",
                metric="clip_score",
                value=means.clip_score,
                threshold=cfg.min_clip_score,
            )
        )
    if below(means.aesthetic, cfg.min_aesthetic):
        recs.append(
            Recommendation(
                stage="forge",
                issue="technical/aesthetic quality low",
                action="revisit training config (learning rate / steps) or the base checkpoint",
                metric="aesthetic",
                value=means.aesthetic,
                threshold=cfg.min_aesthetic,
            )
        )
    if below(means.preference, cfg.min_preference):
        recs.append(
            Recommendation(
                stage="forge",
                issue="human-preference proxy low",
                action="the LoRA's outputs are under-preferred — revisit training data quality / config",
                metric="preference",
                value=means.preference,
                threshold=cfg.min_preference,
            )
        )
    if below(report.aggregate.diversity, cfg.min_diversity):
        recs.append(
            Recommendation(
                stage="grid",
                issue="low output diversity",
                action="widen the token/prompt axes; if the LoRA reproduces training data, reduce training (overfit)",
                metric="diversity",
                value=report.aggregate.diversity,
                threshold=cfg.min_diversity,
            )
        )
    if cfg.max_unsafe_rate is not None:
        rate = _unsafe_rate(report, cfg.unsafe_safety_floor)
        if rate > cfg.max_unsafe_rate:
            recs.append(
                Recommendation(
                    stage="lens",
                    issue="unsafe outputs",
                    action="filter/re-caption training data to remove unsafe content",
                    metric="safety",
                    value=rate,
                    threshold=cfg.max_unsafe_rate,
                )
            )
    if report.verdict.pending:
        recs.append(
            Recommendation(
                stage="refine",
                issue="borderline results awaiting review",
                action="run a HITL review on the needs-review band before deciding",
                metric="pass_rate",
                value=report.aggregate.pass_rate,
            )
        )
    if store is not None:
        recs.extend(_best_across_runs(store, "base_checkpoint", "checkpoint"))
        recs.extend(_best_across_runs(store, "lora_weight", "weight"))
    return recs


def _best_across_runs(store: _Sliceable, dimension: str, stage: RecommendStage) -> list[Recommendation]:
    """Recommend the best-performing value of *dimension* across the store, if it
    varies (needs ≥2 cells to be a meaningful choice)."""
    slices = store.slice_pass_rate(dimension)
    if len(slices) < 2:
        return []
    best = slices[0]  # already ranked by CI lower bound
    return [
        Recommendation(
            stage=stage,
            issue=f"{dimension} outcome varies across runs",
            action=f"prefer {dimension}={best.value!r} (highest evidence-adjusted pass-rate, lower bound {best.ci_low:.2f})",
            metric="pass_rate",
            value=best.pass_rate,
        )
    ]
