"""Turn eval findings into routed recommendations (#12, findings half).

The CI gate (:mod:`argus_proof.acceptance`) answers *did it pass?*; this answers
*what should I change, and where?* — mapping weak metrics to the suite stage that
owns the fix, so a failing eval loops back concretely instead of vaguely:

* identity / aesthetic didn't transfer → **forge** (training config / data)
* prompt adherence low → **lens** (captioning) and the prompt **grid**
* low diversity → the **grid** (token axes) or forge (overfit)
* pose collapse → **curator** when the training poses are themselves narrow (or the outputs
  match their mix), else the **grid** (generation)
* unsafe outputs → **lens** (data/caption filtering)
* which checkpoint / LoRA weight to prefer → the cross-run **store**
* borderline band → **refine** (run HITL)

Pure rules over an :class:`~argus_proof.models.EvalReport` (+ an optional
:class:`~argus_proof.crossrun.CrossRunStore` for cross-run picks); no heavy deps.

The floor thresholds live in :class:`RecommendConfig`; use
:meth:`RecommendConfig.from_acceptance` to keep them in lock-step with the CI
gate's :class:`~argus_proof.models.AcceptanceThresholds` so the two can't
silently disagree about the same report.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol

from pydantic import BaseModel, Field

from argus_proof.acceptance import (
    _unsafe_rate,  # single source of "what counts as unsafe"
    pose_collapsed,
    pose_unmeasurable,
)
from argus_proof.models import AcceptanceThresholds, EvalReport
from argus_proof.scoring.scorers.safety import is_unsafe

if TYPE_CHECKING:
    from argus_proof.crossrun import SliceStats

RecommendStage = Literal["curator", "forge", "lens", "grid", "weight", "checkpoint", "refine"]

# A coverage divergence (Jensen-Shannon, base 2, 0..1) at or under this means the outputs' pose mix matches the
# training set's. Not calibrated on a real run yet.
POSE_LOW_COVERAGE_DIVERGENCE = 0.1


class Recommendation(BaseModel):
    """One routed suggestion: which stage to act on, why, and what to do.

    ``value`` and ``threshold`` are the measured metric and the floor it missed
    (both ``[0,1]`` in the same "higher = better" sense as the metric), except for
    the ``unsafe_rate`` metric where ``value`` is the fraction of unsafe images
    and ``threshold`` is the allowed ceiling, and ``pose_largest_cluster_share``,
    the share of poses in the largest cluster against its ceiling.
    """

    stage: RecommendStage
    issue: str
    action: str
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None


class RecommendConfig(BaseModel):
    """Floors below which a metric triggers a recommendation (``None`` = skip).

    Bounds mirror :class:`~argus_proof.models.AcceptanceThresholds` (``[0,1]``) so
    an out-of-range floor is rejected rather than silently misfiring.
    """

    min_identity: float | None = Field(default=0.6, ge=0.0, le=1.0)
    # Off by default: DINOv2 cosine has no calibrated floor yet (unrelated images sit around 0.1-0.4).
    min_likeness: float | None = Field(default=None, ge=0.0, le=1.0)
    min_clip_score: float | None = Field(default=0.5, ge=0.0, le=1.0)
    min_aesthetic: float | None = Field(default=0.5, ge=0.0, le=1.0)
    min_preference: float | None = Field(default=None, ge=0.0, le=1.0)
    min_diversity: float | None = Field(default=0.3, ge=0.0, le=1.0)
    max_unsafe_rate: float | None = Field(default=0.0, ge=0.0, le=1.0)
    unsafe_safety_floor: float = Field(default=0.8, ge=0.0, le=1.0)  # == safety.UNSAFE_SAFETY_FLOOR (640m, #84)
    # Off by default, like the gate: the pose thresholds are calibrated on synthetic poses only.
    min_pose_diversity: float | None = Field(default=None, ge=0.0, le=1.0)
    max_pose_cluster_share: float = Field(default=0.75, gt=0.0, le=1.0)

    @classmethod
    def from_acceptance(cls, thresholds: AcceptanceThresholds, **overrides: float | None) -> RecommendConfig:
        """A config whose shared floors track the CI gate's *thresholds*.

        Keeps the recommender and :func:`~argus_proof.acceptance.evaluate_acceptance`
        from disagreeing about the same report: the identity and likeness floors, unsafe ceiling,
        safety floor and pose-collapse thresholds are taken from *thresholds*. The recommend-only floors
        (clip_score / aesthetic / preference / diversity) keep their defaults
        unless supplied in *overrides*.
        """
        fields: dict[str, float | None] = {
            "min_identity": thresholds.min_identity_mean,
            "min_likeness": thresholds.min_likeness_mean,
            "max_unsafe_rate": thresholds.max_unsafe_rate,
            "unsafe_safety_floor": thresholds.unsafe_safety_floor,
            "min_pose_diversity": thresholds.min_pose_diversity,
            "max_pose_cluster_share": thresholds.max_pose_cluster_share,
        }
        fields.update(overrides)
        return cls(**fields)


class _Sliceable(Protocol):
    def slice_pass_rate(self, dimension: str) -> list[SliceStats]: ...


class _MetricRule(NamedTuple):
    """A below-floor metric and the stage(s) that own its fix (data-driven so a
    new metric is a table row, not another copy-pasted block)."""

    metric: str
    floor_attr: str
    get: Callable[[EvalReport], float | None]
    targets: tuple[tuple[RecommendStage, str, str], ...]  # (stage, issue, action)


# Order matters only for readability here; safety is emitted first (below), and
# these follow. clip_score routes to two stages (lens + grid) in one rule.
_METRIC_RULES: tuple[_MetricRule, ...] = (
    _MetricRule(
        "identity",
        "min_identity",
        lambda r: r.aggregate.means.identity,
        (
            (
                "forge",
                "identity didn't transfer",
                "add/curate more identity-representative training images, or adjust training steps/rank",
            ),
        ),
    ),
    _MetricRule(
        "likeness",
        "min_likeness",
        lambda r: r.aggregate.means.likeness,
        (
            (
                "forge",
                "likeness didn't transfer",
                "the subject or set doesn't match its references: curate more representative training images, "
                "or adjust training steps/rank",
            ),
        ),
    ),
    _MetricRule(
        "clip_score",
        "min_clip_score",
        lambda r: r.aggregate.means.clip_score,
        (
            (
                "lens",
                "prompt adherence low",
                "outputs drift from the prompt — revisit the captioning strategy (more descriptive captions)",
            ),
            ("grid", "prompt adherence low", "try different prompt / token combinations in the grid"),
        ),
    ),
    _MetricRule(
        "aesthetic",
        "min_aesthetic",
        lambda r: r.aggregate.means.aesthetic,
        (
            (
                "forge",
                "technical/aesthetic quality low",
                "revisit training config (learning rate / steps) or the base checkpoint",
            ),
        ),
    ),
    _MetricRule(
        "preference",
        "min_preference",
        lambda r: r.aggregate.means.preference,
        (
            (
                "forge",
                "human-preference proxy low",
                "the LoRA's outputs are under-preferred — revisit training data quality / config",
            ),
        ),
    ),
    _MetricRule(
        "diversity",
        "min_diversity",
        lambda r: r.aggregate.diversity,
        (
            (
                "grid",
                "low output diversity",
                "widen the token/prompt axes; if the LoRA reproduces training data, reduce training (overfit)",
            ),
        ),
    ),
)


def _below(value: float | None, floor: float | None) -> bool:
    """True when *value* is measured and misses *floor*.

    A ``None`` floor (check disabled) or ``None`` value (metric not measured)
    never triggers. A ``NaN`` value counts as a miss — a broken/degenerate scorer
    should surface a recommendation, not be silently read as passing (the CI gate
    fails NaN in the same, safe direction)."""
    return floor is not None and value is not None and (math.isnan(value) or value < floor)


def recommend(
    report: EvalReport,
    *,
    config: RecommendConfig | None = None,
    store: _Sliceable | None = None,
) -> list[Recommendation]:
    """Routed recommendations for improving a run's outcome.

    Each below-floor metric maps to the stage that owns the fix (safety first, as
    the highest-stakes axis). If a *store* is given, the best checkpoint / LoRA
    weight across runs is surfaced too — but only when the evidence separates a
    clear winner (non-overlapping CIs); a statistically tied field yields nothing.
    Returns ``[]`` when nothing is actionable.
    """
    cfg = config or RecommendConfig()
    means = report.aggregate.means
    recs: list[Recommendation] = []

    # Safety first — the highest-stakes axis. Prefer the per-image unsafe rate;
    # fall back to the aggregate mean for reports that carry no per-image rows
    # (an aggregate-only report would otherwise never be checked for safety).
    if cfg.max_unsafe_rate is not None:
        if report.images:
            rate = _unsafe_rate(report, cfg.unsafe_safety_floor)
            if rate > cfg.max_unsafe_rate:
                recs.append(
                    Recommendation(
                        stage="lens",
                        issue="unsafe outputs",
                        action="filter/re-caption training data to remove unsafe content",
                        metric="unsafe_rate",
                        value=rate,
                        threshold=cfg.max_unsafe_rate,
                    )
                )
        elif _below(means.safety, cfg.unsafe_safety_floor) or (
            means.safety is not None and is_unsafe(means.safety, cfg.unsafe_safety_floor)
        ):
            recs.append(
                Recommendation(
                    stage="lens",
                    issue="unsafe outputs",
                    action="filter/re-caption training data to remove unsafe content",
                    metric="safety",
                    value=means.safety,
                    threshold=cfg.unsafe_safety_floor,
                )
            )

    for rule in _METRIC_RULES:
        value = rule.get(report)
        floor = getattr(cfg, rule.floor_attr)
        if _below(value, floor):
            recs.extend(
                Recommendation(
                    stage=stage, issue=issue, action=action, metric=rule.metric, value=value, threshold=floor
                )
                for stage, issue, action in rule.targets
            )

    recs.extend(_pose_collapse(report, cfg))

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


def _pose_collapse(report: EvalReport, cfg: RecommendConfig) -> list[Recommendation]:
    """Route a pose collapse to the stage that owns it.

    A collapse onto a pose the training set is itself dominated by (the training poses fail the same collapse test,
    or the outputs' pose mix matches the training mix: a low coverage divergence) is a dataset-balance problem for
    **curator** (rebalance the export's poses, then retrain). Otherwise generation narrowed it: the **grid**.
    Nothing when the floor is off or the pose block isn't measurable (the gate reports that).
    """
    floor = cfg.min_pose_diversity
    pose = report.pose
    if floor is None or pose is None or pose_unmeasurable(pose) is not None:
        return []
    if not pose_collapsed(pose, floor, cfg.max_pose_cluster_share):
        return []
    if pose.diversity is not None and pose.diversity < floor:
        evidence = {"metric": "pose_diversity", "value": pose.diversity, "threshold": floor}
    else:  # collapsed on the cluster share alone
        evidence = {
            "metric": "pose_largest_cluster_share",
            "value": pose.largest_cluster_share,
            "threshold": cfg.max_pose_cluster_share,
        }
    onto = f" (largest cluster around {pose.largest_cluster_medoids[0]})" if pose.largest_cluster_medoids else ""
    cmp = pose.comparison
    narrow_training = cmp is not None and (
        (cmp.training_diversity is not None and cmp.training_diversity < floor)
        or (
            cmp.training_largest_cluster_share is not None
            and cmp.training_largest_cluster_share >= cfg.max_pose_cluster_share
        )
    )
    matches_training = (
        cmp is not None
        and cmp.coverage_divergence is not None
        and cmp.coverage_divergence <= POSE_LOW_COVERAGE_DIVERGENCE
    )
    if narrow_training or matches_training:
        why = "the training poses are narrow" if narrow_training else "the outputs match the training pose mix"
        rec = Recommendation(
            stage="curator",
            issue=f"pose collapse{onto}: {why}",
            action="rebalance the dataset's poses (curator's coverage panel: angle, shot scale, framing), "
            "re-export and retrain",
            **evidence,
        )
    else:
        context = "the training set wasn't compared" if cmp is None else "the training poses are broader"
        rec = Recommendation(
            stage="grid",
            issue=f"pose collapse{onto}: {context}",
            action="vary pose and composition tokens in the prompt grid; if the LoRA forces one pose at every "
            "prompt, train less (overfit)",
            **evidence,
        )
    return [rec]


def _best_across_runs(store: _Sliceable, dimension: str, stage: RecommendStage) -> list[Recommendation]:
    """Recommend the best-performing value of *dimension* across the store — but
    only when the evidence actually separates a winner.

    Drops the ``value=None`` cohort (e.g. the multi-/no-LoRA rows for
    ``lora_weight``): a null isn't something to "prefer". Requires the top cell's
    CI lower bound to clear the runner-up's upper bound, so a field of
    statistically indistinguishable cells (overlapping CIs) yields nothing rather
    than a confident pick chasing noise.
    """
    cells = [s for s in store.slice_pass_rate(dimension) if s.value is not None]
    if len(cells) < 2:
        return []
    best = cells[0]  # slice_pass_rate ranks by CI lower bound, best first
    # Against every other cell's upper bound, not just the runner-up's: a small
    # cell ranks low on ci_low yet its wide interval can still reach above best.
    if best.ci_low <= max(c.ci_high for c in cells[1:]):
        return []  # CIs overlap — the data can't distinguish the cells

    return [
        Recommendation(
            stage=stage,
            issue=f"{dimension} outcome varies across runs",
            action=f"prefer {dimension}={best.value!r} (highest evidence-adjusted pass-rate, lower bound {best.ci_low:.2f})",
            metric="pass_rate",
            value=best.pass_rate,
        )
    ]
