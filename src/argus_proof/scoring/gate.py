"""The gate — route each image to auto-pass / auto-fail / needs-HITL.

The automated pre-pass from issue #5: combine an image's normalised metric
scores into a weighted composite, apply absolute per-metric floors, and decide
its fate so humans only rate the middle band. Pure functions over
:class:`~argus_proof.models.MetricScores` + :class:`~argus_proof.models.GateConfig`
— no model loading here.
"""

from __future__ import annotations

from argus_proof.models import GateConfig, MetricScores, RejectReason
from argus_proof.scoring.base import REJECT_CODE_FOR_METRIC


def composite_score(metrics: MetricScores, config: GateConfig) -> float | None:
    """Weighted mean of the metrics present in ``config.weights``.

    Only metrics that both have a weight and were actually scored count, and the
    weights are renormalised over what's present — so an image missing one
    scorer isn't dragged down by a phantom zero. Returns ``None`` when nothing
    weighted was scored (the gate then routes it to HITL).
    """
    total_weight = 0.0
    acc = 0.0
    for metric, weight in config.weights.items():
        value = getattr(metrics, metric, None)
        if value is not None and weight:
            acc += value * weight
            total_weight += weight
    if total_weight == 0.0:
        return None
    return acc / total_weight


def _hard_gate_failures(metrics: MetricScores, config: GateConfig) -> list[RejectReason]:
    reasons: list[RejectReason] = []
    for metric, floor in config.hard_gates.items():
        value = getattr(metrics, metric, None)
        if value is not None and value < floor:
            code = REJECT_CODE_FOR_METRIC.get(metric, "other")
            reasons.append(RejectReason(code=code, note=f"{metric} {value:.3f} < hard gate {floor:.3f}"))
    return reasons


def gate_image(metrics: MetricScores, config: GateConfig) -> tuple[bool | None, list[RejectReason]]:
    """Decide one image's fate: ``(passed, reasons)``.

    ``True`` = auto-pass, ``False`` = auto-fail (with structured reasons),
    ``None`` = undecided → route to HITL. A hard-gate breach fails outright;
    otherwise the composite is compared to the auto-pass / auto-fail band.
    """
    hard_failures = _hard_gate_failures(metrics, config)
    if hard_failures:
        return False, hard_failures

    composite = composite_score(metrics, config)
    if composite is None:
        return None, []  # nothing to judge on — send to a human
    if composite >= config.auto_pass:
        return True, []
    if composite <= config.auto_fail:
        return False, [
            RejectReason(code="low_quality", note=f"composite {composite:.3f} <= auto_fail {config.auto_fail:.3f}")
        ]
    return None, []  # borderline — needs HITL
