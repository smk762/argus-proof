"""The CI acceptance gate (#12): a machine-readable pass/fail on an EvalReport.

Closes the loop for CI — "was this dataset/LoRA good enough?" becomes an
automatable yes/no against declared :class:`~argus_proof.models.AcceptanceThresholds`
(a min pass-rate, a statistically-defensible pass-rate lower bound, an identity
floor, a safety ceiling). Only the configured checks run; the run is accepted
iff every check passes.
"""

from __future__ import annotations

from argus_proof.models import (
    AcceptanceThresholds,
    EvalReport,
    GateCheck,
    GateResult,
)
from argus_proof.stats import wilson_lower_bound


def _unsafe_rate(report: EvalReport, safety_floor: float) -> float:
    """Fraction of images that are unsafe.

    An image counts as unsafe if its ``safety`` metric is below *safety_floor*
    OR it carries an ``unsafe`` reject reason. Keying off the metric (not only
    the reject reason) matters: the ``unsafe`` reason is only produced when the
    gate has a ``safety`` hard gate, so a run scored with the default gate could
    otherwise report a 0% unsafe rate while containing unsafe imagery.
    """
    n = len(report.images)
    if n == 0:
        return 0.0
    unsafe = sum(
        1
        for img in report.images
        if (img.metrics.safety is not None and img.metrics.safety < safety_floor)
        or any(r.code == "unsafe" for r in img.reject_reasons)
    )
    return unsafe / n


def evaluate_acceptance(report: EvalReport, thresholds: AcceptanceThresholds) -> GateResult:
    """Evaluate *report* against *thresholds* → a :class:`GateResult`.

    A configured metric that wasn't measured (e.g. ``min_identity_mean`` with no
    identity scorer) fails its check rather than passing silently — you can't
    accept on evidence you don't have.
    """
    checks: list[GateCheck] = []
    agg = report.aggregate
    # The pass-rate is over near-dup groups when deduped, else over images.
    n = agg.n_groups if agg.n_groups is not None else agg.n_images
    n_passed = min(agg.n_passed, n)  # defensive: never let successes exceed the denominator

    if thresholds.min_pass_rate is not None:
        ok = agg.pass_rate >= thresholds.min_pass_rate
        checks.append(
            GateCheck(
                name="pass_rate",
                passed=ok,
                actual=agg.pass_rate,
                threshold=thresholds.min_pass_rate,
                detail=f"pass_rate {agg.pass_rate:.3f} {'>=' if ok else '<'} {thresholds.min_pass_rate:.3f}",
            )
        )

    if thresholds.min_pass_rate_ci_lower is not None:
        lb = wilson_lower_bound(n_passed, n, thresholds.confidence)
        ok = lb >= thresholds.min_pass_rate_ci_lower
        checks.append(
            GateCheck(
                name="pass_rate_ci_lower",
                passed=ok,
                actual=lb,
                threshold=thresholds.min_pass_rate_ci_lower,
                detail=(
                    f"{thresholds.confidence:.0%} Wilson lower bound {lb:.3f} "
                    f"({n_passed}/{n}) {'>=' if ok else '<'} {thresholds.min_pass_rate_ci_lower:.3f}"
                ),
            )
        )

    if thresholds.min_identity_mean is not None:
        val = agg.means.identity
        ok = val is not None and val >= thresholds.min_identity_mean
        detail = (
            "identity not measured"
            if val is None
            else f"identity mean {val:.3f} {'>=' if ok else '<'} {thresholds.min_identity_mean:.3f}"
        )
        checks.append(
            GateCheck(
                name="identity_mean", passed=ok, actual=val, threshold=thresholds.min_identity_mean, detail=detail
            )
        )

    if thresholds.max_unsafe_rate is not None:
        rate = _unsafe_rate(report, thresholds.unsafe_safety_floor)
        ok = rate <= thresholds.max_unsafe_rate
        checks.append(
            GateCheck(
                name="unsafe_rate",
                passed=ok,
                actual=rate,
                threshold=thresholds.max_unsafe_rate,
                detail=f"unsafe rate {rate:.3f} {'<=' if ok else '>'} {thresholds.max_unsafe_rate:.3f}",
            )
        )

    if not checks:
        # No thresholds configured -> nothing was verified. Refuse rather than
        # accept on zero evidence (all([]) would otherwise be True).
        return GateResult(
            passed=False,
            checks=[GateCheck(name="no_checks", passed=False, detail="no acceptance thresholds configured")],
        )

    return GateResult(passed=all(c.passed for c in checks), checks=checks)
