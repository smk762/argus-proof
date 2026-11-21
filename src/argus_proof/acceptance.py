"""The CI acceptance gate (#12): a machine-readable pass/fail on an EvalReport.

Closes the loop for CI — "was this dataset/LoRA good enough?" becomes an
automatable yes/no against declared :class:`~argus_proof.models.AcceptanceThresholds`
(a min pass-rate, a statistically-defensible pass-rate lower bound, identity and
likeness floors, a safety ceiling, a pose-collapse gate). Only the configured checks run; the run is
accepted iff every check passes.

The pose gate (``min_pose_diversity``, #122) follows argus-puppet's suggested gate: the run's pose block must be
measurable first (enough usable, comparable poses), else the check fails; then a diversity under the floor or a
largest pose cluster at or above ``max_pose_cluster_share`` is a ``pose_collapse``. Pose-copying evidence
(``close_excess`` against the training set's own baseline) is only ever a ``pose_copying`` flag for review on a
passing check: a pose-conditioned generator reproduces training poses by design. The thresholds are calibrated on
synthetic poses only.
"""

from __future__ import annotations

from argus_proof.models import (
    AcceptanceThresholds,
    EvalReport,
    GateCheck,
    GateResult,
    PoseReport,
)
from argus_proof.scoring.scorers.safety import is_unsafe
from argus_proof.stats import wilson_lower_bound

# When a pose block measures enough of the run to be read (argus-puppet docs/metrics.md, "Suggested gate").
POSE_MIN_USABLE = 10
POSE_MIN_USABLE_SHARE = 0.5
POSE_MIN_COMPARABLE_SHARE = 0.8
# Pose-copying evidence: outputs closer to training frames than the frames are to each other, on enough poses.
POSE_COPYING_MIN_EXCESS = 0.1
POSE_COPYING_MIN_ELIGIBLE = 20
# Run-level reason codes (GateCheck.code), kept out of the per-image RejectReasonCode.
POSE_COLLAPSE = "pose_collapse"
POSE_COPYING = "pose_copying"


def pose_unmeasurable(pose: PoseReport | None) -> str | None:
    """Why *pose* can't be read (``None`` when it can): no block, too few usable poses, too small a usable share of
    the run, or too few comparable pairs. An unmeasurable pose block is never a pass."""
    if pose is None:
        return "no pose block (PROOF_PUPPET_URL unset when scored, or a report from before proof 1.10)"
    if pose.usable is None:
        why = pose.provenance.not_checked[0] if pose.provenance.not_checked else "puppet returned no set stats"
        return f"pose not measured: {why}"
    if pose.usable < POSE_MIN_USABLE:
        return f"{pose.usable} usable pose(s), need at least {POSE_MIN_USABLE}"
    if pose.count == 0 or pose.usable / pose.count < POSE_MIN_USABLE_SHARE:
        return f"{pose.usable}/{pose.count} images have a usable pose, need at least {POSE_MIN_USABLE_SHARE:.0%}"
    if pose.comparable_share is None or pose.comparable_share < POSE_MIN_COMPARABLE_SHARE:
        share = "unknown" if pose.comparable_share is None else f"{pose.comparable_share:.2f}"
        return f"comparable pose share {share}, need at least {POSE_MIN_COMPARABLE_SHARE:.2f}"
    if pose.diversity is None or pose.largest_cluster_share is None:
        return "pose diversity not measured"
    return None


def pose_collapsed(pose: PoseReport, floor: float, max_share: float) -> bool:
    """Whether a measurable *pose* block is a collapse: diversity under *floor* or a largest cluster at or above
    *max_share*."""
    return (pose.diversity is not None and pose.diversity < floor) or (
        pose.largest_cluster_share is not None and pose.largest_cluster_share >= max_share
    )


def _pose_checks(pose: PoseReport | None, floor: float, max_share: float) -> list[GateCheck]:
    why = pose_unmeasurable(pose)
    if why is not None:
        return [GateCheck(name="pose_measurable", passed=False, detail=f"pose gate not measurable: {why}")]
    assert pose is not None and pose.diversity is not None and pose.largest_cluster_share is not None
    medoids = ", ".join(pose.largest_cluster_medoids[:3]) or "unknown"
    div_ok = pose.diversity >= floor
    share_ok = pose.largest_cluster_share < max_share
    checks = [
        GateCheck(
            name="pose_measurable",
            passed=True,
            detail=f"{pose.usable}/{pose.count} usable poses, comparable share {pose.comparable_share:.2f}",
        ),
        GateCheck(
            name="pose_diversity",
            passed=div_ok,
            actual=pose.diversity,
            threshold=floor,
            detail=f"pose diversity {pose.diversity:.3f} {'>=' if div_ok else '<'} {floor:.3f}"
            + ("" if div_ok else f"; the outputs collapsed toward {medoids}"),
            code=None if div_ok else POSE_COLLAPSE,
        ),
        GateCheck(
            name="pose_largest_cluster",
            passed=share_ok,
            actual=pose.largest_cluster_share,
            threshold=max_share,
            detail=f"largest pose cluster {pose.largest_cluster_share:.2f} {'<' if share_ok else '>='} "
            f"{max_share:.2f} (medoid {medoids})",
            code=None if share_ok else POSE_COLLAPSE,
        ),
    ]
    cmp = pose.comparison
    if (
        cmp is not None
        and cmp.close_excess is not None
        and cmp.close_excess >= POSE_COPYING_MIN_EXCESS
        and cmp.close_eligible >= POSE_COPYING_MIN_ELIGIBLE
        and cmp.baseline_eligible >= POSE_COPYING_MIN_ELIGIBLE
    ):
        checks.append(
            GateCheck(
                name="pose_copying",
                passed=True,  # informational: never a reject
                actual=cmp.close_excess,
                threshold=POSE_COPYING_MIN_EXCESS,
                detail=(
                    f"flagged for review, not a reject: close excess {cmp.close_excess:.2f} over the training set's "
                    f"own baseline, {cmp.n_close} output(s) a close pose match for a training frame; compare them at "
                    "image level (a pose-conditioned generator reproduces training poses by design)"
                ),
                code=POSE_COPYING,
            )
        )
    return checks


def _unsafe_rate(report: EvalReport, safety_floor: float) -> float:
    """Fraction of images that are unsafe.

    An image counts as unsafe if its ``safety`` metric is at or below *safety_floor*
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
        if (img.metrics.safety is not None and is_unsafe(img.metrics.safety, safety_floor))
        or any(r.code == "unsafe" for r in img.reject_reasons)
    )
    return unsafe / n


def evaluate_acceptance(report: EvalReport, thresholds: AcceptanceThresholds) -> GateResult:
    """Evaluate *report* against *thresholds* → a :class:`GateResult`.

    A configured metric that wasn't measured (e.g. ``min_identity_mean`` with no
    identity scorer, or ``min_pose_diversity`` with no measurable pose block) fails
    its check rather than passing silently — you can't accept on evidence you don't have.
    """
    checks: list[GateCheck] = []
    agg = report.aggregate
    # The pass-rate is over near-dup groups when deduped, else over images.
    n = agg.n_groups if agg.n_groups is not None else agg.n_images
    # n_passed is counted per group (the same basis as n). More successes than
    # groups means the aggregate is inconsistent, e.g. passes counted per image;
    # clamping would hide that and inflate the Wilson bound, so it fails instead.
    n_passed = agg.n_passed
    consistent = n_passed <= n

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

    if thresholds.min_pass_rate_ci_lower is not None and not consistent:
        checks.append(
            GateCheck(
                name="pass_rate_ci_lower",
                passed=False,
                threshold=thresholds.min_pass_rate_ci_lower,
                detail=f"inconsistent report: {n_passed} passed > {n} groups; the pass count is on another basis",
            )
        )
    elif thresholds.min_pass_rate_ci_lower is not None:
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

    # The mean-similarity floors: ArcFace identity (faces) and DINOv2 likeness (any subject).
    for metric, floor in (("identity", thresholds.min_identity_mean), ("likeness", thresholds.min_likeness_mean)):
        if floor is None:
            continue
        val = getattr(agg.means, metric)
        ok = val is not None and val >= floor
        detail = (
            f"{metric} not measured" if val is None else f"{metric} mean {val:.3f} {'>=' if ok else '<'} {floor:.3f}"
        )
        checks.append(GateCheck(name=f"{metric}_mean", passed=ok, actual=val, threshold=floor, detail=detail))

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

    if thresholds.min_pose_diversity is not None:
        checks.extend(_pose_checks(report.pose, thresholds.min_pose_diversity, thresholds.max_pose_cluster_share))

    if not checks:
        # No thresholds configured -> nothing was verified. Refuse rather than
        # accept on zero evidence (all([]) would otherwise be True).
        return GateResult(
            passed=False,
            checks=[GateCheck(name="no_checks", passed=False, detail="no acceptance thresholds configured")],
        )

    return GateResult(passed=all(c.passed for c in checks), checks=checks)
