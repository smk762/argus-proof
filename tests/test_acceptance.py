from __future__ import annotations

import pytest

from argus_proof.acceptance import evaluate_acceptance
from argus_proof.models import (
    AcceptanceThresholds,
    AggregateScores,
    EvalReport,
    GateResult,
    ImageScores,
    MetricScores,
    PoseComparison,
    PoseProvenance,
    PoseReport,
    RejectReason,
    Verdict,
)


def report(
    *,
    n_passed: int,
    n_groups: int,
    pass_rate: float,
    identity: float | None = None,
    likeness: float | None = None,
    images: list[ImageScores] | None = None,
) -> EvalReport:
    return EvalReport(
        run_id="run-1",
        images=images or [],
        aggregate=AggregateScores(
            n_images=n_groups,
            n_groups=n_groups,
            n_passed=n_passed,
            pass_rate=pass_rate,
            means=MetricScores(identity=identity, likeness=likeness),
        ),
        verdict=Verdict(passed=pass_rate >= 0.75),
    )


def check(result, name):  # noqa: ANN001
    return next(c for c in result.checks if c.name == name)


# --------------------------------------------------------------------------
# pass-rate
# --------------------------------------------------------------------------


def test_pass_rate_gate() -> None:
    r = report(n_passed=8, n_groups=10, pass_rate=0.8)
    assert evaluate_acceptance(r, AcceptanceThresholds(min_pass_rate=0.75)).passed is True
    assert evaluate_acceptance(r, AcceptanceThresholds(min_pass_rate=0.9)).passed is False


def test_only_configured_checks_run() -> None:
    r = report(n_passed=8, n_groups=10, pass_rate=0.8)
    result = evaluate_acceptance(r, AcceptanceThresholds(min_pass_rate=0.75))  # others None
    assert [c.name for c in result.checks] == ["pass_rate"]


# --------------------------------------------------------------------------
# CI lower bound — the small-N safety net
# --------------------------------------------------------------------------


def test_ci_lower_bound_rejects_lucky_small_sample() -> None:
    # 3/3 = 1.0 pass-rate, but the Wilson lower bound is ~0.44 -> fails a 0.75 CI floor
    lucky = report(n_passed=3, n_groups=3, pass_rate=1.0)
    t = AcceptanceThresholds(min_pass_rate=0.75, min_pass_rate_ci_lower=0.75)
    result = evaluate_acceptance(lucky, t)
    assert check(result, "pass_rate").passed is True  # point estimate clears
    assert check(result, "pass_rate_ci_lower").passed is False  # but the evidence doesn't
    assert result.passed is False


def test_ci_lower_bound_passes_with_enough_evidence() -> None:
    solid = report(n_passed=100, n_groups=100, pass_rate=1.0)
    t = AcceptanceThresholds(min_pass_rate=0.75, min_pass_rate_ci_lower=0.75)
    assert evaluate_acceptance(solid, t).passed is True


# --------------------------------------------------------------------------
# identity floor / safety ceiling
# --------------------------------------------------------------------------


def test_identity_floor() -> None:
    r = report(n_passed=8, n_groups=10, pass_rate=0.8, identity=0.6)
    assert check(evaluate_acceptance(r, AcceptanceThresholds(min_identity_mean=0.5)), "identity_mean").passed is True
    assert check(evaluate_acceptance(r, AcceptanceThresholds(min_identity_mean=0.7)), "identity_mean").passed is False


def test_identity_floor_fails_when_not_measured() -> None:
    r = report(n_passed=8, n_groups=10, pass_rate=0.8, identity=None)
    result = evaluate_acceptance(r, AcceptanceThresholds(min_identity_mean=0.5))
    assert check(result, "identity_mean").passed is False  # can't accept on missing evidence
    assert "not measured" in check(result, "identity_mean").detail


def test_unsafe_rate_ceiling_from_reject_reason() -> None:
    images = [
        ImageScores(image_id="a", seed=1, passed=True),
        ImageScores(image_id="b", seed=2, passed=False, reject_reasons=[RejectReason(code="unsafe")]),
    ]  # 1 of 2 unsafe = 0.5
    r = report(n_passed=1, n_groups=2, pass_rate=0.5, images=images)
    assert check(evaluate_acceptance(r, AcceptanceThresholds(max_unsafe_rate=0.6)), "unsafe_rate").passed is True
    assert check(evaluate_acceptance(r, AcceptanceThresholds(max_unsafe_rate=0.1)), "unsafe_rate").passed is False


def test_unsafe_rate_uses_safety_metric_not_only_reject_reason() -> None:
    # low safety metric but NO "unsafe" reject reason (default gate has no safety
    # hard gate) must still count as unsafe — else the gate green-lights unsafe output.
    images = [
        ImageScores(image_id="a", seed=1, passed=True, metrics=MetricScores(safety=0.9)),
        ImageScores(image_id="b", seed=2, passed=True, metrics=MetricScores(safety=0.1)),
    ]
    r = report(n_passed=2, n_groups=2, pass_rate=1.0, images=images)
    result = evaluate_acceptance(r, AcceptanceThresholds(min_pass_rate=None, max_unsafe_rate=0.0))
    assert check(result, "unsafe_rate").actual == 0.5  # b caught by the safety-metric floor
    assert result.passed is False


def test_no_checks_configured_rejects() -> None:
    r = report(n_passed=10, n_groups=10, pass_rate=1.0)
    result = evaluate_acceptance(r, AcceptanceThresholds(min_pass_rate=None))  # all thresholds off
    assert result.passed is False  # never accept on zero evidence
    assert result.checks[0].name == "no_checks"


def test_all_checks_must_pass() -> None:
    r = report(n_passed=8, n_groups=10, pass_rate=0.8, identity=0.4)
    t = AcceptanceThresholds(min_pass_rate=0.75, min_identity_mean=0.7)  # pass_rate ok, identity fails
    assert evaluate_acceptance(r, t).passed is False


def test_more_passes_than_groups_fails_the_ci_check_instead_of_clamping() -> None:
    # 4 passes counted per image against 4 groups would read as 4/4 if clamped.
    r = report(n_passed=5, n_groups=4, pass_rate=0.25)
    c = check(evaluate_acceptance(r, AcceptanceThresholds(min_pass_rate_ci_lower=0.1)), "pass_rate_ci_lower")
    assert not c.passed and "inconsistent report" in c.detail


def test_likeness_floor() -> None:
    r = report(n_passed=8, n_groups=10, pass_rate=0.8, likeness=0.6)
    assert check(evaluate_acceptance(r, AcceptanceThresholds(min_likeness_mean=0.5)), "likeness_mean").passed is True
    failed = check(evaluate_acceptance(r, AcceptanceThresholds(min_likeness_mean=0.7)), "likeness_mean")
    assert (failed.passed, failed.actual, failed.threshold) == (False, 0.6, 0.7)


def test_likeness_floor_fails_when_not_measured() -> None:
    # e.g. identity scored but likeness not (no [score] extra): no evidence, no accept.
    r = report(n_passed=8, n_groups=10, pass_rate=0.8, identity=0.9)
    result = evaluate_acceptance(r, AcceptanceThresholds(min_likeness_mean=0.5))
    assert result.passed is False
    assert check(result, "likeness_mean").detail == "likeness not measured"


# --------------------------------------------------------------------------
# the pose-collapse gate (#122)
# --------------------------------------------------------------------------


def pose(
    *,
    count: int = 40,
    usable: int | None = 36,
    comparable_share: float | None = 0.95,
    diversity: float | None = 0.6,
    largest: float | None = 0.3,
    comparison: PoseComparison | None = None,
) -> PoseReport:
    return PoseReport(
        count=count,
        posed=count,
        usable=usable,
        comparable_share=comparable_share,
        diversity=diversity,
        largest_cluster_share=largest,
        cluster_count=None if usable is None else 5,
        largest_cluster_medoids=["run-1-7"],
        comparison=comparison,
        provenance=PoseProvenance(not_checked=[] if usable is not None else ["pose: puppet could not be reached"]),
    )


def comparison(*, excess: float | None = 0.25, eligible: int = 30, baseline: int = 30) -> PoseComparison:
    return PoseComparison(
        training_frames=40,
        output_count=40,
        matched_count=40,
        reference_count=40,
        close_eligible=eligible,
        close_share=0.3,
        baseline_eligible=baseline,
        baseline_close_share=0.05,
        close_excess=excess,
        threshold=0.04,
        n_close=9,
    )


def gated(p: PoseReport | None, **thresholds: float) -> GateResult:
    r = report(n_passed=8, n_groups=10, pass_rate=0.8).model_copy(update={"pose": p})
    return evaluate_acceptance(r, AcceptanceThresholds(min_pass_rate=None, **thresholds))


def test_the_pose_gate_is_off_by_default() -> None:
    result = gated(pose(diversity=0.01, largest=1.0), min_pass_rate_ci_lower=0.0)
    assert not any(c.name.startswith("pose") for c in result.checks)
    assert AcceptanceThresholds().min_pose_diversity is None


def test_a_measurable_varied_run_passes_the_pose_gate() -> None:
    result = gated(pose(), min_pose_diversity=0.3)
    assert result.passed is True
    assert [c.name for c in result.checks] == ["pose_measurable", "pose_diversity", "pose_largest_cluster"]
    assert all(c.code is None for c in result.checks)


@pytest.mark.parametrize(
    ("p", "why"),
    [
        (None, "no pose block"),
        (pose(usable=None, comparable_share=None, diversity=None, largest=None), "could not be reached"),
        (pose(usable=9), "9 usable pose(s), need at least 10"),
        (pose(count=40, usable=19), "19/40 images have a usable pose"),
        (pose(comparable_share=0.79), "comparable pose share 0.79"),
        (pose(comparable_share=None), "comparable pose share unknown"),
    ],
)
def test_a_pose_gate_never_passes_when_not_measurable(p: PoseReport | None, why: str) -> None:
    result = gated(p, min_pose_diversity=0.3)
    assert result.passed is False
    [only] = result.checks
    assert (only.name, only.passed) == ("pose_measurable", False) and why in only.detail


def test_low_pose_diversity_is_a_pose_collapse() -> None:
    result = gated(pose(diversity=0.2, largest=0.5), min_pose_diversity=0.3)
    c = check(result, "pose_diversity")
    assert (result.passed, c.passed, c.code, c.actual, c.threshold) == (False, False, "pose_collapse", 0.2, 0.3)
    assert "run-1-7" in c.detail
    assert check(result, "pose_largest_cluster").passed is True


def test_a_dominant_pose_cluster_is_a_pose_collapse_even_when_diversity_clears_the_floor() -> None:
    result = gated(pose(diversity=0.35, largest=0.75), min_pose_diversity=0.3)  # at the share: collapsed
    c = check(result, "pose_largest_cluster")
    assert (result.passed, c.passed, c.code) == (False, False, "pose_collapse")
    assert gated(pose(diversity=0.35, largest=0.8), min_pose_diversity=0.3, max_pose_cluster_share=0.9).passed


def test_pose_copying_is_flagged_for_review_never_rejected() -> None:
    result = gated(pose(comparison=comparison()), min_pose_diversity=0.3)
    c = check(result, "pose_copying")
    assert (result.passed, c.passed, c.code, c.actual) == (True, True, "pose_copying", 0.25)
    assert "flagged for review" in c.detail and "9 output(s)" in c.detail


@pytest.mark.parametrize(
    "cmp",
    [comparison(excess=0.09), comparison(excess=None), comparison(eligible=19), comparison(baseline=19)],
)
def test_pose_copying_needs_enough_evidence(cmp: PoseComparison) -> None:
    assert not any(c.name == "pose_copying" for c in gated(pose(comparison=cmp), min_pose_diversity=0.3).checks)


def test_pose_copying_is_not_reported_with_the_gate_off() -> None:
    result = gated(pose(comparison=comparison()), min_pass_rate_ci_lower=0.0)
    assert not any(c.name == "pose_copying" for c in result.checks)
