from __future__ import annotations

from argus_proof.acceptance import evaluate_acceptance
from argus_proof.models import (
    AcceptanceThresholds,
    AggregateScores,
    EvalReport,
    ImageScores,
    MetricScores,
    RejectReason,
    Verdict,
)


def report(
    *,
    n_passed: int,
    n_groups: int,
    pass_rate: float,
    identity: float | None = None,
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
            means=MetricScores(identity=identity),
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


def test_unsafe_rate_ceiling() -> None:
    images = [
        ImageScores(image_id="a", seed=1, passed=True),
        ImageScores(image_id="b", seed=2, passed=False, reject_reasons=[RejectReason(code="unsafe")]),
    ]  # 1 of 2 unsafe = 0.5
    r = report(n_passed=1, n_groups=2, pass_rate=0.5, images=images)
    assert check(evaluate_acceptance(r, AcceptanceThresholds(max_unsafe_rate=0.6)), "unsafe_rate").passed is True
    assert check(evaluate_acceptance(r, AcceptanceThresholds(max_unsafe_rate=0.1)), "unsafe_rate").passed is False


def test_all_checks_must_pass() -> None:
    r = report(n_passed=8, n_groups=10, pass_rate=0.8, identity=0.4)
    t = AcceptanceThresholds(min_pass_rate=0.75, min_identity_mean=0.7)  # pass_rate ok, identity fails
    assert evaluate_acceptance(r, t).passed is False
