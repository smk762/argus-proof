from __future__ import annotations

import pytest

from argus_proof.stats import normal_ppf, wilson_interval, wilson_lower_bound


def test_normal_ppf_known_quantiles() -> None:
    assert normal_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert normal_ppf(0.975) == pytest.approx(1.959964, abs=1e-4)
    assert normal_ppf(0.95) == pytest.approx(1.644854, abs=1e-4)


def test_normal_ppf_symmetry() -> None:
    assert normal_ppf(0.1) == pytest.approx(-normal_ppf(0.9), abs=1e-6)


def test_normal_ppf_out_of_range_raises() -> None:
    for p in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            normal_ppf(p)


def test_wilson_no_evidence_is_widest() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_bounds_are_valid() -> None:
    lo, hi = wilson_interval(8, 10)
    assert 0.0 <= lo < 0.8 < hi <= 1.0


def test_wilson_lower_bound_rewards_more_evidence() -> None:
    # 0.75 both ways, but 300/400 is far more evidence than 3/4 -> tighter, higher LB
    assert wilson_lower_bound(300, 400) > wilson_lower_bound(3, 4)


def test_wilson_extremes() -> None:
    assert wilson_lower_bound(0, 10) == pytest.approx(0.0, abs=0.05)  # near 0
    assert wilson_interval(10, 10)[1] == 1.0  # all-success upper bound is 1


def test_wilson_invalid_successes() -> None:
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
