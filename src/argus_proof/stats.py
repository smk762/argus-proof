"""Small, dependency-free statistics for defensible pass-rate reporting.

A bare pass-rate (``n_passed / n``) hides how much evidence is behind it — 3/4
and 300/400 are both 0.75 but mean very different things. The Wilson score
interval gives a confidence interval on a proportion that behaves well at small
N and near 0/1, so the CI gate (#12) can require a statistically defensible
*lower bound*, not just a point estimate. Implemented with stdlib ``math`` (an
inverse-normal approximation) so the core needs no scipy/statsmodels.
"""

from __future__ import annotations

import math

# Acklam's inverse-normal-CDF coefficients (rational approximation, ~1e-8 without
# the optional Halley refinement — far tighter than any gate decision needs).
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)


def normal_ppf(p: float) -> float:
    """Inverse standard-normal CDF (quantile) for ``0 < p < 1``.

    The z-score such that ``P(Z <= z) = p`` — e.g. ``normal_ppf(0.975) ≈ 1.96``.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5])
        * q
        / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)
    )


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval ``(lo, hi)`` for a proportion.

    Returns ``(0.0, 1.0)`` for ``n == 0`` (no evidence — the widest interval).
    """
    if successes < 0 or successes > n:
        raise ValueError(f"successes ({successes}) must be in [0, n] ([0, {n}])")
    if n == 0:
        return (0.0, 1.0)
    z = normal_ppf(1 - (1 - confidence) / 2)
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def wilson_lower_bound(successes: int, n: int, confidence: float = 0.95) -> float:
    """The lower end of the Wilson interval — a defensible worst-case pass-rate."""
    return wilson_interval(successes, n, confidence)[0]
