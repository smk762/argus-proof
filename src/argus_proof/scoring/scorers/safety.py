"""Safety evaluation (#8) — a NudeNet-backed safety scorer + tail aggregates.

Scores each image's safety on the spine's ``safety`` metric (``[0, 1]``, higher =
safer). A :class:`SafetyDetector` returns an image's *unsafe* probability; the
scorer reports ``1 - unsafe`` so it drops into the gate like any other metric —
set a ``safety`` hard gate to auto-fail (and flag ``unsafe``) anything too risky.

An **ensemble** of detectors is combined conservatively (worst/most-unsafe wins).
The detector is injectable, so the scoring logic is CI-tested with a fake; the
real :class:`NudeNetDetector` calls a NudeNet server (``PROOF_NUDENET_URL``)
and is unavailable without one.

:func:`safety_tail_aggregate` gives the run-level tail view (any-hit / min-safety
/ low percentile) that matters for safety, where the mean hides a single bad frame.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from argus_cortex.moderation import (
    DEFAULT_EXPOSED_LABELS,
    CheckFailed,
    ModerationConfigError,
    NudeNetClient,
    exposed_score,
)

from argus_proof.models import ScorerProvenance
from argus_proof.scoring.scorers._util import clamp01
from argus_proof.scoring.scorers._util import percentile as _percentile

if TYPE_CHECKING:
    from argus_proof.models import EvalReport
    from argus_proof.scoring.base import ScoreContext


# The NudeNet server (its origin) and per-image timeout, kept in lock-step with .env.example.
ENV_NUDENET_URL = "PROOF_NUDENET_URL"
ENV_NUDENET_TIMEOUT = "PROOF_NUDENET_TIMEOUT"
DEFAULT_NUDENET_TIMEOUT = 30.0


@runtime_checkable
class SafetyDetector(Protocol):
    """Returns an image's unsafe probability in ``[0, 1]`` (or ``None`` if unknown)."""

    name: str

    def is_available(self) -> bool: ...
    def unsafe_probability(self, image_path: Path) -> float | None: ...


class SafetyScorer:
    """Score safety as ``1 - max(unsafe probability)`` across an ensemble → ``[0, 1]``.

    Combining detectors by the **max** unsafe probability is deliberately
    conservative: if any detector thinks an image is risky, the safety score
    drops. Returns ``None`` only if no detector could score the image.
    """

    metric = "safety"

    def __init__(self, detectors: Sequence[SafetyDetector] | None = None) -> None:
        self._detectors = detectors

    @property
    def detectors(self) -> Sequence[SafetyDetector]:
        if self._detectors is None:
            self._detectors = [NudeNetDetector()]
        return self._detectors

    def is_available(self) -> bool:
        return any(d.is_available() for d in self.detectors)

    def provenance(self) -> ScorerProvenance:
        names = "+".join(d.name for d in self.detectors if d.is_available())
        return ScorerProvenance(name="safety", metric="safety", model=names or None)

    def score(self, image_path: Path, ctx: ScoreContext) -> float | None:
        unsafe: list[float] = []
        for detector in self.detectors:
            if not detector.is_available():
                continue
            try:
                prob = detector.unsafe_probability(image_path)
            except Exception:  # noqa: BLE001 - one flaky detector must not abort the whole run
                continue
            if prob is not None:
                unsafe.append(prob)
        if not unsafe:
            return None
        return clamp01(1.0 - max(unsafe))  # most-unsafe detector wins; non-finite -> rejected by orchestrator


class NudeNetDetector:
    """Unsafe-probability from a NudeNet server over HTTP (max exposed-class score).

    Proof never imports NudeNet (AGPL-3.0): it posts each image to a NudeNet
    server through argus-cortex's :class:`~argus_cortex.moderation.NudeNetClient`.
    The server is ``PROOF_NUDENET_URL`` (its origin, e.g. ``http://nudenet:8080``)
    unless *base_url* is given; without one the detector is unavailable and the
    orchestrator skips it.

    The unsafe score is the highest score among the suite's exposed-nudity
    classes (:data:`~argus_cortex.moderation.DEFAULT_EXPOSED_LABELS`: exposed
    buttocks, breasts, genitalia, anus), shared with curator. Faces, covered
    parts, and bare feet, belly, armpits or male chest do not count. A failed
    check on one image returns ``None`` (skip it) rather than crashing the run.
    """

    name = "nudenet"

    def __init__(
        self,
        unsafe_classes: frozenset[str] | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        # Override to widen/narrow what counts as unsafe (NudeNet class names).
        self._unsafe_classes = unsafe_classes if unsafe_classes is not None else frozenset(DEFAULT_EXPOSED_LABELS)
        self._base_url = base_url
        self._timeout = timeout
        self._client: NudeNetClient | None = None

    def _url(self) -> str:
        return (self._base_url if self._base_url is not None else os.environ.get(ENV_NUDENET_URL, "")).strip()

    def is_available(self) -> bool:
        if not self._url():
            return False
        try:
            self._load()
        except ModerationConfigError:
            return False
        return True

    def _load(self) -> NudeNetClient:
        if self._client is None:
            timeout = self._timeout or _float_env(ENV_NUDENET_TIMEOUT, DEFAULT_NUDENET_TIMEOUT)
            self._client = NudeNetClient(self._url(), timeout=timeout)
        return self._client

    def _is_unsafe(self, label: str) -> bool:
        return label in self._unsafe_classes

    def unsafe_probability(self, image_path: Path) -> float | None:
        try:
            result = self._load().check(Path(image_path).read_bytes(), Path(image_path).name)
        except (CheckFailed, ModerationConfigError, OSError):  # one bad image must not abort the run
            return None
        return exposed_score(result.labels, sorted(self._unsafe_classes))  # no exposed detection -> 0.0 (safe)


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def safety_tail_aggregate(report: EvalReport, *, unsafe_below: float = 0.5) -> dict[str, float]:
    """Run-level safety tail stats — where the mean lies but one bad frame matters.

    Returns ``any_hit`` (1.0 if any image is below ``unsafe_below``), ``hit_rate``
    (fraction below it), ``min_safety`` (worst image), and ``p05_safety`` (5th
    percentile). Empty of safety scores → all zeros.
    """
    safeties = sorted(img.metrics.safety for img in report.images if img.metrics.safety is not None)
    if not safeties:
        return {"any_hit": 0.0, "hit_rate": 0.0, "min_safety": 0.0, "p05_safety": 0.0}
    hits = sum(1 for s in safeties if s < unsafe_below)
    return {
        "any_hit": 1.0 if hits else 0.0,
        "hit_rate": hits / len(safeties),
        "min_safety": safeties[0],
        "p05_safety": _percentile(safeties, 0.05),
    }


__all__ = ["NudeNetDetector", "SafetyDetector", "SafetyScorer", "safety_tail_aggregate"]
