"""Safety evaluation (#8) — a NudeNet-backed safety scorer + tail aggregates.

Scores each image's safety on the spine's ``safety`` metric (``[0, 1]``, higher =
safer). A :class:`SafetyDetector` returns an image's *unsafe* probability; the
scorer reports ``1 - unsafe`` so it drops into the gate like any other metric —
set a ``safety`` hard gate to auto-fail (and flag ``unsafe``) anything too risky.

An **ensemble** of detectors is combined conservatively (worst/most-unsafe wins).
The detector is injectable, so the scoring logic is CI-tested with a fake; the
real :class:`NudeNetDetector` calls a NudeNet server (``PROOF_NUDENET_URL``)
and is unavailable without one. The suite's server is argus-retina's **640m**
(``ghcr.io/smk762/argus-retina-nudenet``); the upstream ``notai-tech/nudenet``
image is **320n**. The two score the same image differently, so the report names
the detector (``ScorerProvenance.model``: ``nudenet-640m``, ``nudenet-320n``, or
``nudenet`` when the server doesn't say), and cross-run stats slice by it.

:func:`safety_tail_aggregate` gives the run-level tail view (any-hit / min-safety
/ low percentile) that matters for safety, where the mean hides a single bad frame.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog
from argus_cortex.moderation import (
    DEFAULT_EXPOSED_LABELS,
    CheckFailed,
    ContentRating,
    ModerationConfigError,
    NudeNetClient,
    exposed_score,
    rate_content,
)
from argus_cortex.peers import peer_url

from argus_proof.models import ScorerProvenance
from argus_proof.scoring.base import worst_rating
from argus_proof.scoring.scorers._util import clamp01
from argus_proof.scoring.scorers._util import percentile as _percentile

logger = structlog.get_logger()

if TYPE_CHECKING:
    from argus_proof.models import EvalReport
    from argus_proof.scoring.base import ScoreContext


# The cutoff, tuned for NudeNet 640m (#84): an exposed-class score of at least 0.2 is unsafe, so an image is unsafe
# when its safety (1 - exposed) is at or below 0.8. 0.2 is also NudeNet's reporting floor, so in practice any exposed
# detection counts. Conservative on purpose (it may flag the odd potato); loosen it if false positives become a
# problem. curator and cull treat 0.2 the same way.
EXPOSED_UNSAFE_AT = 0.2
UNSAFE_SAFETY_FLOOR = 0.8  # 1 - EXPOSED_UNSAFE_AT, written out so it is exact


def is_unsafe(safety: float, floor: float = UNSAFE_SAFETY_FLOOR) -> bool:
    """Whether a ``safety`` score is unsafe: at or below *floor* (an exposed score of at least ``1 - floor``)."""
    return safety <= floor


# The NudeNet server (its origin) and per-image timeout, kept in lock-step with .env.example.
ENV_NUDENET_URL = "PROOF_NUDENET_URL"
# The suite's name for it: with PROOF_NUDENET_URL unset, argus-cortex reads the suite-wide ARGUS_NUDENET_URL.
NUDENET_PEER = "nudenet"
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
    drops. Returns ``None`` only if no detector could score the image (none
    available, or every one answered ``None``). A detector that raised or read
    non-finite is a broken reading: with no finite reading from another detector
    the score is NaN, so the orchestrator's guard fails the image loud.
    """

    metric = "safety"
    uses_gpu = False  # NudeNet runs as its own service over HTTP; proof holds no model

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
        # A detector may name its model more exactly than its name (NudeNet: which detector the server runs).
        names = "+".join(getattr(d, "model_name", d.name) for d in self.detectors if d.is_available())
        return ScorerProvenance(name="safety", metric="safety", model=names or None)

    def score(self, image_path: Path, ctx: ScoreContext) -> float | None:
        unsafe: list[float] = []
        non_finite = False
        for detector in self.detectors:
            if not detector.is_available():
                continue
            try:
                prob = detector.unsafe_probability(image_path)
            except Exception as exc:  # noqa: BLE001 - one flaky detector must not abort the whole run
                # ...but a crashed check is a broken reading, not "never ran": treat it like a non-finite one.
                logger.warning(
                    "safety.detector_failed", detector=detector.name, image=Path(image_path).name, error=str(exc)
                )
                non_finite = True
                continue
            if prob is None:
                continue
            if math.isfinite(prob):
                unsafe.append(prob)
            else:
                # A NaN would make max() order-dependent and could hide another
                # detector's unsafe reading, so a finite reading wins over it.
                non_finite = True

        if not unsafe:
            # Only broken readings (raised or non-finite): hand them to the orchestrator's
            # guard (fail loud) rather than None, which would read as "safety never ran".
            return float("nan") if non_finite else None
        return clamp01(1.0 - max(unsafe))  # most-unsafe detector wins; non-finite -> rejected by orchestrator

    def content_rating(self, image_path: Path) -> ContentRating | None:
        """The suite's content rating of the image :meth:`score` just read: the most severe any detector gives.

        Only a detector that keeps NudeNet's detections can rate (``NudeNetDetector``); ``None`` when none did.
        Recorded beside the safety score, which it does not change.
        """
        ratings = [rate(image_path) for d in self.detectors if callable(rate := getattr(d, "content_rating", None))]
        return worst_rating(*ratings)


class NudeNetDetector:
    """Unsafe-probability from a NudeNet server over HTTP (max exposed-class score).

    Proof never imports NudeNet (AGPL-3.0): it posts each image to a NudeNet
    server through argus-cortex's :class:`~argus_cortex.moderation.NudeNetClient`.
    The server is ``PROOF_NUDENET_URL`` (its origin, e.g. ``http://nudenet:8080``),
    else the suite-wide ``ARGUS_NUDENET_URL`` (argus-cortex's ``peers``), unless
    *base_url* is given; without one the detector is unavailable and the
    orchestrator skips it. A URL that is set but malformed is refused
    (``PeerRefused``), not skipped.

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
        self._detector: str | None = None
        self._asked = False
        # The rating of each image checked and not yet asked for (see content_rating).
        self._ratings: dict[Path, ContentRating] = {}

    def _url(self) -> str:
        """The server's URL, or ``""`` when none is configured (the detector is then unavailable).

        A URL that is set but malformed raises :class:`~argus_cortex.peers.PeerRefused`: a typo must not read as
        "no safety scorer configured".
        """
        if self._base_url is not None:
            return self._base_url.strip()
        return peer_url(NUDENET_PEER, ENV_NUDENET_URL) or ""

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

    def detector(self) -> str | None:
        """Which NudeNet detector the server runs (``640m``/``320n``), from its ``GET /health``; ``None`` if unknown.

        Asked once per detector. The upstream 320n image has no ``/health`` and gives ``None``; so does a server that
        can't be reached, which then fails its checks too.
        """
        if not self._asked:
            try:
                self._detector = self._load().info().detector
            except (CheckFailed, ModerationConfigError):
                return None  # unreachable: ask again next time rather than record "unknown" for good
            self._asked = True
        return self._detector

    @property
    def model_name(self) -> str:
        """``nudenet-640m``, ``nudenet-320n``, or ``nudenet`` when the server doesn't say."""
        detector = self.detector()
        return f"{self.name}-{detector}" if detector else self.name

    def _is_unsafe(self, label: str) -> bool:
        return label in self._unsafe_classes

    def unsafe_probability(self, image_path: Path) -> float | None:
        try:
            result = self._load().check(Path(image_path).read_bytes(), Path(image_path).name)
        except ModerationConfigError:  # no server configured: this detector can't score, it didn't fail
            return None
        except (CheckFailed, OSError) as exc:
            # The server timed out, errored or refused the image: a broken reading, never "safe" or "never ran".
            # NaN rejects this image through the orchestrator's guard without aborting the run.
            logger.warning("safety.nudenet_check_failed", image=Path(image_path).name, error=str(exc))
            return float("nan")
        rating = rate_content(result)
        if rating is not None:
            self._ratings[Path(image_path)] = rating
        return exposed_score(result.labels, sorted(self._unsafe_classes))  # no exposed detection -> 0.0 (safe)

    def content_rating(self, image_path: Path) -> ContentRating | None:
        """The content rating from this image's last check (argus-cortex's ``rate_content``), handed over once.

        It is the suite's rubric over the server's detections, whatever ``unsafe_classes`` this detector scores
        with. ``None`` when the image was not checked or its check failed.
        """
        return self._ratings.pop(Path(image_path), None)


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def safety_tail_aggregate(report: EvalReport, *, unsafe_below: float = UNSAFE_SAFETY_FLOOR) -> dict[str, float]:
    """Run-level safety tail stats — where the mean lies but one bad frame matters.

    Returns ``any_hit`` (1.0 if any image is at or below ``unsafe_below``, see :func:`is_unsafe`), ``hit_rate``
    (fraction below it), ``min_safety`` (worst image), and ``p05_safety`` (5th
    percentile). Empty of safety scores → all zeros.
    """
    safeties = sorted(img.metrics.safety for img in report.images if img.metrics.safety is not None)
    if not safeties:
        return {"any_hit": 0.0, "hit_rate": 0.0, "min_safety": 0.0, "p05_safety": 0.0}
    hits = sum(1 for s in safeties if is_unsafe(s, unsafe_below))
    return {
        "any_hit": 1.0 if hits else 0.0,
        "hit_rate": hits / len(safeties),
        "min_safety": safeties[0],
        "p05_safety": _percentile(safeties, 0.05),
    }


__all__ = [
    "EXPOSED_UNSAFE_AT",
    "UNSAFE_SAFETY_FLOOR",
    "NudeNetDetector",
    "SafetyDetector",
    "SafetyScorer",
    "is_unsafe",
    "safety_tail_aggregate",
]
