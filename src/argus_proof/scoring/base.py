"""The scoring interfaces — the spine the Phase 2 scorers plug into.

Objective scoring turns "did the concept transfer, and is the output any good?"
into numbers, so identity/quality are measured continuously and HITL only sees
the borderline band. Concrete scorers (InsightFace identity, CLIPScore, pyiqa,
DreamSim, ImageReward, phash dedup) are heavy and land behind the ``[score]``
extra; this module is the light, dependency-free contract they implement and the
orchestrator composes.

Three roles:

* :class:`ImageScorer` — scores one image on one metric (identity / clip_score /
  aesthetic / preference / safety), returning a **normalised ``[0, 1]``** score
  (higher = better) so the gate can combine metrics, or ``None`` when it doesn't
  apply (an identity scorer on a non-identity run, missing references).
* :class:`Deduper` — groups near-identical outputs so a Monte-Carlo cluster
  counts once toward the pass rate.
* :class:`DiversityScorer` — rewards output variety, penalises mode collapse.

Every scorer reports :class:`~argus_proof.models.ScorerProvenance` (its model +
version) so a report is auditable and cross-run comparison stays valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from argus_core.taxonomy import TargetProfile

from argus_proof.models import RejectReasonCode

if TYPE_CHECKING:
    from argus_proof.models import GeneratedImage, ScorerProvenance

# The metric axes a per-image scorer can fill (mirrors MetricScores fields).
METRIC_FIELDS: tuple[str, ...] = ("identity", "clip_score", "aesthetic", "preference", "safety")

# Which structured reject reason a failing metric maps to, so an auto-fail names
# a cause downstream stats can group by.
REJECT_CODE_FOR_METRIC: dict[str, RejectReasonCode] = {
    "identity": "identity_mismatch",
    "clip_score": "prompt_mismatch",
    "aesthetic": "low_quality",
    "preference": "low_quality",
    "safety": "unsafe",
}


@dataclass
class ScoreContext:
    """Per-run inputs a scorer may need beyond the image itself.

    ``reference_images`` is the **held-out** subject reference set for identity
    scoring — it must NOT overlap the training set, or you measure memorisation
    instead of likeness. ``dataset_images`` is the training distribution used for
    CLIP distribution-match. ``profile.target_category`` decides which similarity
    scorers apply (face identity vs. perceptual similarity for wardrobe/setting).
    """

    prompt: str
    profile: TargetProfile = field(default_factory=TargetProfile)
    reference_images: list[Path] = field(default_factory=list)
    dataset_images: list[Path] = field(default_factory=list)


@runtime_checkable
class ImageScorer(Protocol):
    """Scores one image on one metric, normalised to ``[0, 1]`` (higher better)."""

    metric: str

    def provenance(self) -> ScorerProvenance: ...
    def is_available(self) -> bool: ...
    def score(self, image_path: Path, ctx: ScoreContext) -> float | None: ...


@runtime_checkable
class Deduper(Protocol):
    """Groups near-identical images; returns a group id per image (parallel list)."""

    def provenance(self) -> ScorerProvenance: ...
    def is_available(self) -> bool: ...
    def group(self, images: list[GeneratedImage]) -> list[int]: ...


@runtime_checkable
class DiversityScorer(Protocol):
    """Scores output variety across the run in ``[0, 1]`` (higher = more varied)."""

    def provenance(self) -> ScorerProvenance: ...
    def is_available(self) -> bool: ...
    def score(self, images: list[GeneratedImage], ctx: ScoreContext) -> float: ...
