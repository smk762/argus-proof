"""Objective scoring for a generation run — the Phase 2 spine.

The framework that turns generated images into a scored
:class:`~argus_proof.models.EvalReport`: per-image metric scorers, near-duplicate
collapse, diversity, and an auto-pass / auto-fail / needs-HITL gate. Concrete
scorers (InsightFace, CLIPScore, pyiqa, DreamSim, ImageReward, phash) live behind
the ``[score]`` extra and implement the protocols here.
"""

from __future__ import annotations

from argus_proof.scoring.base import (
    METRIC_FIELDS,
    REJECT_CODE_FOR_METRIC,
    Deduper,
    DiversityScorer,
    ImageScorer,
    ScoreContext,
)
from argus_proof.scoring.gate import composite_score, gate_image
from argus_proof.scoring.orchestrator import score_run

__all__ = [
    "METRIC_FIELDS",
    "REJECT_CODE_FOR_METRIC",
    "Deduper",
    "DiversityScorer",
    "ImageScorer",
    "ScoreContext",
    "composite_score",
    "gate_image",
    "score_run",
]
