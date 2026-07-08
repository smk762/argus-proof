"""Concrete scorers implementing the :mod:`argus_proof.scoring` protocols.

Each scorer lazy-imports its heavy dependencies (behind the ``[score]`` extra)
and reports ``is_available()`` so the orchestrator skips it cleanly when the
extra isn't installed. Perceptual-hash dedup/diversity land first (lightweight,
CPU-only); identity/quality scorers (the torch stack) follow with #4/#5.
"""

from __future__ import annotations

from argus_proof.scoring.scorers.identity import Embedder, IdentityScorer, InsightFaceEmbedder
from argus_proof.scoring.scorers.phash import PhashDeduper, PhashDiversityScorer
from argus_proof.scoring.scorers.quality import (
    ModelScorer,
    ScoreModel,
    clip_score_scorer,
    image_reward_scorer,
    pyiqa_scorer,
)

__all__ = [
    "Embedder",
    "IdentityScorer",
    "InsightFaceEmbedder",
    "ModelScorer",
    "PhashDeduper",
    "PhashDiversityScorer",
    "ScoreModel",
    "clip_score_scorer",
    "image_reward_scorer",
    "pyiqa_scorer",
]
