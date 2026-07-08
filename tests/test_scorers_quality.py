from __future__ import annotations

from pathlib import Path

import pytest

from argus_proof.models import GeneratedImage
from argus_proof.scoring import ScoreContext, score_run
from argus_proof.scoring.scorers import (
    ModelScorer,
    clip_score_scorer,
    image_reward_scorer,
    pyiqa_scorer,
)
from argus_proof.scoring.scorers.quality import linear_normalize
from tests.test_scoring import manifest


class FakeModel:
    """Returns a fixed raw score per image stem (None = couldn't score)."""

    name = "fake-model"

    def __init__(self, raw: dict[str, float | None], available: bool = True) -> None:
        self.raw = raw
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def score(self, image_path: Path, ctx: ScoreContext) -> float | None:
        return self.raw.get(Path(image_path).stem)


def gen(image_id: str, seed: int = 1) -> GeneratedImage:
    return GeneratedImage(image_id=image_id, run_id="run-1", seed=seed, path=f"{image_id}.png", width=64, height=64)


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------


def test_linear_normalize_maps_and_clamps() -> None:
    assert linear_normalize(20.0, 20.0, 35.0) == 0.0  # at lo
    assert linear_normalize(35.0, 20.0, 35.0) == 1.0  # at hi
    assert linear_normalize(27.5, 20.0, 35.0) == pytest.approx(0.5)  # midpoint
    assert linear_normalize(10.0, 20.0, 35.0) == 0.0  # below lo clamps
    assert linear_normalize(99.0, 20.0, 35.0) == 1.0  # above hi clamps


def test_linear_normalize_handles_degenerate_range() -> None:
    assert linear_normalize(0.5, 1.0, 1.0) == 0.0  # below the step
    assert linear_normalize(1.0, 1.0, 1.0) == 1.0  # at/above the step


def test_image_reward_style_negative_range() -> None:
    assert linear_normalize(0.0, -2.0, 2.0) == pytest.approx(0.5)
    assert linear_normalize(-2.0, -2.0, 2.0) == 0.0
    assert linear_normalize(2.0, -2.0, 2.0) == 1.0


# --------------------------------------------------------------------------
# ModelScorer
# --------------------------------------------------------------------------


def test_model_scorer_normalizes_raw() -> None:
    scorer = ModelScorer("clip_score", FakeModel({"gen": 27.5}), lo=20.0, hi=35.0)
    assert scorer.score(Path("gen.png"), ctx=None) == pytest.approx(0.5)  # type: ignore[arg-type]


def test_model_scorer_passes_none_through() -> None:
    scorer = ModelScorer("aesthetic", FakeModel({"gen": None}), lo=0.0, hi=1.0)
    assert scorer.score(Path("gen.png"), ctx=None) is None  # type: ignore[arg-type]


def test_model_scorer_availability_and_metric() -> None:
    scorer = ModelScorer("preference", FakeModel({}, available=False), lo=-2.0, hi=2.0)
    assert scorer.metric == "preference"
    assert scorer.is_available() is False
    assert scorer.provenance().metric == "preference"


# --------------------------------------------------------------------------
# factories (real deps absent in CI -> unavailable, but wiring is checked)
# --------------------------------------------------------------------------


def test_factories_set_expected_metric_and_defaults() -> None:
    assert clip_score_scorer().metric == "clip_score"
    assert (clip_score_scorer().lo, clip_score_scorer().hi) == (20.0, 35.0)
    assert pyiqa_scorer().metric == "aesthetic"
    assert image_reward_scorer().metric == "preference"
    assert (image_reward_scorer().lo, image_reward_scorer().hi) == (-2.0, 2.0)


def test_factories_accept_calibration_overrides() -> None:
    s = clip_score_scorer(lo=25.0, hi=30.0)
    assert (s.lo, s.hi) == (25.0, 30.0)


def test_real_scorers_unavailable_without_extra() -> None:
    # torch/pyiqa/ImageReward aren't in the CI env, so these skip cleanly.
    assert clip_score_scorer().is_available() is False
    assert pyiqa_scorer().is_available() is False
    assert image_reward_scorer().is_available() is False


# --------------------------------------------------------------------------
# end-to-end
# --------------------------------------------------------------------------


def test_score_run_fills_quality_metrics() -> None:
    clip = ModelScorer("clip_score", FakeModel({"run-1-1": 35.0}), lo=20.0, hi=35.0)
    aesthetic = ModelScorer("aesthetic", FakeModel({"run-1-1": 0.8}), lo=0.0, hi=1.0)
    report = score_run(manifest(), [gen("run-1-1")], scorers=[clip, aesthetic])
    m = report.images[0].metrics
    assert m.clip_score == 1.0
    assert m.aesthetic == pytest.approx(0.8)
