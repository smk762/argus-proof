from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_scoring import manifest

from argus_proof.models import GeneratedImage
from argus_proof.scoring import score_run
from argus_proof.scoring.scorers._util import clamp01, module_available, phash_of

# --------------------------------------------------------------------------
# clamp01
# --------------------------------------------------------------------------


def test_clamp01_clamps_finite() -> None:
    assert clamp01(0.5) == 0.5
    assert clamp01(1.5) == 1.0
    assert clamp01(-0.2) == 0.0


def test_clamp01_passes_non_finite_through() -> None:
    # crucially NOT 1.0 — a naive max(0,min(1,nan)) would return 1.0 (a perfect score)
    assert math.isnan(clamp01(float("nan")))
    assert clamp01(float("inf")) == float("inf")
    assert clamp01(float("-inf")) == float("-inf")


# --------------------------------------------------------------------------
# module_available
# --------------------------------------------------------------------------


def test_module_available() -> None:
    assert module_available("math", "json") is True
    assert module_available("definitely_not_a_module_xyz") is False
    assert module_available("math", "definitely_not_a_module_xyz") is False


# --------------------------------------------------------------------------
# phash_of cache
# --------------------------------------------------------------------------


def test_phash_of_is_cached(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(p)
    first = phash_of(str(p), 8)
    second = phash_of(str(p), 8)
    assert first is second  # cache hit returns the same object


# --------------------------------------------------------------------------
# orchestrator guards (via the base branch)
# --------------------------------------------------------------------------


class _Fake:
    def __init__(self, metric: str, value: float) -> None:
        self.metric = metric
        self._value = value

    def provenance(self):  # noqa: ANN202
        from argus_proof.models import ScorerProvenance

        return ScorerProvenance(name=f"fake-{self.metric}", metric=self.metric)

    def is_available(self) -> bool:
        return True

    def score(self, image_path: Path, ctx) -> float:  # noqa: ANN001
        return self._value


def _img() -> GeneratedImage:
    return GeneratedImage(image_id="run-1-1", run_id="run-1", seed=1, path="run-1-1.png", width=64, height=64)


def test_duplicate_metric_scorers_rejected() -> None:
    with pytest.raises(ValueError, match="both target metric"):
        score_run(manifest(), [_img()], scorers=[_Fake("aesthetic", 0.9), _Fake("aesthetic", 0.1)])


def test_orchestrator_rejects_nan_score() -> None:
    # A NaN reaching the orchestrator (not pre-clamped to 1.0) is refused loudly.
    with pytest.raises(ValueError, match=r"normalised to \[0, 1\]"):
        score_run(manifest(), [_img()], scorers=[_Fake("aesthetic", float("nan"))])
