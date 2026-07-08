"""Quality / prompt-adherence scoring (#5).

An automated pre-pass so HITL only sees the borderline band: prompt adherence
(CLIPScore), technical/aesthetic quality (pyiqa), and a human-preference proxy
(ImageReward). Each is a thin model wrapped by :class:`ModelScorer`, which
applies a **configurable linear normalization** into the spine's ``[0, 1]``
contract.

The raw scales differ per model and a "good" value is library-specific, so the
default ``lo``/``hi`` per scorer are **calibration placeholders** — tune them
against real generations (this is what the CI-gate phase, #12, calibrates). The
model backend is injectable (:class:`ScoreModel`) so the normalization + wiring
is CI-tested without torch; the real adapters are lazy and behind ``[score]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from argus_proof.models import ScorerProvenance

if TYPE_CHECKING:
    from argus_proof.scoring.base import ScoreContext


def linear_normalize(raw: float, lo: float, hi: float) -> float:
    """Map ``raw`` from ``[lo, hi]`` onto ``[0, 1]``, clamped. ``hi==lo`` → 0/1 step."""
    if hi <= lo:
        return 1.0 if raw >= hi else 0.0
    return max(0.0, min(1.0, (raw - lo) / (hi - lo)))


@runtime_checkable
class ScoreModel(Protocol):
    """A model producing a RAW (un-normalised) quality score for an image."""

    name: str

    def is_available(self) -> bool: ...
    def score(self, image_path: Path, ctx: ScoreContext) -> float | None: ...


class ModelScorer:
    """Wrap a :class:`ScoreModel`, normalising its raw score to ``[0, 1]``.

    ``lo``/``hi`` are the raw values that map to 0 and 1 (linear, clamped). A
    model returning ``None`` (couldn't score) passes ``None`` through so the
    metric is simply absent for that image.
    """

    def __init__(self, metric: str, model: ScoreModel, lo: float, hi: float) -> None:
        self.metric = metric
        self.model = model
        self.lo = lo
        self.hi = hi

    def provenance(self) -> ScorerProvenance:
        return ScorerProvenance(name=self.model.name, metric=self.metric, model=self.model.name)

    def is_available(self) -> bool:
        return self.model.is_available()

    def score(self, image_path: Path, ctx: ScoreContext) -> float | None:
        raw = self.model.score(image_path, ctx)
        if raw is None:
            return None
        return linear_normalize(raw, self.lo, self.hi)


# ---------------------------------------------------------------------------
# Real model adapters — lazy, behind the [score] extra (untested in CI)
# ---------------------------------------------------------------------------


class ClipScoreModel:
    """Prompt adherence via torchmetrics ``CLIPScore`` (raw ≈ ``100·max(cos,0)``)."""

    name = "clip-score"

    def __init__(self, clip_model: str = "openai/clip-vit-base-patch16") -> None:
        self.clip_model = clip_model
        self._metric = None

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import torchmetrics  # noqa: F401
            from PIL import Image  # noqa: F401

            return True
        except ImportError:
            return False

    def _load(self):  # noqa: ANN202
        if self._metric is None:
            from torchmetrics.multimodal.clip_score import CLIPScore

            self._metric = CLIPScore(model_name_or_path=self.clip_model)
        return self._metric

    def score(self, image_path: Path, ctx: ScoreContext) -> float | None:
        import numpy as np
        import torch
        from PIL import Image

        with Image.open(image_path) as im:
            arr = np.array(im.convert("RGB"))
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC uint8 -> CHW
        return float(self._load()(tensor, ctx.prompt))


class PyiqaModel:
    """No-reference technical/aesthetic quality via ``pyiqa`` (default CLIP-IQA, ~[0,1])."""

    def __init__(self, metric_name: str = "clipiqa") -> None:
        self.metric_name = metric_name
        self._metric = None

    @property
    def name(self) -> str:
        return f"pyiqa-{self.metric_name}"

    def is_available(self) -> bool:
        try:
            import pyiqa  # noqa: F401

            return True
        except ImportError:
            return False

    def _load(self):  # noqa: ANN202
        if self._metric is None:
            import pyiqa

            self._metric = pyiqa.create_metric(self.metric_name)
        return self._metric

    def score(self, image_path: Path, ctx: ScoreContext) -> float | None:
        return float(self._load()(str(image_path)).item())


class ImageRewardModel:
    """Human-preference proxy via ImageReward (raw ≈ ``[-2, 2]``)."""

    name = "imagereward"

    def __init__(self, model_name: str = "ImageReward-v1.0") -> None:
        self.model_name = model_name
        self._model = None

    def is_available(self) -> bool:
        try:
            import ImageReward  # noqa: F401

            return True
        except ImportError:
            return False

    def _load(self):  # noqa: ANN202
        if self._model is None:
            import ImageReward

            self._model = ImageReward.load(self.model_name)
        return self._model

    def score(self, image_path: Path, ctx: ScoreContext) -> float | None:
        return float(self._load().score(ctx.prompt, str(image_path)))


# ---------------------------------------------------------------------------
# Factories with placeholder default normalizations (CALIBRATE against real data)
# ---------------------------------------------------------------------------


def clip_score_scorer(lo: float = 20.0, hi: float = 35.0, **kwargs: object) -> ModelScorer:
    """Prompt-adherence scorer (metric ``clip_score``). Default lo/hi for the
    torchmetrics 0–100 scale (good match ≈ 25–35) — placeholder; calibrate."""
    return ModelScorer("clip_score", ClipScoreModel(**kwargs), lo, hi)  # type: ignore[arg-type]


def pyiqa_scorer(lo: float = 0.0, hi: float = 1.0, metric_name: str = "clipiqa") -> ModelScorer:
    """Aesthetic/IQA scorer (metric ``aesthetic``). CLIP-IQA is already ~[0,1]."""
    return ModelScorer("aesthetic", PyiqaModel(metric_name), lo, hi)


def image_reward_scorer(lo: float = -2.0, hi: float = 2.0, **kwargs: object) -> ModelScorer:
    """Preference scorer (metric ``preference``). Default lo/hi for ImageReward's
    ≈[-2,2] range — placeholder; calibrate."""
    return ModelScorer("preference", ImageRewardModel(**kwargs), lo, hi)  # type: ignore[arg-type]


__all__ = [
    "ClipScoreModel",
    "ImageRewardModel",
    "ModelScorer",
    "PyiqaModel",
    "ScoreModel",
    "clip_score_scorer",
    "image_reward_scorer",
    "linear_normalize",
    "pyiqa_scorer",
]
