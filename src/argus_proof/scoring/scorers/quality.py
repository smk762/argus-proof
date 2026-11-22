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
Once a model has loaded, the weights files it loaded are pinned by SHA256 in the
scorer's provenance (CLIPScore and pyiqa; ImageReward runs in its own
environment and pins nothing yet).
"""

from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog

from argus_proof.models import ModelRef, ScorerProvenance
from argus_proof.scoring.scorers._util import clamp01, hf_cached_file, module_available, pin_file, weight_pins

if TYPE_CHECKING:
    from argus_proof.scoring.base import ScoreContext

logger = structlog.get_logger()

# The weights files transformers looks for in a checkpoint, in the order it prefers them.
HF_WEIGHTS_FILES = ("model.safetensors", "pytorch_model.bin")


def linear_normalize(raw: float, lo: float, hi: float) -> float:
    """Map ``raw`` from ``[lo, hi]`` onto ``[0, 1]``, clamped. ``hi<=lo`` → 0/1 step.

    A non-finite ``raw`` (NaN/inf from a broken model) passes through via
    :func:`clamp01` so the orchestrator's ``[0, 1]`` guard rejects it, rather
    than a NaN silently clamping to a perfect 1.0.
    """
    if not math.isfinite(raw):
        return raw  # every branch hands a broken score to the orchestrator's guard
    if hi <= lo:
        return 1.0 if raw >= hi else 0.0
    return clamp01((raw - lo) / (hi - lo))


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
        if hi <= lo:
            raise ValueError(f"normalization needs lo < hi, got lo={lo}, hi={hi} (swapped calibration?)")
        self.metric = metric
        self.model = model
        self.lo = lo
        self.hi = hi

    # Every shipped model (CLIPScore, CLIP-IQA, ImageReward) is torch, on the GPU when there is one.
    uses_gpu = True

    def provenance(self) -> ScorerProvenance:
        model_file, extra_files = weight_pins(self.model)
        return ScorerProvenance(
            name=self.model.name,
            metric=self.metric,
            model=self.model.name,
            model_file=model_file,
            extra_files=extra_files,
        )

    def unload(self) -> None:
        """Drop the model (see :func:`~argus_proof.scoring.scorers._util.release_models`)."""
        unload = getattr(self.model, "unload", None)
        if callable(unload):
            unload()

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
        return module_available("torch", "torchmetrics", "PIL", "numpy")

    def unload(self) -> None:
        self._metric = None

    def _load(self):  # noqa: ANN202
        if self._metric is None:
            from torchmetrics.multimodal.clip_score import CLIPScore

            self._metric = CLIPScore(model_name_or_path=self.clip_model)
        return self._metric

    def weight_refs(self) -> list[ModelRef]:
        """The CLIP checkpoint's weights file pinned by SHA256; empty before a load.

        torchmetrics loads the checkpoint through ``transformers``, so this is the
        local directory's or the Hugging Face cache's copy (never downloaded here).
        """
        if self._metric is None:
            return []
        path = hf_cached_file(self.clip_model, HF_WEIGHTS_FILES)
        if path is None:
            logger.warning("clip_score.weights_unpinned", model=self.clip_model)
            return []
        return [pin_file(f"{self.clip_model}/{path.name}", path)]

    def score(self, image_path: Path, ctx: ScoreContext) -> float | None:
        import numpy as np
        import torch
        from PIL import Image

        with Image.open(image_path) as im:
            arr = np.array(im.convert("RGB"))
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC uint8 -> CHW
        metric = self._load()
        metric.reset()  # CLIPScore accumulates state; reset so this is a per-image score, not a running mean
        return float(metric(tensor, ctx.prompt))


class PyiqaModel:
    """No-reference technical/aesthetic quality via ``pyiqa`` (default CLIP-IQA, ~[0,1]).

    pyiqa metrics disagree on direction: NIQE, BRISQUE and PIQE are lower-is-better.
    For those (``metric.lower_better``) the raw score is negated, so the value this
    returns is always higher-is-better and :class:`ModelScorer`'s ``lo``/``hi``
    calibrate on that negated scale.
    """

    def __init__(self, metric_name: str = "clipiqa") -> None:
        self.metric_name = metric_name
        self._metric = None
        self._files: list[Path] = []

    @property
    def name(self) -> str:
        return f"pyiqa-{self.metric_name}"

    def is_available(self) -> bool:
        return module_available("pyiqa")

    def unload(self) -> None:
        self._metric = None
        self._files = []

    def _load(self):  # noqa: ANN202
        if self._metric is None:
            import pyiqa

            with _pyiqa_downloads() as files:
                self._metric = pyiqa.create_metric(self.metric_name)
            self._files = files
        return self._metric

    def weight_refs(self) -> list[ModelRef]:
        """The weights files pyiqa fetched (or found cached) for this metric, pinned by SHA256.

        Empty before a load. A metric that loads its weights some other way
        (a local path, the Hugging Face Hub) records nothing here, and says so.
        """
        if self._metric is None:
            return []
        if not self._files:
            logger.warning("pyiqa.weights_unpinned", metric=self.metric_name)
        return [pin_file(f"{self.name}/{path.name}", path) for path in self._files if path.is_file()]

    def score(self, image_path: Path, ctx: ScoreContext) -> float | None:
        metric = self._load()
        raw = float(metric(str(image_path)).item())
        return -raw if getattr(metric, "lower_better", False) else raw


# pyiqa's helpers that resolve a weights URL to a file in its cache, downloading it
# only when it isn't there.
_PYIQA_FETCHERS = (("pyiqa.utils.download_util", "load_file_from_url"), ("pyiqa.archs.clip_model", "_download"))


def _pyiqa_modules() -> list[object]:
    return [m for n, m in list(sys.modules.items()) if m is not None and (n == "pyiqa" or n.startswith("pyiqa."))]


@contextmanager
def _pyiqa_downloads() -> Iterator[list[Path]]:
    """Record the file every pyiqa weights fetch returns while the block runs.

    pyiqa keeps no record of what it loaded, so its fetch helpers (the generic
    URL cache and its CLIP loader's) are wrapped for one ``create_metric`` call.
    pyiqa imports its architectures lazily, inside that call, so each helper is
    patched where it is defined (a module importing it during the block binds
    the wrapper) and in every module that already bound it; on the way out,
    every wrapper is swapped back, wherever it ended up. The wrappers call
    through unchanged. A helper missing from this pyiqa version is skipped.
    """
    files: list[Path] = []

    def recording(fn: Callable[..., object]) -> Callable[..., object]:
        def wrapper(*args: object, **kwargs: object) -> object:
            path = fn(*args, **kwargs)
            if isinstance(path, str | Path) and Path(path) not in files:
                files.append(Path(path))
            return path

        wrapper._proof_wraps = fn  # type: ignore[attr-defined]
        return wrapper

    wrappers: dict[int, Callable[..., object]] = {}
    for module_name, attr in _PYIQA_FETCHERS:
        try:
            fn = getattr(importlib.import_module(module_name), attr, None)
        except Exception as exc:  # noqa: BLE001 - a pyiqa layout change must not stop scoring
            logger.warning("pyiqa.fetcher_missing", module=module_name, error=str(exc))
            continue
        if callable(fn):
            wrappers[id(fn)] = recording(fn)
    for module in _pyiqa_modules():
        for attr, value in list(vars(module).items()):
            if id(value) in wrappers and wrappers[id(value)]._proof_wraps is value:  # type: ignore[attr-defined]
                setattr(module, attr, wrappers[id(value)])
    try:
        yield files
    finally:
        for module in _pyiqa_modules():
            for attr, value in list(vars(module).items()):
                original = getattr(value, "_proof_wraps", None)
                if original is not None and callable(value):
                    setattr(module, attr, original)


class ImageRewardModel:
    """Human-preference proxy via ImageReward (raw ≈ ``[-2, 2]``)."""

    name = "imagereward"

    def __init__(self, model_name: str = "ImageReward-v1.0") -> None:
        self.model_name = model_name
        self._model = None

    def is_available(self) -> bool:
        return module_available("ImageReward")

    def unload(self) -> None:
        self._model = None

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


def pyiqa_scorer(lo: float | None = None, hi: float | None = None, metric_name: str = "clipiqa") -> ModelScorer:
    """Aesthetic/IQA scorer (metric ``aesthetic``). CLIP-IQA is already ~[0,1].

    Any other pyiqa metric has its own range (MUSIQ 0–100, NIMA 1–10) and may be
    lower-is-better (negated, see :class:`PyiqaModel`), so it needs an explicit
    ``lo``/``hi`` calibration rather than CLIP-IQA's ``[0, 1]``.
    """
    if (lo is None or hi is None) and metric_name != "clipiqa":
        raise ValueError(
            f"pyiqa metric {metric_name!r} needs explicit lo/hi: only clipiqa's [0, 1] range is the default"
        )
    # Fill only the bound that wasn't given: an explicit lo or hi alone is kept.
    lo = 0.0 if lo is None else lo
    hi = 1.0 if hi is None else hi
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
