from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from test_scoring import manifest

from argus_proof.models import GeneratedImage
from argus_proof.scoring import ScoreContext, score_run
from argus_proof.scoring.scorers import (
    ModelScorer,
    clip_score_scorer,
    image_reward_scorer,
    pyiqa_scorer,
)
from argus_proof.scoring.scorers.quality import linear_normalize


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


def test_linear_normalize_nan_passes_through() -> None:
    # NaN must NOT clamp to 1.0 (the review's headline bug) — it passes through
    # so the orchestrator rejects it.
    assert math.isnan(linear_normalize(float("nan"), 20.0, 35.0))


def test_model_scorer_nan_not_scored_as_perfect() -> None:
    scorer = ModelScorer("clip_score", FakeModel({"gen": float("nan")}), lo=20.0, hi=35.0)
    assert math.isnan(scorer.score(Path("gen.png"), ctx=None))  # type: ignore[arg-type]


def test_model_scorer_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="lo < hi"):
        ModelScorer("clip_score", FakeModel({}), lo=35.0, hi=20.0)


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


def test_real_scorers_unavailable_without_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the extras to be missing (a None entry in sys.modules makes the import
    # fail), so this holds on a machine that has them installed too.
    for name in ("torch", "torchmetrics", "pyiqa", "ImageReward"):
        monkeypatch.setitem(sys.modules, name, None)
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


@pytest.mark.parametrize("raw", [float("nan"), float("inf"), float("-inf")])
def test_step_normalization_passes_a_non_finite_score_through(raw: float) -> None:
    out = linear_normalize(raw, 0.5, 0.5)  # hi <= lo: the step branch
    assert not math.isfinite(out)


def test_a_non_clipiqa_pyiqa_metric_needs_explicit_calibration() -> None:
    with pytest.raises(ValueError, match="needs explicit lo/hi"):
        pyiqa_scorer(metric_name="musiq")
    s = pyiqa_scorer(lo=20.0, hi=80.0, metric_name="musiq")
    assert (s.lo, s.hi) == (20.0, 80.0)


def test_clipiqa_keeps_a_single_explicit_bound() -> None:
    assert (pyiqa_scorer(lo=0.2).lo, pyiqa_scorer(lo=0.2).hi) == (0.2, 1.0)
    assert (pyiqa_scorer(hi=0.9).lo, pyiqa_scorer(hi=0.9).hi) == (0.0, 0.9)


def test_a_lower_is_better_pyiqa_metric_is_negated(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus_proof.scoring.scorers.quality import PyiqaModel

    class Tensor:
        def __init__(self, v: float) -> None:
            self.v = v

        def item(self) -> float:
            return self.v

    class Niqe:
        lower_better = True

        def __call__(self, path: str) -> Tensor:
            return Tensor(4.0)

    model = PyiqaModel("niqe")
    model._metric = Niqe()
    assert model.score(Path("x.png"), None) == -4.0  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# provenance: the weights each model loaded, pinned by SHA256 (#105)
# --------------------------------------------------------------------------


class PinnedModel(FakeModel):
    def weight_refs(self) -> list:
        from argus_proof.models import ModelRef

        return [ModelRef(name="a", sha256="aa" * 32), ModelRef(name="b", sha256="bb" * 32)]


def test_model_scorer_provenance_pins_the_model_file_and_the_extras() -> None:
    from argus_proof.models import ModelRef

    prov = ModelScorer("aesthetic", PinnedModel({}), 0.0, 1.0).provenance()
    assert prov.model_file == ModelRef(name="a", sha256="aa" * 32)
    assert prov.extra_files == [ModelRef(name="b", sha256="bb" * 32)]
    plain = ModelScorer("aesthetic", FakeModel({}), 0.0, 1.0).provenance()
    assert (plain.model_file, plain.extra_files) == (None, [])


def test_clip_score_pins_the_checkpoint_it_loaded(tmp_path: Path) -> None:
    from argus_proof.hashing import sha256_file
    from argus_proof.models import ModelRef
    from argus_proof.scoring.scorers.quality import ClipScoreModel

    (tmp_path / "pytorch_model.bin").write_bytes(b"bin weights")
    (tmp_path / "model.safetensors").write_bytes(b"safetensors weights")
    model = ClipScoreModel(clip_model=str(tmp_path))
    assert model.weight_refs() == []  # nothing loaded yet: nothing hashed
    model._metric = object()  # stands in for a loaded CLIPScore: no torch here
    # transformers prefers safetensors when a checkpoint carries both.
    assert model.weight_refs() == [
        ModelRef(name=f"{tmp_path}/model.safetensors", sha256=sha256_file(tmp_path / "model.safetensors"))
    ]


def test_clip_score_with_no_weights_file_pins_nothing(tmp_path: Path) -> None:
    from argus_proof.scoring.scorers.quality import ClipScoreModel

    model = ClipScoreModel(clip_model=str(tmp_path))
    model._metric = object()
    assert model.weight_refs() == []


def _fake_pyiqa(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, object]:
    """A stand-in pyiqa whose create_metric fetches two weights files, one through
    an architecture module it imports lazily (as pyiqa 0.1.16 does)."""
    from types import ModuleType

    def load_file_from_url(url: str, model_dir: str | None = None) -> str:
        path = tmp_path / url.rsplit("/", 1)[-1]
        path.write_bytes(url.encode())
        return str(path)

    def _download(url: str, root: str) -> str:
        return load_file_from_url(url)

    download_util = ModuleType("pyiqa.utils.download_util")
    download_util.load_file_from_url = load_file_from_url  # type: ignore[attr-defined]
    clip_model = ModuleType("pyiqa.archs.clip_model")
    clip_model._download = _download  # type: ignore[attr-defined]
    clip_model.load = lambda: clip_model._download("https://x/RN50.pt", "/")  # type: ignore[attr-defined]
    pyiqa = ModuleType("pyiqa")

    def create_metric(name: str) -> object:
        # The arch module binds load_file_from_url at its import, inside create_metric.
        arch = ModuleType("pyiqa.archs.clipiqa_arch")
        arch.load_file_from_url = download_util.load_file_from_url  # type: ignore[attr-defined]
        # setitem records an undo even when the key is new. A bare sys.modules assignment
        # survives the fixture: delitem(raising=False) on a missing key records nothing.
        monkeypatch.setitem(sys.modules, "pyiqa.archs.clipiqa_arch", arch)
        clip_model.load()  # type: ignore[attr-defined]
        arch.load_file_from_url("https://x/prompts.pth")  # type: ignore[attr-defined]
        return object()

    pyiqa.create_metric = create_metric  # type: ignore[attr-defined]
    for name, module in {
        "pyiqa": pyiqa,
        "pyiqa.utils": ModuleType("pyiqa.utils"),
        "pyiqa.utils.download_util": download_util,
        "pyiqa.archs": ModuleType("pyiqa.archs"),
        "pyiqa.archs.clip_model": clip_model,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return {"load_file_from_url": load_file_from_url, "_download": _download}


def test_fake_pyiqa_does_not_leave_clipiqa_arch_imported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from argus_proof.scoring.scorers.quality import PyiqaModel

    _fake_pyiqa(monkeypatch, tmp_path)
    PyiqaModel("clipiqa+")._load()
    assert "pyiqa.archs.clipiqa_arch" in sys.modules
    # undo drops a setitem, including a key that did not exist before. A direct
    # sys.modules assignment is not recorded, so the fake would still be here.
    monkeypatch.undo()
    assert "pyiqa.archs.clipiqa_arch" not in sys.modules


def test_pyiqa_pins_every_weights_file_it_fetched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from argus_proof.hashing import sha256_file
    from argus_proof.models import ModelRef
    from argus_proof.scoring.scorers.quality import PyiqaModel

    originals = _fake_pyiqa(monkeypatch, tmp_path)
    model = PyiqaModel("clipiqa+")
    assert model.weight_refs() == []  # nothing loaded yet
    model._load()
    assert model.weight_refs() == [
        ModelRef(name="pyiqa-clipiqa+/RN50.pt", sha256=sha256_file(tmp_path / "RN50.pt")),
        ModelRef(name="pyiqa-clipiqa+/prompts.pth", sha256=sha256_file(tmp_path / "prompts.pth")),
    ]
    prov = ModelScorer("aesthetic", model, 0.0, 1.0).provenance()
    assert prov.model_file is not None and prov.model_file.name == "pyiqa-clipiqa+/RN50.pt"
    assert [f.name for f in prov.extra_files] == ["pyiqa-clipiqa+/prompts.pth"]
    # Every wrapper is undone, including the one the lazily imported arch module bound.
    assert sys.modules["pyiqa.utils.download_util"].load_file_from_url is originals["load_file_from_url"]
    assert sys.modules["pyiqa.archs.clip_model"]._download is originals["_download"]
    assert sys.modules["pyiqa.archs.clipiqa_arch"].load_file_from_url is originals["load_file_from_url"]


def test_pyiqa_unload_forgets_its_pins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from argus_proof.scoring.scorers.quality import PyiqaModel

    _fake_pyiqa(monkeypatch, tmp_path)
    model = PyiqaModel()
    model._load()
    model.unload()
    assert model.weight_refs() == []
