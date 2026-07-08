from __future__ import annotations

from pathlib import Path

import pytest

from argus_proof.models import (
    AggregateScores,
    EvalReport,
    ImageScores,
    MetricScores,
    Verdict,
)
from argus_proof.scoring import score_run
from argus_proof.scoring.scorers import SafetyScorer, safety_tail_aggregate


class FakeDetector:
    """Returns a fixed unsafe probability per image stem (None = couldn't score)."""

    def __init__(self, name: str, unsafe: dict[str, float | None], available: bool = True) -> None:
        self.name = name
        self.unsafe = unsafe
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def unsafe_probability(self, image_path: Path) -> float | None:
        return self.unsafe.get(Path(image_path).stem)


def test_safe_image_scores_high() -> None:
    s = SafetyScorer([FakeDetector("d", {"gen": 0.1})])
    assert s.score(Path("gen.png"), ctx=None) == pytest.approx(0.9)  # type: ignore[arg-type]


def test_unsafe_image_scores_low() -> None:
    s = SafetyScorer([FakeDetector("d", {"gen": 0.95})])
    assert s.score(Path("gen.png"), ctx=None) == pytest.approx(0.05)  # type: ignore[arg-type]


def test_ensemble_takes_most_unsafe() -> None:
    # two detectors disagree; the more-unsafe one wins (conservative)
    s = SafetyScorer([FakeDetector("a", {"gen": 0.1}), FakeDetector("b", {"gen": 0.8})])
    assert s.score(Path("gen.png"), ctx=None) == pytest.approx(0.2)  # 1 - max(0.1, 0.8)  # type: ignore[arg-type]


def test_unavailable_detectors_skipped() -> None:
    s = SafetyScorer([FakeDetector("a", {"gen": 0.9}, available=False), FakeDetector("b", {"gen": 0.1})])
    assert s.score(Path("gen.png"), ctx=None) == pytest.approx(0.9)  # only b counts  # type: ignore[arg-type]


def test_no_detector_can_score_returns_none() -> None:
    s = SafetyScorer([FakeDetector("d", {"gen": None})])
    assert s.score(Path("gen.png"), ctx=None) is None  # type: ignore[arg-type]


def test_availability_and_provenance() -> None:
    s = SafetyScorer([FakeDetector("nudenet", {})])
    assert s.metric == "safety"
    assert s.is_available() is True
    assert s.provenance().metric == "safety"
    assert "nudenet" in (s.provenance().model or "")


def test_score_run_fills_safety_and_hard_gate_flags_unsafe() -> None:
    from argus_proof.models import GateConfig, GeneratedImage

    images = [
        GeneratedImage(image_id="run-1-1", run_id="run-1", seed=1, path="run-1-1.png", width=64, height=64),
        GeneratedImage(image_id="run-1-2", run_id="run-1", seed=2, path="run-1-2.png", width=64, height=64),
    ]
    from test_scoring import manifest

    detector = FakeDetector("d", {"run-1-1": 0.1, "run-1-2": 0.9})  # img2 unsafe
    report = score_run(
        manifest(),
        images,
        scorers=[SafetyScorer([detector])],
        gate=GateConfig(hard_gates={"safety": 0.5}),  # safety below 0.5 -> auto-fail + "unsafe"
    )
    assert report.images[0].metrics.safety == pytest.approx(0.9)
    assert report.images[1].metrics.safety == pytest.approx(0.1)
    assert report.images[1].passed is False
    assert any(r.code == "unsafe" for r in report.images[1].reject_reasons)


# --------------------------------------------------------------------------
# tail aggregate
# --------------------------------------------------------------------------


def _report(safeties: list[float | None]) -> EvalReport:
    images = [ImageScores(image_id=f"i{i}", seed=i, metrics=MetricScores(safety=s)) for i, s in enumerate(safeties)]
    return EvalReport(
        run_id="run-1",
        images=images,
        aggregate=AggregateScores(n_images=len(images), n_passed=0, pass_rate=0.0),
        verdict=Verdict(passed=False),
    )


def test_tail_aggregate_flags_single_bad_frame() -> None:
    # mean is high, but one 0.1 frame must show up in the tail
    agg = safety_tail_aggregate(_report([0.95, 0.98, 0.99, 0.1]), unsafe_below=0.5)
    assert agg["any_hit"] == 1.0
    assert agg["hit_rate"] == pytest.approx(0.25)
    assert agg["min_safety"] == pytest.approx(0.1)


def test_tail_aggregate_all_safe() -> None:
    agg = safety_tail_aggregate(_report([0.9, 0.95, 0.99]), unsafe_below=0.5)
    assert agg["any_hit"] == 0.0
    assert agg["hit_rate"] == 0.0


def test_tail_aggregate_no_safety_scores() -> None:
    agg = safety_tail_aggregate(_report([None, None]))
    assert agg == {"any_hit": 0.0, "hit_rate": 0.0, "min_safety": 0.0, "p05_safety": 0.0}
