from __future__ import annotations

from pathlib import Path

from argus_proof.models import (
    GateConfig,
    GeneratedImage,
    MetricScores,
    ModelRef,
    RunManifest,
    SamplingParams,
    ScorerProvenance,
)
from argus_proof.scoring import ScoreContext, composite_score, gate_image, score_run

SHA = "a" * 64


def manifest() -> RunManifest:
    return RunManifest(
        run_id="run-1",
        base_checkpoint=ModelRef(name="base.safetensors", sha256=SHA),
        sampling=SamplingParams(sampler="euler", scheduler="normal", steps=20, cfg=6.0, width=64, height=64),
        prompt="a photo of sks",
        seeds=[1],
        engine="comfyui",
        engine_version="0.3.0",
    )


def img(image_id: str, seed: int) -> GeneratedImage:
    return GeneratedImage(image_id=image_id, run_id="run-1", seed=seed, path=f"{image_id}.png", width=64, height=64)


class FakeScorer:
    """Scores by image stem (== image_id here); None means "no score for it"."""

    def __init__(self, metric: str, values: dict[str, float], *, available: bool = True) -> None:
        self.metric = metric
        self.values = values
        self._available = available

    def provenance(self) -> ScorerProvenance:
        return ScorerProvenance(name=f"fake-{self.metric}", metric=self.metric, version="0.1")

    def is_available(self) -> bool:
        return self._available

    def score(self, image_path: Path, ctx: ScoreContext) -> float | None:
        return self.values.get(Path(image_path).stem)


class FakeDeduper:
    def __init__(self, labels: list[int]) -> None:
        self.labels = labels

    def provenance(self) -> ScorerProvenance:
        return ScorerProvenance(name="fake-dedup", metric="duplicate")

    def is_available(self) -> bool:
        return True

    def group(self, images: list[GeneratedImage]) -> list[int]:
        return self.labels


class FakeDiversity:
    def __init__(self, value: float) -> None:
        self.value = value

    def provenance(self) -> ScorerProvenance:
        return ScorerProvenance(name="fake-diversity", metric="diversity")

    def is_available(self) -> bool:
        return True

    def score(self, images: list[GeneratedImage], ctx: ScoreContext) -> float:
        return self.value


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------


def test_composite_renormalizes_over_present_metrics() -> None:
    cfg = GateConfig()  # equal weights on identity/clip_score/aesthetic/preference
    # only two of four present -> mean of those two, not diluted by absent ones
    m = MetricScores(identity=0.8, clip_score=0.6)
    assert composite_score(m, cfg) == 0.7


def test_composite_none_when_nothing_scored() -> None:
    assert composite_score(MetricScores(), GateConfig()) is None


def test_gate_auto_pass_fail_and_hitl_band() -> None:
    cfg = GateConfig(auto_pass=0.7, auto_fail=0.4)
    assert gate_image(MetricScores(aesthetic=0.9), cfg)[0] is True
    assert gate_image(MetricScores(aesthetic=0.2), cfg)[0] is False
    assert gate_image(MetricScores(aesthetic=0.55), cfg)[0] is None  # borderline -> HITL


def test_gate_auto_fail_carries_reason() -> None:
    passed, reasons = gate_image(MetricScores(aesthetic=0.1), GateConfig())
    assert passed is False
    assert reasons and reasons[0].code == "low_quality"


def test_hard_gate_fails_regardless_of_composite() -> None:
    cfg = GateConfig(hard_gates={"identity": 0.5})
    # composite would pass, but identity is below the hard floor
    passed, reasons = gate_image(MetricScores(identity=0.3, aesthetic=0.95), cfg)
    assert passed is False
    assert reasons[0].code == "identity_mismatch"
    assert "hard gate" in (reasons[0].note or "")


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------


def test_score_run_fills_metrics_and_provenance() -> None:
    images = [img("run-1-1", 1), img("run-1-2", 2)]
    scorer = FakeScorer("aesthetic", {"run-1-1": 0.9, "run-1-2": 0.2})
    report = score_run(manifest(), images, scorers=[scorer])

    assert report.images[0].metrics.aesthetic == 0.9
    assert report.images[0].passed is True
    assert report.images[1].passed is False
    assert report.images[1].reject_reasons[0].code == "low_quality"
    assert [p.name for p in report.scorers] == ["fake-aesthetic"]


def test_unavailable_scorer_is_skipped() -> None:
    images = [img("run-1-1", 1)]
    scorer = FakeScorer("aesthetic", {"run-1-1": 0.9}, available=False)
    report = score_run(manifest(), images, scorers=[scorer])
    assert report.images[0].metrics.aesthetic is None
    assert report.scorers == []
    assert report.images[0].passed is None  # nothing scored -> HITL


def test_pass_rate_counts_groups_not_frames() -> None:
    # 4 near-dup passing frames (one group) + 1 distinct failing frame (another).
    images = [img(f"run-1-{i}", i) for i in range(1, 6)]
    scorer = FakeScorer(
        "aesthetic",
        {"run-1-1": 0.9, "run-1-2": 0.9, "run-1-3": 0.9, "run-1-4": 0.9, "run-1-5": 0.1},
    )
    deduper = FakeDeduper([0, 0, 0, 0, 1])  # first four collapse
    report = score_run(manifest(), images, scorers=[scorer], deduper=deduper)

    agg = report.aggregate
    assert agg.n_images == 5
    assert agg.n_groups == 2
    assert agg.n_passed == 1
    assert agg.pass_rate == 0.5  # 1/2 groups, NOT 4/5 frames
    assert [r.duplicate_group for r in report.images] == [0, 0, 0, 0, 1]


def test_group_passes_if_any_member_passes() -> None:
    images = [img("run-1-1", 1), img("run-1-2", 2)]
    scorer = FakeScorer("aesthetic", {"run-1-1": 0.9, "run-1-2": 0.1})  # one pass, one fail
    deduper = FakeDeduper([0, 0])  # same group
    report = score_run(manifest(), images, scorers=[scorer], deduper=deduper)
    assert report.aggregate.n_groups == 1
    assert report.aggregate.n_passed == 1
    assert report.aggregate.pass_rate == 1.0


def test_needs_hitl_counted_at_group_level() -> None:
    images = [img("run-1-1", 1)]
    scorer = FakeScorer("aesthetic", {"run-1-1": 0.55})  # borderline
    report = score_run(manifest(), images, scorers=[scorer])
    assert report.aggregate.n_needs_hitl == 1
    assert report.aggregate.n_passed == 0
    assert report.images[0].passed is None


def test_diversity_recorded_in_aggregate_and_verdict() -> None:
    images = [img("run-1-1", 1)]
    report = score_run(
        manifest(), images, scorers=[FakeScorer("aesthetic", {"run-1-1": 0.9})], diversity=FakeDiversity(0.42)
    )
    assert report.aggregate.diversity == 0.42
    assert any("diversity 0.42" in r for r in report.verdict.reasons)
    assert "fake-diversity" in [p.name for p in report.scorers]


def test_metric_means_ignore_none() -> None:
    images = [img("run-1-1", 1), img("run-1-2", 2)]
    scorer = FakeScorer("identity", {"run-1-1": 0.8})  # only first image scored
    report = score_run(manifest(), images, scorers=[scorer])
    assert report.aggregate.means.identity == 0.8  # mean of the one present value


def test_run_verdict_uses_pass_rate_threshold() -> None:
    images = [img(f"run-1-{i}", i) for i in range(1, 5)]
    scorer = FakeScorer("aesthetic", {"run-1-1": 0.9, "run-1-2": 0.9, "run-1-3": 0.9, "run-1-4": 0.1})
    # 3/4 groups pass = 0.75
    assert score_run(manifest(), images, scorers=[scorer], gate=GateConfig(run_pass_rate=0.75)).verdict.passed is True
    assert score_run(manifest(), images, scorers=[scorer], gate=GateConfig(run_pass_rate=0.8)).verdict.passed is False


def test_report_round_trips_with_new_fields() -> None:
    from argus_proof.models import EvalReport

    images = [img("run-1-1", 1)]
    report = score_run(
        manifest(), images, scorers=[FakeScorer("aesthetic", {"run-1-1": 0.9})], diversity=FakeDiversity(0.5)
    )
    assert EvalReport.model_validate_json(report.model_dump_json()) == report


def test_empty_images_do_not_crash() -> None:
    report = score_run(manifest(), [])
    assert report.aggregate.n_images == 0
    assert report.aggregate.n_groups == 0
    assert report.aggregate.pass_rate == 0.0
    assert report.verdict.passed is False


def test_score_context_defaults_to_manifest_prompt() -> None:
    captured: list[str] = []

    class PromptCapture(FakeScorer):
        def score(self, image_path: Path, ctx: ScoreContext) -> float | None:
            captured.append(ctx.prompt)
            return 0.9

    score_run(manifest(), [img("run-1-1", 1)], scorers=[PromptCapture("aesthetic", {})])
    assert captured == ["a photo of sks"]
