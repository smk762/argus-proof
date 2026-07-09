from __future__ import annotations

from argus_proof.models import (
    AggregateScores,
    EvalReport,
    ImageScores,
    MetricScores,
    Verdict,
)
from argus_proof.recommend import RecommendConfig, recommend


def _report(
    *,
    means: MetricScores | None = None,
    diversity: float | None = None,
    pending: bool = False,
    images: list[ImageScores] | None = None,
    pass_rate: float = 1.0,
) -> EvalReport:
    return EvalReport(
        run_id="run-1",
        images=images or [],
        aggregate=AggregateScores(
            n_images=1, n_groups=1, n_passed=1, pass_rate=pass_rate, means=means or MetricScores(), diversity=diversity
        ),
        verdict=Verdict(passed=not pending, pending=pending),
    )


def stages(recs) -> set[str]:  # noqa: ANN001
    return {r.stage for r in recs}


def test_no_recommendations_when_healthy() -> None:
    healthy = _report(means=MetricScores(identity=0.9, clip_score=0.8, aesthetic=0.8), diversity=0.7)
    assert recommend(healthy) == []


def test_low_identity_routes_to_forge() -> None:
    recs = recommend(_report(means=MetricScores(identity=0.3)))
    assert any(r.stage == "forge" and r.metric == "identity" for r in recs)


def test_low_clip_score_routes_to_lens_and_grid() -> None:
    recs = recommend(_report(means=MetricScores(clip_score=0.2)))
    assert {"lens", "grid"} <= stages(recs)
    assert all(r.metric == "clip_score" for r in recs)


def test_low_aesthetic_routes_to_forge() -> None:
    recs = recommend(_report(means=MetricScores(aesthetic=0.2)))
    assert any(r.stage == "forge" and r.metric == "aesthetic" for r in recs)


def test_low_diversity_routes_to_grid() -> None:
    recs = recommend(_report(diversity=0.1))
    assert any(r.stage == "grid" and r.metric == "diversity" for r in recs)


def test_unsafe_outputs_route_to_lens() -> None:
    images = [
        ImageScores(image_id="a", seed=1, metrics=MetricScores(safety=0.9)),
        ImageScores(image_id="b", seed=2, metrics=MetricScores(safety=0.1)),  # unsafe
    ]
    recs = recommend(_report(images=images))
    unsafe = next(r for r in recs if r.stage == "lens" and r.issue == "unsafe outputs")
    assert unsafe.metric == "unsafe_rate"  # per-image path reports the rate, not a safety score
    assert unsafe.value == 0.5


def test_unsafe_via_reject_reason_only() -> None:
    # safety metric fine (or absent) but an explicit unsafe reject reason still counts
    from argus_proof.models import RejectReason

    images = [
        ImageScores(image_id="a", seed=1, metrics=MetricScores(safety=0.9)),
        ImageScores(image_id="b", seed=2, reject_reasons=[RejectReason(code="unsafe")]),
    ]
    recs = recommend(_report(images=images))
    assert any(r.stage == "lens" and r.issue == "unsafe outputs" for r in recs)


def test_unsafe_from_aggregate_mean_when_no_images() -> None:
    # An aggregate-only report (images=[]) must still be safety-checked off the mean.
    recs = recommend(_report(means=MetricScores(identity=0.9, safety=0.05), images=[]))
    unsafe = next(r for r in recs if r.issue == "unsafe outputs")
    assert unsafe.stage == "lens" and unsafe.metric == "safety" and unsafe.value == 0.05


def test_nan_metric_is_flagged_not_silently_passed() -> None:
    recs = recommend(_report(means=MetricScores(identity=float("nan"))))
    assert any(r.metric == "identity" for r in recs)  # NaN is a miss, not a pass


def test_metric_exactly_at_floor_not_flagged() -> None:
    # value == floor is acceptable (strict `<`); a passing run gets no advice
    recs = recommend(_report(means=MetricScores(identity=0.6, clip_score=0.9, aesthetic=0.9), diversity=0.9))
    assert recs == []


def test_out_of_range_floor_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="less than or equal to 1"):
        RecommendConfig(min_identity=1.5)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        RecommendConfig(max_unsafe_rate=-0.1)


def test_from_acceptance_tracks_gate_floors() -> None:
    from argus_proof.models import AcceptanceThresholds

    cfg = RecommendConfig.from_acceptance(AcceptanceThresholds(min_identity_mean=0.7, max_unsafe_rate=0.02))
    assert cfg.min_identity == 0.7 and cfg.max_unsafe_rate == 0.02


def test_pending_routes_to_refine() -> None:
    recs = recommend(_report(means=MetricScores(identity=0.9), pending=True))
    assert any(r.stage == "refine" for r in recs)


def test_config_none_floor_skips_metric() -> None:
    # min_identity None -> a low identity produces no identity recommendation
    recs = recommend(_report(means=MetricScores(identity=0.1)), config=RecommendConfig(min_identity=None))
    assert not any(r.metric == "identity" for r in recs)


def test_missing_metric_does_not_trigger() -> None:
    # identity not measured (None) -> no identity rec, even with a floor set
    recs = recommend(_report(means=MetricScores(identity=None, clip_score=0.9, aesthetic=0.9)))
    assert not any(r.metric == "identity" for r in recs)


class _FakeStore:
    def __init__(self, slices: dict[str, list]) -> None:
        self._slices = slices

    def slice_pass_rate(self, dimension: str):  # noqa: ANN201
        return self._slices.get(dimension, [])


def test_store_surfaces_best_checkpoint() -> None:
    from argus_proof.crossrun import SliceStats

    store = _FakeStore(
        {
            "base_checkpoint": [
                SliceStats(
                    dimension="base_checkpoint",
                    value="A",
                    n_runs=2,
                    n_groups=100,
                    n_passed=95,
                    pass_rate=0.95,
                    ci_low=0.9,
                    ci_high=0.98,
                ),
                SliceStats(
                    dimension="base_checkpoint",
                    value="B",
                    n_runs=1,
                    n_groups=10,
                    n_passed=3,
                    pass_rate=0.3,
                    ci_low=0.1,
                    ci_high=0.6,
                ),
            ]
        }
    )
    recs = recommend(_report(means=MetricScores(identity=0.9)), store=store)
    ckpt = next(r for r in recs if r.stage == "checkpoint")
    assert "A" in ckpt.action  # best cell surfaced


def test_store_single_cell_no_recommendation() -> None:
    from argus_proof.crossrun import SliceStats

    store = _FakeStore(
        {
            "base_checkpoint": [
                SliceStats(
                    dimension="base_checkpoint",
                    value="A",
                    n_runs=1,
                    n_groups=1,
                    n_passed=1,
                    pass_rate=1.0,
                    ci_low=0.2,
                    ci_high=1.0,
                )
            ]
        }
    )
    assert not any(r.stage == "checkpoint" for r in recommend(_report(means=MetricScores(identity=0.9)), store=store))


def _slice(value, ci_low, ci_high, *, dimension="base_checkpoint", pass_rate=0.9):  # noqa: ANN001, ANN202
    from argus_proof.crossrun import SliceStats

    return SliceStats(
        dimension=dimension,
        value=value,
        n_runs=1,
        n_groups=100,
        n_passed=90,
        pass_rate=pass_rate,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def test_store_tied_cells_yield_no_pick() -> None:
    # Overlapping CIs -> the data can't separate a winner -> no recommendation.
    store = _FakeStore({"base_checkpoint": [_slice("A", 0.49, 0.94), _slice("B", 0.40, 0.89)]})
    assert not any(r.stage == "checkpoint" for r in recommend(_report(means=MetricScores(identity=0.9)), store=store))


def test_store_skips_none_value_cell() -> None:
    # A value=None cohort (e.g. no-/multi-LoRA rows) must never become "prefer X=None".
    store = _FakeStore(
        {
            "lora_weight": [
                _slice(None, 0.95, 1.0, dimension="lora_weight"),
                _slice("0.8", 0.1, 0.4, dimension="lora_weight"),
            ]
        }
    )
    recs = [r for r in recommend(_report(means=MetricScores(identity=0.9)), store=store) if r.stage == "weight"]
    assert recs == []  # only one non-None cell remains -> nothing to compare


def test_store_weight_branch_against_real_store(tmp_path) -> None:  # noqa: ANN001
    # Exercise the lora_weight -> "weight" branch through the real CrossRunStore
    # (ranking + None handling), not a hand-ordered fake.
    from argus_proof.crossrun import CrossRunStore, RunStats

    store = CrossRunStore(tmp_path / "s.parquet")
    store.append(
        [
            RunStats(
                run_id="r1",
                base_checkpoint="sdxl",
                base_checkpoint_sha="x",
                lora="e.safetensors",
                lora_weight=1.0,
                n_groups=100,
                n_passed=95,
                pass_rate=0.95,
            ),
            RunStats(
                run_id="r2",
                base_checkpoint="sdxl",
                base_checkpoint_sha="x",
                lora="e.safetensors",
                lora_weight=0.6,
                n_groups=100,
                n_passed=40,
                pass_rate=0.40,
            ),
        ]
    )
    recs = [r for r in recommend(_report(means=MetricScores(identity=0.9)), store=store) if r.stage == "weight"]
    assert len(recs) == 1
    assert "1.0" in recs[0].action  # the clearly-better weight is surfaced


def test_healthy_run_with_indistinct_store_is_empty() -> None:
    # A passing run + a store with no separable winner -> [] (nothing actionable).
    store = _FakeStore({"base_checkpoint": [_slice("A", 0.5, 0.95), _slice("B", 0.45, 0.9)]})
    healthy = _report(means=MetricScores(identity=0.9, clip_score=0.9, aesthetic=0.9), diversity=0.9)
    assert recommend(healthy, store=store) == []
