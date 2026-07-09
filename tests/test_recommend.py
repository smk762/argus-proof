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
    assert any(r.stage == "lens" and r.metric == "safety" for r in recs)


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
