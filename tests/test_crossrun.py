from __future__ import annotations

import math
from pathlib import Path

import pytest

from argus_proof.crossrun import CrossRunStore, krippendorff_alpha, run_stats
from argus_proof.models import (
    AggregateScores,
    EvalReport,
    ImageScores,
    LoRARef,
    MetricScores,
    ModelRef,
    RunManifest,
    SamplingParams,
    Verdict,
)

SHA = "a" * 64
SHB = "b" * 64


def manifest(run_id: str, *, checkpoint: str = "base.safetensors", lora: str = "sub.safetensors", weight: float = 1.0):
    return RunManifest(
        run_id=run_id,
        base_checkpoint=ModelRef(name=checkpoint, sha256=SHA),
        loras=[LoRARef(name=lora, sha256=SHB, weight=weight)],
        sampling=SamplingParams(sampler="euler", scheduler="normal", steps=20, cfg=6.0, width=64, height=64),
        prompt="a photo of sks",
        seeds=[1],
        engine="comfyui",
        engine_version="0.3.0",
    )


def report(run_id: str, *, n_passed: int, n_groups: int, safety: list[float] | None = None) -> EvalReport:
    images = [ImageScores(image_id=f"{run_id}-{i}", seed=i) for i in range(n_groups)]
    if safety is not None:
        images = [
            ImageScores(image_id=f"{run_id}-{i}", seed=i, metrics=MetricScores(safety=s)) for i, s in enumerate(safety)
        ]
    return EvalReport(
        run_id=run_id,
        images=images,
        aggregate=AggregateScores(
            n_images=len(images), n_groups=n_groups, n_passed=n_passed, pass_rate=n_passed / n_groups
        ),
        verdict=Verdict(passed=False),
    )


# --------------------------------------------------------------------------
# run_stats
# --------------------------------------------------------------------------


def test_run_stats_extracts_identity_and_ci() -> None:
    s = run_stats(manifest("r1", checkpoint="ckA", lora="loraA", weight=0.8), report("r1", n_passed=3, n_groups=4))
    assert s.base_checkpoint == "ckA"
    assert (s.lora, s.lora_weight) == ("loraA", 0.8)
    assert s.pass_rate == 0.75
    assert 0.0 < s.pass_rate_ci_low < 0.75 < s.pass_rate_ci_high <= 1.0  # CI brackets the estimate


def test_run_stats_ci_tightens_with_more_evidence() -> None:
    small = run_stats(manifest("r1"), report("r1", n_passed=3, n_groups=4))
    big = run_stats(manifest("r2"), report("r2", n_passed=300, n_groups=400))
    assert big.pass_rate_ci_low > small.pass_rate_ci_low  # same 0.75, more evidence -> higher floor


def test_run_stats_safety_none_when_unmeasured() -> None:
    s = run_stats(manifest("r1"), report("r1", n_passed=1, n_groups=1))
    assert s.safety_min is None and s.safety_hit_rate is None


def test_run_stats_safety_from_metric() -> None:
    s = run_stats(manifest("r1"), report("r1", n_passed=2, n_groups=2, safety=[0.9, 0.1]))
    assert s.safety_min == pytest.approx(0.1)
    assert s.safety_hit_rate == pytest.approx(0.5)  # one of two below 0.5


# --------------------------------------------------------------------------
# CrossRunStore
# --------------------------------------------------------------------------


def test_store_append_and_frame(tmp_path: Path) -> None:
    store = CrossRunStore(tmp_path / "store.parquet")
    store.append(run_stats(manifest("r1"), report("r1", n_passed=1, n_groups=1)))
    store.append(run_stats(manifest("r2"), report("r2", n_passed=0, n_groups=1)))
    df = store.frame()
    assert sorted(df["run_id"].to_list()) == ["r1", "r2"]


def test_store_reappend_replaces_run(tmp_path: Path) -> None:
    store = CrossRunStore(tmp_path / "s.parquet")
    store.append(run_stats(manifest("r1"), report("r1", n_passed=0, n_groups=4)))
    store.append(run_stats(manifest("r1"), report("r1", n_passed=4, n_groups=4)))  # same run_id, updated
    df = store.frame()
    assert df.height == 1
    assert df["pass_rate"].to_list() == [1.0]


def test_store_empty_frame(tmp_path: Path) -> None:
    assert CrossRunStore(tmp_path / "none.parquet").frame().is_empty()


def test_slice_pass_rate_pools_and_orders(tmp_path: Path) -> None:
    store = CrossRunStore(tmp_path / "s.parquet")
    # checkpoint A: 8/10 across two runs; checkpoint B: 2/10
    store.append(run_stats(manifest("a1", checkpoint="A"), report("a1", n_passed=5, n_groups=5)))
    store.append(run_stats(manifest("a2", checkpoint="A"), report("a2", n_passed=3, n_groups=5)))
    store.append(run_stats(manifest("b1", checkpoint="B"), report("b1", n_passed=2, n_groups=10)))
    slices = store.slice_pass_rate("base_checkpoint")
    assert [s.value for s in slices] == ["A", "B"]  # sorted by pass-rate desc
    a = slices[0]
    assert (a.n_runs, a.n_groups, a.n_passed) == (2, 10, 8)
    assert a.pass_rate == pytest.approx(0.8)
    assert a.ci_low < 0.8 < a.ci_high


def test_slice_invalid_dimension_raises(tmp_path: Path) -> None:
    store = CrossRunStore(tmp_path / "s.parquet")
    store.append(run_stats(manifest("r1"), report("r1", n_passed=1, n_groups=1)))
    with pytest.raises(ValueError, match="cannot slice by"):
        store.slice_pass_rate("pass_rate")


# --------------------------------------------------------------------------
# krippendorff_alpha
# --------------------------------------------------------------------------


def test_alpha_perfect_agreement() -> None:
    units = [{"r1": 5.0, "r2": 5.0}, {"r1": 2.0, "r2": 2.0}, {"r1": 4.0, "r2": 4.0}]
    assert krippendorff_alpha(units) == pytest.approx(1.0)


def test_alpha_matches_reference_implementation() -> None:
    # wrapper correctness: our per-unit dict form == the package's matrix form
    import krippendorff as kd

    units = [{"r1": 1.0, "r2": 2.0}, {"r1": 3.0, "r2": 3.0}, {"r1": 4.0, "r2": 5.0}, {"r1": 2.0, "r2": 1.0}]
    ours = krippendorff_alpha(units, level="interval")
    matrix = [[1.0, 3.0, 4.0, 2.0], [2.0, 3.0, 5.0, 1.0]]
    theirs = float(kd.alpha(reliability_data=matrix, level_of_measurement="interval"))
    assert ours == pytest.approx(theirs)


def test_alpha_undefined_returns_nan() -> None:
    assert math.isnan(krippendorff_alpha([{"r1": 5.0}]))  # only one rater -> undefined
    assert math.isnan(krippendorff_alpha([]))
