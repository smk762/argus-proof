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


def test_slice_ranks_by_ci_lower_bound_not_point_estimate(tmp_path: Path) -> None:
    # lucky 3/3 (pass 1.0, wide CI) must NOT outrank well-evidenced 380/400 (0.95)
    store = CrossRunStore(tmp_path / "s.parquet")
    store.append(run_stats(manifest("lucky", checkpoint="Lucky"), report("lucky", n_passed=3, n_groups=3)))
    store.append(run_stats(manifest("solid", checkpoint="Solid"), report("solid", n_passed=380, n_groups=400)))
    assert [s.value for s in store.slice_pass_rate("base_checkpoint")] == ["Solid", "Lucky"]


def test_slice_keeps_none_distinct_from_empty(tmp_path: Path) -> None:
    store = CrossRunStore(tmp_path / "s.parquet")
    m = manifest("r1").model_copy(update={"loras": []})  # a run with no LoRA -> lora is None
    store.append(run_stats(m, report("r1", n_passed=1, n_groups=1)))
    assert store.slice_pass_rate("lora")[0].value is None  # never coerced to ""


def test_multi_lora_is_its_own_cell() -> None:
    from argus_proof.models import LoRARef

    m = manifest("r1").model_copy(
        update={"loras": [LoRARef(name="style", sha256=SHB, weight=0.8), LoRARef(name="subj", sha256=SHA, weight=0.5)]}
    )
    s = run_stats(m, report("r1", n_passed=1, n_groups=1))
    assert s.lora == "style+subj"  # full set, not just the first
    assert s.lora_weight is None  # weight only recorded for a single-LoRA run


def test_run_stats_pass_rate_matches_ci_bounds() -> None:
    s = run_stats(manifest("r1"), report("r1", n_passed=3, n_groups=4))
    assert s.pass_rate_ci_low <= s.pass_rate <= s.pass_rate_ci_high  # internally consistent row


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


# --------------------------------------------------------------------------
# experiment attribution: step_config + labels are sliceable (issue #36)
# --------------------------------------------------------------------------


def test_run_stats_records_experiment_arm() -> None:
    s = run_stats(
        manifest("r1"),
        report("r1", n_passed=1, n_groups=1),
        step_config="quality",
        labels={"caption_strategy": "florence"},
    )
    assert s.step_config == "quality"
    assert s.labels == {"caption_strategy": "florence"}


def test_slice_by_step_config(tmp_path: Path) -> None:
    store = CrossRunStore(tmp_path / "s.parquet")
    store.append(run_stats(manifest("r1"), report("r1", n_passed=9, n_groups=10), step_config="quality"))
    store.append(run_stats(manifest("r2"), report("r2", n_passed=2, n_groups=10), step_config="fast"))
    cells = {c.value: c for c in store.slice_pass_rate("step_config")}
    assert set(cells) == {"quality", "fast"}
    assert cells["quality"].pass_rate == 0.9 and cells["fast"].pass_rate == 0.2
    assert store.slice_pass_rate("step_config")[0].value == "quality"  # ranked by CI lower bound


def test_slice_by_label_pools_runs_sharing_an_upstream_factor(tmp_path: Path) -> None:
    store = CrossRunStore(tmp_path / "s.parquet")
    for i, (strategy, passed) in enumerate([("florence", 9), ("florence", 9), ("wd14", 1)]):
        store.append(
            run_stats(
                manifest(f"r{i}"),
                report(f"r{i}", n_passed=passed, n_groups=10),
                labels={"caption_strategy": strategy},
            )
        )
    cells = {c.value: c for c in store.slice_pass_rate("label:caption_strategy")}
    assert cells["florence"].n_runs == 2 and cells["florence"].n_passed == 18  # pooled across both runs
    assert cells["wd14"].n_runs == 1 and cells["wd14"].pass_rate == 0.1


def test_slice_by_label_keeps_unlabelled_runs_as_a_none_cell(tmp_path: Path) -> None:
    store = CrossRunStore(tmp_path / "s.parquet")
    store.append(run_stats(manifest("r1"), report("r1", n_passed=1, n_groups=1), labels={"caption_strategy": "x"}))
    store.append(run_stats(manifest("r2"), report("r2", n_passed=1, n_groups=1)))  # no labels
    values = {c.value for c in store.slice_pass_rate("label:caption_strategy")}
    assert values == {"x", None}  # the unlabelled run isn't silently dropped


def test_labels_are_stored_as_decodable_json(tmp_path: Path) -> None:
    import json

    store = CrossRunStore(tmp_path / "s.parquet")
    store.append(run_stats(manifest("r1"), report("r1", n_passed=1, n_groups=1), labels={"a": "1", "b": "2"}))
    # a JSON text column (arbitrary keys need no fixed schema) — decodes back to the dict
    assert json.loads(store.frame()["labels"][0]) == {"a": "1", "b": "2"}


def test_invalid_dimension_raises_even_on_an_empty_store(tmp_path: Path) -> None:
    # A typo'd dimension must fail fast, not read as "no data yet" — the name is
    # validated before the store is even read.
    store = CrossRunStore(tmp_path / "nothing-here.parquet")
    with pytest.raises(ValueError, match="cannot slice by"):
        store.slice_pass_rate("caption_strategy")  # missing the label: prefix
    with pytest.raises(ValueError, match="invalid label key"):
        store.slice_pass_rate("label:not a key$")


def test_slicing_a_store_from_an_older_build_says_so(tmp_path: Path) -> None:
    # A parquet written before step_config/labels existed lacks those columns; say
    # that plainly rather than surfacing a bare polars ColumnNotFoundError.
    import polars as pl

    path = tmp_path / "legacy.parquet"
    pl.DataFrame([{"run_id": "r1", "base_checkpoint": "ck", "n_groups": 2, "n_passed": 1}]).write_parquet(path)
    store = CrossRunStore(path)
    with pytest.raises(ValueError, match="older build"):
        store.slice_pass_rate("step_config")
    with pytest.raises(ValueError, match="older build"):
        store.slice_pass_rate("label:caption_strategy")
    assert store.slice_pass_rate("base_checkpoint")[0].value == "ck"  # existing columns still work


def test_invalid_label_key_rejected(tmp_path: Path) -> None:
    store = CrossRunStore(tmp_path / "s.parquet")
    store.append(run_stats(manifest("r1"), report("r1", n_passed=1, n_groups=1), labels={"a": "1"}))
    with pytest.raises(ValueError, match="invalid label key"):
        store.slice_pass_rate("label:not a key$")


def test_unknown_dimension_names_the_label_escape_hatch(tmp_path: Path) -> None:
    store = CrossRunStore(tmp_path / "s.parquet")
    store.append(run_stats(manifest("r1"), report("r1", n_passed=1, n_groups=1)))
    with pytest.raises(ValueError, match="label:<key>"):
        store.slice_pass_rate("caption_strategy")  # a label needs the label: prefix
