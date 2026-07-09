from __future__ import annotations

import pytest

from argus_proof.experiment import (
    ExperimentError,
    ExperimentMatrix,
    StepConfig,
    expand_experiment,
    optuna_search,
)
from argus_proof.models import SamplingParams


def _sampling(steps: int = 30) -> SamplingParams:
    return SamplingParams(
        sampler="dpmpp_2m", scheduler="karras", steps=steps, cfg=7.0, clip_skip=2, width=1024, height=1024
    )


def _matrix(**overrides) -> ExperimentMatrix:
    defaults = dict(
        base_checkpoints=["sdxl_a.safetensors", "sdxl_b.safetensors"],
        step_configs=[
            StepConfig(name="fast", sampling=_sampling(20)),
            StepConfig(name="quality", sampling=_sampling(40)),
        ],
        lora_checkpoints=["e10.safetensors", "e20.safetensors"],
        lora_weights=[0.8, 1.0],
        seeds=[1, 2, 3],
    )
    defaults.update(overrides)
    return ExperimentMatrix(**defaults)


PROMPTS = ["a photo of sks person", "sks person in a park"]


def test_expands_to_cartesian_of_outer_factors() -> None:
    plan = expand_experiment(_matrix(), PROMPTS)
    # 2 base checkpoints × 2 step configs = 4 cells
    assert plan.estimate.n_cells == 4
    assert len(plan.cells) == 4


def test_cell_ids_are_unique_and_prefixed() -> None:
    plan = expand_experiment(_matrix(run_id_prefix="myexp"), PROMPTS)
    ids = [c.cell_id for c in plan.cells]
    assert len(ids) == len(set(ids))
    assert all(cid.startswith("myexp-") for cid in ids)


def test_run_ids_across_cells_do_not_collide() -> None:
    plan = expand_experiment(_matrix(), PROMPTS)
    all_run_ids = [spec.run_id for cell in plan.cells for spec in cell.plan.specs]
    assert len(all_run_ids) == len(set(all_run_ids))


def test_each_cell_uses_its_step_config_sampling() -> None:
    plan = expand_experiment(_matrix(), PROMPTS)
    by_step = {c.step_config: c for c in plan.cells if c.base_checkpoint == "sdxl_a.safetensors"}
    assert by_step["fast"].plan.specs[0].sampling.steps == 20
    assert by_step["quality"].plan.specs[0].sampling.steps == 40


def test_cost_aggregates_across_cells() -> None:
    matrix = _matrix(seconds_per_image=10.0)
    plan = expand_experiment(matrix, PROMPTS)
    per_cell_images = sum(c.plan.estimate.n_images for c in plan.cells)
    assert plan.estimate.n_images == per_cell_images
    assert plan.estimate.est_gpu_seconds == pytest.approx(per_cell_images * 10.0)
    assert plan.estimate.est_gpu_hours == pytest.approx(per_cell_images * 10.0 / 3600.0)
    assert plan.estimate.per_cell == {c.cell_id: c.plan.estimate.n_images for c in plan.cells}


def test_max_gpu_hours_guardrail_refuses_intractable_matrix() -> None:
    matrix = _matrix(seconds_per_image=100.0)
    with pytest.raises(ExperimentError, match="GPU-hours"):
        expand_experiment(matrix, PROMPTS, max_gpu_hours=0.01)


def test_generous_budget_passes() -> None:
    plan = expand_experiment(_matrix(), PROMPTS, max_gpu_hours=1000.0)
    assert plan.estimate.n_cells == 4


def test_upstream_labels_ride_on_every_cell() -> None:
    matrix = _matrix(labels={"caption_strategy": "florence"})
    plan = expand_experiment(matrix, PROMPTS)
    for cell in plan.cells:
        assert cell.labels["caption_strategy"] == "florence"
        assert cell.labels["step_config"] == cell.step_config  # own factor annotated too


def test_no_prompts_raises_experiment_error_naming_the_cell() -> None:
    with pytest.raises(ExperimentError, match="could not be built"):
        expand_experiment(_matrix(), [])


def test_duplicate_step_names_rejected() -> None:
    with pytest.raises(ValueError, match="unique names"):
        _matrix(
            step_configs=[
                StepConfig(name="fast", sampling=_sampling(20)),
                StepConfig(name="fast", sampling=_sampling(40)),
            ]
        )


def test_distinct_checkpoints_that_slug_alike_stay_distinct() -> None:
    # Same stem, different directory -> same slug, but the checkpoint index keeps
    # the cell ids (and thus run ids) unique.
    plan = expand_experiment(
        _matrix(
            base_checkpoints=["a/sdxl.safetensors", "b/sdxl.safetensors"],
            step_configs=[StepConfig(name="q", sampling=_sampling())],
        ),
        PROMPTS,
    )
    ids = [c.cell_id for c in plan.cells]
    assert len(ids) == len(set(ids)) == 2


def test_search_space_lists_categorical_levels() -> None:
    space = _matrix().search_space()
    assert space["base_checkpoint"] == ["sdxl_a.safetensors", "sdxl_b.safetensors"]
    assert space["step_config"] == ["fast", "quality"]
    assert space["lora_checkpoint"] == ["e10.safetensors", "e20.safetensors"]
    assert space["lora_weight"] == [0.8, 1.0]


def test_optuna_search_finds_the_best_arm() -> None:
    pytest.importorskip("optuna")
    matrix = _matrix()

    # A trivial objective that peaks on one specific factor combination; a guided
    # search over a small space should recover it and report a valid choice.
    def objective(choice: dict) -> float:
        score = 0.0
        if choice["lora_weight"] == 1.0:
            score += 1.0
        if choice["step_config"] == "quality":
            score += 1.0
        return score

    result = optuna_search(matrix, objective, n_trials=25, seed=7)
    assert result.n_trials == 25
    assert result.best_value == 2.0
    assert result.best_params["lora_weight"] == 1.0
    assert result.best_params["step_config"] == "quality"
    # Every suggested value comes from the declared levels.
    assert result.best_params["base_checkpoint"] in matrix.base_checkpoints


def test_optuna_search_choices_are_within_declared_levels() -> None:
    pytest.importorskip("optuna")
    matrix = _matrix()
    seen: list[dict] = []

    def objective(choice: dict) -> float:
        seen.append(choice)
        return 0.0

    optuna_search(matrix, objective, n_trials=10, seed=1)
    space = matrix.search_space()
    for choice in seen:
        for factor, levels in space.items():
            assert choice[factor] in levels
