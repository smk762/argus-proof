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


def test_max_gpu_hours_boundary_is_inclusive() -> None:
    # A matrix costing exactly the budget is accepted (strict `>`); a hair over is refused.
    exact = _matrix().model_copy(update={"seconds_per_image": 6.0})
    plan = expand_experiment(exact, PROMPTS)
    budget = plan.estimate.est_gpu_hours
    assert expand_experiment(exact, PROMPTS, max_gpu_hours=budget).estimate.n_cells == 4  # == budget: ok
    with pytest.raises(ExperimentError, match="GPU-hours"):
        expand_experiment(exact, PROMPTS, max_gpu_hours=budget * 0.999)


def test_guardrail_refuses_before_materializing_specs() -> None:
    # A matrix whose full expansion would be enormous must be refused from the
    # arithmetic estimate alone — without building (and OOMing on) the specs.
    huge = _matrix(
        base_checkpoints=[f"c{i}.safetensors" for i in range(20)],
        lora_checkpoints=[f"e{i}.safetensors" for i in range(50)],
        lora_weights=[round(0.1 * i, 2) for i in range(1, 11)],
        seeds=list(range(10)),
        token_axes={"setting": [f"s{i}" for i in range(50)], "wardrobe": [f"w{i}" for i in range(50)]},
        # max_token_combos left None: 2500 combos/prompt — millions of specs if built
    )
    with pytest.raises(ExperimentError, match="GPU-hours"):
        expand_experiment(huge, PROMPTS, max_gpu_hours=1.0)


def test_per_step_seconds_per_image_costs_each_cell_correctly() -> None:
    matrix = _matrix(
        step_configs=[
            StepConfig(name="fast", sampling=_sampling(20), seconds_per_image=4.0),
            StepConfig(name="quality", sampling=_sampling(60), seconds_per_image=12.0),
        ],
        seconds_per_image=6.0,  # matrix default — overridden per step
    )
    plan = expand_experiment(matrix, PROMPTS)
    secs = {c.step_config: c.plan.estimate.seconds_per_image for c in plan.cells}
    assert secs == {"fast": 4.0, "quality": 12.0}
    # quality cells cost 3x the fast cells at equal image counts
    n_per_cell = next(iter(plan.estimate.per_cell.values()))
    assert plan.estimate.est_gpu_seconds == pytest.approx(n_per_cell * 2 * (4.0 + 12.0))


def test_estimate_matches_built_grids() -> None:
    # The arithmetic pre-flight estimate must equal what build_grid actually produces.
    plan = expand_experiment(_matrix(token_axes={"setting": ["indoor", "outdoor"]}), PROMPTS)
    assert plan.estimate.n_runs == sum(c.plan.estimate.n_runs for c in plan.cells)
    assert plan.estimate.n_images == sum(c.plan.estimate.n_images for c in plan.cells)
    for c in plan.cells:
        assert plan.estimate.per_cell[c.cell_id] == c.plan.estimate.n_images


def test_upstream_labels_ride_on_every_cell() -> None:
    matrix = _matrix(labels={"caption_strategy": "florence"})
    plan = expand_experiment(matrix, PROMPTS)
    for cell in plan.cells:
        assert cell.labels["caption_strategy"] == "florence"
        assert cell.labels["step_config"] == cell.step_config  # own factor annotated too


def test_no_prompts_raises_experiment_error() -> None:
    with pytest.raises(ExperimentError, match="no prompts to generate"):
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
    # Same stem, different extension -> same slug, but the checkpoint index keeps
    # the cell ids (and thus run ids) unique.
    plan = expand_experiment(
        _matrix(
            base_checkpoints=["sdxl.safetensors", "sdxl.ckpt"],
            step_configs=[StepConfig(name="q", sampling=_sampling())],
        ),
        PROMPTS,
    )
    ids = [c.cell_id for c in plan.cells]
    assert len(ids) == len(set(ids)) == 2
    assert all("sdxl" in cid for cid in ids)


def test_step_name_is_slugged_into_run_ids() -> None:
    # A step name with path/OS-illegal chars must not leak into RunSpec.run_id.
    plan = expand_experiment(
        _matrix(step_configs=[StepConfig(name="hi/res fast", sampling=_sampling())]),
        PROMPTS,
    )
    for cell in plan.cells:
        assert "/" not in cell.cell_id and " " not in cell.cell_id
        for spec in cell.plan.specs:
            assert "/" not in spec.run_id and " " not in spec.run_id


def test_step_names_colliding_after_slug_rejected() -> None:
    with pytest.raises(ValueError, match="after slugification"):
        _matrix(
            step_configs=[
                StepConfig(name="hi res", sampling=_sampling()),
                StepConfig(name="hi-res", sampling=_sampling()),
            ]
        )


def test_cell_ids_unique_beyond_100_checkpoints() -> None:
    # The delimiter after the zero-padded index prevents the c{ci}{slug} boundary
    # collision (index 10 + slug '0zz' vs index 100 + slug 'zz').
    cks = [f"c{i}.safetensors" for i in range(101)]
    cks[10], cks[100] = "0zz.safetensors", "zz.safetensors"
    ids = [
        cid
        for cid, *_ in _matrix(
            base_checkpoints=cks, step_configs=[StepConfig(name="q", sampling=_sampling())]
        ).cell_configs()
    ]
    assert len(ids) == len(set(ids)) == 101


def test_slug_preserves_version_suffix() -> None:
    # A version dot is not an extension; both versions stay distinct in the id.
    ids = [
        cid
        for cid, *_ in _matrix(
            base_checkpoints=["base_v2.0", "base_v2.5"], step_configs=[StepConfig(name="q", sampling=_sampling())]
        ).cell_configs()
    ]
    assert any("v2-5" in cid for cid in ids) and any("v2-0" in cid for cid in ids)


def test_sampling_not_aliased_across_cells() -> None:
    plan = expand_experiment(_matrix(), PROMPTS)
    samplings = [c.plan.specs[0].sampling for c in plan.cells]
    # distinct instances -> mutating one cell's sampling can't bleed into another
    assert len({id(s) for s in samplings}) == len(samplings)


def test_token_axes_reach_the_grid() -> None:
    # token_axes must be forwarded into each cell's GridConfig and multiply prompts.
    plain = expand_experiment(_matrix(), PROMPTS)
    tokened = expand_experiment(_matrix(token_axes={"setting": ["indoor", "outdoor"]}), PROMPTS)
    assert tokened.estimate.n_images == plain.estimate.n_images * 2


# The GridAxes fields a cell legitimately overrides rather than inheriting verbatim.
_PER_CELL = {"run_id_prefix", "seconds_per_image"}


def test_every_shared_axis_is_forwarded_into_each_cell() -> None:
    # The guard for issue #37: cell_configs forwards every inherited axis, so a new
    # GridAxes field is accepted on the matrix AND reaches the cell's GridConfig
    # without a hand-written pass-through (which is what silently drifted before).
    #
    # It also asserts each axis is exercised with a NON-DEFAULT value: otherwise a
    # newly-added axis would sit at its default on both sides and the equality below
    # would pass vacuously even if forwarding dropped it — i.e. the guard would
    # quietly stop guarding. A new axis fails here until it's added to the fixture.
    from argus_proof.models import GridAxes, GridConfig

    matrix = _matrix(
        token_axes={"setting": ["indoor"]},
        combo_seed=7,
        max_base_prompts=3,
        max_token_combos=2,
        flexibility_prompts=["a novel scene"],
        negative_prompt="blurry",
        source_manifest="export/manifest.jsonl",
        source_manifest_version="1.0",
        training_run_id="train-9",
        lora_weights=[0.6, 0.9],
    )
    _cell_id, _ckpt, _step, config = matrix.cell_configs()[0]
    for field in set(GridAxes.model_fields) - _PER_CELL:
        info = GridConfig.model_fields[field]
        if not info.is_required():
            default = info.get_default(call_default_factory=True)
            assert getattr(matrix, field) != default, (
                f"{field} is left at its default here, so this test can't prove it's forwarded — "
                f"set a non-default {field} on the fixture above"
            )
        assert getattr(config, field) == getattr(matrix, field), f"{field} not forwarded to the cell"


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


def test_optuna_search_honours_direction() -> None:
    pytest.importorskip("optuna")
    matrix = _matrix()

    # minimize -> the LOW weight should win (guards against a hardcoded 'maximize')
    def objective(choice: dict) -> float:
        return float(choice["lora_weight"])

    result = optuna_search(matrix, objective, n_trials=25, direction="minimize", seed=3)
    assert result.best_params["lora_weight"] == 0.8  # the smallest declared weight


def test_count_prompt_items_matches_actual_expansion() -> None:
    # The arithmetic pre-flight count must equal what build_grid materializes.
    from argus_proof.grid import build_grid, count_prompt_items

    for cid, _ckpt, _step, config in _matrix(
        token_axes={"setting": ["indoor", "outdoor"], "wardrobe": ["a", "b", "c"]},
        max_token_combos=4,
        flexibility_prompts=["a novel scene"],
        max_base_prompts=1,
    ).cell_configs():
        plan = build_grid(config, PROMPTS)
        n_items = plan.estimate.n_runs // (len(config.lora_checkpoints) * len(config.lora_weights))
        assert count_prompt_items(config, PROMPTS) == n_items, cid
