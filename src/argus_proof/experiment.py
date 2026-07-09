"""A/B experiment matrix — expand factors × levels into a multi-cell grid.

The prompt-grid builder (:mod:`argus_proof.grid`) sweeps one *cell*: the LoRA
checkpoint × weight × prompt × seed axes under a **fixed** base checkpoint and
sampler. An experiment adds the outer factors that vary a whole grid — the base
checkpoint and named **step configs** (e.g. a fast vs. a quality sampler) — so
one declarative config expands to the full cartesian of grids, each cell a
self-contained :class:`~argus_proof.models.GridPlan`.

The expansion is pure and deterministic, reuses the grid builder's per-cell
:class:`~argus_proof.models.GridEstimate`, and aggregates cost across cells so
the total GPU-hour bill is known **before launch** — with an optional
``max_gpu_hours`` guardrail that refuses an intractable matrix rather than
silently queueing thousands of images.

**Upstream factors** (caption strategy, source-image variation) are *not*
proof's to vary — a LoRA is already trained under one caption strategy, so proof
can only compare LoRAs trained under different ones. Those live in
:attr:`ExperimentMatrix.labels` and ride along on each cell. Note: the cross-run
store's sliceable columns don't yet include ``labels``/``step_config``, so
comparing arms by those is a follow-up (wire them into ``RunStats`` +
``CrossRunStore.SLICEABLE``); today the labels are carried metadata for that.

For a matrix too large to brute-force, :func:`optuna_search` (optional ``[opt]``
extra) does sample-efficient search over the same factor levels.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator

from argus_proof.grid import build_grid, count_prompt_items
from argus_proof.models import GridConfig, GridPlan, ProofError, SamplingParams

if TYPE_CHECKING:
    from typing import Self


class ExperimentError(ProofError):
    """An experiment matrix could not be expanded or exceeds its cost budget."""


_SLUG_RE = re.compile(r"[^a-z0-9]+")
# Only these are stripped as extensions; a bare version dot ("model_v2.5") is
# NOT an extension and must survive into the slug (as "-") rather than be lost.
_MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf")


def _slug(text: str) -> str:
    """A filesystem/run-id-safe slug from an arbitrary checkpoint or step name.

    Strips a known model extension (``.safetensors`` …) but keeps version dots;
    every other non-alphanumeric run — path separators, spaces, colons — collapses
    to ``-`` (rather than being dropped), so no directory or unsafe char leaks into
    a run_id and distinct inputs stay distinct (``a/sdxl`` vs ``b/sdxl``).
    """
    low = text.lower()
    for ext in _MODEL_EXTS:  # strip a known model extension, but keep version dots
        if low.endswith(ext):
            text = text[: -len(ext)]
            break
    return _SLUG_RE.sub("-", text.lower()).strip("-") or "x"


class StepConfig(BaseModel):
    """A named sampler variant — one outer factor level (e.g. ``fast``/``quality``).

    Naming the sampler lets a matrix compare "same LoRA, cheap 20-step preview vs.
    full 40-step render" as an explicit factor, with the ``name`` flowing into
    each cell's run-id prefix and cross-run labels so the two are distinguishable.
    """

    name: str = Field(min_length=1)
    sampling: SamplingParams
    # Per-step cost override: a 60-step "quality" pass renders slower than a
    # 20-step "fast" one, so a single matrix-wide rate mis-costs the very axis
    # this varies. None -> fall back to the matrix's seconds_per_image.
    seconds_per_image: float | None = Field(default=None, gt=0)


class ExperimentMatrix(BaseModel):
    """Factors × levels declaring a reproducible multi-cell experiment.

    **Outer factors** — each combination becomes one grid cell:

    * ``base_checkpoints`` — the base model(s) the LoRA rides on.
    * ``step_configs`` — named sampler variants (``fast``/``quality`` …).

    **Inner axes** — shared by every cell, expanded within it by the grid builder
    (``lora_checkpoints`` × ``lora_weights`` × prompts × ``seeds``); these mirror
    :class:`~argus_proof.models.GridConfig`.

    **Labels** — upstream factors proof can only *observe*, not vary (caption
    strategy, source set). They annotate every cell so the cross-run store can
    slice results by them once scored.
    """

    # Outer factors — the cartesian of these enumerates the cells.
    base_checkpoints: list[str] = Field(min_length=1)
    step_configs: list[StepConfig] = Field(min_length=1)

    # Inner axes — passed through to each cell's GridConfig.
    lora_checkpoints: list[str] = Field(min_length=1)
    lora_weights: list[float] = Field(default_factory=lambda: [1.0], min_length=1)
    seeds: list[int] = Field(min_length=1)
    token_axes: dict[str, list[str]] = Field(default_factory=dict)
    max_token_combos: int | None = Field(default=None, ge=1)
    max_base_prompts: int | None = Field(default=None, ge=1)
    flexibility_prompts: list[str] = Field(default_factory=list)
    negative_prompt: str = ""
    combo_seed: int = 0
    seconds_per_image: float = 6.0

    run_id_prefix: str = "exp"
    source_manifest: str | None = None
    source_manifest_version: str | None = None
    training_run_id: str | None = None

    # Upstream factors proof observes but does not drive (caption strategy, …).
    labels: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_step_names(self) -> Self:
        names = [s.name for s in self.step_configs]
        if len(names) != len(set(names)):
            raise ValueError("step_configs must have unique names")
        slugs = [_slug(s.name) for s in self.step_configs]
        if len(slugs) != len(set(slugs)):
            raise ValueError("step_configs names must stay distinct after slugification (e.g. 'hi res' vs 'hi-res')")
        return self

    def cell_configs(self) -> list[tuple[str, str, str, GridConfig]]:
        """Enumerate ``(cell_id, base_checkpoint, step_name, GridConfig)`` per cell.

        One entry per ``base_checkpoint × step_config``. The ``cell_id`` is
        ``{prefix}-c{index}-{checkpoint-slug}-{step-slug}`` — delimited (so the
        index/slug boundary can't collide once the index needs 3 digits) and
        fully slugged (so a step name with a slash/space can't produce an unsafe
        ``RunSpec`` run_id) — and becomes the cell's ``run_id_prefix`` so no two
        cells' :class:`RunSpec` ids collide.
        """
        cells: list[tuple[str, str, str, GridConfig]] = []
        for ci, checkpoint in enumerate(self.base_checkpoints):
            for step in self.step_configs:
                cell_id = f"{self.run_id_prefix}-c{ci:02d}-{_slug(checkpoint)}-{_slug(step.name)}"
                config = GridConfig(
                    base_checkpoint=checkpoint,
                    lora_checkpoints=self.lora_checkpoints,
                    lora_weights=self.lora_weights,
                    # copy so cells (and the specs they expand into) don't alias one
                    # mutable SamplingParams — pydantic v2 doesn't copy nested models.
                    sampling=step.sampling.model_copy(),
                    negative_prompt=self.negative_prompt,
                    seeds=self.seeds,
                    token_axes=self.token_axes,
                    max_token_combos=self.max_token_combos,
                    max_base_prompts=self.max_base_prompts,
                    flexibility_prompts=self.flexibility_prompts,
                    combo_seed=self.combo_seed,
                    seconds_per_image=step.seconds_per_image
                    if step.seconds_per_image is not None
                    else self.seconds_per_image,
                    run_id_prefix=cell_id,
                    source_manifest=self.source_manifest,
                    source_manifest_version=self.source_manifest_version,
                    training_run_id=self.training_run_id,
                )
                cells.append((cell_id, checkpoint, step.name, config))
        return cells

    def search_space(self) -> dict[str, list[Any]]:
        """The categorical factor levels an optimiser samples (see :func:`optuna_search`)."""
        return {
            "base_checkpoint": list(self.base_checkpoints),
            "step_config": [s.name for s in self.step_configs],
            "lora_checkpoint": list(self.lora_checkpoints),
            "lora_weight": list(self.lora_weights),
        }


class ExperimentEstimate(BaseModel):
    """Aggregate cost across every cell, reported before any generation.

    ``per_cell`` maps each ``cell_id`` to its image count so an outsized cell is
    visible before it is queued.
    """

    n_cells: int
    n_runs: int
    n_images: int
    seconds_per_image: float
    est_gpu_seconds: float
    est_gpu_hours: float
    per_cell: dict[str, int] = Field(default_factory=dict)


class ExperimentCell(BaseModel):
    """One grid cell: its fixed outer factors, comparison labels, and full plan."""

    cell_id: str
    base_checkpoint: str
    step_config: str
    labels: dict[str, str] = Field(default_factory=dict)
    plan: GridPlan


class ExperimentPlan(BaseModel):
    """A fully enumerated experiment: every cell's grid plus the aggregate cost."""

    run_id_prefix: str
    estimate: ExperimentEstimate
    cells: list[ExperimentCell] = Field(default_factory=list)


def expand_experiment(
    matrix: ExperimentMatrix,
    base_prompts: list[str],
    *,
    max_gpu_hours: float | None = None,
) -> ExperimentPlan:
    """Expand *matrix* over *base_prompts* into a deterministic :class:`ExperimentPlan`.

    The cost is computed **before any RunSpec is built** — from the axis
    cardinalities, using each cell's per-step ``seconds_per_image`` — so an
    intractable matrix is refused (``max_gpu_hours``) without first materializing
    the millions of specs it would expand to. Only a within-budget matrix is then
    built into per-cell :class:`GridPlan`\\ s (reusing the grid builder). Each
    cell carries ``labels`` (the matrix's upstream factors plus its own
    ``step_config``) as metadata.

    Raises :class:`ExperimentError` if the aggregate exceeds ``max_gpu_hours``,
    or if the expansion yields no prompts.
    """
    configs = matrix.cell_configs()

    # Pre-flight cost from cardinalities — no RunSpec is built yet. Prompt fields
    # are matrix-level, so the prompt count is identical for every cell.
    n_prompts = count_prompt_items(configs[0][3], base_prompts)
    if n_prompts == 0:
        raise ExperimentError(
            "no prompts to generate — the export had no captions and no flexibility_prompts were given"
        )
    n_runs_per_cell = len(matrix.lora_checkpoints) * len(matrix.lora_weights) * n_prompts
    n_images_per_cell = n_runs_per_cell * len(matrix.seeds)
    gpu_seconds = sum(n_images_per_cell * cfg.seconds_per_image for *_, cfg in configs)
    gpu_hours = gpu_seconds / 3600.0

    if max_gpu_hours is not None and gpu_hours > max_gpu_hours:
        raise ExperimentError(
            f"experiment needs {gpu_hours:.1f} GPU-hours across {len(configs)} cells "
            f"({n_images_per_cell * len(configs)} images) > budget {max_gpu_hours} — trim factors/levels, "
            f"cap prompts (max_base_prompts), or use optuna_search for guided search"
        )

    # Within budget → materialize the grids for the returned plan.
    cells: list[ExperimentCell] = []
    for cell_id, checkpoint, step_name, config in configs:
        try:
            plan = build_grid(config, base_prompts)
        except ProofError as exc:  # defensive: n_prompts > 0 means build_grid has prompts
            raise ExperimentError(f"cell {cell_id!r} could not be built: {exc}") from exc
        cells.append(
            ExperimentCell(
                cell_id=cell_id,
                base_checkpoint=checkpoint,
                step_config=step_name,
                labels={**matrix.labels, "step_config": step_name},
                plan=plan,
            )
        )

    estimate = ExperimentEstimate(
        n_cells=len(cells),
        n_runs=n_runs_per_cell * len(cells),
        n_images=n_images_per_cell * len(cells),
        seconds_per_image=matrix.seconds_per_image,
        est_gpu_seconds=gpu_seconds,
        est_gpu_hours=gpu_hours,
        per_cell={c.cell_id: n_images_per_cell for c in cells},
    )
    return ExperimentPlan(run_id_prefix=matrix.run_id_prefix, estimate=estimate, cells=cells)


class OptunaResult(BaseModel):
    """The outcome of a guided search: the best factor choice and its score."""

    best_params: dict[str, Any]
    best_value: float
    n_trials: int


def optuna_search(
    matrix: ExperimentMatrix,
    objective: Callable[[dict[str, Any]], float],
    *,
    n_trials: int,
    direction: str = "maximize",
    seed: int | None = None,
) -> OptunaResult:
    """Sample-efficient search over *matrix*'s factor levels via Optuna (``[opt]`` extra).

    For a matrix too large to brute-force with :func:`expand_experiment`, this
    lets an optimiser propose factor combinations instead of running the full
    cartesian. Each trial draws one categorical value per factor from
    :meth:`ExperimentMatrix.search_space` and passes the choice dict to
    *objective*, which the caller implements — typically: build the single cell,
    generate + score it, and return a scalar (pass-rate, or its Wilson lower
    bound) to ``maximize``.

    Optuna is an optional dependency; this raises :class:`ExperimentError` with an
    install hint if it is not present. ``seed`` makes the sampler reproducible.
    """
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - exercised via the [opt] extra
        raise ExperimentError("optuna_search requires: pip install 'argus-proof[opt]'") from exc

    space = matrix.search_space()

    def _objective(trial: optuna.Trial) -> float:
        choice = {name: trial.suggest_categorical(name, levels) for name, levels in space.items()}
        return objective(choice)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(_objective, n_trials=n_trials)
    return OptunaResult(
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        n_trials=len(study.trials),
    )
