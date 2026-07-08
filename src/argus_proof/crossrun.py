"""Cross-run statistics store (#10).

Turns per-run :class:`~argus_proof.models.EvalReport`s into defensible, comparable
numbers and a durable store, so "which checkpoint / LoRA weight / token wins?" is
answered with evidence:

* :func:`run_stats` flattens a (manifest, report) pair into one tidy
  :class:`RunStats` row — the run's identity (checkpoint / LoRA / weight / prompt),
  its group-collapsed pass-rate **with a Wilson confidence interval** (small
  per-cell N is misleading as a bare average), diversity, and safety tail.
* :class:`CrossRunStore` accumulates those rows in a **parquet** file keyed by
  ``run_id`` + versions, and :meth:`~CrossRunStore.slice_pass_rate` pools them by
  any dimension (checkpoint / lora / weight / prompt) with a CI per cell.
* :func:`krippendorff_alpha` gives inter-rater reliability when multiple raters
  reviewed the same images.

The parquet store and inter-rater need the ``[stats]`` extra (polars,
krippendorff); the pass-rate CIs themselves use the dependency-free
:mod:`argus_proof.stats`.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from argus_proof.scoring.scorers.safety import safety_tail_aggregate
from argus_proof.stats import wilson_interval

if TYPE_CHECKING:
    from argus_proof.models import EvalReport, RunManifest


class RunStats(BaseModel):
    """One run's flattened, comparable statistics — a row in the cross-run store."""

    run_id: str
    training_run_id: str | None = None
    base_checkpoint: str
    base_checkpoint_sha: str
    lora: str | None = None
    lora_sha: str | None = None
    lora_weight: float | None = None
    prompt: str = ""
    n_images: int = 0
    n_groups: int = 0
    n_passed: int = 0
    n_needs_hitl: int = 0
    pass_rate: float = 0.0
    pass_rate_ci_low: float = 0.0
    pass_rate_ci_high: float = 0.0
    diversity: float | None = None
    safety_min: float | None = None
    safety_hit_rate: float | None = None
    proof_version: str = ""
    scorers: str | None = None
    created_at: str | None = None


class SliceStats(BaseModel):
    """Pooled pass-rate + CI for one cell of a cross-run slice (e.g. one checkpoint)."""

    dimension: str
    value: str
    n_runs: int
    n_groups: int
    n_passed: int
    pass_rate: float
    ci_low: float
    ci_high: float


def run_stats(manifest: RunManifest, report: EvalReport, *, confidence: float = 0.95) -> RunStats:
    """Flatten a (manifest, report) pair into a :class:`RunStats` row.

    The pass-rate is over near-dup *groups* (as scored); its CI is the Wilson
    interval over ``n_passed / n_groups`` — so a 3/3 run reads as far less certain
    than a 300/400 one. Safety min/hit-rate come from the run's safety metric when
    it was scored, else stay ``None``.
    """
    agg = report.aggregate
    n_groups = agg.n_groups if agg.n_groups is not None else agg.n_images
    ci_low, ci_high = wilson_interval(min(agg.n_passed, n_groups), n_groups, confidence)
    lora = manifest.loras[0] if manifest.loras else None

    safety_min = safety_hit_rate = None
    if any(img.metrics.safety is not None for img in report.images):
        tail = safety_tail_aggregate(report)
        safety_min, safety_hit_rate = tail["min_safety"], tail["hit_rate"]

    scorers = ",".join(f"{p.name}@{p.version}" if p.version else p.name for p in report.scorers) or None
    return RunStats(
        run_id=report.run_id,
        training_run_id=manifest.training_run_id,
        base_checkpoint=manifest.base_checkpoint.name,
        base_checkpoint_sha=manifest.base_checkpoint.sha256,
        lora=lora.name if lora else None,
        lora_sha=lora.sha256 if lora else None,
        lora_weight=lora.weight if lora else None,
        prompt=manifest.prompt,
        n_images=agg.n_images,
        n_groups=n_groups,
        n_passed=agg.n_passed,
        n_needs_hitl=agg.n_needs_hitl,
        pass_rate=agg.pass_rate,
        pass_rate_ci_low=ci_low,
        pass_rate_ci_high=ci_high,
        diversity=agg.diversity,
        safety_min=safety_min,
        safety_hit_rate=safety_hit_rate,
        proof_version=report.proof_version,
        scorers=scorers,
        created_at=report.created_at,
    )


class CrossRunStore:
    """A parquet file of :class:`RunStats` rows, keyed by ``run_id`` (re-append updates)."""

    # Columns a slice can group by (identity dimensions, not measured outcomes).
    SLICEABLE = ("base_checkpoint", "lora", "lora_weight", "prompt", "training_run_id")

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def append(self, stats: RunStats | Iterable[RunStats]) -> None:
        """Add row(s); a run_id already present is replaced (latest wins)."""
        import polars as pl

        rows = [stats] if isinstance(stats, RunStats) else list(stats)
        if not rows:
            return
        frame = pl.DataFrame([s.model_dump() for s in rows])
        if self.path.exists():
            frame = pl.concat([pl.read_parquet(self.path), frame], how="diagonal_relaxed")
        frame = frame.unique(subset=["run_id"], keep="last", maintain_order=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.tmp")
        frame.write_parquet(tmp)
        os.replace(tmp, self.path)

    def frame(self):  # noqa: ANN201 - polars is optional; avoid importing it at module load
        """The store as a polars DataFrame (empty if nothing stored yet)."""
        import polars as pl

        return pl.read_parquet(self.path) if self.path.exists() else pl.DataFrame()

    def slice_pass_rate(self, dimension: str, *, confidence: float = 0.95) -> list[SliceStats]:
        """Pooled pass-rate + Wilson CI per value of *dimension*, across all runs.

        Pools ``n_passed`` / ``n_groups`` within each cell (so cells with more
        runs get tighter intervals) and sorts by pass-rate descending — the
        "which wins?" ordering, but with the CI to show how much to trust it.
        """
        if dimension not in self.SLICEABLE:
            raise ValueError(f"cannot slice by {dimension!r}; choose one of {self.SLICEABLE}")
        import polars as pl

        df = self.frame()
        if df.is_empty():
            return []
        grouped = (
            df.group_by(dimension).agg(pl.len().alias("n_runs"), pl.sum("n_groups"), pl.sum("n_passed")).sort(dimension)
        )
        out: list[SliceStats] = []
        for row in grouped.iter_rows(named=True):
            n_groups, n_passed = row["n_groups"] or 0, row["n_passed"] or 0
            lo, hi = wilson_interval(min(n_passed, n_groups), n_groups, confidence)
            out.append(
                SliceStats(
                    dimension=dimension,
                    value="" if row[dimension] is None else str(row[dimension]),
                    n_runs=row["n_runs"],
                    n_groups=n_groups,
                    n_passed=n_passed,
                    pass_rate=n_passed / n_groups if n_groups else 0.0,
                    ci_low=lo,
                    ci_high=hi,
                )
            )
        return sorted(out, key=lambda s: s.pass_rate, reverse=True)


def krippendorff_alpha(units: Sequence[dict[str, float]], *, level: str = "interval") -> float:
    """Inter-rater reliability (Krippendorff's alpha) over per-unit ratings.

    *units* is one ``{rater_id: rating}`` dict per unit (image); raters may differ
    or be absent. Returns alpha in ``(-inf, 1]`` (1.0 = perfect agreement), or
    ``nan`` when it's undefined (no unit rated by ≥2 raters). Needs ``[stats]``.
    """
    import krippendorff
    import numpy as np

    raters = sorted({rater for unit in units for rater in unit})
    scorable = sum(1 for unit in units if len(unit) >= 2)
    if len(raters) < 2 or scorable == 0:
        return float("nan")
    matrix = [[unit.get(rater, np.nan) for unit in units] for rater in raters]
    return float(krippendorff.alpha(reliability_data=matrix, level_of_measurement=level))
