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

import json
import os
import re
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

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
    # The experiment arm this run belongs to (argus_proof.experiment): the named
    # sampler variant, plus the upstream factors proof observes but can't vary
    # (caption strategy, source set). Both are sliceable — see slice_pass_rate.
    step_config: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)


class SliceStats(BaseModel):
    """Pooled pass-rate + CI for one cell of a cross-run slice (e.g. one checkpoint)."""

    dimension: str
    value: str | None  # None is kept distinct from a literal "" dimension value
    n_runs: int
    n_groups: int
    n_passed: int
    pass_rate: float
    ci_low: float
    ci_high: float


def run_stats(
    manifest: RunManifest,
    report: EvalReport,
    *,
    confidence: float = 0.95,
    step_config: str | None = None,
    labels: dict[str, str] | None = None,
) -> RunStats:
    """Flatten a (manifest, report) pair into a :class:`RunStats` row.

    The pass-rate is over near-dup *groups* (as scored); its CI is the Wilson
    interval over ``n_passed / n_groups`` — so a 3/3 run reads as far less certain
    than a 300/400 one. Safety min/hit-rate come from the run's safety metric when
    it was scored, else stay ``None``.

    *step_config* and *labels* attribute the run to its experiment arm — pass an
    :class:`~argus_proof.experiment.ExperimentCell`'s ``step_config``/``labels``
    so the store can compare arms (``slice_pass_rate("step_config")`` /
    ``slice_pass_rate("label:caption_strategy")``).
    """
    agg = report.aggregate
    n_groups = agg.n_groups if agg.n_groups is not None else agg.n_images
    n_passed = min(agg.n_passed, n_groups)  # clamp so pass_rate, CI, and n_passed stay consistent
    ci_low, ci_high = wilson_interval(n_passed, n_groups, confidence)

    # A stacked (multi-LoRA) run is its OWN comparison cell, not the same as its
    # first LoRA alone — key it by the full set so slicing can't misattribute it.
    loras = manifest.loras
    lora = "+".join(lo.name for lo in loras) or None
    single = loras[0] if len(loras) == 1 else None
    lora_sha = single.sha256 if single else None
    lora_weight = round(single.weight, 4) if single else None

    safety_min = safety_hit_rate = None
    if any(img.metrics.safety is not None for img in report.images):
        from argus_proof.scoring.scorers.safety import safety_tail_aggregate  # lazy: keeps [stats] off scoring

        tail = safety_tail_aggregate(report)
        safety_min, safety_hit_rate = tail["min_safety"], tail["hit_rate"]

    scorers = ",".join(f"{p.name}@{p.version}" if p.version else p.name for p in report.scorers) or None
    return RunStats(
        run_id=report.run_id,
        training_run_id=manifest.training_run_id,
        base_checkpoint=manifest.base_checkpoint.name,
        base_checkpoint_sha=manifest.base_checkpoint.sha256,
        lora=lora,
        lora_sha=lora_sha,
        lora_weight=lora_weight,
        prompt=manifest.prompt,
        n_images=agg.n_images,
        n_groups=n_groups,
        n_passed=n_passed,
        n_needs_hitl=agg.n_needs_hitl,
        pass_rate=n_passed / n_groups if n_groups else 0.0,
        pass_rate_ci_low=ci_low,
        pass_rate_ci_high=ci_high,
        diversity=agg.diversity,
        safety_min=safety_min,
        safety_hit_rate=safety_hit_rate,
        proof_version=report.proof_version,
        scorers=scorers,
        created_at=report.created_at,
        step_config=step_config,
        labels=dict(labels or {}),
    )


# Label keys are embedded in a JSONPath, so keep them to a safe charset.
_LABEL_KEY_RE = re.compile(r"[A-Za-z0-9_-]+")


def _row_for_parquet(stats: RunStats) -> dict:
    """A :class:`RunStats` flattened for the parquet store.

    ``labels`` is arbitrary user-defined keys, so it becomes a **JSON text column**
    rather than a struct (whose schema would differ run-to-run and break the
    concat); :meth:`CrossRunStore.slice_pass_rate` extracts a key from it on demand.
    """
    row = stats.model_dump()
    row["labels"] = json.dumps(row.get("labels") or {}, sort_keys=True)
    return row


class CrossRunStore:
    """A parquet file of :class:`RunStats` rows, keyed by ``run_id`` (re-append updates)."""

    # Columns a slice can group by (identity dimensions, not measured outcomes).
    SLICEABLE = ("base_checkpoint", "lora", "lora_weight", "prompt", "training_run_id", "step_config")

    # Arbitrary experiment labels are sliced as "label:<key>" (they have no fixed
    # schema, so they're stored as a JSON text column rather than one column each).
    LABEL_PREFIX = "label:"

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        """Serialize the read-modify-write against concurrent appenders (unix flock).

        Parallel eval workers sharing one store would otherwise each read the same
        snapshot and last-writer-wins would drop rows. Best-effort: platforms
        without ``fcntl`` (Windows) fall back to no locking.
        """
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-unix
            yield
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_name(f"{self.path.name}.lock"), "w") as lockfile:
            fcntl.flock(lockfile, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lockfile, fcntl.LOCK_UN)

    def append(self, stats: RunStats | Iterable[RunStats]) -> None:
        """Add row(s); a run_id already present is replaced (latest wins).

        Concurrency-safe: the read-modify-write is flock-guarded and the write is
        atomic (unique temp + ``os.replace``), so parallel appends don't lose rows
        or leave a torn file. Note: re-reads the whole store per call (O(n)), fine
        for accumulating hundreds–thousands of runs.
        """
        import polars as pl

        rows = [stats] if isinstance(stats, RunStats) else list(stats)
        if not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock():
            frame = pl.DataFrame([_row_for_parquet(s) for s in rows])
            if self.path.exists():
                frame = pl.concat([pl.read_parquet(self.path), frame], how="diagonal_relaxed")
            frame = frame.unique(subset=["run_id"], keep="last", maintain_order=True)
            fd, tmp_name = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp")
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                frame.write_parquet(tmp)
                os.replace(tmp, self.path)
            finally:
                tmp.unlink(missing_ok=True)  # no-op after a successful replace; cleans a partial write

    def frame(self):  # noqa: ANN201 - polars is optional; avoid importing it at module load
        """The store as a polars DataFrame (empty if nothing stored yet)."""
        import polars as pl

        return pl.read_parquet(self.path) if self.path.exists() else pl.DataFrame()

    def _slice_expr(self, dimension: str):  # noqa: ANN202 - a polars expression
        """The column expression *dimension* groups by: a fixed :data:`SLICEABLE`
        column, or ``label:<key>`` extracted from the labels JSON."""
        import polars as pl

        if dimension.startswith(self.LABEL_PREFIX):
            key = dimension[len(self.LABEL_PREFIX) :]
            if not _LABEL_KEY_RE.fullmatch(key):
                raise ValueError(
                    f"invalid label key {key!r}; use {self.LABEL_PREFIX}<key> with key matching [A-Za-z0-9_-]+"
                )
            if "labels" not in self.frame().columns:  # store predates labels
                raise ValueError(f"cannot slice by {dimension!r}; this store has no labels column")
            return pl.col("labels").str.json_path_match(f"$.{key}")
        if dimension not in self.SLICEABLE:
            raise ValueError(
                f"cannot slice by {dimension!r}; choose one of {self.SLICEABLE} or '{self.LABEL_PREFIX}<key>'"
            )
        return pl.col(dimension)

    def slice_pass_rate(self, dimension: str, *, confidence: float = 0.95) -> list[SliceStats]:
        """Pooled pass-rate + Wilson CI per value of *dimension*, across all runs.

        *dimension* is one of :data:`SLICEABLE` (including ``step_config``, the
        experiment's sampler arm) or ``label:<key>`` to compare by an upstream
        factor an experiment recorded (e.g. ``label:caption_strategy``); a run that
        carries no such label falls into the ``None`` cell rather than being dropped.

        Cells are **ranked by the CI lower bound** (not the point estimate), so a
        well-evidenced 380/400 outranks a lucky 3/3 — the ranking honours the
        uncertainty. The CI pools every group in the cell as one sample, so it
        reflects within-sample sampling error but NOT between-run variance: for a
        handful of disagreeing runs, inspect the per-run rows (``frame()``) too.
        """
        import polars as pl

        df = self.frame()
        if df.is_empty():
            return []
        expr = self._slice_expr(dimension)
        grouped = (
            df.with_columns(expr.alias("_slice"))
            .group_by("_slice")
            .agg(pl.len().alias("n_runs"), pl.sum("n_groups"), pl.sum("n_passed"))
            .sort("_slice")
        )
        out: list[SliceStats] = []
        for row in grouped.iter_rows(named=True):
            n_groups, n_passed = row["n_groups"] or 0, row["n_passed"] or 0
            lo, hi = wilson_interval(min(n_passed, n_groups), n_groups, confidence)
            out.append(
                SliceStats(
                    dimension=dimension,
                    value=None if row["_slice"] is None else str(row["_slice"]),
                    n_runs=row["n_runs"],
                    n_groups=n_groups,
                    n_passed=n_passed,
                    pass_rate=n_passed / n_groups if n_groups else 0.0,
                    ci_low=lo,
                    ci_high=hi,
                )
            )
        # Rank by the CI lower bound (evidence-adjusted), tie-break on point estimate.
        return sorted(out, key=lambda s: (s.ci_low, s.pass_rate), reverse=True)


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
