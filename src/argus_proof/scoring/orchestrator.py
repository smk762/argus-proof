"""Assemble scorers into an EvalReport for one generation run.

The spine that ties the pieces together: run each available :class:`ImageScorer`
over every image, collapse near-duplicates into groups, gate each image, and roll
it all up — with the pass rate computed over *groups*, not raw frames, so a
Monte-Carlo cluster can't inflate it. Works with any mix of scorers (including
none), so it's exercised end-to-end with fakes before a single model is loaded.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import structlog

from argus_proof.models import (
    AggregateScores,
    EvalReport,
    GateConfig,
    GeneratedImage,
    ImageScores,
    MetricScores,
    RunManifest,
    ScorerProvenance,
    Verdict,
)
from argus_proof.scoring.base import (
    METRIC_FIELDS,
    Deduper,
    DiversityScorer,
    ImageScorer,
    ScoreContext,
)
from argus_proof.scoring.gate import gate_image

logger = structlog.get_logger()


def _score_images(images: list[GeneratedImage], scorers: Sequence[ImageScorer], ctx: ScoreContext) -> list[ImageScores]:
    live = [s for s in scorers if s.is_available()]
    rows: list[ImageScores] = []
    for img in images:
        metrics = MetricScores()
        for scorer in live:
            value = scorer.score(Path(img.path), ctx)
            if value is not None:
                setattr(metrics, scorer.metric, value)
        rows.append(ImageScores(image_id=img.image_id, seed=img.seed, metrics=metrics))
    return rows


def _apply_dedup(rows: list[ImageScores], images: list[GeneratedImage], deduper: Deduper | None) -> list[int]:
    """Return a group label per image; stamp it onto each row. No deduper → each
    image is its own group."""
    if deduper is not None and deduper.is_available():
        labels = deduper.group(images)
        if len(labels) != len(images):
            raise ValueError(f"deduper returned {len(labels)} labels for {len(images)} images")
    else:
        labels = list(range(len(images)))
    for row, label in zip(rows, labels, strict=True):
        row.duplicate_group = label
    return labels


def _group_outcomes(rows: list[ImageScores], labels: list[int]) -> dict[int, bool | None]:
    """Collapse rows into groups and decide each group's outcome.

    A group passes if any member auto-passed; else it needs HITL if any member is
    undecided; else it fails. This is what makes a near-dup cluster count once.
    """
    members: dict[int, list[bool | None]] = {}
    for row, label in zip(rows, labels, strict=True):
        members.setdefault(label, []).append(row.passed)
    outcomes: dict[int, bool | None] = {}
    for label, verdicts in members.items():
        if any(v is True for v in verdicts):
            outcomes[label] = True
        elif any(v is None for v in verdicts):
            outcomes[label] = None
        else:
            outcomes[label] = False
    return outcomes


def _metric_means(rows: list[ImageScores]) -> MetricScores:
    means = MetricScores()
    for field in METRIC_FIELDS:
        values = [v for row in rows if (v := getattr(row.metrics, field)) is not None]
        if values:
            setattr(means, field, sum(values) / len(values))
    return means


def score_run(
    manifest: RunManifest,
    images: list[GeneratedImage],
    *,
    scorers: Sequence[ImageScorer] = (),
    deduper: Deduper | None = None,
    diversity: DiversityScorer | None = None,
    context: ScoreContext | None = None,
    gate: GateConfig | None = None,
) -> EvalReport:
    """Score a run's images into an :class:`EvalReport` (per-image + aggregate).

    ``context`` defaults to the run's prompt; ``gate`` to :class:`GateConfig`
    defaults. Only *available* scorers contribute (and are recorded in the
    report's provenance).
    """
    ctx = context or ScoreContext(prompt=manifest.prompt)
    config = gate or GateConfig()

    rows = _score_images(images, scorers, ctx)
    labels = _apply_dedup(rows, images, deduper)

    for row in rows:
        row.passed, row.reject_reasons = gate_image(row.metrics, config)

    outcomes = _group_outcomes(rows, labels)
    n_groups = len(outcomes)
    n_passed = sum(1 for o in outcomes.values() if o is True)
    n_needs_hitl = sum(1 for o in outcomes.values() if o is None)
    pass_rate = n_passed / n_groups if n_groups else 0.0

    div = diversity.score(images, ctx) if diversity is not None and diversity.is_available() else None

    aggregate = AggregateScores(
        n_images=len(images),
        n_groups=n_groups,
        n_passed=n_passed,
        n_needs_hitl=n_needs_hitl,
        pass_rate=pass_rate,
        means=_metric_means(rows),
        diversity=div,
    )

    provenance: list[ScorerProvenance] = [s.provenance() for s in scorers if s.is_available()]
    if deduper is not None and deduper.is_available():
        provenance.append(deduper.provenance())
    if diversity is not None and diversity.is_available():
        provenance.append(diversity.provenance())

    passed = pass_rate >= config.run_pass_rate
    reasons = [
        f"pass_rate {pass_rate:.2f} {'>=' if passed else '<'} threshold {config.run_pass_rate:.2f} "
        f"({n_passed}/{n_groups} groups passed, {n_needs_hitl} need review)"
    ]
    if div is not None:
        reasons.append(f"diversity {div:.2f}")

    logger.info("scoring.run", run_id=manifest.run_id, pass_rate=pass_rate, n_groups=n_groups, diversity=div)
    return EvalReport(
        run_id=manifest.run_id,
        images=rows,
        aggregate=aggregate,
        scorers=provenance,
        verdict=Verdict(passed=passed, reasons=reasons),
    )
