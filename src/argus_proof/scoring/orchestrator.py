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
    EvalReport,
    GateConfig,
    GeneratedImage,
    ImageScores,
    MetricScores,
    RunManifest,
    ScorerProvenance,
)
from argus_proof.scoring.base import (
    METRIC_FIELDS,
    Deduper,
    DiversityScorer,
    ImageScorer,
    ScoreContext,
)
from argus_proof.scoring.gate import gate_image
from argus_proof.scoring.summary import summarise

logger = structlog.get_logger()


def _score_images(images: list[GeneratedImage], scorers: Sequence[ImageScorer], ctx: ScoreContext) -> list[ImageScores]:
    live = [s for s in scorers if s.is_available()]
    # Validate the scorers up front so a misconfigured metric fails fast with a
    # clear message, not an opaque pydantic error partway through the run.
    seen_metrics: set[str] = set()
    for scorer in live:
        if scorer.metric not in METRIC_FIELDS:
            raise ValueError(f"scorer metric {scorer.metric!r} is not one of {METRIC_FIELDS}")
        if scorer.metric in seen_metrics:
            raise ValueError(
                f"two available scorers both target metric {scorer.metric!r}; one would silently "
                "overwrite the other — supply at most one scorer per metric"
            )
        seen_metrics.add(scorer.metric)
    rows: list[ImageScores] = []
    for img in images:
        metrics = MetricScores()
        image_path = Path(img.path)
        for scorer in live:
            value = scorer.score(image_path, ctx)
            if value is None:
                continue
            # Enforce the [0,1] contract here: an un-normalised scorer (e.g. a raw
            # pyiqa 1–10 or ImageReward score) would otherwise silently dominate
            # the gate's weighted composite and auto-pass garbage.
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"scorer for {scorer.metric!r} returned {value}; scores must be normalised to [0, 1]")
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
    _apply_dedup(rows, images, deduper)

    for row in rows:
        row.passed, row.reject_reasons = gate_image(row.metrics, config)

    div = diversity.score(images, ctx) if diversity is not None and diversity.is_available() else None

    # Group-collapsed aggregate + verdict — the same roll-up the HITL flow reuses.
    aggregate, verdict = summarise(rows, gate=config, diversity=div, n_images=len(images))

    provenance: list[ScorerProvenance] = [s.provenance() for s in scorers if s.is_available()]
    if deduper is not None and deduper.is_available():
        provenance.append(deduper.provenance())
    if diversity is not None and diversity.is_available():
        provenance.append(diversity.provenance())

    logger.info(
        "scoring.run",
        run_id=manifest.run_id,
        pass_rate=aggregate.pass_rate,
        n_groups=aggregate.n_groups,
        pending=verdict.pending,
        diversity=div,
    )
    return EvalReport(
        run_id=manifest.run_id,
        images=rows,
        aggregate=aggregate,
        scorers=provenance,
        verdict=verdict,
    )
