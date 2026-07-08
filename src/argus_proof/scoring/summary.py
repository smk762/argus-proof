"""Roll per-image score rows up into an aggregate + verdict.

Shared by :func:`~argus_proof.scoring.orchestrator.score_run` (the initial
automated pass) and the HITL review flow (:func:`argus_proof.reports.apply_hitl`,
which re-rolls a report after a human rates the borderline band). Keeping the
group-collapsed pass-rate math and the passed/pending verdict logic here means
both paths compute a run's outcome exactly the same way — a near-dup cluster
counts once, and "pending review" stays distinct from "failed".
"""

from __future__ import annotations

from argus_proof.models import AggregateScores, GateConfig, ImageScores, MetricScores, Verdict
from argus_proof.scoring.base import METRIC_FIELDS


def group_labels(rows: list[ImageScores]) -> list[int]:
    """A group label per row: its ``duplicate_group``, or a unique own-group when
    unset. Distinct negatives keep un-deduped rows from colliding with real group
    ids (which a deduper assigns from ``0``)."""
    labels: list[int] = []
    solo = -1
    for row in rows:
        if row.duplicate_group is not None:
            labels.append(row.duplicate_group)
        else:
            labels.append(solo)
            solo -= 1
    return labels


def group_outcomes(rows: list[ImageScores]) -> dict[int, bool | None]:
    """Collapse rows into near-dup groups and decide each group's outcome.

    A group passes if any member passed; else it needs HITL if any member is
    undecided; else it fails. This is what makes a near-dup cluster count once.
    """
    members: dict[int, list[bool | None]] = {}
    for row, label in zip(rows, group_labels(rows), strict=True):
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


def metric_means(rows: list[ImageScores]) -> MetricScores:
    """Mean of each metric over the rows that have it; an axis nobody scored stays
    ``None`` rather than reporting a fabricated ``0.0``."""
    means = MetricScores()
    for field in METRIC_FIELDS:
        values = [v for row in rows if (v := getattr(row.metrics, field)) is not None]
        if values:
            setattr(means, field, sum(values) / len(values))
    return means


def build_verdict(aggregate: AggregateScores, gate: GateConfig) -> Verdict:
    """The run verdict from its aggregate.

    ``passed`` = the passed groups alone clear ``run_pass_rate``. ``pending`` =
    it hasn't cleared yet, but if every un-reviewed group ends up passing it could
    — so it's "awaiting review", not "failed".
    """
    n_groups = aggregate.n_groups or 0
    passed = aggregate.pass_rate >= gate.run_pass_rate
    best_case = (aggregate.n_passed + aggregate.n_needs_hitl) / n_groups if n_groups else 0.0
    pending = not passed and aggregate.n_needs_hitl > 0 and best_case >= gate.run_pass_rate
    status = "passed" if passed else ("pending review" if pending else "failed")
    reasons = [
        f"pass_rate {aggregate.pass_rate:.2f} vs threshold {gate.run_pass_rate:.2f} — {status} "
        f"({aggregate.n_passed}/{n_groups} groups passed, {aggregate.n_needs_hitl} need review)"
    ]
    if aggregate.diversity is not None:
        reasons.append(f"diversity {aggregate.diversity:.2f}")
    return Verdict(passed=passed, pending=pending, reasons=reasons)


def summarise(
    rows: list[ImageScores],
    *,
    gate: GateConfig,
    diversity: float | None = None,
    n_images: int | None = None,
) -> tuple[AggregateScores, Verdict]:
    """Aggregate ``rows`` and derive the verdict, both over near-dup *groups*.

    ``n_images`` defaults to ``len(rows)`` — pass it explicitly to preserve the
    original frame count when recomputing from a report whose rows already carry
    their group labels. ``diversity`` is carried through unchanged (recomputing
    after HITL doesn't re-measure it).
    """
    outcomes = group_outcomes(rows)
    n_groups = len(outcomes)
    n_passed = sum(1 for o in outcomes.values() if o is True)
    n_needs_hitl = sum(1 for o in outcomes.values() if o is None)
    aggregate = AggregateScores(
        n_images=len(rows) if n_images is None else n_images,
        n_groups=n_groups,
        n_passed=n_passed,
        n_needs_hitl=n_needs_hitl,
        pass_rate=n_passed / n_groups if n_groups else 0.0,
        means=metric_means(rows),
        diversity=diversity,
    )
    return aggregate, build_verdict(aggregate, gate)
