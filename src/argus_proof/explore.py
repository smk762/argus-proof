"""Optional FiftyOne integration (#7… issue #14): a power-user exploration surface.

Turns a scored :class:`~argus_proof.models.EvalReport` into a
`FiftyOne <https://docs.voxel51.com>`_ dataset — every computed field attached to
its image — so an analyst can visualise embeddings (UMAP/t-SNE) to spot mode
collapse / clusters / outliers, run the uniqueness/near-dup brain, and triage by
tag. It **complements** the ``/proof`` HITL view rather than replacing it, and is
strictly optional: FiftyOne is heavy, so it lives behind the ``[fiftyone]`` extra,
is imported lazily, and :func:`is_available` lets a caller skip it when absent.

The pure mapping (:func:`sample_fields`, :func:`sample_tags`, :func:`ingest_tags`)
has no FiftyOne dependency and is fully unit-tested; only the thin dataset/brain
adapters below need the package installed. The tag round-trip reuses
:func:`~argus_proof.reports.apply_hitl`, so tags fold back into the report through
the exact same path a HITL review does.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

from argus_proof.models import EvalReport, ImageScores, ProofError, RejectReason, RejectReasonCode

if TYPE_CHECKING:
    from collections.abc import Mapping

# The numeric scoring axes carried on MetricScores, exported one FiftyOne field each.
_METRIC_FIELDS: tuple[str, ...] = ("identity", "clip_score", "aesthetic", "preference", "safety")
_REJECT_CODES: frozenset[str] = frozenset(get_args(RejectReasonCode))

# Tag conventions for the round-trip: "reject:<code>" -> a reject reason,
# "rating:<1-5>" -> a HITL star rating.
_REJECT_TAG = "reject:"
_RATING_TAG = "rating:"


# ---------------------------------------------------------------------------
# Pure mapping — EvalReport <-> FiftyOne sample fields/tags (no FiftyOne dep)
# ---------------------------------------------------------------------------


def _verdict_label(passed: bool | None) -> str:
    """The tri-state verdict as a label: pass / fail / needs_review (undecided)."""
    return "passed" if passed is True else "failed" if passed is False else "needs_review"


def sample_fields(img: ImageScores) -> dict[str, Any]:
    """The FiftyOne sample fields for one image — every computed metric + HITL
    value that is set (a ``None`` metric is omitted rather than exported as null)."""
    fields: dict[str, Any] = {"image_id": img.image_id, "seed": img.seed, "verdict": _verdict_label(img.passed)}
    for name in _METRIC_FIELDS:
        value = getattr(img.metrics, name)
        if value is not None:
            fields[name] = value
    if img.hitl_rating is not None:
        fields["hitl_rating"] = img.hitl_rating
    if img.hitl_rater is not None:
        fields["hitl_rater"] = img.hitl_rater
    if img.duplicate_group is not None:
        fields["duplicate_group"] = img.duplicate_group
    if img.reject_reasons:
        fields["reject_reasons"] = [r.code for r in img.reject_reasons]
    return fields


def sample_tags(img: ImageScores) -> list[str]:
    """Triage tags for one image: its verdict plus a ``reject:<code>`` per reason."""
    return [_verdict_label(img.passed), *(f"{_REJECT_TAG}{r.code}" for r in img.reject_reasons)]


def _parse_rating(tags: list[str]) -> int | None:
    """The last valid ``rating:<1-5>`` tag as an int, else ``None``."""
    rating: int | None = None
    for tag in tags:
        if tag.startswith(_RATING_TAG):
            try:
                value = int(tag[len(_RATING_TAG) :])
            except ValueError:
                continue
            if 1 <= value <= 5:
                rating = value
    return rating


def _parse_rejects(tags: list[str]) -> list[RejectReason]:
    """``reject:<code>`` tags whose code is in the taxonomy, de-duplicated in order."""
    seen: set[str] = set()
    out: list[RejectReason] = []
    for tag in tags:
        if tag.startswith(_REJECT_TAG):
            code = tag[len(_REJECT_TAG) :]
            if code in _REJECT_CODES and code not in seen:
                seen.add(code)
                out.append(RejectReason(code=code))  # type: ignore[arg-type]  # guarded by _REJECT_CODES
    return out


def ingest_tags(
    report: EvalReport,
    tags_by_image: Mapping[str, list[str]],
    *,
    rater: str | None = None,
    gate: Any = None,
) -> EvalReport:
    """Fold FiftyOne triage *tags* back into *report* as ratings / reject reasons.

    Recognises ``rating:<1-5>`` and ``reject:<code>`` tags. Only images that carry
    a recognised tag are touched; for those, an absent rating/reject tag keeps the
    image's existing value (so a reject-only triage doesn't wipe a prior rating).
    Unknown image_ids (samples not from this run) are ignored. Delegates to
    :func:`~argus_proof.reports.apply_hitl`, so the aggregate pass-rate and verdict
    are recomputed exactly as a HITL review would.
    """
    from argus_proof.reports import HitlImageUpdate, HitlRequest, apply_hitl

    existing = {img.image_id: img for img in report.images}
    updates: list[HitlImageUpdate] = []
    for image_id, tags in tags_by_image.items():
        img = existing.get(image_id)
        if img is None:
            continue
        rating = _parse_rating(tags)
        rejects = _parse_rejects(tags)
        if rating is None and not rejects:
            continue  # no actionable tag for this image
        updates.append(
            HitlImageUpdate(
                image_id=image_id,
                hitl_rating=rating if rating is not None else img.hitl_rating,
                reject_reasons=rejects if rejects else img.reject_reasons,
            )
        )
    return apply_hitl(report, HitlRequest(rater=rater, gate=gate, updates=updates))


def image_paths_from_dir(directory: str | Path) -> dict[str, str]:
    """Map ``image_id -> filepath`` for a run's images, keyed by filename **stem**.

    A convenience for the common layout where each generated image is written as
    ``<image_id>.<ext>``. Hidden files are skipped; on a stem collision the first
    (sorted) path wins.
    """
    paths: dict[str, str] = {}
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and not path.name.startswith("."):
            paths.setdefault(path.stem, str(path))
    return paths


# ---------------------------------------------------------------------------
# FiftyOne adapters — lazy; require `pip install "argus-proof[fiftyone]"`
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Whether FiftyOne is importable (the ``[fiftyone]`` extra is installed)."""
    return importlib.util.find_spec("fiftyone") is not None


def _fiftyone():  # noqa: ANN202 - the fiftyone module, imported lazily
    try:
        import fiftyone as fo
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ProofError("FiftyOne integration requires: pip install 'argus-proof[fiftyone]'") from exc
    return fo


def to_fiftyone_dataset(
    report: EvalReport,
    image_paths: Mapping[str, str],
    *,
    name: str | None = None,
    persistent: bool = False,
):  # noqa: ANN201 - returns a fiftyone.Dataset
    """Build a FiftyOne dataset from *report*, one sample per image with a known path.

    *image_paths* maps ``image_id -> filepath`` (see :func:`image_paths_from_dir`);
    an image with no path is skipped (you can't show a sample without a file).
    Every computed field rides on the sample (:func:`sample_fields`) with triage
    tags (:func:`sample_tags`). Raises :class:`~argus_proof.models.ProofError` if
    FiftyOne isn't installed.
    """
    fo = _fiftyone()
    dataset = fo.Dataset(name=name, persistent=persistent)
    samples = []
    for img in report.images:
        path = image_paths.get(img.image_id)
        if path is None:
            continue
        sample = fo.Sample(filepath=path, tags=sample_tags(img))
        for key, value in sample_fields(img).items():
            sample[key] = value
        samples.append(sample)
    dataset.add_samples(samples)
    return dataset


def dataset_tags(dataset) -> dict[str, list[str]]:  # noqa: ANN001 - a fiftyone.Dataset
    """Read each sample's tags back out, keyed by its ``image_id`` field."""
    return {sample["image_id"]: list(sample.tags) for sample in dataset}


def ingest_from_dataset(dataset, report: EvalReport, *, rater: str | None = None, gate: Any = None) -> EvalReport:  # noqa: ANN001
    """Round-trip: pull tags off *dataset* and fold them into *report* via :func:`ingest_tags`."""
    return ingest_tags(report, dataset_tags(dataset), rater=rater, gate=gate)


def compute_visualization(dataset, *, brain_key: str = "proof_viz", method: str = "umap", **kwargs):  # noqa: ANN001, ANN201
    """Compute an embedding visualisation (UMAP/t-SNE) via the FiftyOne Brain, so
    clusters / mode collapse / outliers are visible in the App. Thin passthrough to
    ``fiftyone.brain.compute_visualization``; needs the ``[fiftyone]`` extra."""
    _fiftyone()
    import fiftyone.brain as fob

    return fob.compute_visualization(dataset, brain_key=brain_key, method=method, **kwargs)


def compute_uniqueness(dataset, *, uniqueness_field: str = "uniqueness"):  # noqa: ANN001, ANN201
    """Score per-sample uniqueness (surfaces near-duplicates / redundancy) via the
    FiftyOne Brain. Thin passthrough to ``fiftyone.brain.compute_uniqueness``."""
    _fiftyone()
    import fiftyone.brain as fob

    return fob.compute_uniqueness(dataset, uniqueness_field=uniqueness_field)


def launch_app(dataset, **kwargs):  # noqa: ANN001, ANN201
    """Open the FiftyOne App on *dataset* and return the session (thin passthrough)."""
    fo = _fiftyone()
    return fo.launch_app(dataset, **kwargs)
