"""Metadata-only archival of a reviewed run (#9).

Keep the analytical value of a run without hoarding imagery: after review, a run
directory is reduced to

* ``manifest.json`` + ``eval_report.json`` — the run's params and scores;
* ``passing_images.zip`` — the images that passed, retained zipped;
* ``reject_archive.json`` — a :class:`~argus_proof.models.RejectArchive`: params
  (the embedded :class:`~argus_proof.models.RunManifest`) + scores + structured
  reject reasons + rater id for every rejected image, and **no image or
  thumbnail reference** (explicit product decision — the metadata is the useful
  signal for diagnosing failure modes and could later train an auto-classifier).

With ``blank_slate=True`` the loose per-image files that were archived (passing →
zipped, rejected → recorded) are deleted, leaving only the artifacts above.
Images still awaiting review (``passed is None``) are left untouched. Every
record joins back to its run by ``run_id``, so a reject is traceable to its exact
config without the pixels.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from argus_proof.models import (
    EvalReport,
    GeneratedImage,
    RejectArchive,
    RejectRecord,
    RunManifest,
)

logger = structlog.get_logger()

# Version of the on-disk archive layout (the set/naming of artifacts below).
# Bump if the directory contract changes so consumers can refuse an old layout.
ARCHIVE_VERSION = "1.0"

MANIFEST_NAME = "manifest.json"
REPORT_NAME = "eval_report.json"
PASSING_ZIP_NAME = "passing_images.zip"
REJECT_ARCHIVE_NAME = "reject_archive.json"


@dataclass
class ArchiveResult:
    """What :func:`archive_run` produced: counts + the artifact paths."""

    run_id: str
    n_passing: int
    n_rejected: int
    n_pending: int  # images left for review (passed is None)
    passing_zip: Path | None
    reject_archive_path: Path
    deleted_images: list[str] = field(default_factory=list)


def build_reject_archive(report: EvalReport, manifest: RunManifest, *, rater_id: str | None = None) -> RejectArchive:
    """A :class:`RejectArchive` of the report's rejected images (``passed is False``).

    Embeds *manifest* (keyed by ``run_id``) so the archive is self-describing, and
    stamps ``rater_id`` on each record when a human reviewed the run. Carries zero
    image references by construction.
    """
    records = [
        RejectRecord(
            run_id=report.run_id,
            seed=img.seed,
            metrics=img.metrics,
            hitl_rating=img.hitl_rating,
            reasons=img.reject_reasons,
            rater_id=rater_id,
        )
        for img in report.images
        if img.passed is False
    ]
    return RejectArchive(manifests={manifest.run_id: manifest}, records=records)


def archive_run(
    run_dir: Path,
    images: list[GeneratedImage],
    report: EvalReport,
    manifest: RunManifest,
    *,
    rater_id: str | None = None,
    blank_slate: bool = True,
) -> ArchiveResult:
    """Reduce a reviewed *run_dir* to metadata + a zip of the passing images.

    *images* supplies the on-disk paths (joined to *report* by ``image_id``).
    Writes ``manifest.json``, ``eval_report.json``, ``passing_images.zip``, and
    ``reject_archive.json``; with ``blank_slate`` deletes the loose image files
    that were archived (passing + rejected), leaving pending images in place.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    verdict_by_id = {img.image_id: img.passed for img in report.images}
    by_id = {img.image_id: img for img in images}

    (run_dir / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / REPORT_NAME).write_text(report.model_dump_json(indent=2), encoding="utf-8")

    reject_archive = build_reject_archive(report, manifest, rater_id=rater_id)
    reject_path = run_dir / REJECT_ARCHIVE_NAME
    reject_path.write_text(reject_archive.model_dump_json(indent=2), encoding="utf-8")

    # Zip the passing images (that still exist on disk).
    passing = [img for img in images if verdict_by_id.get(img.image_id) is True]
    passing_zip: Path | None = None
    if passing:
        passing_zip = run_dir / PASSING_ZIP_NAME
        with zipfile.ZipFile(passing_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for img in passing:
                src = Path(img.path)
                if src.is_file():
                    zf.write(src, arcname=src.name)

    n_pending = sum(1 for img in report.images if img.passed is None)
    deleted: list[str] = []
    if blank_slate:
        # Delete the loose files that are now archived: passing (in the zip) and
        # rejected (recorded as metadata). Pending images are left for review.
        for image_id, passed in verdict_by_id.items():
            if passed is None:
                continue  # not yet reviewed
            img = by_id.get(image_id)
            if img is None:
                continue
            src = Path(img.path)
            if src.is_file():
                src.unlink()
                deleted.append(src.name)

    result = ArchiveResult(
        run_id=report.run_id,
        n_passing=len(passing),
        n_rejected=len(reject_archive.records),
        n_pending=n_pending,
        passing_zip=passing_zip,
        reject_archive_path=reject_path,
        deleted_images=deleted,
    )
    logger.info(
        "archive.run",
        run_id=result.run_id,
        passing=result.n_passing,
        rejected=result.n_rejected,
        pending=result.n_pending,
        blank_slate=blank_slate,
    )
    return result
