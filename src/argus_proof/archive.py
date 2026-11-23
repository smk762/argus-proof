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
Images still awaiting review (``passed is None``) are left untouched; archiving
the run again once they are reviewed adds to ``passing_images.zip`` rather than
replacing it, and reconciles it with the current verdicts: an image whose verdict
changed from passed to rejected leaves the zip (it is the hand-off to training),
and one sent back to pending is restored to the run directory for review. Every
record joins back to its run by ``run_id``, so a reject is traceable to its exact
config without the pixels.
"""

from __future__ import annotations

import os
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
    n_passing: int  # passing images held in the zip (this call's and earlier ones'), not merely judged passing
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


def _entry_image_id(arcname: str) -> str:
    """The image_id an arcname (``<image_id><suffix>``) was written for."""
    suffix = Path(arcname).suffix
    return arcname[: -len(suffix)] if suffix else arcname


def _write_passing_zip(
    zip_path: Path,
    passing: list[GeneratedImage],
    verdict_by_id: dict[str, bool | None],
    by_id: dict[str, GeneratedImage],
    run_dir: Path,
    run_id: str,
) -> None:
    """Rebuild *zip_path* to hold exactly the images now passing, then swap it into place atomically.

    Arcname is the unique image_id (+ suffix) so two passing images that share a basename in different dirs don't
    collide. A run is archived again after more review, by which time blank_slate deleted the images zipped last
    time, so an existing entry still passing is carried over rather than lost. An entry whose verdict changed is not
    carried: a rejected one is dropped (the zip is the hand-off to training), and a pending one is restored to the
    run directory for review. No zip is left behind when nothing passes.
    """
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    written: set[str] = set()
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        if zip_path.is_file():
            with zipfile.ZipFile(zip_path) as old:
                for info in old.infolist():
                    image_id = _entry_image_id(info.filename)
                    verdict = verdict_by_id.get(image_id)
                    if verdict is True:
                        out.writestr(info, old.read(info))
                        written.add(info.filename)
                    elif verdict is None and image_id in verdict_by_id:
                        _restore_pending(old, info, by_id.get(image_id), run_dir, run_id)
                    else:
                        logger.warning("archive.drop_stale_passing", run_id=run_id, image_id=image_id)
        for img in passing:
            src = Path(img.path)
            arcname = f"{img.image_id}{src.suffix}"
            if src.is_file() and arcname not in written:
                out.write(src, arcname=arcname)
                written.add(arcname)
    if written:
        os.replace(tmp, zip_path)
    else:
        tmp.unlink()
        zip_path.unlink(missing_ok=True)


def _contained(path: Path, run_dir: Path) -> Path | None:
    """*path* when it is a file location inside *run_dir*, else None.

    ``resolve`` follows ``..`` and symlinks, so a zip member or a stored image path
    that climbs out of the run (or is absolute — ``run_dir / "/tmp/x"`` drops
    *run_dir*) does not come back as contained. The run directory itself is not a
    file destination.
    """
    try:
        resolved = path.resolve()
        run_resolved = run_dir.resolve()
    except OSError:
        return None
    if resolved == run_resolved or not resolved.is_relative_to(run_resolved):
        return None
    return path


def _restore_dest(img: GeneratedImage | None, info: zipfile.ZipInfo, run_dir: Path) -> Path | None:
    """Where to put *info* back: the image's own path, or the member's basename under *run_dir*.

    The basename is the fallback when the image has no path here or that path escapes the run.
    It is checked again: a member named ``..`` or an absolute path still does not get written.
    """
    if img is not None:
        inside = _contained(Path(img.path), run_dir)
        if inside is not None:
            return inside
    name = Path(info.filename).name
    if not name or name in {".", ".."}:
        return None
    return _contained(run_dir / name, run_dir)


def _restore_pending(
    zf: zipfile.ZipFile, info: zipfile.ZipInfo, img: GeneratedImage | None, run_dir: Path, run_id: str
) -> None:
    """Put a zipped image whose verdict went back to pending on disk again, so it can be reviewed."""
    dest = _restore_dest(img, info, run_dir)
    if dest is None:
        logger.warning("archive.skip_restore_outside_run_dir", run_id=run_id, filename=info.filename)
        return
    if not dest.exists():
        dest.write_bytes(zf.read(info))
    logger.info("archive.restore_pending", run_id=run_id, image=dest.name)


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

    # Warn about report images with no matching file — they can't be archived.
    missing = [i.image_id for i in report.images if i.passed is not None and i.image_id not in by_id]
    if missing:
        logger.warning("archive.missing_images", run_id=report.run_id, image_ids=missing)

    passing = [img for img in images if verdict_by_id.get(img.image_id) is True]
    zip_path = run_dir / PASSING_ZIP_NAME
    if passing or zip_path.is_file():
        _write_passing_zip(zip_path, passing, verdict_by_id, by_id, run_dir, report.run_id)
    # Report what is archived, not what this call was handed: a zip from an
    # earlier call still counts, and a passing image whose file was missing does not.
    passing_zip = zip_path if zip_path.is_file() else None
    n_passing = 0
    if passing_zip is not None:
        with zipfile.ZipFile(passing_zip) as zf:
            n_passing = len(zf.namelist())

    run_resolved = run_dir.resolve()
    n_pending = sum(1 for img in report.images if img.passed is None)
    deleted: list[str] = []
    if blank_slate:
        # Delete the loose files that are now archived: passing (in the zip) and
        # rejected (recorded as metadata). Pending images are left for review.
        # Only touch files inside run_dir — never delete an image the run doesn't own.
        for image_id, passed in verdict_by_id.items():
            if passed is None:
                continue  # not yet reviewed
            img = by_id.get(image_id)
            if img is None:
                continue
            src = Path(img.path)
            if not src.is_file():
                continue
            if not src.resolve().is_relative_to(run_resolved):
                logger.warning("archive.skip_delete_outside_run_dir", run_id=report.run_id, path=str(src))
                continue
            src.unlink()
            deleted.append(src.name)

    result = ArchiveResult(
        run_id=report.run_id,
        n_passing=n_passing,
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
