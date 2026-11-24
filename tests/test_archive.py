from __future__ import annotations

import json
import zipfile
from pathlib import Path

from argus_proof.archive import (
    PASSING_ZIP_NAME,
    REJECT_ARCHIVE_NAME,
    archive_run,
    build_reject_archive,
)
from argus_proof.models import (
    AggregateScores,
    EvalReport,
    GeneratedImage,
    ImageScores,
    MetricScores,
    ModelRef,
    RejectArchive,
    RejectReason,
    RunManifest,
    SamplingParams,
    Verdict,
)

SHA = "a" * 64


def manifest() -> RunManifest:
    return RunManifest(
        run_id="run-1",
        base_checkpoint=ModelRef(name="base.safetensors", sha256=SHA),
        sampling=SamplingParams(sampler="euler", scheduler="normal", steps=20, cfg=6.0, width=64, height=64),
        prompt="a photo of sks",
        seeds=[1, 2, 3],
        engine="comfyui",
        engine_version="0.3.0",
    )


def image_scores(seed: int, passed: bool | None, reasons: list[RejectReason] | None = None) -> ImageScores:
    return ImageScores(
        image_id=f"run-1-{seed}",
        seed=seed,
        metrics=MetricScores(identity=0.5),
        reject_reasons=reasons or [],
        passed=passed,
    )


def report(images: list[ImageScores]) -> EvalReport:
    n_pass = sum(1 for i in images if i.passed is True)
    return EvalReport(
        run_id="run-1",
        images=images,
        aggregate=AggregateScores(n_images=len(images), n_passed=n_pass, pass_rate=0.0),
        verdict=Verdict(passed=False),
    )


def gen_images(run_dir: Path, seeds: list[int]) -> list[GeneratedImage]:
    out = []
    for s in seeds:
        p = run_dir / f"run-1-{s}.png"
        p.write_bytes(b"fake-png-bytes")
        out.append(GeneratedImage(image_id=f"run-1-{s}", run_id="run-1", seed=s, path=str(p), width=64, height=64))
    return out


# --------------------------------------------------------------------------
# build_reject_archive
# --------------------------------------------------------------------------


def test_build_reject_archive_records_only_failures() -> None:
    rpt = report(
        [
            image_scores(1, True),
            image_scores(2, False, [RejectReason(code="identity_mismatch")]),
            image_scores(3, None),  # pending — not archived
        ]
    )
    archive = build_reject_archive(rpt, manifest(), rater_id="alice")
    assert [r.seed for r in archive.records] == [2]
    assert archive.records[0].reasons[0].code == "identity_mismatch"
    assert archive.records[0].rater_id == "alice"
    assert archive.manifests["run-1"].run_id == "run-1"  # manifest embedded for join


def test_reject_archive_has_zero_image_references() -> None:
    rpt = report([image_scores(2, False)])
    blob = json.dumps(build_reject_archive(rpt, manifest()).model_dump(mode="json")).lower()
    for banned in ("image_id", "thumbnail", ".png", "path"):
        assert banned not in blob, f"reject archive leaked {banned!r}"


# --------------------------------------------------------------------------
# archive_run
# --------------------------------------------------------------------------


def test_archive_run_zips_passing_records_rejects_blank_slates(tmp_path: Path) -> None:
    images = gen_images(tmp_path, [1, 2, 3])
    rpt = report([image_scores(1, True), image_scores(2, False), image_scores(3, None)])

    result = archive_run(tmp_path, images, rpt, manifest(), rater_id="bob")

    assert (result.n_passing, result.n_rejected, result.n_pending) == (1, 1, 1)

    # passing image is in the zip
    with zipfile.ZipFile(tmp_path / PASSING_ZIP_NAME) as zf:
        assert zf.namelist() == ["run-1-1.png"]

    # reject archive written, rater stamped
    archive = RejectArchive.model_validate_json((tmp_path / REJECT_ARCHIVE_NAME).read_text())
    assert [r.seed for r in archive.records] == [2]
    assert archive.records[0].rater_id == "bob"

    # blank slate: passing + rejected loose files deleted; pending left
    assert not (tmp_path / "run-1-1.png").exists()  # passing (zipped) removed
    assert not (tmp_path / "run-1-2.png").exists()  # rejected (recorded) removed
    assert (tmp_path / "run-1-3.png").exists()  # pending kept for review


def test_rearchive_after_more_review_keeps_previously_zipped_images(tmp_path: Path) -> None:
    """Archiving again once pending images are reviewed adds to the zip; it never drops the earlier passes."""
    images = gen_images(tmp_path, [1, 2])
    archive_run(tmp_path, images, report([image_scores(1, True), image_scores(2, None)]), manifest())
    assert not (tmp_path / "run-1-1.png").exists()  # zipped, then blank-slated

    archive_run(tmp_path, images, report([image_scores(1, True), image_scores(2, True)]), manifest())
    with zipfile.ZipFile(tmp_path / PASSING_ZIP_NAME) as zf:
        assert sorted(zf.namelist()) == ["run-1-1.png", "run-1-2.png"]
        assert zf.read("run-1-1.png") == b"fake-png-bytes"

    # a third pass over an unchanged run adds no duplicate entries
    archive_run(tmp_path, images, report([image_scores(1, True), image_scores(2, True)]), manifest())
    with zipfile.ZipFile(tmp_path / PASSING_ZIP_NAME) as zf:
        assert sorted(zf.namelist()) == ["run-1-1.png", "run-1-2.png"]


def test_archive_run_keeps_loose_when_not_blank_slate(tmp_path: Path) -> None:
    images = gen_images(tmp_path, [1, 2])
    rpt = report([image_scores(1, True), image_scores(2, False)])
    archive_run(tmp_path, images, rpt, manifest(), blank_slate=False)
    assert (tmp_path / "run-1-1.png").exists()
    assert (tmp_path / "run-1-2.png").exists()


def test_archive_run_no_passing_writes_no_zip(tmp_path: Path) -> None:
    images = gen_images(tmp_path, [1])
    rpt = report([image_scores(1, False)])
    result = archive_run(tmp_path, images, rpt, manifest())
    assert result.passing_zip is None
    assert not (tmp_path / PASSING_ZIP_NAME).exists()


def test_archive_run_writes_manifest_and_report(tmp_path: Path) -> None:
    images = gen_images(tmp_path, [1])
    rpt = report([image_scores(1, True)])
    archive_run(tmp_path, images, rpt, manifest())
    assert RunManifest.model_validate_json((tmp_path / "manifest.json").read_text()).run_id == "run-1"
    assert EvalReport.model_validate_json((tmp_path / "eval_report.json").read_text()).run_id == "run-1"


def test_zip_arcname_is_unique_per_image_not_basename(tmp_path: Path) -> None:
    # two passing images sharing a basename in different dirs must both survive
    (tmp_path / "d1").mkdir()
    (tmp_path / "d2").mkdir()
    for d in ("d1", "d2"):
        (tmp_path / d / "frame.png").write_bytes(b"x")
    images = [
        GeneratedImage(
            image_id="a", run_id="run-1", seed=1, path=str(tmp_path / "d1" / "frame.png"), width=8, height=8
        ),
        GeneratedImage(
            image_id="b", run_id="run-1", seed=2, path=str(tmp_path / "d2" / "frame.png"), width=8, height=8
        ),
    ]
    rpt = report([ImageScores(image_id="a", seed=1, passed=True), ImageScores(image_id="b", seed=2, passed=True)])
    archive_run(tmp_path, images, rpt, manifest(), blank_slate=False)
    import zipfile

    with zipfile.ZipFile(tmp_path / PASSING_ZIP_NAME) as zf:
        assert sorted(zf.namelist()) == ["a.png", "b.png"]  # both retained, no collision


def test_blank_slate_never_deletes_outside_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "shared_source.png"  # not owned by the run
    outside.write_bytes(b"precious")
    images = [
        GeneratedImage(image_id="x", run_id="run-1", seed=1, path=str(outside), width=8, height=8),
    ]
    rpt = report([ImageScores(image_id="x", seed=1, passed=False)])  # rejected -> would be deleted
    archive_run(run_dir, images, rpt, manifest(), blank_slate=True)
    assert outside.exists()  # a file outside run_dir is never unlinked


def test_report_image_without_matching_file_does_not_crash(tmp_path: Path) -> None:
    images = gen_images(tmp_path, [1])  # only run-1-1
    rpt = report([image_scores(1, True), image_scores(2, True)])  # run-1-2 has no file
    result = archive_run(tmp_path, images, rpt, manifest())
    assert result.n_passing == 1  # only the image with a real file is zipped


def test_the_result_reports_what_the_zip_holds(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    imgs = gen_images(run_dir, [1, 2])
    Path(imgs[1].path).unlink()  # judged passing, but its file is gone: nothing to zip
    first = archive_run(run_dir, imgs, report([image_scores(1, True), image_scores(2, True)]), manifest())
    assert first.n_passing == 1
    # Archived again with no passing image left on disk: the earlier zip still stands.
    again = archive_run(run_dir, [], report([image_scores(1, True), image_scores(2, True)]), manifest())
    assert again.passing_zip == run_dir / "passing_images.zip"
    assert again.n_passing == 1


def test_rearchive_drops_an_image_whose_verdict_flipped_to_rejected(tmp_path: Path) -> None:
    """The zip is the hand-off to training: an image rejected after it was zipped must leave it."""
    images = gen_images(tmp_path, [1, 2])
    archive_run(tmp_path, images, report([image_scores(1, True), image_scores(2, True)]), manifest())

    again = archive_run(tmp_path, images, report([image_scores(1, True), image_scores(2, False)]), manifest())
    with zipfile.ZipFile(tmp_path / PASSING_ZIP_NAME) as zf:
        assert zf.namelist() == ["run-1-1.png"]
    assert (again.n_passing, again.n_rejected) == (1, 1)
    assert not (tmp_path / f"{PASSING_ZIP_NAME}.tmp").exists()

    # every passing image rejected: no zip is left behind
    last = archive_run(tmp_path, images, report([image_scores(1, False), image_scores(2, False)]), manifest())
    assert last.passing_zip is None and last.n_passing == 0
    assert not (tmp_path / PASSING_ZIP_NAME).exists()


def test_a_zip_member_that_escapes_the_run_dir_is_not_written_there(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "escaped.png"
    absolute = tmp_path / "abs.png"
    zip_path = run_dir / PASSING_ZIP_NAME
    payload = b"zip-slip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../escaped.png", payload)
        zf.writestr(str(absolute), payload)
        zf.writestr("..", payload)
    with zipfile.ZipFile(zip_path) as zf:
        assert "../escaped.png" in zf.namelist()

    escaped = image_scores(1, None).model_copy(update={"image_id": "../escaped"})
    absolute_id = image_scores(2, None).model_copy(update={"image_id": str(absolute)[: -len(".png")]})
    dotdot = image_scores(3, None).model_copy(update={"image_id": ".."})
    outside_path = GeneratedImage(image_id="../escaped", run_id="run-1", seed=1, path=str(outside), width=1, height=1)
    archive_run(run_dir, [outside_path], report([escaped, absolute_id, dotdot]), manifest(), blank_slate=False)

    assert not outside.exists()
    assert not absolute.exists()
    # Nothing else was written, so the ".." member (which has no safe basename) was skipped.
    assert [p.name for p in tmp_path.iterdir()] == ["run"]
    assert sorted(p.name for p in run_dir.iterdir()) == sorted(
        ["abs.png", "escaped.png", "eval_report.json", "manifest.json", "reject_archive.json"]
    )
    assert (run_dir / "escaped.png").read_bytes() == payload
    assert (run_dir / "abs.png").read_bytes() == payload


def test_rearchive_restores_an_image_sent_back_to_pending(tmp_path: Path) -> None:
    images = gen_images(tmp_path, [1, 2])
    archive_run(tmp_path, images, report([image_scores(1, True), image_scores(2, True)]), manifest())
    assert not (tmp_path / "run-1-2.png").exists()  # zipped, then blank-slated

    again = archive_run(tmp_path, images, report([image_scores(1, True), image_scores(2, None)]), manifest())
    with zipfile.ZipFile(tmp_path / PASSING_ZIP_NAME) as zf:
        assert zf.namelist() == ["run-1-1.png"]
    assert (tmp_path / "run-1-2.png").read_bytes() == b"fake-png-bytes"
    assert (again.n_passing, again.n_pending) == (1, 1)
