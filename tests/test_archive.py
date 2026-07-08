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
