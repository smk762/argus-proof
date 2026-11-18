from __future__ import annotations

from pathlib import Path

import pytest
from fakebackend import FakeBackend, save_png

from argus_proof.backends import BackendError, ComfyUIBackend
from argus_proof.evaluate import (
    EvaluateError,
    backend_from_env,
    check_held_out,
    discover_images,
    load_manifest,
    reference_images,
    runs_root,
    score_run_dir,
    training_images,
)
from argus_proof.models import LoRASpec, RunSpec, SamplingParams

SAMPLING = SamplingParams(sampler="euler", scheduler="normal", steps=8, cfg=5.0, width=16, height=16)


def _generated_run(tmp_path: Path, run_id: str = "run-1", seeds: list[int] | None = None) -> Path:
    spec = RunSpec(
        run_id=run_id,
        base_checkpoint="sdxl.safetensors",
        loras=[LoRASpec(name="subject.safetensors")],
        sampling=SAMPLING,
        prompt="a photo of sks person",
        seeds=seeds or [1, 2],
    )
    run_dir = tmp_path / run_id
    FakeBackend().generate(spec, run_dir)
    return run_dir


# -- env config --------------------------------------------------------------


def test_runs_root_precedence(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.delenv("ARGUS_PROOF_RUNS_DIR", raising=False)
    assert runs_root() == Path("runs")
    monkeypatch.setenv("ARGUS_PROOF_RUNS_DIR", str(tmp_path))
    assert runs_root() == tmp_path
    assert runs_root(tmp_path / "explicit") == tmp_path / "explicit"


def test_backend_from_env_comfyui_default(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.delenv("PROOF_BACKEND", raising=False)
    monkeypatch.setenv("PROOF_MODELS_DIR", str(tmp_path))
    backend = backend_from_env()
    assert isinstance(backend, ComfyUIBackend)


def test_backend_from_env_needs_models_dir(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("PROOF_MODELS_DIR", raising=False)
    with pytest.raises(EvaluateError, match="PROOF_MODELS_DIR"):
        backend_from_env("comfyui")


def test_backend_from_env_remote_needs_url(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("PROOF_REMOTE_URL", raising=False)
    with pytest.raises(EvaluateError, match="PROOF_REMOTE_URL"):
        backend_from_env("remote")


def test_backend_from_env_unknown_name(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.setenv("PROOF_MODELS_DIR", str(tmp_path))
    with pytest.raises(BackendError, match="unknown generation backend"):
        backend_from_env("imaginary")


def test_models_resolver_searches_every_pathsep_root(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    import os

    from argus_proof.evaluate import models_resolver

    main, extra = tmp_path / "main", tmp_path / "extra"
    (main / "loras").mkdir(parents=True)
    (main / "loras" / "subject.safetensors").write_bytes(b"l")
    (extra / "checkpoints").mkdir(parents=True)
    (extra / "checkpoints" / "base.safetensors").write_bytes(b"c")
    monkeypatch.setenv("PROOF_MODELS_DIR", f"{main}{os.pathsep}{extra}")

    resolve = models_resolver()
    assert resolve("subject.safetensors") == main / "loras" / "subject.safetensors"
    assert resolve("base.safetensors") == extra / "checkpoints" / "base.safetensors"
    with pytest.raises(FileNotFoundError):
        resolve("missing.safetensors")


# -- run-dir loading ---------------------------------------------------------


def test_load_manifest_and_discover_images(tmp_path: Path) -> None:
    run_dir = _generated_run(tmp_path)
    manifest = load_manifest(run_dir)
    images = discover_images(run_dir, manifest)
    assert [img.seed for img in images] == [1, 2]
    assert all(img.width == 16 for img in images)


def test_discover_images_ignores_foreign_files(tmp_path: Path) -> None:
    run_dir = _generated_run(tmp_path)
    save_png(run_dir / "unrelated.png")  # doesn't match <run_id>-<seed>
    manifest = load_manifest(run_dir)
    assert len(discover_images(run_dir, manifest)) == 2


def test_load_manifest_missing_is_evaluate_error(tmp_path: Path) -> None:
    with pytest.raises(EvaluateError, match="manifest.json"):
        load_manifest(tmp_path)


def test_discover_images_empty_run_is_evaluate_error(tmp_path: Path) -> None:
    run_dir = _generated_run(tmp_path)
    for png in run_dir.glob("*.png"):
        png.unlink()
    with pytest.raises(EvaluateError, match="no images"):
        discover_images(run_dir, load_manifest(run_dir))


def test_reference_images_filters_to_images(tmp_path: Path) -> None:
    save_png(tmp_path / "refs" / "a.png")
    (tmp_path / "refs" / "notes.txt").write_text("not an image", encoding="utf-8")
    refs = reference_images(tmp_path / "refs")
    assert [p.name for p in refs] == ["a.png"]


# -- scoring -----------------------------------------------------------------


def test_score_run_dir_produces_report(tmp_path: Path) -> None:
    run_dir = _generated_run(tmp_path)
    report = score_run_dir(run_dir)
    assert report.run_id == "run-1"
    assert {img.seed for img in report.images} == {1, 2}
    # phash dedup/diversity are availability-guarded; either way every image has a group label
    assert all(img.duplicate_group is not None for img in report.images)


# -- identity references are held out from training (CLAUDE.md must-never) -----


def _photo(path: Path, seed: int, size: tuple[int, int] = (320, 240)) -> Path:
    """A structured stand-in for a photo: seeded shapes, so distinct seeds hash far apart under pHash."""
    import random

    from PIL import Image, ImageDraw, ImageFilter

    rng = random.Random(seed)
    im = Image.new("RGB", size, tuple(rng.randrange(256) for _ in range(3)))
    draw = ImageDraw.Draw(im)
    for _ in range(25):
        x, y = rng.randrange(size[0]), rng.randrange(size[1])
        draw.ellipse(
            (x, y, x + rng.randrange(20, 140), y + rng.randrange(20, 140)),
            fill=tuple(rng.randrange(256) for _ in range(3)),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    im.filter(ImageFilter.GaussianBlur(2)).save(path)
    return path


def _export(tmp_path: Path) -> Path:
    export = tmp_path / "export"
    _photo(export / "train" / "a.png", seed=1)
    _photo(export / "references" / "held_out.png", seed=2)
    elsewhere = _photo(tmp_path / "dataset" / "b.png", seed=3)  # a manifest-only row pointing at the dataset
    (export / "manifest.jsonl").write_text(
        f'{{"exported_path": "train/a.png", "abs_path": "{elsewhere}"}}\nnot json\n', encoding="utf-8"
    )
    return export


def test_training_images_skip_references_and_follow_the_manifest(tmp_path: Path) -> None:
    export = _export(tmp_path)
    names = {p.name for p in training_images(export)}
    assert names == {"a.png", "b.png"}


def test_held_out_references_pass(tmp_path: Path) -> None:
    export = _export(tmp_path)
    check_held_out(reference_images(export / "references"), export)
    check_held_out([], export)


def test_a_reference_that_is_a_training_image_is_refused(tmp_path: Path) -> None:
    export = _export(tmp_path)
    copy = tmp_path / "refs" / "renamed.png"  # the same bytes under another name
    copy.parent.mkdir()
    copy.write_bytes((tmp_path / "dataset" / "b.png").read_bytes())
    with pytest.raises(EvaluateError, match=r"1 identity reference\(s\) are training images.*renamed.png.*same bytes"):
        check_held_out([copy], export)


def test_scoring_refuses_overlapping_references(tmp_path: Path) -> None:
    export = _export(tmp_path)
    run_dir = _generated_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = load_manifest(run_dir).model_copy(update={"source_manifest": str(export / "manifest.jsonl")})
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    with pytest.raises(EvaluateError, match="training images"):
        score_run_dir(run_dir, references=[export / "train" / "a.png"])
    report = score_run_dir(run_dir, references=reference_images(export / "references"))
    assert report.run_id == "run-1"
    [check] = [p for p in report.scorers if p.name == "held-out-references"]
    assert check.model == "sha256+phash<=6"  # the report says which check ran


# -- near-duplicates (#87) ------------------------------------------------------


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(lambda im: im.resize((im.width // 2, im.height // 2)), id="downscaled"),
        pytest.param(lambda im: im.resize((im.width * 2, im.height * 2)), id="upscaled"),
        pytest.param(lambda im: im, id="re-encoded"),
    ],
)
def test_a_resized_or_reencoded_training_image_is_refused(tmp_path: Path, transform) -> None:  # noqa: ANN001
    from PIL import Image

    export = _export(tmp_path)
    ref = tmp_path / "refs" / "copy.jpg"
    ref.parent.mkdir()
    with Image.open(export / "train" / "a.png") as im:
        transform(im.convert("RGB")).save(ref, "JPEG", quality=60)
    with pytest.raises(EvaluateError, match=r"copy.jpg = .*a.png \(near-duplicate, \d bits\)"):
        check_held_out([ref], export)


def test_a_different_photo_passes_the_near_duplicate_check(tmp_path: Path) -> None:
    export = _export(tmp_path)
    others = [_photo(tmp_path / "refs" / f"r{seed}.png", seed=seed) for seed in range(10, 20)]
    assert check_held_out(others, export) == "sha256+phash<=6"


def test_without_the_score_extra_only_exact_copies_are_checked(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    import argus_proof.evaluate as evaluate

    monkeypatch.setattr(evaluate, "near_duplicates_checked", lambda: False)
    export = _export(tmp_path)
    ref = tmp_path / "refs" / "copy.jpg"
    ref.parent.mkdir()
    from PIL import Image

    with Image.open(export / "train" / "a.png") as im:
        im.convert("RGB").save(ref, "JPEG", quality=60)
    assert check_held_out([ref], export) == "sha256"  # not caught, and the report will say so


def test_an_unreadable_training_file_is_skipped(tmp_path: Path) -> None:
    export = _export(tmp_path)
    (export / "train" / "broken.png").write_bytes(b"not an image")
    assert check_held_out(reference_images(export / "references"), export) == "sha256+phash<=6"


def test_the_report_says_when_no_check_could_run(tmp_path: Path) -> None:
    export = _export(tmp_path)
    run_dir = _generated_run(tmp_path)
    report = score_run_dir(run_dir, references=reference_images(export / "references"))  # no source_manifest
    [check] = [p for p in report.scorers if p.name == "held-out-references"]
    assert check.model == "not checked: no training export"
    assert not any(p.name == "held-out-references" for p in score_run_dir(run_dir).scorers)  # no references


def test_discover_images_is_in_seed_order_not_filename_order(tmp_path: Path) -> None:
    run_dir = _generated_run(tmp_path, seeds=[10, 2, 1])
    save_png(run_dir / "run-1-1-1.png")  # a second image for seed 1
    images = discover_images(run_dir, load_manifest(run_dir))
    assert [img.image_id for img in images] == ["run-1-10", "run-1-2", "run-1-1", "run-1-1-1"]


def test_one_stem_with_two_extensions_is_refused(tmp_path: Path) -> None:
    run_dir = _generated_run(tmp_path)
    save_png(run_dir / "run-1-1.jpg")
    with pytest.raises(EvaluateError, match="one image_id, two images"):
        discover_images(run_dir, load_manifest(run_dir))


def test_training_images_list_one_file_once_and_skip_non_image_rows(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    export = tmp_path / "export"
    a = _photo(export / "train" / "a.png", seed=1)
    (export / "train" / "a.txt").write_text("a caption", encoding="utf-8")
    (export / "manifest.jsonl").write_text(
        f'{{"exported_path": "train/a.png", "exported_abs_path": "{a.resolve()}"}}\n'
        '{"exported_path": "train/a.txt"}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert training_images(Path("export")) == [a.resolve()]  # relative export dir, absolute manifest path


def test_a_decompression_bomb_is_an_unreadable_image(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    from PIL import Image

    from argus_proof.evaluate import _phash_bits

    path = _photo(tmp_path / "big.png", seed=1)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)  # 320x240 is now > 2x the limit
    assert _phash_bits(path) is None
