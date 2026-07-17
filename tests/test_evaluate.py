from __future__ import annotations

from pathlib import Path

import pytest
from fakebackend import FakeBackend, save_png

from argus_proof.backends import BackendError, ComfyUIBackend
from argus_proof.evaluate import (
    EvaluateError,
    backend_from_env,
    discover_images,
    load_manifest,
    reference_images,
    runs_root,
    score_run_dir,
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
