from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from argus_proof.backends import DiffusersBackend, GenBackend, get_backend
from argus_proof.backends.base import make_dir_resolver
from argus_proof.hashing import sha256_file
from argus_proof.models import LoRASpec, ProgressEvent, RunSpec, SamplingParams


def _spec(seeds: list[int], loras=(("subject.safetensors", 0.8),)) -> RunSpec:  # noqa: ANN001
    return RunSpec(
        run_id="run-1",
        base_checkpoint="base.safetensors",
        loras=[LoRASpec(name=n, weight=w) for n, w in loras],
        sampling=SamplingParams(
            sampler="euler", scheduler="normal", steps=4, cfg=7.0, clip_skip=1, width=64, height=64
        ),
        prompt="a photo of sks person",
        seeds=seeds,
    )


@pytest.fixture
def models(tmp_path: Path) -> Path:
    (tmp_path / "base.safetensors").write_bytes(b"base-weights")
    (tmp_path / "subject.safetensors").write_bytes(b"lora-weights")
    return tmp_path


class FakeRenderer:
    """A renderer that records what it was asked to render and returns a solid image."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def render(self, spec: RunSpec, seed: int) -> Image.Image:
        self.calls.append((spec.run_id, seed))
        return Image.new("RGB", (spec.sampling.width, spec.sampling.height), (seed % 256, 0, 0))

    def engine_version(self) -> str:
        return "diffusers-fake"


def _backend(models: Path, renderer: FakeRenderer) -> DiffusersBackend:
    return DiffusersBackend(resolve_model=make_dir_resolver(models), renderer=renderer)


def test_get_backend_constructs_diffusers() -> None:
    backend = get_backend("diffusers", resolve_model=lambda n: Path(n))
    assert isinstance(backend, DiffusersBackend)
    assert isinstance(backend, GenBackend)
    assert backend.capabilities().name == "diffusers"


def test_generate_produces_one_image_per_seed(models: Path, tmp_path: Path) -> None:
    renderer = FakeRenderer()
    result = _backend(models, renderer).generate(_spec([1, 2, 3]), tmp_path / "run")
    assert [i.seed for i in result.images] == [1, 2, 3]
    assert renderer.calls == [("run-1", 1), ("run-1", 2), ("run-1", 3)]
    for img in result.images:
        assert Path(img.path).is_file() and img.width == 64 and img.height == 64


def test_manifest_pins_models_and_records_engine(models: Path, tmp_path: Path) -> None:
    result = _backend(models, FakeRenderer()).generate(_spec([1]), tmp_path / "run")
    m = result.manifest
    assert m.engine == "diffusers" and m.engine_version == "diffusers-fake"
    assert m.base_checkpoint.sha256 == sha256_file(models / "base.safetensors")
    assert m.loras[0].sha256 == sha256_file(models / "subject.safetensors") and m.loras[0].weight == 0.8
    # manifest.json written next to the images
    assert (tmp_path / "run" / "manifest.json").is_file()


def test_missing_model_fails_fast_with_error_event(tmp_path: Path) -> None:
    events: list[ProgressEvent] = []
    backend = DiffusersBackend(resolve_model=make_dir_resolver(tmp_path), renderer=FakeRenderer())  # empty dir
    with pytest.raises(Exception, match="cannot hash"):
        backend.generate(_spec([1]), tmp_path / "run", events.append)
    assert events[-1].type == "error"


def test_progress_events_streamed(models: Path, tmp_path: Path) -> None:
    events: list[ProgressEvent] = []
    _backend(models, FakeRenderer()).generate(_spec([1, 2]), tmp_path / "run", events.append)
    types = [e.type for e in events]
    assert types[0] == "start" and types[-1] == "done"
    assert types.count("image") == 2
