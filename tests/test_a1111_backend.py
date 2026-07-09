from __future__ import annotations

import base64
from pathlib import Path

import pytest
from pnghelpers import make_png

from argus_proof.backends import A1111Backend, GenBackend, get_backend
from argus_proof.backends.base import BackendError, make_dir_resolver
from argus_proof.hashing import sha256_file
from argus_proof.models import LoRASpec, RunSpec, SamplingParams


def _spec(seeds: list[int]) -> RunSpec:
    return RunSpec(
        run_id="run-1",
        base_checkpoint="base.safetensors",
        loras=[LoRASpec(name="subject.safetensors", weight=0.8)],
        sampling=SamplingParams(
            sampler="DPM++ 2M", scheduler="karras", steps=20, cfg=7.0, clip_skip=2, width=64, height=64
        ),
        prompt="a photo of sks person",
        negative_prompt="blurry",
        seeds=seeds,
    )


@pytest.fixture
def models(tmp_path: Path) -> Path:
    (tmp_path / "base.safetensors").write_bytes(b"base")
    (tmp_path / "subject.safetensors").write_bytes(b"lora")
    return tmp_path


class FakeTransport:
    """Records txt2img payloads and returns a base64 PNG per request."""

    def __init__(self, png: bytes, *, data_uri: bool = False) -> None:
        b64 = base64.b64encode(png).decode()
        self._encoded = f"data:image/png;base64,{b64}" if data_uri else b64
        self.payloads: list[dict] = []

    def post_json(self, path: str, payload: dict) -> dict:
        assert path == "/sdapi/v1/txt2img"
        self.payloads.append(payload)
        return {"images": [self._encoded]}

    def get_json(self, path: str) -> dict:  # pragma: no cover - unused
        return {}

    def get_bytes(self, path: str) -> bytes:  # pragma: no cover - unused
        return b""


def _backend(models: Path, transport: FakeTransport) -> A1111Backend:
    return A1111Backend(resolve_model=make_dir_resolver(models), transport=transport, engine_version="a1111-1.9")


def test_get_backend_constructs_a1111() -> None:
    backend = get_backend("a1111", resolve_model=lambda n: Path(n))
    assert isinstance(backend, A1111Backend)
    assert isinstance(backend, GenBackend)
    assert backend.capabilities().name == "a1111"


def test_generate_decodes_image_per_seed_and_pins_models(models: Path, tmp_path: Path) -> None:
    result = _backend(models, FakeTransport(make_png())).generate(_spec([1, 2]), tmp_path / "run")
    assert [i.seed for i in result.images] == [1, 2]
    assert all(Path(i.path).is_file() for i in result.images)
    assert result.manifest.engine == "a1111" and result.manifest.engine_version == "a1111-1.9"
    assert result.manifest.base_checkpoint.sha256 == sha256_file(models / "base.safetensors")


def test_payload_carries_lora_syntax_seed_and_checkpoint_override(models: Path, tmp_path: Path) -> None:
    transport = FakeTransport(make_png())
    _backend(models, transport).generate(_spec([7]), tmp_path / "run")
    payload = transport.payloads[0]
    assert payload["seed"] == 7
    assert "<lora:subject:0.8>" in payload["prompt"]
    assert payload["sampler_name"] == "DPM++ 2M" and payload["scheduler"] == "karras"
    assert payload["override_settings"]["sd_model_checkpoint"] == "base.safetensors"
    # clip_skip is a setting, not a txt2img field, so it must ride override_settings
    assert payload["override_settings"]["CLIP_stop_at_last_layers"] == 2
    assert payload["negative_prompt"] == "blurry"


def test_payload_pins_vae_and_keeps_lora_subdirectory(tmp_path: Path) -> None:
    (tmp_path / "base.safetensors").write_bytes(b"base")
    (tmp_path / "char").mkdir()
    (tmp_path / "char" / "subject.safetensors").write_bytes(b"lora")
    (tmp_path / "sdxl_vae.safetensors").write_bytes(b"vae")
    from argus_proof.models import SamplingParams

    spec = RunSpec(
        run_id="run-1",
        base_checkpoint="base.safetensors",
        vae="sdxl_vae.safetensors",
        loras=[LoRASpec(name="char/subject.safetensors", weight=0.7)],
        sampling=SamplingParams(
            sampler="euler", scheduler="normal", steps=4, cfg=7.0, clip_skip=1, width=64, height=64
        ),
        prompt="sks",
        seeds=[1],
    )
    transport = FakeTransport(make_png())
    _backend(tmp_path, transport).generate(spec, tmp_path / "run")
    payload = transport.payloads[0]
    assert "<lora:char/subject:0.7>" in payload["prompt"]  # subdir kept, extension dropped
    assert payload["override_settings"]["sd_vae"] == "sdxl_vae.safetensors"


def test_data_uri_prefixed_image_is_decoded(models: Path, tmp_path: Path) -> None:
    result = _backend(models, FakeTransport(make_png(), data_uri=True)).generate(_spec([1]), tmp_path / "run")
    assert Path(result.images[0].path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_no_images_raises(models: Path, tmp_path: Path) -> None:
    class Empty(FakeTransport):
        def post_json(self, path: str, payload: dict) -> dict:
            return {"images": []}

    with pytest.raises(BackendError, match="no images"):
        _backend(models, Empty(make_png())).generate(_spec([1]), tmp_path / "run")
