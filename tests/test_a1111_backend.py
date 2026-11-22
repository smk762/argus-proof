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


def test_a_run_without_a_vae_pins_the_checkpoints_own(models: Path, tmp_path: Path) -> None:
    # settings persist across requests (restore_afterwards=False), so leaving sd_vae out would keep whatever VAE an
    # earlier run pinned while the manifest records none
    transport = FakeTransport(make_png())
    result = _backend(models, transport).generate(_spec([7]), tmp_path / "run")
    assert result.manifest.vae is None
    assert transport.payloads[0]["override_settings"]["sd_vae"] == "None"


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


def test_guidance_is_refused_not_dropped(models: Path, tmp_path: Path) -> None:
    transport = FakeTransport(make_png(8, 8))
    spec = _spec([1])
    spec = spec.model_copy(update={"sampling": spec.sampling.model_copy(update={"guidance": 3.5})})
    with pytest.raises(BackendError, match="guidance"):
        _backend(models, transport).generate(spec, tmp_path / "out")
    assert transport.payloads == []


@pytest.mark.parametrize("field", ["prompt", "negative_prompt"])
@pytest.mark.parametrize("tag", ["<lora:other:1>", "< LyCo :x:0.5>", "<hypernet:h:1>"])
def test_a_prompt_carrying_an_extra_network_tag_is_refused(models: Path, tmp_path: Path, field, tag) -> None:  # noqa: ANN001
    spec = _spec([1]).model_copy(update={field: f"a photo {tag}"})
    transport = FakeTransport(make_png())
    with pytest.raises(BackendError, match="spec.loras"):
        _backend(models, transport).generate(spec, tmp_path / "run")
    assert transport.payloads == []


def test_repeated_seeds_are_refused_before_generating(models: Path, tmp_path: Path) -> None:
    transport = FakeTransport(make_png())
    with pytest.raises(BackendError, match="distinct seeds"):
        _backend(models, transport).generate(_spec([3, 3]), tmp_path / "run")
    assert transport.payloads == []


@pytest.mark.parametrize(
    ("name", "tag"),
    [
        ("char/subject.v2", "char/subject.v2"),
        ("char/subject.v2.safetensors", "char/subject.v2"),
        ("char\\subject.PT", "char/subject"),
        ("subject.ckpt", "subject"),
    ],
)
def test_lora_tag_strips_only_a_weight_extension(name: str, tag: str) -> None:
    from argus_proof.backends.a1111 import _lora_tag_name

    assert _lora_tag_name(name) == tag
