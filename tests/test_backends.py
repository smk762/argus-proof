from __future__ import annotations

from pathlib import Path

import pytest
from pnghelpers import make_png

from argus_proof.backends import ComfyUIBackend, GenBackend, GenResult, get_backend
from argus_proof.backends.base import BackendError, make_dir_resolver
from argus_proof.backends.workflow import example_template
from argus_proof.hashing import sha256_file
from argus_proof.models import (
    PROOF_VERSION,
    LoRASpec,
    ProgressEvent,
    RunManifest,
    RunSpec,
    SamplingParams,
)

# --------------------------------------------------------------------------
# registry + resolver
# --------------------------------------------------------------------------


def test_get_backend_constructs_comfyui() -> None:
    backend = get_backend("comfyui", workflow_template=example_template(), resolve_model=lambda n: Path(n))
    assert isinstance(backend, ComfyUIBackend)
    assert isinstance(backend, GenBackend)  # satisfies the protocol
    assert backend.capabilities().name == "comfyui"


def test_get_backend_unknown_raises() -> None:
    with pytest.raises(BackendError, match="unknown generation backend"):
        get_backend("midjourney")


def test_dir_resolver_finds_by_relative_path_and_basename(tmp_path: Path) -> None:
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "loras").mkdir()
    ckpt = tmp_path / "checkpoints" / "base.safetensors"
    lora = tmp_path / "loras" / "subject.safetensors"
    ckpt.write_bytes(b"ckpt")
    lora.write_bytes(b"lora")
    resolve = make_dir_resolver(tmp_path)
    assert resolve("checkpoints/base.safetensors") == ckpt  # relative path
    assert resolve("subject.safetensors") == lora  # basename fallback


def test_dir_resolver_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        make_dir_resolver(tmp_path)("nope.safetensors")


# --------------------------------------------------------------------------
# ComfyUI backend end-to-end (fake transport)
# --------------------------------------------------------------------------


class FakeTransport:
    """A ComfyUI HTTP stand-in: accepts prompts, reports them done, serves PNGs."""

    def __init__(self, png: bytes, version: str = "0.3.4", ready: bool = True) -> None:
        self.png = png
        self.version = version
        self.ready = ready
        self.posts: list[dict] = []
        self._n = 0

    def post_json(self, path: str, payload: dict) -> dict:
        assert path == "/prompt"
        self.posts.append(payload)
        self._n += 1
        return {"prompt_id": f"pid-{self._n}", "node_errors": {}}

    def get_json(self, path: str) -> dict:
        if path == "/system_stats":
            return {"system": {"comfyui_version": self.version}}
        if path.startswith("/history/"):
            pid = path.rsplit("/", 1)[1]
            if not self.ready:
                return {}
            return {
                pid: {
                    "status": {"status_str": "success", "completed": True},
                    "outputs": {"9": {"images": [{"filename": f"{pid}.png", "subfolder": "", "type": "output"}]}},
                }
            }
        raise AssertionError(f"unexpected GET {path}")

    def get_bytes(self, path: str) -> bytes:
        assert path.startswith("/view?")
        return self.png


@pytest.fixture
def models(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    (root / "checkpoints").mkdir(parents=True)
    (root / "loras").mkdir(parents=True)
    (root / "checkpoints" / "base.safetensors").write_bytes(b"the-checkpoint-bytes")
    (root / "loras" / "subject.safetensors").write_bytes(b"the-lora-bytes")
    return root


def make_spec(seeds: list[int]) -> RunSpec:
    return RunSpec(
        run_id="run-42",
        base_checkpoint="base.safetensors",
        loras=[LoRASpec(name="subject.safetensors", weight=0.75)],
        sampling=SamplingParams(
            sampler="dpmpp_2m", scheduler="karras", steps=25, cfg=7.0, clip_skip=2, width=64, height=48
        ),
        prompt="a photo of sks person",
        negative_prompt="blurry",
        seeds=seeds,
        source_manifest="/exports/run/manifest.jsonl",
        source_manifest_version="2.0",
        training_run_id="forge-xyz",
    )


def build_backend(models: Path, transport: FakeTransport) -> ComfyUIBackend:
    return ComfyUIBackend(
        workflow_template=example_template(),
        resolve_model=make_dir_resolver(models),
        transport=transport,
        poll_interval=0,
    )


def test_generate_produces_image_per_seed(models: Path, tmp_path: Path) -> None:
    png = make_png(width=64, height=48, text={"prompt": '{"seed": 1}'})
    backend = build_backend(models, FakeTransport(png))
    out = tmp_path / "run"

    result = backend.generate(make_spec([1, 2, 3]), out)

    assert isinstance(result, GenResult)
    assert len(result.images) == 3
    for img in result.images:
        assert Path(img.path).is_file()
        assert (img.width, img.height) == (64, 48)
        assert img.pnginfo["prompt"] == '{"seed": 1}'
        assert img.run_id == "run-42"
    assert {img.seed for img in result.images} == {1, 2, 3}
    assert {img.image_id for img in result.images} == {"run-42-1", "run-42-2", "run-42-3"}


def test_manifest_pins_files_by_sha256_and_reconstructs_run(models: Path, tmp_path: Path) -> None:
    backend = build_backend(models, FakeTransport(make_png()))
    out = tmp_path / "run"

    manifest = backend.generate(make_spec([7, 8]), out).manifest

    assert manifest.proof_version == PROOF_VERSION
    assert manifest.engine == "comfyui"
    assert manifest.engine_version == "0.3.4"
    assert manifest.base_checkpoint.sha256 == sha256_file(models / "checkpoints" / "base.safetensors")
    assert manifest.loras[0].sha256 == sha256_file(models / "loras" / "subject.safetensors")
    assert manifest.loras[0].weight == 0.75
    assert manifest.seeds == [7, 8]
    assert manifest.training_run_id == "forge-xyz"

    # The manifest is written to the run dir and round-trips exactly.
    written = RunManifest.model_validate_json((out / "manifest.json").read_text())
    assert written == manifest


def test_seed_is_injected_into_each_submitted_graph(models: Path, tmp_path: Path) -> None:
    transport = FakeTransport(make_png())
    build_backend(models, transport).generate(make_spec([100, 200]), tmp_path / "run")

    submitted_seeds = [
        next(n["inputs"]["seed"] for n in post["prompt"].values() if n["class_type"] == "KSampler")
        for post in transport.posts
    ]
    assert submitted_seeds == [100, 200]
    assert all(post["client_id"] == "argus-proof" for post in transport.posts)


def test_progress_events_streamed(models: Path, tmp_path: Path) -> None:
    events: list[ProgressEvent] = []
    build_backend(models, FakeTransport(make_png())).generate(make_spec([1, 2]), tmp_path / "run", events.append)

    types = [e.type for e in events]
    assert types[0] == "start"
    assert types[-1] == "done"
    assert types.count("image") == 2
    assert events[0].total == 2
    assert events[-1].completed == 2


def test_node_errors_raise_and_emit_error_event(models: Path, tmp_path: Path) -> None:
    class Rejecting(FakeTransport):
        def post_json(self, path: str, payload: dict) -> dict:
            return {"prompt_id": "pid-1", "node_errors": {"3": "bad sampler"}}

    events: list[ProgressEvent] = []
    backend = build_backend(models, Rejecting(make_png()))
    with pytest.raises(BackendError, match="rejected the workflow"):
        backend.generate(make_spec([1]), tmp_path / "run", events.append)
    assert events[-1].type == "error"


def test_run_that_never_finishes_times_out(models: Path, tmp_path: Path) -> None:
    backend = ComfyUIBackend(
        workflow_template=example_template(),
        resolve_model=make_dir_resolver(models),
        transport=FakeTransport(make_png(), ready=False),
        poll_interval=0,
        timeout=0,
    )
    with pytest.raises(BackendError, match="did not finish"):
        backend.generate(make_spec([1]), tmp_path / "run")


def test_missing_model_file_raises(tmp_path: Path) -> None:
    backend = ComfyUIBackend(
        workflow_template=example_template(),
        resolve_model=make_dir_resolver(tmp_path),  # empty dir
        transport=FakeTransport(make_png()),
        poll_interval=0,
    )
    with pytest.raises(BackendError, match="cannot hash"):
        backend.generate(make_spec([1]), tmp_path / "run")
