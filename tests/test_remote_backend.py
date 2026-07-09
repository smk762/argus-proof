from __future__ import annotations

import base64
from pathlib import Path

import pytest
from pnghelpers import make_png

from argus_proof.backends import GenBackend, RemoteBackend, get_backend
from argus_proof.backends.base import BackendError
from argus_proof.models import ModelRef, RunManifest, RunSpec, SamplingParams


def _spec(seeds: list[int]) -> RunSpec:
    return RunSpec(
        run_id="run-1",
        base_checkpoint="base.safetensors",
        sampling=SamplingParams(
            sampler="euler", scheduler="normal", steps=4, cfg=7.0, clip_skip=1, width=64, height=64
        ),
        prompt="a photo of sks person",
        seeds=seeds,
    )


def _manifest_json(run_id: str = "run-1") -> dict:
    return RunManifest(
        run_id=run_id,
        base_checkpoint=ModelRef(name="base.safetensors", sha256="a" * 64),
        sampling=SamplingParams(
            sampler="euler", scheduler="normal", steps=4, cfg=7.0, clip_skip=1, width=64, height=64
        ),
        prompt="a photo of sks person",
        seeds=[1],
        engine="remote-svc",
        engine_version="svc-2.0",
    ).model_dump(mode="json")


class FakeTransport:
    """A hosted proof-gen endpoint that returns a manifest + base64 images."""

    def __init__(self, *, manifest: dict | None = None, images: list[dict] | None = None) -> None:
        png_b64 = base64.b64encode(make_png()).decode()
        self._manifest = manifest if manifest is not None else _manifest_json()
        self._images = images if images is not None else [{"seed": 1, "content_base64": png_b64}]
        self.posted: list[tuple[str, dict]] = []

    def post_json(self, path: str, payload: dict) -> dict:
        self.posted.append((path, payload))
        return {"manifest": self._manifest, "images": self._images}

    def get_json(self, path: str) -> dict:  # pragma: no cover - unused
        return {}

    def get_bytes(self, path: str) -> bytes:  # pragma: no cover - unused
        return b""


def test_get_backend_constructs_remote() -> None:
    backend = get_backend("remote", base_url="http://svc.example")
    assert isinstance(backend, RemoteBackend)
    assert isinstance(backend, GenBackend)
    assert backend.capabilities().name == "remote"


def test_api_key_becomes_bearer_header() -> None:
    backend = RemoteBackend("http://svc.example", api_key="secret")
    assert backend.transport.headers["Authorization"] == "Bearer secret"


def test_generate_posts_spec_and_trusts_service_manifest(tmp_path: Path) -> None:
    transport = FakeTransport()
    result = RemoteBackend("http://svc", transport=transport).generate(_spec([1]), tmp_path / "run")
    # the spec was POSTed to /generate
    assert transport.posted[0][0] == "/generate"
    assert transport.posted[0][1]["spec"]["run_id"] == "run-1"
    # the service's manifest is trusted (its engine/hashes), and images saved
    assert result.manifest.engine == "remote-svc" and result.manifest.engine_version == "svc-2.0"
    assert result.images[0].seed == 1 and Path(result.images[0].path).is_file()
    assert (tmp_path / "run" / "manifest.json").is_file()


def test_manifest_run_id_mismatch_raises(tmp_path: Path) -> None:
    transport = FakeTransport(manifest=_manifest_json(run_id="other"))
    with pytest.raises(BackendError, match="run_id"):
        RemoteBackend("http://svc", transport=transport).generate(_spec([1]), tmp_path / "run")


def test_invalid_manifest_raises(tmp_path: Path) -> None:
    bad = _manifest_json()
    bad["base_checkpoint"]["sha256"] = "not-a-sha256"  # violates the wire contract
    transport = FakeTransport(manifest=bad)
    with pytest.raises(BackendError, match="invalid manifest"):
        RemoteBackend("http://svc", transport=transport).generate(_spec([1]), tmp_path / "run")


def test_incompatible_major_manifest_raises(tmp_path: Path) -> None:
    bad = _manifest_json()
    bad["proof_version"] = "99.0"  # a major this build doesn't understand
    transport = FakeTransport(manifest=bad)
    with pytest.raises(BackendError, match="invalid manifest"):
        RemoteBackend("http://svc", transport=transport).generate(_spec([1]), tmp_path / "run")


def test_no_images_raises(tmp_path: Path) -> None:
    transport = FakeTransport(images=[])
    with pytest.raises(BackendError, match="no images"):
        RemoteBackend("http://svc", transport=transport).generate(_spec([1]), tmp_path / "run")


def test_image_missing_fields_raises(tmp_path: Path) -> None:
    transport = FakeTransport(images=[{"seed": 1}])  # no content_base64
    with pytest.raises(BackendError, match="content_base64"):
        RemoteBackend("http://svc", transport=transport).generate(_spec([1]), tmp_path / "run")


def test_malformed_image_item_raises_backend_error(tmp_path: Path) -> None:
    transport = FakeTransport(images=[None])  # not an object -> BackendError, not raw TypeError
    with pytest.raises(BackendError, match="must be an object"):
        RemoteBackend("http://svc", transport=transport).generate(_spec([1]), tmp_path / "run")


def test_manifest_not_matching_spec_raises(tmp_path: Path) -> None:
    bad = _manifest_json()
    bad["prompt"] = "a completely different prompt"  # service substituted the request
    transport = FakeTransport(manifest=bad)
    with pytest.raises(BackendError, match="does not match the requested spec"):
        RemoteBackend("http://svc", transport=transport).generate(_spec([1]), tmp_path / "run")


def test_unsafe_image_id_is_rejected(tmp_path: Path) -> None:
    png_b64 = base64.b64encode(make_png()).decode()
    transport = FakeTransport(images=[{"seed": 1, "image_id": "../../evil", "content_base64": png_b64}])
    with pytest.raises(BackendError, match="unsafe image_id"):
        RemoteBackend("http://svc", transport=transport).generate(_spec([1]), tmp_path / "run")


def test_duplicate_seed_images_do_not_overwrite(tmp_path: Path) -> None:
    png_b64 = base64.b64encode(make_png()).decode()
    transport = FakeTransport(
        images=[{"seed": 1, "content_base64": png_b64}, {"seed": 1, "content_base64": png_b64}]  # same seed, no id
    )
    result = RemoteBackend("http://svc", transport=transport).generate(_spec([1]), tmp_path / "run")
    paths = {i.path for i in result.images}
    assert len(paths) == 2 and all(Path(p).is_file() for p in paths)  # neither clobbered the other
