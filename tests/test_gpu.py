"""GPU leases (argus-iris, #79): what proof leases, where the work goes, when it yields, and what a refusal does."""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
from argus_cortex.gpu import GpuLease, GpuLeaseRefused, GpuLeaseTimeout, NullCoordinator
from argus_cortex.gpu_wire import load_tenant_manifest
from pnghelpers import make_png
from test_backends import FakeTransport, make_spec
from test_backends import models as models  # noqa: F401 - the fixture
from test_diffusers_backend import FakeRenderer

from argus_proof import gpu
from argus_proof.backends import ComfyUIBackend, DiffusersBackend
from argus_proof.backends.base import BackendError, make_dir_resolver
from argus_proof.backends.workflow import example_template
from argus_proof.models import ProgressEvent

MANIFEST = Path(__file__).resolve().parent.parent / "gpu-tenant.yaml"


class Recording:
    """A coordinator that grants at once and records each lease, its outcome, and when it was held.

    *grants* are the leases handed out in order (the last repeats); *raises* fails every lease.
    """

    name = "broker"

    def __init__(self, grants: list[GpuLease] | None = None, raises: Exception | None = None) -> None:
        self.grants = grants or [GpuLease(via="broker")]
        self.raises = raises
        self.calls: list[dict] = []
        self.exits: list[type[BaseException] | None] = []
        self.held = False

    @contextlib.contextmanager
    def lease(self, **kw):
        assert not self.held, "leases must not nest"
        self.calls.append(kw)
        if self.raises is not None:
            raise self.raises
        grant = self.grants[min(len(self.calls), len(self.grants)) - 1]
        self.held = True
        try:
            yield grant
        except BaseException as exc:
            self.exits.append(type(exc))
            raise
        else:
            self.exits.append(None)
        finally:
            self.held = False

    def alease(self, **kw):  # pragma: no cover - proof leases from threads only
        raise NotImplementedError


def comfyui(models: Path, transport: FakeTransport, coordinator) -> ComfyUIBackend:  # noqa: ANN001
    return ComfyUIBackend(
        workflow_template=example_template(),
        resolve_model=make_dir_resolver(models),
        transport=transport,
        poll_interval=0,
        timeout=0.05,
        gpu_coordinator=coordinator,
    )


# --- the manifest and the figures ----------------------------------------------------------------------------


def test_the_manifest_validates_and_matches_the_lease_figures() -> None:
    manifest = load_tenant_manifest(MANIFEST)
    assert manifest.tenant == gpu.TENANT
    assert manifest.residents == ()
    assert {k.kind for k in manifest.kinds} == {gpu.GENERATE.kind, gpu.GENERATE_LOCAL.kind}
    for footprint in (gpu.GENERATE, gpu.GENERATE_LOCAL):
        kind = manifest.kind_for(footprint.kind)
        assert kind is not None, footprint.kind
        assert (kind.target, kind.vram_mb, kind.model_mb) == (footprint.target, footprint.vram_mb, footprint.model_mb)
        assert kind.priority == gpu.DEFAULT_PRIORITY == "batch"
        assert kind.preemptible is footprint.preemptible is True


def test_no_coordinator_configured_means_no_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARGUS_GPU_COORDINATOR", raising=False)
    assert isinstance(gpu.default_coordinator(), NullCoordinator)


# --- comfyui -------------------------------------------------------------------------------------------------


def test_a_comfyui_run_holds_one_batch_lease_on_comfyui(models: Path, tmp_path: Path) -> None:
    coordinator = Recording()
    comfyui(models, FakeTransport(make_png()), coordinator).generate(make_spec([1, 2, 3]), tmp_path / "run")
    [call] = coordinator.calls
    assert call["caller"] == "run-42"
    assert (call["kind"], call["target"], call["priority"], call["preemptible"]) == (
        "generate",
        "comfyui",
        "batch",
        True,
    )
    assert (call["min_vram_mb"], call["model_mb"]) == (gpu.GENERATE.vram_mb, gpu.GENERATE.model_mb)
    assert call["constraints"] is None
    assert coordinator.exits == [None]


def test_a_missing_model_fails_before_any_lease(tmp_path: Path) -> None:
    coordinator = Recording()
    with pytest.raises(BackendError):
        comfyui(tmp_path, FakeTransport(make_png()), coordinator).generate(make_spec([1]), tmp_path / "run")
    assert coordinator.calls == []


def test_a_yield_releases_and_requeues_on_the_same_node(models: Path, tmp_path: Path) -> None:
    first = GpuLease(via="broker", node="dragon")
    second = GpuLease(via="broker", node="dragon")
    transport = FakeTransport(make_png())

    class YieldAfterFirstSeed(FakeTransport):
        def get_bytes(self, path: str) -> bytes:
            first._yield.set()  # the broker asks for the GPU back while seed 1 renders
            return transport.get_bytes(path)

    coordinator = Recording([first, second])
    result = comfyui(models, YieldAfterFirstSeed(make_png()), coordinator).generate(make_spec([1, 2, 3]), tmp_path)
    assert len(result.images) == 3
    assert len(coordinator.calls) == 2  # given back once, between seeds 1 and 2
    assert coordinator.calls[0]["constraints"] is None
    assert coordinator.calls[1]["constraints"].node == "dragon"  # pinned: the manifest hashed one node's models
    assert coordinator.exits == [None, None]


def test_the_run_goes_to_the_comfyui_the_grant_names(models: Path, tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    placed = FakeTransport(make_png(), version="0.9.9")
    configured = FakeTransport(make_png())
    backend = comfyui(models, configured, Recording([GpuLease(via="broker", base_url="http://dragon:8188/")]))
    seen: list[str] = []

    def transport_for(base_url):  # noqa: ANN001, ANN202
        seen.append(base_url)
        return placed

    monkeypatch.setattr(backend, "_transport_for", transport_for)
    result = backend.generate(make_spec([1, 2]), tmp_path)
    assert seen == ["http://dragon:8188/"] * 2
    assert len(placed.posts) == 2 and configured.posts == []
    assert result.manifest.engine_version == "0.9.9"  # the ComfyUI that rendered it


def test_a_grant_without_a_url_uses_the_configured_comfyui(models: Path, tmp_path: Path) -> None:
    configured = FakeTransport(make_png())
    backend = comfyui(models, configured, Recording())
    assert backend._transport_for(None) is configured
    assert backend._transport_for("http://dragon:8188") is backend._transport_for("http://dragon:8188/")
    with pytest.raises(BackendError, match="unusable ComfyUI URL"):
        backend._transport_for("file:///etc")


def test_a_failed_seed_is_reported_to_the_broker_as_failed(models: Path, tmp_path: Path) -> None:
    coordinator = Recording()
    with pytest.raises(BackendError):
        comfyui(models, FakeTransport(make_png(), ready=False), coordinator).generate(make_spec([1]), tmp_path)
    assert coordinator.exits == [BackendError]


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (GpuLeaseRefused("never fits"), "the GPU broker refused the lease: never fits"),
        (GpuLeaseTimeout("queue"), "no GPU lease in time: queue"),
    ],
)
def test_a_lease_failure_is_a_backend_error_that_says_so(models: Path, tmp_path: Path, exc, message) -> None:  # noqa: ANN001
    events: list[ProgressEvent] = []
    transport = FakeTransport(make_png())
    with pytest.raises(BackendError, match=message):
        comfyui(models, transport, Recording(raises=exc)).generate(make_spec([1]), tmp_path, events.append)
    assert transport.posts == []
    assert events[-1].type == "error" and events[-1].message == message


# --- diffusers -----------------------------------------------------------------------------------------------


def _diffusers(models: Path, coordinator) -> DiffusersBackend:  # noqa: ANN001
    return DiffusersBackend(
        resolve_model=make_dir_resolver(models), renderer=FakeRenderer(), gpu_coordinator=coordinator
    )


def test_a_diffusers_run_leases_in_process(models: Path, tmp_path: Path) -> None:
    coordinator = Recording()
    _diffusers(models, coordinator).generate(make_spec([1, 2]), tmp_path)
    [call] = coordinator.calls
    assert (call["kind"], call["target"], call["priority"]) == ("generate-local", "in-process", "batch")
    assert coordinator.exits == [None]


def test_a_diffusers_lease_failure_is_a_backend_error(models: Path, tmp_path: Path) -> None:
    with pytest.raises(BackendError, match="refused the lease"):
        _diffusers(models, Recording(raises=GpuLeaseRefused("no"))).generate(make_spec([1]), tmp_path)
