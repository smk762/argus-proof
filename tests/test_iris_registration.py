"""argus-iris registration (argus-cortex 0.15): ``argus-proof serve`` registers its manifest at startup."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import httpx
import pytest
from argus_cortex.gpu_broker import BrokerCoordinator
from argus_cortex.gpu_wire import load_tenant_manifest
from fastapi.testclient import TestClient

from argus_proof import gpu
from argus_proof.server import create_app

MANIFEST = Path(__file__).resolve().parent.parent / "gpu-tenant.yaml"


class FakeIris:
    """argus-iris's tenant routes over an httpx mock transport, keeping every request it saw."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str, dict]] = []
        self.registered = threading.Event()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.seen.append((request.method, request.url.path, body))
        if request.method == "POST" and request.url.path == "/tenants":
            self.registered.set()
        return httpx.Response(200, json={})

    def coordinator(self) -> BrokerCoordinator:
        return BrokerCoordinator(
            "http://iris:8120",
            tenant=gpu.TENANT,
            token="caller-token",
            manifest=load_tenant_manifest(MANIFEST),
            async_transport=httpx.MockTransport(self),
            detect_container=False,
        )


def test_the_server_registers_with_iris_at_startup_and_signs_off_at_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    iris = FakeIris()
    monkeypatch.setattr(gpu, "default_coordinator", iris.coordinator)
    with TestClient(create_app(reports_dir=str(tmp_path))) as client:
        assert iris.registered.wait(5), "no registration at startup"
        assert client.get("/health").status_code == 200
    method, path, body = iris.seen[0]
    assert (method, path, body["tenant"]) == ("POST", "/tenants", "argus-proof")
    assert {kind["kind"] for kind in body["kinds"]} >= {"generate", "score"}
    assert body["residents"] == []
    assert ("DELETE", "/tenants/argus-proof") in [(m, p) for m, p, _ in iris.seen]
    assert not [p for _, p, _ in iris.seen if p.startswith("/leases")]


def test_a_read_only_server_stays_off_the_broker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    iris = FakeIris()
    monkeypatch.setattr(gpu, "default_coordinator", iris.coordinator)
    with TestClient(create_app(reports_dir=str(tmp_path), read_only=True)) as client:
        assert client.get("/health").status_code == 200
    assert iris.seen == []


def test_an_unreachable_broker_does_not_stop_the_server_starting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(MANIFEST.parent)
    monkeypatch.setenv("ARGUS_GPU_BROKER_URL", "http://127.0.0.1:1")
    with TestClient(create_app(reports_dir=str(tmp_path))) as client:
        assert client.get("/health").status_code == 200
