from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from argus_proof import __version__
from argus_proof.server import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(cors=True))


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["service"] == "argus-proof"
    assert body["version"] == __version__


def test_cors_header(client: TestClient) -> None:
    resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
    # allow_credentials=True makes CORSMiddleware echo the origin rather than "*"
    assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"
