from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_real_puppet(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reaches a real argus-puppet: a developer's PROOF_PUPPET_URL would otherwise turn on the pose pass."""
    monkeypatch.delenv("PROOF_PUPPET_URL", raising=False)


@pytest.fixture(autouse=True)
def _no_real_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reaches a real argus-iris: since argus-cortex 0.15 a broker URL alone selects the broker."""
    for name in ("ARGUS_GPU_COORDINATOR", "ARGUS_GPU_BROKER_URL", "ARGUS_GPU_TENANT", "ARGUS_GPU_TENANT_MANIFEST"):
        monkeypatch.delenv(name, raising=False)
