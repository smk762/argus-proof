from __future__ import annotations

import pytest

# Every service URL proof resolves through argus-cortex's peers: its own variable, then the suite-wide one.
PEER_VARIABLES = (
    "PROOF_NUDENET_URL",
    "PROOF_PUPPET_URL",
    "PROOF_ECHO_URL",
    "COMFYUI_BASE_URL",
    "ARGUS_NUDENET_URL",
    "ARGUS_PUPPET_URL",
    "ARGUS_XTTS_URL",
    "ARGUS_TTS_URL",
    "ARGUS_COMFYUI_URL",
)


@pytest.fixture(autouse=True)
def _no_suite_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's suite file (~/.config/argus/suite.env) is never read: it would configure peers and the broker."""
    monkeypatch.setenv("ARGUS_ENV_FILE", "off")


@pytest.fixture(autouse=True)
def _no_real_peers(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reaches a real service: a developer's PROOF_PUPPET_URL or ARGUS_NUDENET_URL would otherwise turn on
    the pose pass or the safety scorer."""
    for name in PEER_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_real_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reaches a real argus-iris: since argus-cortex 0.15 a broker URL alone selects the broker."""
    for name in ("ARGUS_GPU_COORDINATOR", "ARGUS_GPU_BROKER_URL", "ARGUS_GPU_TENANT", "ARGUS_GPU_TENANT_MANIFEST"):
        monkeypatch.delenv(name, raising=False)
