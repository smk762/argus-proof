from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_real_puppet(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reaches a real argus-puppet: a developer's PROOF_PUPPET_URL would otherwise turn on the pose pass."""
    monkeypatch.delenv("PROOF_PUPPET_URL", raising=False)
