from __future__ import annotations

import argus_proof


def test_version_is_exposed() -> None:
    assert isinstance(argus_proof.__version__, str)
    assert argus_proof.__version__
