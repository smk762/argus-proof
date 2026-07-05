from __future__ import annotations

import pytest
from typer.testing import CliRunner

from argus_proof.cli import app

runner = CliRunner()


def test_help_lists_verbs() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for verb in ("inspect", "run", "score", "report", "serve"):
        assert verb in result.output


@pytest.mark.parametrize(
    "argv",
    [
        ["inspect", "/tmp/run"],
        ["run", "/tmp/lora.safetensors", "/tmp/manifest.jsonl"],
        ["score", "/tmp/run"],
        ["report", "/tmp/run"],
    ],
)
def test_stub_verbs_exit_2_and_point_at_tracking_issue(argv: list[str]) -> None:
    result = runner.invoke(app, argv)
    assert result.exit_code == 2
    assert "not implemented yet" in result.output
    assert "github.com/smk762/argus-studio" in result.output
