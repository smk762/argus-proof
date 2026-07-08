from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from argus_proof.cli import app
from argus_proof.models import AggregateScores, EvalReport, MetricScores, Verdict

runner = CliRunner()


def test_help_lists_verbs() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for verb in ("inspect", "run", "score", "report", "gate", "schema", "serve"):
        assert verb in result.output


def _write_report(path: Path, *, n_passed: int, n_groups: int, pass_rate: float) -> None:
    report = EvalReport(
        run_id="run-1",
        aggregate=AggregateScores(
            n_images=n_groups, n_groups=n_groups, n_passed=n_passed, pass_rate=pass_rate, means=MetricScores()
        ),
        verdict=Verdict(passed=pass_rate >= 0.75),
    )
    path.write_text(report.model_dump_json(), encoding="utf-8")


def test_gate_accepts_and_exits_zero(tmp_path: Path) -> None:
    report = tmp_path / "eval_report.json"
    _write_report(report, n_passed=90, n_groups=100, pass_rate=0.9)
    result = runner.invoke(app, ["gate", str(report), "--min-pass-rate", "0.75"])
    assert result.exit_code == 0
    assert "ACCEPTED" in result.output


def test_gate_rejects_and_exits_one(tmp_path: Path) -> None:
    report = tmp_path / "eval_report.json"
    _write_report(report, n_passed=40, n_groups=100, pass_rate=0.4)
    result = runner.invoke(app, ["gate", str(report), "--min-pass-rate", "0.75"])
    assert result.exit_code == 1
    assert "REJECTED" in result.output


def test_gate_unreadable_report_exits_two(tmp_path: Path) -> None:
    result = runner.invoke(app, ["gate", str(tmp_path / "nope.json")])
    assert result.exit_code == 2


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
