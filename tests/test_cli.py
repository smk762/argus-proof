from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from argus_proof.cli import app
from argus_proof.models import AggregateScores, EvalReport, MetricScores, Verdict

runner = CliRunner()


def test_help_lists_verbs() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for verb in ("inspect", "run", "score", "report", "gate", "recommend", "experiment", "explore", "schema", "serve"):
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


def test_gate_invalid_confidence_exits_two_cleanly(tmp_path: Path) -> None:
    report = tmp_path / "eval_report.json"
    _write_report(report, n_passed=90, n_groups=100, pass_rate=0.9)
    result = runner.invoke(app, ["gate", str(report), "--confidence", "1.0"])
    assert result.exit_code == 2  # clean exit, not an unhandled traceback


def test_recommend_prints_routed_suggestions(tmp_path: Path) -> None:
    from argus_proof.models import AggregateScores, EvalReport, MetricScores, Verdict

    report = tmp_path / "eval_report.json"
    low_identity = EvalReport(
        run_id="run-1",
        aggregate=AggregateScores(n_images=1, n_groups=1, n_passed=0, pass_rate=0.0, means=MetricScores(identity=0.2)),
        verdict=Verdict(passed=False),
    )
    report.write_text(low_identity.model_dump_json(), encoding="utf-8")
    result = runner.invoke(app, ["recommend", str(report)])
    assert result.exit_code == 0
    assert "forge" in result.output  # low identity routes to forge


def test_recommend_unreadable_exits_two(tmp_path: Path) -> None:
    result = runner.invoke(app, ["recommend", str(tmp_path / "nope.json")])
    assert result.exit_code == 2
    assert "cannot read EvalReport" in result.output


def test_recommend_healthy_reports_nothing_actionable(tmp_path: Path) -> None:
    from argus_proof.models import AggregateScores, EvalReport, MetricScores, Verdict

    report = tmp_path / "eval_report.json"
    healthy = EvalReport(
        run_id="run-1",
        aggregate=AggregateScores(
            n_images=1,
            n_groups=1,
            n_passed=1,
            pass_rate=1.0,
            means=MetricScores(identity=0.9, clip_score=0.9, aesthetic=0.9),
            diversity=0.9,
        ),
        verdict=Verdict(passed=True),
    )
    report.write_text(healthy.model_dump_json(), encoding="utf-8")
    result = runner.invoke(app, ["recommend", str(report)])
    assert result.exit_code == 0
    assert "nothing actionable" in result.output


def _write_matrix(path: Path, *, seconds_per_image: float = 6.0) -> None:
    matrix = {
        "base_checkpoints": ["sdxl.safetensors"],
        "step_configs": [
            {
                "name": "quality",
                "sampling": {
                    "sampler": "dpmpp_2m",
                    "scheduler": "karras",
                    "steps": 30,
                    "cfg": 7.0,
                    "clip_skip": 2,
                    "width": 1024,
                    "height": 1024,
                },
            }
        ],
        "lora_checkpoints": ["e10.safetensors"],
        "lora_weights": [1.0],
        "seeds": [1, 2],
        "seconds_per_image": seconds_per_image,
    }
    path.write_text(json.dumps(matrix), encoding="utf-8")


def _export_with_prompt(tmp_path: Path) -> Path:
    export = tmp_path / "export"
    export.mkdir()
    (export / "img.txt").write_text("a photo of sks person", encoding="utf-8")
    return export


def test_experiment_reports_cost_estimate(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    _write_matrix(matrix)
    export = _export_with_prompt(tmp_path)
    result = runner.invoke(app, ["experiment", str(matrix), "--export", str(export)])
    assert result.exit_code == 0
    assert "1 cells" in result.output
    assert "GPU-hours" in result.output


def test_experiment_over_budget_exits_one(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    _write_matrix(matrix, seconds_per_image=1000.0)
    export = _export_with_prompt(tmp_path)
    result = runner.invoke(app, ["experiment", str(matrix), "--export", str(export), "--max-gpu-hours", "0.001"])
    assert result.exit_code == 1
    assert "cannot expand experiment" in result.output


def test_experiment_unreadable_matrix_exits_two(tmp_path: Path) -> None:
    export = _export_with_prompt(tmp_path)
    result = runner.invoke(app, ["experiment", str(tmp_path / "nope.json"), "--export", str(export)])
    assert result.exit_code == 2


def test_explore_without_fiftyone_exits_two(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    # When the [fiftyone] extra is absent, the verb should exit 2 with an install hint
    # (not crash). Force is_available() False so the test is deterministic in CI.
    import argus_proof.explore as fo_explore

    monkeypatch.setattr(fo_explore, "is_available", lambda: False)
    result = runner.invoke(app, ["explore", str(tmp_path / "r.json"), "--images", str(tmp_path)])
    assert result.exit_code == 2
    assert "fiftyone extra" in result.output


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
