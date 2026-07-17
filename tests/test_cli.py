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


# ---------------------------------------------------------------------------
# run / score / report / inspect — the executing eval verbs
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_backend(monkeypatch):  # noqa: ANN201
    """Route the run verb's backend construction to the in-memory FakeBackend."""
    from fakebackend import FakeBackend

    import argus_proof.evaluate as evaluate

    backend = FakeBackend()
    monkeypatch.setattr(evaluate, "backend_from_env", lambda *a, **k: backend)
    return backend


def _run_grid(tmp_path: Path) -> tuple[Path, Path]:
    """Invoke `run` against a one-prompt export; return (runs_root, run_dir)."""
    export = _export_with_prompt(tmp_path)
    out = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "run",
            "subject.safetensors",
            str(export),
            "--checkpoint",
            "sdxl.safetensors",
            "--out",
            str(out),
            "--seed",
            "1",
            "--seed",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    run_dirs = [p for p in out.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    return out, run_dirs[0]


def test_run_generates_grid_and_manifest(tmp_path: Path, fake_backend) -> None:  # noqa: ANN001
    _, run_dir = _run_grid(tmp_path)
    assert (run_dir / "manifest.json").is_file()
    assert len(list(run_dir.glob("*.png"))) == 2
    (spec,) = fake_backend.generated
    assert spec.loras[0].name == "subject.safetensors"
    assert spec.prompt == "a photo of sks person"


def test_run_explicit_prompt_skips_export_captions(tmp_path: Path, fake_backend) -> None:  # noqa: ANN001
    export = _export_with_prompt(tmp_path)
    result = runner.invoke(
        app,
        [
            "run",
            "l.safetensors",
            str(export),
            "-c",
            "ckpt.safetensors",
            "--out",
            str(tmp_path / "r"),
            "--prompt",
            "override",
        ],
    )
    assert result.exit_code == 0, result.output
    assert fake_backend.generated[0].prompt == "override"


def test_run_empty_export_exits_one(tmp_path: Path, fake_backend) -> None:  # noqa: ANN001
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(
        app, ["run", "l.safetensors", str(empty), "-c", "c.safetensors", "--out", str(tmp_path / "r")]
    )
    assert result.exit_code == 1
    assert "no prompts" in result.output


def test_run_bad_prefix_exits_two(tmp_path: Path, fake_backend) -> None:  # noqa: ANN001
    export = _export_with_prompt(tmp_path)
    result = runner.invoke(
        app, ["run", "l", str(export), "-c", "c", "--out", str(tmp_path / "r"), "--run-prefix", "../evil"]
    )
    assert result.exit_code == 2
    assert "run-prefix" in result.output


def test_score_then_report_round_trip(tmp_path: Path, fake_backend) -> None:  # noqa: ANN001
    _, run_dir = _run_grid(tmp_path)
    reports = tmp_path / "reports"

    scored = runner.invoke(app, ["score", str(run_dir), "--reports-dir", str(reports)])
    assert scored.exit_code == 0, scored.output
    assert "pass rate" in scored.output
    assert len(list(reports.glob("*.json"))) == 1

    listed = runner.invoke(app, ["report", "--reports-dir", str(reports)])
    assert listed.exit_code == 0
    assert run_dir.name in listed.output

    detail = runner.invoke(app, ["report", run_dir.name, "--reports-dir", str(reports), "--json"])
    assert detail.exit_code == 0
    assert json.loads(detail.output)["run_id"] == run_dir.name


def test_score_missing_run_dir_exits_one(tmp_path: Path) -> None:
    result = runner.invoke(app, ["score", str(tmp_path / "nope")])
    assert result.exit_code == 1
    assert "manifest.json" in result.output


def test_report_unknown_run_exits_two(tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", "ghost", "--reports-dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "ghost" in result.output


def test_inspect_run_and_export_dirs(tmp_path: Path, fake_backend) -> None:  # noqa: ANN001
    _, run_dir = _run_grid(tmp_path)
    inspected = runner.invoke(app, ["inspect", str(run_dir)])
    assert inspected.exit_code == 0, inspected.output
    assert "sdxl.safetensors" in inspected.output
    assert "2 image(s)" in inspected.output

    export = tmp_path / "export"  # created by _run_grid
    inspected = runner.invoke(app, ["inspect", str(export)])
    assert inspected.exit_code == 0
    assert "1 base prompt(s)" in inspected.output


def test_inspect_unrecognised_dir_exits_two(tmp_path: Path) -> None:
    result = runner.invoke(app, ["inspect", str(tmp_path)])
    assert result.exit_code == 2
