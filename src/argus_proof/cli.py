"""argus-proof CLI — ``inspect``, ``run``, ``score``, ``report``, ``serve``.

Phase 0 scaffold: the eval verbs are stubs that name the epic issue that will
implement them. Only ``serve`` (health endpoint on :8104) is live.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog

try:
    import typer
    from typer import Argument, Option
except ImportError as _exc:  # pragma: no cover
    print("CLI requires: pip install argus-proof[cli]", file=sys.stderr)
    raise SystemExit(1) from _exc

EPIC = "https://github.com/smk762/argus-studio/issues/6"

app = typer.Typer(
    name="argus-proof",
    help="Post-training LoRA evaluation: generate samples and score them against the curated dataset.",
    no_args_is_help=True,
)


@app.callback()
def _cli(verbose: bool = Option(False, "--verbose", "-v", help="Show info/debug logs")) -> None:
    """Keep stdout clean for --json output; ``serve`` re-enables info logs."""
    level = logging.DEBUG if verbose else logging.WARNING
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(level))


def _stub(verb: str, issue: str) -> None:
    typer.echo(f"argus-proof {verb} is not implemented yet — tracked in {issue} (epic: {EPIC})", err=True)
    raise typer.Exit(2)


def _load_report(report_json: Path):  # noqa: ANN202 - EvalReport imported lazily
    """Load a scored EvalReport, exiting 2 with a message if it can't be read."""
    from argus_proof.models import EvalReport, ProofError

    try:
        return EvalReport.model_validate_json(report_json.read_text(encoding="utf-8"))
    except (OSError, ValueError, ProofError) as exc:
        typer.echo(f"cannot read EvalReport {report_json}: {exc}", err=True)
        raise typer.Exit(2) from exc


@app.command()
def inspect(
    run_dir: Path = Argument(..., help="Proof run dir (or export dir + LoRA pair) to summarise"),
) -> None:
    """Summarise a proof run: manifest, prompt grid, scores, verdicts. [stub]"""
    _stub("inspect", "https://github.com/smk762/argus-studio/issues/8")


@app.command()
def run(
    lora: Path = Argument(..., help="Trained LoRA safetensors"),
    manifest: Path = Argument(..., help="Export manifest.jsonl the LoRA was trained from"),
) -> None:
    """Generate a sample grid from a trained LoRA via the configured backend. [stub]"""
    _stub("run", "https://github.com/smk762/argus-studio/issues/9")


@app.command()
def score(
    run_dir: Path = Argument(..., help="Proof run dir containing generated samples"),
) -> None:
    """Score generated samples: identity, quality, diversity, safety. [stub]"""
    _stub("score", "https://github.com/smk762/argus-studio/issues/11")


@app.command()
def report(
    run_dir: Path = Argument(..., help="Scored proof run dir"),
) -> None:
    """Render an EvalReport (JSON + HTML) with a pass/fail verdict. [stub]"""
    _stub("report", "https://github.com/smk762/argus-studio/issues/19")


@app.command()
def gate(
    report_json: Path = Argument(..., help="Scored EvalReport JSON to gate"),
    min_pass_rate: float = Option(0.75, help="Minimum aggregate pass-rate"),
    min_pass_rate_ci_lower: float | None = Option(None, help="Minimum Wilson lower bound on the pass-rate"),
    min_identity: float | None = Option(None, help="Minimum mean identity score"),
    max_unsafe_rate: float | None = Option(None, help="Maximum fraction of images flagged unsafe"),
    confidence: float = Option(0.95, help="Confidence level for the CI lower bound"),
) -> None:
    """Gate a scored EvalReport against acceptance thresholds; exit non-zero if rejected (for CI)."""
    from argus_proof.acceptance import evaluate_acceptance
    from argus_proof.models import AcceptanceThresholds

    report = _load_report(report_json)

    try:
        thresholds = AcceptanceThresholds(
            min_pass_rate=min_pass_rate,
            min_pass_rate_ci_lower=min_pass_rate_ci_lower,
            min_identity_mean=min_identity,
            max_unsafe_rate=max_unsafe_rate,
            confidence=confidence,
        )
    except ValueError as exc:  # out-of-range option (e.g. --confidence 1.0)
        typer.echo(f"invalid threshold: {exc}", err=True)
        raise typer.Exit(2) from exc

    result = evaluate_acceptance(report, thresholds)
    for check in result.checks:
        typer.echo(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    typer.echo("ACCEPTED" if result.passed else "REJECTED")
    raise typer.Exit(0 if result.passed else 1)


@app.command()
def recommend(
    report_json: Path = Argument(..., help="Scored EvalReport JSON to analyse"),
    store: Path | None = Option(
        None, "--store", help="CrossRunStore parquet: also recommend the best checkpoint / LoRA weight (needs [stats])"
    ),
) -> None:
    """Suggest routed fixes (which suite stage to act on) from a scored EvalReport."""
    from argus_proof.recommend import recommend as _recommend

    report = _load_report(report_json)

    cross_run = None
    if store is not None:
        try:
            import polars  # noqa: F401  (the store needs the [stats] extra)
        except ImportError as exc:
            typer.echo("--store needs the stats extra: pip install 'argus-proof[stats]'", err=True)
            raise typer.Exit(2) from exc
        from argus_proof.crossrun import CrossRunStore

        cross_run = CrossRunStore(store)

    recommendations = _recommend(report, store=cross_run)
    if not recommendations:
        typer.echo("no recommendations — nothing actionable")
        return
    for rec in recommendations:
        typer.echo(_format_recommendation(rec))


def _format_recommendation(rec) -> str:  # noqa: ANN001 - Recommendation, avoids an import at module load
    """One line per recommendation, surfacing the numeric evidence when present."""
    detail = ""
    if rec.metric is not None and rec.value is not None:
        threshold = f" vs {rec.threshold:.2f}" if rec.threshold is not None else ""
        detail = f" [{rec.metric} {rec.value:.2f}{threshold}]"
    return f"[{rec.stage}] {rec.issue}{detail}: {rec.action}"


DEFAULT_SCHEMA_PATH = Path("schema/proof-wire.schema.json")


@app.command()
def schema(
    output: Path = Option(DEFAULT_SCHEMA_PATH, "--output", "-o", help="Where to write the JSON Schema"),
    check: bool = Option(False, "--check", help="Exit non-zero if the committed schema is stale (for CI)"),
) -> None:
    """Emit the wire-contract JSON Schema consumers codegen against."""
    from argus_proof.models import render_wire_schema

    rendered = render_wire_schema()

    if check:
        existing = output.read_text(encoding="utf-8") if output.exists() else ""
        if existing != rendered:
            typer.echo(f"{output} is stale — run `argus-proof schema` and commit the result.", err=True)
            raise typer.Exit(1)
        typer.echo(f"{output} is up to date.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    typer.echo(f"Wrote wire schema -> {output}")


@app.command()
def serve(
    port: int = Option(8104, "--port", "-p", help="Port to listen on"),
    host: str = Option("0.0.0.0", "--host", help="Host to bind to"),
    cors: bool = Option(False, "--cors", help="Enable CORS (allow all origins)"),
) -> None:
    """Start the argus-proof micro-server (FastAPI) on :8104."""
    try:
        import uvicorn
    except ImportError as _exc:  # pragma: no cover
        typer.echo("Server requires: pip install argus-proof[server]", err=True)
        raise typer.Exit(1) from _exc

    from argus_proof.server import create_app

    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
    application = create_app(cors=cors)
    uvicorn.run(application, host=host, port=port)


if __name__ == "__main__":
    app()
