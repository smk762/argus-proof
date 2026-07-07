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


DEFAULT_SCHEMA_PATH = Path("schema/proof-wire.schema.json")


@app.command()
def schema(
    output: Path = Option(DEFAULT_SCHEMA_PATH, "--output", "-o", help="Where to write the JSON Schema"),
    check: bool = Option(False, "--check", help="Exit non-zero if the committed schema is stale (for CI)"),
) -> None:
    """Emit the wire-contract JSON Schema consumers codegen against."""
    import json

    from argus_proof.models import wire_schema

    rendered = json.dumps(wire_schema(), indent=2, sort_keys=True) + "\n"

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
