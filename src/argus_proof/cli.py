"""argus-proof CLI — ``inspect``, ``run``, ``score``, ``report``, ``serve`` and friends.

The eval verbs wire the library together: ``run`` expands a prompt grid from a
curator export and generates it through the configured backend
(``$PROOF_BACKEND``), ``score`` turns a run dir into a stored
:class:`~argus_proof.models.EvalReport`, ``report`` browses stored reports, and
``inspect`` summarises an export or run dir. ``gate`` / ``recommend`` /
``experiment`` / ``explore`` / ``schema`` / ``serve`` operate on those artifacts.
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


def _load_model(path: Path, model_cls, label: str):  # noqa: ANN001, ANN202 - generic pydantic loader
    """Load a pydantic model from JSON at *path*, exiting 2 with a message on failure."""
    from argus_proof.models import ProofError

    try:
        return model_cls.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ProofError) as exc:
        typer.echo(f"cannot read {label} {path}: {exc}", err=True)
        raise typer.Exit(2) from exc


def _load_report(report_json: Path):  # noqa: ANN202 - EvalReport imported lazily
    """Load a scored EvalReport, exiting 2 with a message if it can't be read."""
    from argus_proof.models import EvalReport

    return _load_model(report_json, EvalReport, "EvalReport")


_RUN_PREFIX_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


def _echo_progress(event) -> None:  # noqa: ANN001 - ProgressEvent, imported lazily by callers
    """Print a backend ProgressEvent as a compact CLI line."""
    if event.type == "image":
        typer.echo(f"  seed {event.seed}: {event.image_id}")
    elif event.type == "error":
        typer.echo(f"  error: {event.message}", err=True)


def _echo_report(rep) -> None:  # noqa: ANN001 - EvalReport, imported lazily by callers
    """The human digest of a scored report: verdict line + one line per image."""
    agg, verdict = rep.aggregate, rep.verdict
    status = "PENDING" if verdict.pending else ("PASSED" if verdict.passed else "FAILED")
    groups = agg.n_groups if agg.n_groups is not None else agg.n_images
    typer.echo(
        f"{rep.run_id}: {status} — pass rate {agg.pass_rate:.0%} "
        f"({agg.n_passed}/{groups} groups, {agg.n_images} images, {agg.n_needs_hitl} need review)"
    )
    for reason in verdict.reasons:
        typer.echo(f"  reason: {reason}")
    for img in rep.images:
        mark = "?" if img.passed is None else ("+" if img.passed else "-")
        rating = f" hitl={img.hitl_rating}" if img.hitl_rating is not None else ""
        rejects = f" rejects={','.join(r.code for r in img.reject_reasons)}" if img.reject_reasons else ""
        typer.echo(f"  [{mark}] {img.image_id} (seed {img.seed}){rating}{rejects}")


@app.command()
def inspect(
    path: Path = Argument(..., help="A proof run dir (manifest.json) or a curator export dir (manifest.jsonl)"),
) -> None:
    """Summarise a proof run dir or a curator export dir."""
    from argus_proof.backends.base import MANIFEST_NAME
    from argus_proof.evaluate import EvaluateError, discover_images, load_manifest
    from argus_proof.grid import read_export_prompts

    if (path / MANIFEST_NAME).is_file():
        try:
            manifest = load_manifest(path)
            images = discover_images(path, manifest)
        except EvaluateError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(2) from exc
        typer.echo(f"run {manifest.run_id} — {manifest.engine} {manifest.engine_version}")
        typer.echo(f"  checkpoint: {manifest.base_checkpoint.name} ({manifest.base_checkpoint.sha256[:12]}…)")
        for lora_ref in manifest.loras:
            typer.echo(f"  lora: {lora_ref.name} @ {lora_ref.weight} ({lora_ref.sha256[:12]}…)")
        typer.echo(f"  prompt: {manifest.prompt}")
        typer.echo(f"  seeds: {', '.join(str(s) for s in manifest.seeds)} — {len(images)} image(s) on disk")
        if manifest.source_manifest:
            typer.echo(f"  source export: {manifest.source_manifest}")
        return

    if path.is_dir():
        prompts = read_export_prompts(path)
        if prompts:
            typer.echo(f"export {path} — {len(prompts)} base prompt(s)")
            for prompt in prompts[:5]:
                typer.echo(f"  {prompt[:100]}{'…' if len(prompt) > 100 else ''}")
            if len(prompts) > 5:
                typer.echo(f"  … and {len(prompts) - 5} more")
            return
        typer.echo(f"{path} has no {MANIFEST_NAME}, captions, or .txt sidecars — not a run or export dir", err=True)
        raise typer.Exit(2)

    typer.echo(f"{path} is not a directory", err=True)
    raise typer.Exit(2)


@app.command()
def run(
    lora: str = Argument(..., help="Trained LoRA, named as the engine loads it (e.g. subject.safetensors)"),
    export: Path = Argument(..., help="Curator export dir the LoRA was trained from (prompt source)"),
    checkpoint: str = Option(..., "--checkpoint", "-c", help="Base checkpoint, named as the engine loads it"),
    out: Path | None = Option(None, "--out", "-o", help="Runs root (default $ARGUS_PROOF_RUNS_DIR or ./runs)"),
    weight: list[float] = Option([1.0], "--weight", "-w", help="LoRA weight(s) to sweep (repeatable)"),
    seed: list[int] = Option([1, 2, 3], "--seed", "-s", help="Control seed-set, one image per seed (repeatable)"),
    prompt: str | None = Option(None, "--prompt", help="Explicit prompt (skips the export's captions)"),
    negative: str = Option("", "--negative", help="Negative prompt"),
    max_prompts: int = Option(1, "--max-prompts", min=1, help="Base prompts to take from the export"),
    steps: int = Option(25, min=1),
    cfg: float = Option(7.0),
    sampler: str = Option("dpmpp_2m"),
    scheduler: str = Option("karras"),
    width: int = Option(1024, min=64),
    height: int = Option(1024, min=64),
    clip_skip: int = Option(1),
    vae: str | None = Option(None, help="Explicit VAE (template needs a $vae slot)"),
    backend: str | None = Option(None, "--backend", help="Override $PROOF_BACKEND (comfyui/diffusers/a1111/remote)"),
    prefix: str = Option("proof", "--run-prefix", help="run_id prefix for the generated runs"),
) -> None:
    """Generate a sample grid from a trained LoRA via the configured backend."""
    import re as _re

    from argus_proof.backends import BackendError
    from argus_proof.evaluate import EvaluateError, backend_from_env, runs_root
    from argus_proof.grid import GridError, build_grid, read_export_prompts
    from argus_proof.models import GridConfig, SamplingParams

    if not _re.match(_RUN_PREFIX_RE, prefix):
        typer.echo(f"invalid --run-prefix {prefix!r}: use letters, digits, '.', '_' or '-'", err=True)
        raise typer.Exit(2)

    base_prompts = [prompt] if prompt else read_export_prompts(export)
    export_manifest = export / "manifest.jsonl"
    config = GridConfig(
        base_checkpoint=checkpoint,
        lora_checkpoints=[lora],
        lora_weights=weight,
        sampling=SamplingParams(
            sampler=sampler, scheduler=scheduler, steps=steps, cfg=cfg, clip_skip=clip_skip, width=width, height=height
        ),
        negative_prompt=negative,
        seeds=seed,
        max_base_prompts=max_prompts,
        run_id_prefix=prefix,
        source_manifest=str(export_manifest) if export_manifest.is_file() else None,
    )

    try:
        plan = build_grid(config, base_prompts)
        engine = backend_from_env(backend)
    except (GridError, EvaluateError, BackendError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    est = plan.estimate
    typer.echo(f"grid: {est.n_runs} run(s), {est.n_images} image(s), est. {est.est_gpu_hours:.2f} GPU-hours")

    out_root = runs_root(out)
    for spec in plan.specs:
        run_dir = out_root / spec.run_id
        typer.echo(f"[{spec.run_id}] generating {len(spec.seeds)} image(s) -> {run_dir}")
        try:
            result = engine.generate(spec, run_dir, progress=_echo_progress)
        except BackendError as exc:
            typer.echo(f"[{spec.run_id}] failed: {exc}", err=True)
            raise typer.Exit(1) from exc
        typer.echo(f"[{spec.run_id}] done: {len(result.images)} image(s)")
    typer.echo(f"score with: argus-proof score {out_root / plan.specs[0].run_id}")


@app.command()
def score(
    run_dir: Path = Argument(..., help="Proof run dir containing manifest.json + generated samples"),
    references: Path | None = Option(
        None, "--references", help="Held-out reference image dir for identity scoring (must not overlap training)"
    ),
    reports_dir: Path | None = Option(None, "--reports-dir", help="Report store (default $ARGUS_PROOF_REPORTS_DIR)"),
    save: bool = Option(True, "--save/--no-save", help="Persist the report into the report store"),
    json_out: bool = Option(False, "--json", help="Print the full EvalReport JSON to stdout"),
) -> None:
    """Score a generated run into an EvalReport: identity, quality, diversity, safety."""
    from argus_proof.evaluate import EvaluateError, reference_images, score_run_dir
    from argus_proof.models import ProofError
    from argus_proof.reports import ReportStore

    try:
        refs = reference_images(references) if references else []
        rep = score_run_dir(run_dir, references=refs)
    except (EvaluateError, ProofError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    if save:
        path = ReportStore(reports_dir).save(rep)
        typer.echo(f"report -> {path}", err=json_out)  # keep stdout clean for --json
    if json_out:
        typer.echo(rep.model_dump_json(indent=2))
    else:
        _echo_report(rep)


@app.command()
def report(
    run_id: str = Argument("", help="Run to show; omit to list every stored report"),
    reports_dir: Path | None = Option(None, "--reports-dir", help="Report store (default $ARGUS_PROOF_REPORTS_DIR)"),
    json_out: bool = Option(False, "--json", help="Print JSON instead of the human digest"),
) -> None:
    """Show a stored EvalReport (or list all of them) with its pass/fail verdict."""
    from argus_proof.models import ProofError
    from argus_proof.reports import ReportStore

    store = ReportStore(reports_dir)
    if not run_id:
        summaries = store.list()
        if json_out:
            import json as _json

            typer.echo(_json.dumps([s.model_dump() for s in summaries], indent=2))
            return
        if not summaries:
            typer.echo(f"no reports in {store.root}")
            return
        for s in summaries:
            status = "PENDING" if s.pending else ("PASSED" if s.passed else "FAILED")
            typer.echo(
                f"{s.run_id}: {status} — pass rate {s.pass_rate:.0%}, {s.n_images} images, {s.n_needs_hitl} need review"
            )
        return

    try:
        rep = store.get(run_id)
    except FileNotFoundError as exc:
        typer.echo(f"no report for run {run_id!r} in {store.root}", err=True)
        raise typer.Exit(2) from exc
    except ProofError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc

    if json_out:
        typer.echo(rep.model_dump_json(indent=2))
    else:
        _echo_report(rep)


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


@app.command()
def experiment(
    matrix_json: Path = Argument(..., help="ExperimentMatrix JSON (factors × levels)"),
    export_dir: Path = Option(..., "--export", help="Curator export dir to source base prompts from"),
    max_gpu_hours: float | None = Option(None, help="Refuse to expand if the matrix exceeds this GPU-hour budget"),
) -> None:
    """Expand an A/B experiment matrix and report the up-front per-cell cost estimate."""
    from argus_proof.experiment import ExperimentError, ExperimentMatrix, expand_experiment
    from argus_proof.grid import GridError, read_export_prompts

    matrix = _load_model(matrix_json, ExperimentMatrix, "ExperimentMatrix")
    prompts = read_export_prompts(export_dir)

    try:
        plan = expand_experiment(matrix, prompts, max_gpu_hours=max_gpu_hours)
    except (ExperimentError, GridError) as exc:
        typer.echo(f"cannot expand experiment: {exc}", err=True)
        raise typer.Exit(1) from exc

    est = plan.estimate
    typer.echo(f"experiment {plan.run_id_prefix}: {est.n_cells} cells, {est.n_runs} runs, {est.n_images} images")
    for cell in plan.cells:
        typer.echo(f"  [{cell.cell_id}] {cell.plan.estimate.n_images} images ({cell.plan.estimate.n_runs} runs)")
    typer.echo(f"est. {est.est_gpu_hours:.1f} GPU-hours @ {est.seconds_per_image:g}s/image")


@app.command()
def explore(
    report_json: Path = Argument(..., help="Scored EvalReport JSON to open in FiftyOne"),
    images: Path = Option(..., "--images", help="Dir of generated images named <image_id>.<ext>"),
    name: str | None = Option(None, "--name", help="FiftyOne dataset name (default: auto)"),
    umap: bool = Option(False, "--umap", help="Compute a UMAP embedding visualisation (needs umap-learn)"),
    launch: bool = Option(True, "--launch/--no-launch", help="Open the FiftyOne App (blocks until closed)"),
    ingest: Path | None = Option(
        None, "--ingest", help="After the App closes, fold rating:/reject: tags back and write the report here"
    ),
    rater: str | None = Option(None, "--rater", help="Rater id stamped on ingested tags"),
) -> None:
    """Open a scored run in FiftyOne with metrics attached, for embedding viz + tag triage."""
    from argus_proof import explore as fo_explore
    from argus_proof.models import ProofError

    if not fo_explore.is_available():
        typer.echo("explore needs the fiftyone extra: pip install 'argus-proof[fiftyone]'", err=True)
        raise typer.Exit(2)

    report = _load_report(report_json)
    try:
        paths = fo_explore.image_paths_from_dir(images)
        dataset = fo_explore.to_fiftyone_dataset(report, paths, name=name, overwrite=True)
        typer.echo(f"dataset {dataset.name!r}: {len(dataset)} samples ({len(paths)} images found)")
        if umap:
            fo_explore.compute_visualization(dataset)
            typer.echo("computed UMAP visualisation (brain_key='proof_viz')")
        if launch:
            fo_explore.launch_app(dataset).wait()  # block until the App is closed
        if ingest is not None:
            updated = fo_explore.ingest_from_dataset(dataset, report, rater=rater)
            ingest.write_text(updated.model_dump_json(indent=2) + "\n", encoding="utf-8")
            typer.echo(f"ingested tags -> {ingest} (verdict: {'pass' if updated.verdict.passed else 'fail'})")
    except (OSError, ProofError) as exc:
        typer.echo(f"explore failed: {exc}", err=True)
        raise typer.Exit(2) from exc


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
    read_only: bool | None = Option(
        None,
        "--read-only/--no-read-only",
        help="Replay/demo mode: serve stored reports but 403 all live eval + writes (default: $ARGUS_PROOF_READ_ONLY).",
    ),
) -> None:
    """Start the argus-proof micro-server (FastAPI) on :8104."""
    try:
        import uvicorn
    except ImportError as _exc:  # pragma: no cover
        typer.echo("Server requires: pip install argus-proof[server]", err=True)
        raise typer.Exit(1) from _exc

    from argus_proof.server import create_app

    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
    application = create_app(cors=cors, read_only=read_only)
    uvicorn.run(application, host=host, port=port)


if __name__ == "__main__":
    app()
