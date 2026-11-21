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
    """Keep stdout clean for --json output; ``serve`` re-enables info logs.

    Logs go to stderr at every level: structlog's default factory prints to
    stdout, where a warning would corrupt a ``--json`` payload.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(level), logger_factory=_stderr_logger)


def _stderr_logger(*_args: object) -> structlog.PrintLogger:
    # Resolve sys.stderr per logger rather than once at configure time, so a
    # swapped stream (a test runner, a redirect) is honoured, never a closed one.
    return structlog.PrintLogger(file=sys.stderr)


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
    if rep.pose is not None:
        typer.echo(f"  pose: {_pose_line(rep.pose)}")
    for img in rep.images:
        mark = "?" if img.passed is None else ("+" if img.passed else "-")
        rating = f" hitl={img.hitl_rating}" if img.hitl_rating is not None else ""
        rejects = f" rejects={','.join(r.code for r in img.reject_reasons)}" if img.reject_reasons else ""
        typer.echo(f"  [{mark}] {img.image_id} (seed {img.seed}){rating}{rejects}")


def _pose_line(pose) -> str:  # noqa: ANN001 - PoseReport, imported lazily by callers
    """One line for the run-level pose block."""

    def num(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    usable = "?" if pose.usable is None else str(pose.usable)
    line = (
        f"diversity {num(pose.diversity)}, largest cluster {num(pose.largest_cluster_share)} "
        f"({usable}/{pose.count} usable{', partial' if pose.partial else ''})"
    )
    if pose.comparison is not None:
        line += f", close excess vs training {num(pose.comparison.close_excess)}"
    if pose.provenance.not_checked:
        line += f"; not checked: {pose.provenance.not_checked[0]}"
    return line


@app.command()
def inspect(
    path: Path = Argument(..., help="A proof run dir (manifest.json) or a curator export dir (manifest.jsonl)"),
) -> None:
    """Summarise a proof run dir or a curator export dir."""
    from argus_proof.backends.base import FAILURE_NAME, MANIFEST_NAME
    from argus_proof.evaluate import EvaluateError, discover_images, load_manifest
    from argus_proof.grid import read_export_prompts

    # A failed run has failed.json in place of a manifest: load_manifest reports why.
    if (path / MANIFEST_NAME).is_file() or (path / FAILURE_NAME).is_file():
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
    cfg: float | None = Option(None, help="CFG scale (default: the checkpoint family's — SDXL 7, Flux 1)"),
    sampler: str | None = Option(None, help="Sampler (default: SDXL dpmpp_2m, Flux euler)"),
    scheduler: str | None = Option(None, help="Scheduler (default: SDXL karras, Flux simple)"),
    guidance: float | None = Option(
        None, min=0, help="Flux guidance, the template's $guidance slot (default: Flux 3.5)"
    ),
    no_guidance: bool = Option(
        False, "--no-guidance", help="Send no guidance, even for a Flux checkpoint (a template with no $guidance slot)"
    ),
    width: int = Option(1024, min=64),
    height: int = Option(1024, min=64),
    clip_skip: int = Option(1),
    vae: str | None = Option(
        None, help="VAE file (template needs a $vae slot; default: Flux ae, SDXL the checkpoint's)"
    ),
    text_encoder: list[str] | None = Option(
        None,
        "--text-encoder",
        help="Text encoder file, repeat in slot order (default: Flux t5xxl_fp8_e4m3fn + clip_l, SDXL none)",
    ),
    backend: str | None = Option(None, "--backend", help="Override $PROOF_BACKEND (comfyui/diffusers/a1111/remote)"),
    prefix: str = Option("proof", "--run-prefix", help="run_id prefix for the generated runs"),
    fail_fast: bool = Option(
        False, "--fail-fast", help="Stop at the first failed run instead of generating the rest of the grid"
    ),
) -> None:
    """Generate a sample grid from a trained LoRA via the configured backend.

    A run that fails (after the backend's own retries) leaves a ``failed.json`` in
    its run dir and the grid carries on; the command exits 1 at the end, naming
    every failed run. ``--fail-fast`` stops at the first failure instead.
    """
    from argus_cortex.fs import PathRefused, safe_name

    from argus_proof.backends import BackendError
    from argus_proof.backends.base import clear_failed, mark_failed
    from argus_proof.backends.workflow import model_defaults, sampling_defaults
    from argus_proof.evaluate import EvaluateError, backend_from_env, runs_root
    from argus_proof.grid import GridError, build_grid, read_export_prompts
    from argus_proof.models import GridConfig, SamplingParams

    try:
        safe_name(prefix)
    except PathRefused:
        typer.echo(f"invalid --run-prefix {prefix!r}: use letters, digits, '.', '_' or '-'", err=True)
        raise typer.Exit(2) from None
    if no_guidance and guidance is not None:
        typer.echo("--guidance and --no-guidance are mutually exclusive", err=True)
        raise typer.Exit(2)

    try:
        base_prompts = [prompt] if prompt else read_export_prompts(export)
    except GridError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    defaults = sampling_defaults(checkpoint)  # by the checkpoint's family folder
    models = model_defaults(checkpoint)
    export_manifest = export / "manifest.jsonl"
    config = GridConfig(
        base_checkpoint=checkpoint,
        vae=vae if vae is not None else models["vae"],
        text_encoders=text_encoder or models["text_encoders"],
        lora_checkpoints=[lora],
        lora_weights=weight,
        sampling=SamplingParams(
            sampler=sampler if sampler is not None else defaults["sampler"],
            scheduler=scheduler if scheduler is not None else defaults["scheduler"],
            steps=steps,
            cfg=cfg if cfg is not None else defaults["cfg"],
            guidance=None if no_guidance else guidance if guidance is not None else defaults["guidance"],
            clip_skip=clip_skip,
            width=width,
            height=height,
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
    done: list[str] = []
    failed: list[tuple[str, str]] = []
    for spec in plan.specs:
        run_dir = out_root / spec.run_id
        typer.echo(f"[{spec.run_id}] generating {len(spec.seeds)} image(s) -> {run_dir}")
        try:
            result = engine.generate(spec, run_dir, progress=_echo_progress)
        except BackendError as exc:
            typer.echo(f"[{spec.run_id}] failed: {exc}", err=True)
            try:
                mark_failed(run_dir, spec.run_id, str(exc))
            except OSError as write_exc:
                typer.echo(f"[{spec.run_id}] cannot record the failure in {run_dir}: {write_exc}", err=True)
            failed.append((spec.run_id, str(exc)))
            if fail_fast:
                break
            continue
        clear_failed(run_dir)
        done.append(spec.run_id)
        typer.echo(f"[{spec.run_id}] done: {len(result.images)} image(s)")

    if done:
        typer.echo(f"score with: argus-proof score {out_root / done[0]}")
    if failed:
        skipped = len(plan.specs) - len(done) - len(failed)
        tail = f", {skipped} not attempted (--fail-fast)" if skipped else ""
        typer.echo(f"{len(failed)} of {len(plan.specs)} run(s) failed{tail}:", err=True)
        for run_id, error in failed:
            typer.echo(f"  {run_id}: {error}", err=True)
        raise typer.Exit(1)


@app.command()
def score(
    run_dir: Path = Argument(..., help="Proof run dir containing manifest.json + generated samples"),
    references: Path | None = Option(
        None,
        "--references",
        help="Held-out reference image dir for identity and likeness scoring (must not overlap training)",
    ),
    reports_dir: Path | None = Option(None, "--reports-dir", help="Report store (default $ARGUS_PROOF_REPORTS_DIR)"),
    save: bool = Option(True, "--save/--no-save", help="Persist the report into the report store"),
    json_out: bool = Option(False, "--json", help="Print the full EvalReport JSON to stdout"),
    require_scorers: list[str] | None = Option(
        None,
        "--require-scorers",
        help="Metrics whose scorer must be available, else exit 2 (repeatable or comma-separated, e.g. "
        "identity,likeness,clip_score,aesthetic,safety,pose)",
    ),
) -> None:
    """Score a generated run into an EvalReport: identity, likeness, quality, diversity, safety, pose.

    A default scorer that can't run here (its extra isn't installed, or its service
    isn't configured) leaves its metric empty; that is warned about on stderr, and
    ``--require-scorers`` turns it into a refusal.
    """
    from argus_proof.evaluate import (
        EvaluateError,
        all_reporting_scorers,
        reference_images,
        score_run_dir,
        unavailable_metrics,
    )
    from argus_proof.models import ProofError
    from argus_proof.reports import ReportStore

    required = [m.strip() for item in require_scorers or [] for m in item.split(",") if m.strip()]
    known = [sc.provenance().metric for sc in all_reporting_scorers()]
    unknown = [m for m in required if m not in known]
    if unknown:
        typer.echo(
            f"unknown metric(s) for --require-scorers: {', '.join(unknown)} (known: {', '.join(known)})", err=True
        )
        raise typer.Exit(2)
    missing = unavailable_metrics()
    missing_required = [m for m in required if m in missing]
    if missing_required:
        typer.echo(
            f"required scorer(s) unavailable: {', '.join(missing_required)} — install argus-proof[score] "
            "(and set PROOF_NUDENET_URL for safety, PROOF_PUPPET_URL for pose), or use the -score image",
            err=True,
        )
        raise typer.Exit(2)
    if missing:
        typer.echo(
            f"warning: no scorer available for {', '.join(missing)}; those metrics will be empty and their "
            "groups left for review (install argus-proof[score], or use the -score image; safety and pose need "
            "PROOF_NUDENET_URL and PROOF_PUPPET_URL)",
            err=True,
        )

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
    min_identity: float | None = Option(None, help="Minimum mean identity score (ArcFace, human faces)"),
    min_likeness: float | None = Option(
        None, help="Minimum mean likeness score (DINOv2 vs the references: masked or non-human subjects, scenes)"
    ),
    max_unsafe_rate: float | None = Option(None, help="Maximum fraction of images flagged unsafe"),
    confidence: float = Option(0.95, help="Confidence level for the CI lower bound"),
    min_pose_diversity: float | None = Option(
        None,
        help="Turn on the pose-collapse gate: minimum run-level pose diversity (argus-puppet; suggested 0.3, "
        "calibrated on synthetic poses only). A run whose pose block isn't measurable fails it",
    ),
    max_pose_cluster_share: float = Option(
        0.75, help="With --min-pose-diversity: a largest pose cluster at or above this share is a collapse"
    ),
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
            min_likeness_mean=min_likeness,
            max_unsafe_rate=max_unsafe_rate,
            confidence=confidence,
            min_pose_diversity=min_pose_diversity,
            max_pose_cluster_share=max_pose_cluster_share,
        )
    except ValueError as exc:  # out-of-range option (e.g. --confidence 1.0)
        typer.echo(f"invalid threshold: {exc}", err=True)
        raise typer.Exit(2) from exc

    result = evaluate_acceptance(report, thresholds)
    for check in result.checks:
        # A passing check that carries a code is a flag for review (pose_copying), never a reject.
        mark = ("FLAG" if check.code else "PASS") if check.passed else "FAIL"
        code = f" ({check.code})" if check.code else ""
        typer.echo(f"[{mark}] {check.name}{code}: {check.detail}")
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

    try:
        prompts = read_export_prompts(export_dir)
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
    overwrite: bool = Option(
        False, "--overwrite", help="Replace an existing FiftyOne dataset of the same --name (its tags are lost)"
    ),
) -> None:
    """Open a scored run in FiftyOne with metrics attached, for embedding viz + tag triage."""
    from argus_proof import explore as fo_explore
    from argus_proof.models import ProofError

    if not fo_explore.is_available():
        typer.echo("explore needs the fiftyone extra: pip install 'argus-proof[fiftyone]'", err=True)
        raise typer.Exit(2)

    report = _load_report(report_json)
    try:
        if name is not None and not overwrite and fo_explore.dataset_exists(name):
            # FiftyOne datasets share one database and hold the App's triage tags:
            # never delete one just because a name was reused.
            typer.echo(f"a FiftyOne dataset named {name!r} already exists; pass --overwrite to replace it", err=True)
            raise typer.Exit(2)
        paths = fo_explore.image_paths_from_dir(images)
        dataset = fo_explore.to_fiftyone_dataset(report, paths, name=name, overwrite=overwrite)

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
    devmode: bool | None = Option(
        None,
        "--devmode/--no-devmode",
        help="Register the /purge routes that delete runs and their reports (default: $ARGUS_PROOF_DEVMODE).",
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
    application = create_app(cors=cors, read_only=read_only, devmode=devmode)
    uvicorn.run(application, host=host, port=port)


if __name__ == "__main__":
    app()
