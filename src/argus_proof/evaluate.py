"""Env-configured evaluation pipeline — the spine behind ``run`` / ``score``.

The shared assembly the CLI verbs and the server's ``POST /run/stream`` both
use: construct the configured generation backend, load a completed run dir
back into (manifest, images), and score those images with the default
availability-guarded scorer set. Configuration comes from the ``PROOF_*``
environment (kept in lock-step with ``.env.example``), so the CLI, the
server, and a compose deployment build the same pipeline from the same knobs.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from argus_proof.backends import GenBackend, get_backend
from argus_proof.backends.base import MANIFEST_NAME, ModelResolver, make_dir_resolver
from argus_proof.backends.pnginfo import read_dimensions
from argus_proof.backends.workflow import example_template, load_template
from argus_proof.models import EvalReport, GateConfig, GeneratedImage, ProofError, RunManifest
from argus_proof.scoring import ScoreContext, score_run


class EvaluateError(ProofError):
    """The pipeline could not be assembled, or a run dir is not a proof run."""


# Environment knobs (kept in lock-step with .env.example).
ENV_BACKEND = "PROOF_BACKEND"
ENV_MODELS_DIR = "PROOF_MODELS_DIR"
ENV_WORKFLOW_TEMPLATE = "PROOF_WORKFLOW_TEMPLATE"
ENV_COMFYUI_URL = "COMFYUI_BASE_URL"
ENV_A1111_URL = "A1111_BASE_URL"
ENV_REMOTE_URL = "PROOF_REMOTE_URL"
ENV_REMOTE_API_KEY = "PROOF_REMOTE_API_KEY"
ENV_RUNS_DIR = "ARGUS_PROOF_RUNS_DIR"

DEFAULT_BACKEND = "comfyui"
DEFAULT_RUNS_DIR = "runs"

# Image files a backend may have written into a run dir (ComfyUI writes PNG;
# other engines may differ) and that reference dirs may contain.
IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")


def runs_root(root: str | os.PathLike[str] | None = None) -> Path:
    """The runs directory: explicit *root* > ``$ARGUS_PROOF_RUNS_DIR`` > default."""
    return Path(root or os.environ.get(ENV_RUNS_DIR, DEFAULT_RUNS_DIR))


def models_roots(models_dir: str | os.PathLike[str] | None = None) -> list[Path]:
    """The configured model tree root(s): explicit > ``$PROOF_MODELS_DIR``.

    The value may name several roots separated by ``os.pathsep`` (PATH-style),
    mirroring ComfyUI's main + ``extra_model_paths`` trees — e.g.
    ``/data/models:/data/models-extra``. Empty when nothing is configured.
    """
    value = models_dir or os.environ.get(ENV_MODELS_DIR) or ""
    return [Path(part).expanduser() for part in str(value).split(os.pathsep) if part.strip()]


def models_resolver(models_dir: str | os.PathLike[str] | None = None) -> ModelResolver:
    """A resolver over the local model tree(s), for SHA256-pinning the manifest.

    Explicit *models_dir* > ``$PROOF_MODELS_DIR`` (each may list several roots,
    see :func:`models_roots`); raises :class:`EvaluateError` when neither is
    set, because a local backend cannot hash checkpoints/LoRAs into a
    reproducible manifest without knowing where the files live.
    """
    roots = models_roots(models_dir)
    if not roots:
        raise EvaluateError(
            "no model directory configured — set PROOF_MODELS_DIR (or pass models_dir) so "
            "checkpoints/LoRAs can be resolved and SHA256-pinned into the run manifest"
        )
    if len(roots) == 1:
        return make_dir_resolver(roots[0])
    resolvers = [make_dir_resolver(root) for root in roots]

    def resolve(name: str) -> Path:
        for resolver in resolvers:
            try:
                return resolver(name)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"no model named {name!r} found under any of {[str(r) for r in roots]}")

    return resolve


def _comfy_template(path: str | os.PathLike[str] | None = None) -> dict:
    """The ComfyUI workflow template: explicit path > ``$PROOF_WORKFLOW_TEMPLATE``
    > the packaged SDXL+LoRA example."""
    value = path or os.environ.get(ENV_WORKFLOW_TEMPLATE)
    if value:
        return load_template(Path(value))
    return example_template()


def backend_from_env(
    name: str | None = None,
    *,
    models_dir: str | os.PathLike[str] | None = None,
    workflow_template: str | os.PathLike[str] | None = None,
) -> GenBackend:
    """Construct the configured generation backend from the environment.

    *name* overrides ``$PROOF_BACKEND`` (default ``comfyui``). Local backends
    (comfyui / diffusers / a1111) need a model dir for manifest hashing; the
    remote backend needs ``$PROOF_REMOTE_URL``. Raises
    :class:`~argus_proof.backends.BackendError` for an unknown name and
    :class:`EvaluateError` for missing configuration.
    """
    resolved = (name or os.environ.get(ENV_BACKEND) or DEFAULT_BACKEND).strip().lower()
    if resolved == "comfyui":
        return get_backend(
            "comfyui",
            workflow_template=_comfy_template(workflow_template),
            resolve_model=models_resolver(models_dir),
            base_url=os.environ.get(ENV_COMFYUI_URL, "http://127.0.0.1:8188"),
        )
    if resolved == "a1111":
        return get_backend(
            "a1111",
            resolve_model=models_resolver(models_dir),
            base_url=os.environ.get(ENV_A1111_URL, "http://127.0.0.1:7860"),
        )
    if resolved == "diffusers":
        return get_backend("diffusers", resolve_model=models_resolver(models_dir))
    if resolved == "remote":
        url = os.environ.get(ENV_REMOTE_URL)
        if not url:
            raise EvaluateError("PROOF_BACKEND=remote needs PROOF_REMOTE_URL set to the hosted generation endpoint")
        return get_backend("remote", base_url=url, api_key=os.environ.get(ENV_REMOTE_API_KEY) or None)
    # Unknown name: let get_backend raise its BackendError naming the known set.
    return get_backend(resolved)


# ---------------------------------------------------------------------------
# run-dir loading — a completed generation back into (manifest, images)
# ---------------------------------------------------------------------------


def load_manifest(run_dir: Path) -> RunManifest:
    """The :class:`RunManifest` a backend wrote into *run_dir*, or :class:`EvaluateError`."""
    path = Path(run_dir) / MANIFEST_NAME
    if not path.is_file():
        raise EvaluateError(
            f"{run_dir} has no {MANIFEST_NAME} — not a proof run dir (generate one with `argus-proof run`)"
        )
    try:
        return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ProofError) as exc:
        raise EvaluateError(f"cannot read {path}: {exc}") from exc


def discover_images(run_dir: Path, manifest: RunManifest) -> list[GeneratedImage]:
    """Rebuild the run's :class:`GeneratedImage` list from the files on disk.

    Backends name images ``<run_id>-<seed>[-<index>].<ext>`` (see
    ``ComfyUIBackend._collect_images``), so the seed is recovered from the
    filename; dimensions are read back from the bytes (PNG) with the manifest's
    sampling size as the fallback. Files that don't match the run's naming are
    ignored, so a stray file in the dir can't be scored into the report.
    """
    run_dir = Path(run_dir)
    pattern = re.compile(rf"^{re.escape(manifest.run_id)}-(\d+)(?:-\d+)?$")
    images: list[GeneratedImage] = []
    for path in sorted(run_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = pattern.match(path.stem)
        if match is None:
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise EvaluateError(f"cannot read image {path}: {exc}") from exc
        dims = read_dimensions(data) or (manifest.sampling.width, manifest.sampling.height)
        images.append(
            GeneratedImage(
                image_id=path.stem,
                run_id=manifest.run_id,
                seed=int(match.group(1)),
                path=str(path),
                width=dims[0],
                height=dims[1],
            )
        )
    if not images:
        raise EvaluateError(f"{run_dir} contains no images named {manifest.run_id!r}-<seed>.*")
    return images


def reference_images(directory: Path) -> list[Path]:
    """The held-out reference images under *directory* (recursive, sorted).

    These drive identity scoring and must NOT overlap the training set — see
    :class:`~argus_proof.scoring.base.ScoreContext`.
    """
    return sorted(p for p in Path(directory).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


# ---------------------------------------------------------------------------
# scoring — the default availability-guarded scorer set
# ---------------------------------------------------------------------------


def default_scorers() -> list:
    """The standard per-image scorer set, availability-guarded.

    Identity (InsightFace), prompt adherence (CLIPScore), aesthetic (CLIP-IQA)
    and safety (NudeNet) — each reports ``is_available()`` so the orchestrator
    skips any whose extra isn't installed, and identity additionally no-ops
    without reference images. Construction is lazy/cheap; nothing heavy loads
    until a scorer actually scores.
    """
    from argus_proof.scoring.scorers import IdentityScorer, SafetyScorer, clip_score_scorer, pyiqa_scorer

    return [IdentityScorer(), clip_score_scorer(), pyiqa_scorer(), SafetyScorer()]


def default_phash_scorers() -> tuple:
    """The (deduper, diversity) pair a default run applies — the single source
    both :func:`score_images` and :func:`all_reporting_scorers` build from."""
    from argus_proof.scoring.scorers import PhashDeduper, PhashDiversityScorer

    return PhashDeduper(), PhashDiversityScorer()


def all_reporting_scorers() -> list:
    """Every scorer a default run applies (per-image set + phash dedup/diversity),
    flat — so a capability probe like the server's ``/scorers`` reports exactly
    what :func:`score_images` will run, with no separate list to drift."""
    return [*default_scorers(), *default_phash_scorers()]


def score_images(
    manifest: RunManifest,
    images: list[GeneratedImage],
    *,
    references: list[Path] | None = None,
    gate: GateConfig | None = None,
) -> EvalReport:
    """Score *images* with the default scorers + phash dedup/diversity."""
    deduper, diversity = default_phash_scorers()
    ctx = ScoreContext(prompt=manifest.prompt, reference_images=list(references or []))
    return score_run(
        manifest,
        images,
        scorers=default_scorers(),
        deduper=deduper,
        diversity=diversity,
        context=ctx,
        gate=gate,
    )


def score_run_dir(
    run_dir: Path,
    *,
    references: list[Path] | None = None,
    gate: GateConfig | None = None,
) -> EvalReport:
    """Load *run_dir* (manifest + images) and score it into an :class:`EvalReport`."""
    manifest = load_manifest(run_dir)
    images = discover_images(run_dir, manifest)
    return score_images(manifest, images, references=references, gate=gate)
