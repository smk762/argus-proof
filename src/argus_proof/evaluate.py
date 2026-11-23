"""Env-configured evaluation pipeline — the spine behind ``run`` / ``score``.

The shared assembly the CLI verbs and the server's ``POST /run/stream`` both
use: construct the configured generation backend, load a completed run dir
back into (manifest, images), and score those images with the default
availability-guarded scorer set. Configuration comes from the ``PROOF_*``
environment (kept in lock-step with ``.env.example``), so the CLI, the
server, and a compose deployment build the same pipeline from the same knobs.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from collections.abc import Collection
from pathlib import Path
from typing import get_args

import structlog
from argus_cortex.gpu import GpuCoordinator
from argus_cortex.peers import peer_url
from argus_cortex.taxonomy import TargetCategory, TargetProfile

from argus_proof import gpu
from argus_proof.backends import GenBackend, get_backend
from argus_proof.backends.base import FAILURE_NAME, MANIFEST_NAME, ModelResolver, make_dir_resolver
from argus_proof.backends.pnginfo import read_dimensions
from argus_proof.backends.workflow import example_template, load_template, packaged_templates
from argus_proof.hashing import sha256_cached
from argus_proof.models import (
    CropBox,
    EvalReport,
    GateConfig,
    GeneratedImage,
    ProofError,
    RunManifest,
    RunScorerName,
    ScoredImage,
    ScoreResult,
    ScorerProvenance,
)
from argus_proof.scoring import ScoreContext, score_run

logger = structlog.get_logger()


class EvaluateError(ProofError):
    """The pipeline could not be assembled, or a run dir is not a proof run."""


class GpuUnavailable(EvaluateError):
    """The scoring pass's GPU lease was refused or timed out, or the broker failed."""


class ScorerUnavailable(EvaluateError):
    """A requested scorer can't run here (its extra isn't installed)."""


# Environment knobs (kept in lock-step with .env.example).
ENV_BACKEND = "PROOF_BACKEND"
ENV_MODELS_DIR = "PROOF_MODELS_DIR"
ENV_WORKFLOW_TEMPLATE = "PROOF_WORKFLOW_TEMPLATE"
ENV_COMFYUI_URL = "COMFYUI_BASE_URL"
COMFYUI_PEER = "comfyui"  # argus-cortex's peer name: ARGUS_COMFYUI_URL is read when COMFYUI_BASE_URL is unset
DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"
ENV_A1111_URL = "A1111_BASE_URL"
ENV_REMOTE_URL = "PROOF_REMOTE_URL"
ENV_REMOTE_API_KEY = "PROOF_REMOTE_API_KEY"
ENV_RUNS_DIR = "ARGUS_PROOF_RUNS_DIR"
ENV_LIKENESS_MODEL = "PROOF_LIKENESS_MODEL"

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


def check_peer_urls() -> None:
    """Refuse a malformed peer URL now (``PeerRefused``, naming the variable) rather than at the first run.

    Resolves every service URL proof reads, in proof's own variable and its suite-wide ``ARGUS_<PEER>_URL``. An
    unset one is fine (that scorer or backend is simply not configured); nothing is contacted.
    """
    from argus_proof.backends.echo import ECHO_PEERS, ENV_ECHO_URL
    from argus_proof.scoring.scorers.pose import ENV_PUPPET_URL, PUPPET_PEER
    from argus_proof.scoring.scorers.safety import ENV_NUDENET_URL, NUDENET_PEER

    peer_url(NUDENET_PEER, ENV_NUDENET_URL)
    peer_url(PUPPET_PEER, ENV_PUPPET_URL)
    peer_url(COMFYUI_PEER, ENV_COMFYUI_URL)
    for peer in ECHO_PEERS.values():
        peer_url(peer, ENV_ECHO_URL)


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
        explicit = workflow_template or os.environ.get(ENV_WORKFLOW_TEMPLATE)
        return get_backend(
            "comfyui",
            workflow_template=_comfy_template(workflow_template),
            # A user's template applies to every checkpoint; the packaged
            # defaults pick SDXL or Flux by the checkpoint's family folder.
            family_templates=None if explicit else packaged_templates(),
            resolve_model=models_resolver(models_dir),
            # COMFYUI_BASE_URL, else the suite-wide ARGUS_COMFYUI_URL; a malformed one raises PeerRefused.
            base_url=peer_url(COMFYUI_PEER, ENV_COMFYUI_URL, default=DEFAULT_COMFYUI_URL),
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
        failure = Path(run_dir) / FAILURE_NAME
        if failure.is_file():
            try:
                error = json.loads(failure.read_text(encoding="utf-8")).get("error", "unknown error")
            except (OSError, ValueError, AttributeError):
                error = "unreadable failure record"
            raise EvaluateError(f"{run_dir} failed to generate, so there is nothing to score: {error}")
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

    The list is in generation order (the manifest's seed order, then the image's
    index within its seed), not filename order, where ``-10`` sorts before ``-2``:
    positional lookups (``image_at``) depend on it. Two files with one stem but
    different extensions would share an ``image_id``, so they raise.
    """
    run_dir = Path(run_dir)
    pattern = re.compile(rf"^{re.escape(manifest.run_id)}-(\d+)(?:-(\d+))?$")
    seed_order: dict[int, int] = {}
    for position, seed in enumerate(manifest.seeds):
        seed_order.setdefault(seed, position)
    found: list[tuple[tuple[int, int, int, str], Path, re.Match[str]]] = []
    by_stem: dict[str, Path] = {}
    for path in run_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = pattern.fullmatch(path.stem)
        if match is None:
            continue
        if path.stem in by_stem:
            first, second = sorted((by_stem[path.stem].name, path.name))
            raise EvaluateError(f"{run_dir} holds both {first} and {second}: one image_id, two images")
        by_stem[path.stem] = path
        seed, index = int(match.group(1)), int(match.group(2) or 0)
        found.append(((seed_order.get(seed, len(seed_order)), seed, index, path.name), path, match))
    images: list[GeneratedImage] = []
    for _key, path, match in sorted(found, key=lambda item: item[0]):
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


# The subdir of a curator export that holds its held-out identity references (for identity and likeness).
REFERENCES_DIR = "references"


def training_images(export_dir: Path) -> list[Path]:
    """The training images of the curator export at *export_dir*.

    Every image under the export except its ``references/`` subdir, plus each
    ``manifest.jsonl`` row's ``exported_path`` (relative to the export),
    ``exported_abs_path`` and ``abs_path`` that exist here: a manifest-only
    export points at the dataset where it lives. Every key is kept on purpose:
    the ``abs_path`` source of an exported crop is training data too (see
    :func:`check_held_out`). A manifest row is not filtered out of
    ``references/``: a row there means a reference *is* a training image, which
    the held-out check must see.

    Paths are resolved, so one file reached by a relative and an absolute path
    is listed once; a manifest row that is not an image (a caption sidecar) is
    skipped.
    """
    export_dir = Path(export_dir)
    found: set[Path] = set()
    for path in export_dir.rglob("*"):
        rel = path.relative_to(export_dir)
        if rel.parts and rel.parts[0] == REFERENCES_DIR:
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            found.add(path.resolve())
    manifest = export_dir / "manifest.jsonl"
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            for key, base in (("exported_path", export_dir), ("exported_abs_path", None), ("abs_path", None)):
                value = row.get(key)
                if isinstance(value, str) and value:
                    candidate = base / value if base is not None else Path(value)
                    if candidate.suffix.lower() in IMAGE_SUFFIXES and candidate.is_file():
                        found.add(candidate.resolve())
    return sorted(found)


# Near-duplicate tier of the held-out check (#87): a 64-bit pHash of the whole image, and a Hamming distance at or
# below this is the same photograph. Measured on real photos: a resize plus JPEG re-encode lands at 0-2 bits, and
# the pairs at or under 6 were re-scans and re-uploads of one photo. A crop is NOT caught: trimming even 5% moves
# the hash 8+ bits, into the range of unrelated photos.
NEAR_DUP_BITS = 6
HELD_OUT_CHECK = "held-out-references"


def _phash_bits(path: Path) -> int | None:
    """64-bit pHash of *path* as an int, or None when it can't be read as an image."""
    import imagehash  # noqa: PLC0415 - the [score] extra
    from PIL import Image, UnidentifiedImageError  # noqa: PLC0415

    try:
        with Image.open(path) as im:
            im.draft("L", (256, 256))  # JPEG: decode at reduced size; pHash downsamples to 32x32 anyway
            return int(str(imagehash.phash(im.convert("L"))), 16)
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError):
        # DecompressionBombError is not an OSError: an oversized file is skipped, not a crashed run.
        return None


def near_duplicates_checked() -> bool:
    """Whether :func:`check_held_out` can compare perceptual hashes (``imagehash`` + ``PIL``, the ``[score]`` extra)."""
    from argus_proof.scoring.scorers._util import module_available  # noqa: PLC0415

    return module_available("imagehash", "PIL")


def check_held_out(references: list[Path], export_dir: Path) -> str:
    """Refuse identity *references* that are training images of the export at *export_dir*.

    A reference the LoRA trained on measures memorisation, not likeness, and
    makes every identity and likeness score meaningless (both scorers use the
    one reference set, so this one check covers both). Two tiers:

    * the SHA256 of the bytes, so an exact copy is caught under any name;
    * with the ``[score]`` extra, a whole-image pHash within :data:`NEAR_DUP_BITS`,
      so a resized or re-encoded copy is caught too.

    A crop is not caught. The training set includes each manifest row's
    ``abs_path`` source, so a reference that is the source of a curator crop is.

    Raises :class:`EvaluateError` naming each overlap. Returns which tiers ran
    (``"sha256+phash<=6"`` or ``"sha256"``), for the report.
    """
    if not references:
        return "sha256"
    training = training_images(export_dir)
    by_digest = {sha256_cached(p): p for p in training}
    overlap: list[tuple[Path, Path, str]] = []
    for ref in references:
        match = by_digest.get(sha256_cached(ref))
        if match is not None:
            overlap.append((ref, match, "same bytes"))
    method = "sha256"
    if near_duplicates_checked():
        method = f"sha256+phash<={NEAR_DUP_BITS}"
        exact = {ref for ref, _, _ in overlap}
        train_hashes = [(p, h) for p in training if (h := _phash_bits(p)) is not None]
        for ref in references:
            if ref in exact or (ref_hash := _phash_bits(ref)) is None:
                continue
            best = min(train_hashes, key=lambda t: (t[1] ^ ref_hash).bit_count(), default=None)
            if best is not None and (dist := (best[1] ^ ref_hash).bit_count()) <= NEAR_DUP_BITS:
                overlap.append((ref, best[0], f"near-duplicate, {dist} bits"))
    else:
        logger.warning("held_out.near_duplicates_unchecked", reason="imagehash/PIL not installed ([score] extra)")
    if overlap:
        pairs = ", ".join(f"{ref.name} = {train} ({how})" for ref, train, how in overlap[:5])
        more = f" (+{len(overlap) - 5} more)" if len(overlap) > 5 else ""
        raise EvaluateError(
            f"{len(overlap)} identity reference(s) are training images, so identity would measure memorisation: "
            f"{pairs}{more}. References must be held out from the training set."
        )
    return method


# ---------------------------------------------------------------------------
# scoring — the default availability-guarded scorer set
# ---------------------------------------------------------------------------


def default_scorers() -> list:
    """The standard per-image scorer set, availability-guarded.

    Identity (InsightFace), likeness (DINOv2, ``$PROOF_LIKENESS_MODEL``), prompt
    adherence (CLIPScore), aesthetic (CLIP-IQA) and safety (NudeNet) — each
    reports ``is_available()`` so the orchestrator skips any whose extra isn't
    installed, and identity and likeness additionally no-op without reference
    images. Construction is lazy/cheap; nothing heavy loads until a scorer
    actually scores.
    """
    from argus_proof.scoring.scorers import (
        IdentityScorer,
        SafetyScorer,
        clip_score_scorer,
        pyiqa_scorer,
    )

    return [IdentityScorer(), likeness_scorer(), clip_score_scorer(), pyiqa_scorer(), SafetyScorer()]


def likeness_scorer():  # noqa: ANN201 - LikenessScorer, imported lazily
    """The likeness scorer on ``$PROOF_LIKENESS_MODEL`` (default DINOv2-base)."""
    from argus_proof.scoring.scorers import DEFAULT_LIKENESS_MODEL, Dinov2Embedder, LikenessScorer

    return LikenessScorer(Dinov2Embedder(os.environ.get(ENV_LIKENESS_MODEL) or DEFAULT_LIKENESS_MODEL))


def reference_scorers(names: list[str]) -> list:
    """The reference-set scorers *names* asks for (``identity``, ``likeness``), in that order."""
    from argus_proof.scoring.scorers import IdentityScorer

    factories = {"identity": IdentityScorer, "likeness": likeness_scorer}
    unknown = [n for n in names if n not in factories]
    if unknown:
        raise EvaluateError(f"unknown reference scorer(s) {unknown}; expected some of {sorted(factories)}")
    return [factories[n]() for n in names]


def default_phash_scorers() -> tuple:
    """The (deduper, diversity) pair a default run applies — the single source
    both :func:`score_images` and :func:`all_reporting_scorers` build from."""
    from argus_proof.scoring.scorers import PhashDeduper, PhashDiversityScorer

    return PhashDeduper(), PhashDiversityScorer()


def default_pose_scorer():  # noqa: ANN201 - PoseScorer, imported lazily
    """The run-level pose scorer (argus-puppet at ``$PROOF_PUPPET_URL``; unavailable without one)."""
    from argus_proof.scoring.scorers import PoseScorer

    return PoseScorer()


def all_reporting_scorers() -> list:
    """Every scorer a default run applies (per-image set + phash dedup/diversity +
    run-level pose), flat — so a capability probe like the server's ``/scorers``
    reports exactly what :func:`score_images` will run, with no separate list to drift."""
    return [*default_scorers(), *default_phash_scorers(), default_pose_scorer()]


def unavailable_metrics() -> list[str]:
    """The metrics a default run would leave empty because their scorer can't run here.

    Probing imports each scorer's extra (no weights load). A run scored without them
    still produces a report, but with those axes ``None`` and every group pending
    review, which reads like a real verdict unless someone says so (#99).
    """
    return [s.provenance().metric for s in all_reporting_scorers() if not s.is_available()]


def score_images(
    manifest: RunManifest,
    images: list[GeneratedImage],
    *,
    references: list[Path] | None = None,
    gate: GateConfig | None = None,
    scorers: Collection[str] | None = None,
    target_category: TargetCategory = "identity",
    gpu_coordinator: GpuCoordinator | None = None,
    priority: gpu.PriorityClass = gpu.SCORE_PRIORITY,
) -> EvalReport:
    """Score *images* with the default scorers + phash dedup/diversity.

    *scorers* narrows the run to those metrics (:data:`~argus_proof.models.RunScorerName`:
    the default per-image set plus ``pose``); ``None`` runs them all, and phash
    dedup and diversity always run. An unknown name raises :class:`ValueError`.
    *target_category* is the run's target profile: ArcFace ``identity`` scores
    only an ``identity`` run, so a non-face subject (a scene, a helmeted or
    non-human character) can say so and leave ``identity`` ``None`` (#106).

    The scoring pass holds one ``score`` GPU lease (:func:`gpu.score_lease`, from
    *gpu_coordinator*, default the ``ARGUS_GPU_*`` environment) at *priority*
    (``interactive``; a batch run sends ``batch``), taken only when an available
    scorer runs on the GPU. The scorers load their models inside it and free them
    before it ends (:func:`release_models`). A lease that isn't granted raises
    :class:`EvaluateError`.

    With *references*, the run's training export (``manifest.source_manifest``'s
    directory, when it is readable here: :func:`training_export`) is checked first:
    a reference that is a training image raises :class:`EvaluateError`
    (:func:`check_held_out`). The report's ``scorers`` records which check ran, or
    that none could, as ``held-out-references``.

    With ``$PROOF_PUPPET_URL`` set, the run-level pose block (``report.pose``) is
    filled from argus-puppet afterwards, outside the lease (puppet takes its own),
    comparing against the training frames under the same readability rule; *priority*
    goes along as ``X-Argus-Priority``. Puppet being down never fails the run.
    """
    picked = check_run_scorers(scorers)
    export_dir, unreadable = training_export(manifest)
    held_out: str | None = None
    if references:
        held_out = check_held_out(references, export_dir) if export_dir is not None else unreadable
    deduper, diversity = default_phash_scorers()
    ctx = ScoreContext(
        prompt=manifest.prompt,
        profile=TargetProfile(target_category=target_category),
        reference_images=list(references or []),
    )
    live = [s for s in default_scorers() if s.metric in picked]
    report = _score_leased(
        live,
        lambda: score_run(manifest, images, scorers=live, deduper=deduper, diversity=diversity, context=ctx, gate=gate),
        caller=manifest.run_id,
        coordinator=gpu_coordinator,
        priority=priority,
    )
    if held_out is not None:
        report.scorers.append(ScorerProvenance(name=HELD_OUT_CHECK, metric="identity", model=held_out))
    pose = default_pose_scorer()
    if "pose" in picked and pose.is_available():
        # After the lease: puppet takes its own. Never raises for a puppet failure; the block says what happened.
        report.pose = pose.score(
            images,
            training=training_images(export_dir) if export_dir is not None else None,
            export_dir=export_dir,
            training_note=unreadable,
            priority=priority,
        )
        report.scorers.append(pose.provenance())
    return report


def check_run_scorers(scorers: Collection[str] | None) -> set[str]:
    """The metric names a run scores: every :data:`~argus_proof.models.RunScorerName` for ``None``.

    Raises :class:`ValueError` on a name that isn't one, or an empty selection.
    """
    known = get_args(RunScorerName)
    if scorers is None:
        return set(known)
    unknown = sorted(set(scorers) - set(known))
    if unknown:
        raise ValueError(f"unknown scorer(s): {', '.join(unknown)} (known: {', '.join(known)})")
    if not scorers:
        raise ValueError(f"pick at least one scorer (known: {', '.join(known)})")
    return set(scorers)


def training_export(manifest: RunManifest) -> tuple[Path | None, str | None]:
    """The run's curator export dir when it is readable on this host, else ``(None, why not)``.

    The export is ``manifest.source_manifest``'s directory. Both the held-out
    reference check and the pose comparison use it only under this one rule.
    """
    if not manifest.source_manifest:
        return None, "not checked: no training export"
    export_dir = Path(manifest.source_manifest).parent
    if not export_dir.is_dir():
        return None, "not checked: training export not readable here"
    return export_dir, None


def _score_leased(
    scorers: list,
    work,  # noqa: ANN001 - () -> T
    *,
    caller: str,
    coordinator: GpuCoordinator | None,
    priority: gpu.PriorityClass,
):  # noqa: ANN202
    """Run *work* under one ``score`` lease when an available scorer uses the GPU, freeing the models before it ends."""
    from argus_proof.scoring.scorers._util import release_models

    needs_gpu = any(getattr(s, "uses_gpu", True) and s.is_available() for s in scorers)
    lease = (
        gpu.score_lease(coordinator or gpu.default_coordinator(), caller=caller, priority=priority)
        if needs_gpu
        else contextlib.nullcontext()
    )
    # Only acquiring the lease is translated: an error from the work itself (a scorer's own broker call included)
    # propagates as raised, never read as "GPU unavailable".
    stack = contextlib.ExitStack()
    try:
        stack.enter_context(lease)
    except gpu.LEASE_ERRORS as exc:
        raise GpuUnavailable(gpu.lease_failure(exc)) from exc
    with stack:
        try:
            return work()
        finally:
            release_models(scorers)  # inside the lease: the GPU goes back clean


# ---------------------------------------------------------------------------
# scoring images a caller already has (POST /score)
# ---------------------------------------------------------------------------


def prepare_scoring_image(data: bytes, crop: CropBox | None, dest: Path, *, max_side: int) -> tuple[int, int]:
    """Decode an uploaded image, crop it, and write it to *dest* as PNG (no metadata).

    The crop is in the image's own pixels after its EXIF orientation is applied.
    Raises :class:`EvaluateError` for bytes that aren't an image, an image with a
    side over *max_side*, or a crop that isn't inside it. Returns the scored size.
    """
    import io

    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as im:
            if max(im.size) > max_side:
                raise EvaluateError(f"image is {im.width}x{im.height}; at most {max_side} px a side")
            upright = ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise EvaluateError(f"not a readable image: {exc}") from exc
    if crop is not None:
        if crop.x + crop.width > upright.width or crop.y + crop.height > upright.height:
            raise EvaluateError(
                f"crop {crop.x},{crop.y} {crop.width}x{crop.height} is outside the {upright.width}x{upright.height} image"
            )
        upright = upright.crop((crop.x, crop.y, crop.x + crop.width, crop.y + crop.height))
    upright.save(dest, "PNG")
    return upright.width, upright.height


def score_files(
    images: list[Path],
    *,
    export_dir: Path,
    scorers: list[str],
    target_category: str = "identity",
    filenames: list[str | None] | None = None,
    crops: list[CropBox | None] | None = None,
    caller: str = "score",
    gpu_coordinator: GpuCoordinator | None = None,
    priority: gpu.PriorityClass = gpu.SCORE_PRIORITY,
    built: list | None = None,
) -> ScoreResult:
    """Score *images* (already cropped) against the references of the export at *export_dir*.

    The references are ``<export_dir>/references/``; the held-out check runs first
    (:func:`check_held_out`, raising :class:`EvaluateError` on an overlap), then
    each requested scorer must be available (:class:`ScorerUnavailable`), then the
    pass runs under one ``score`` GPU lease (:class:`GpuUnavailable` if refused)
    with the models freed before it ends. Nothing is stored. *built* injects the
    scorer objects (tests); *filenames* and *crops* are echoed per image.
    """
    from argus_proof.scoring.orchestrator import score_metrics
    from argus_proof.scoring.summary import metric_means

    references = reference_images(Path(export_dir) / REFERENCES_DIR)
    if not references:
        raise EvaluateError(f"export {Path(export_dir).name!r} has no {REFERENCES_DIR}/ images to score against")
    held_out = check_held_out(references, Path(export_dir))
    live = built if built is not None else reference_scorers(scorers)
    missing = [s.metric for s in live if not s.is_available()]
    if missing:
        raise ScorerUnavailable(f"scorer(s) not available here: {', '.join(missing)} (install argus-proof[score])")
    ctx = ScoreContext(
        prompt="",
        profile=TargetProfile(target_category=target_category),  # type: ignore[arg-type]
        reference_images=references,
    )
    gens = [
        GeneratedImage(image_id=f"image-{i}", run_id=caller, seed=i, path=str(p), width=0, height=0)
        for i, p in enumerate(images)
    ]

    def work() -> tuple[list, list[ScorerProvenance]]:
        rows = score_metrics(gens, live, ctx)
        return rows, [s.provenance() for s in live]  # while loaded: provenance pins the weights

    rows, provenance = _score_leased(live, work, caller=caller, coordinator=gpu_coordinator, priority=priority)
    names = filenames or [None] * len(images)
    boxes = crops or [None] * len(images)
    return ScoreResult(
        images=[
            ScoredImage(index=i, filename=names[i], crop=boxes[i], metrics=row.metrics) for i, row in enumerate(rows)
        ],
        means=metric_means(rows),
        n_images=len(rows),
        scorers=[*provenance, ScorerProvenance(name=HELD_OUT_CHECK, metric="identity", model=held_out)],
    )


def score_run_dir(
    run_dir: Path,
    *,
    references: list[Path] | None = None,
    gate: GateConfig | None = None,
    scorers: Collection[str] | None = None,
    target_category: TargetCategory = "identity",
) -> EvalReport:
    """Load *run_dir* (manifest + images) and score it into an :class:`EvalReport` (see :func:`score_images`)."""
    manifest = load_manifest(run_dir)
    images = discover_images(run_dir, manifest)
    return score_images(
        manifest, images, references=references, gate=gate, scorers=scorers, target_category=target_category
    )
