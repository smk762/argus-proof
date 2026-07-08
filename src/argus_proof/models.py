"""Pydantic models — the proof stage's data contract.

argus-proof is the last stage of the suite: it takes a trained LoRA plus the
curator export manifest it was trained from, generates a sample grid, scores it
(identity / quality / diversity / safety + HITL), and emits a pass/fail verdict
that loops back to improve curation, captioning, and training config.

This module defines the three versioned wire schemas everything else in the
suite reads and writes, mirroring the ``MANIFEST_VERSION`` discipline
argus-curator and argus-forge already use:

* :class:`RunManifest` — a fully reproducible generation run (checkpoint + VAE +
  LoRA(s) with SHA256 hashes, sampling params, prompt, seed-set, engine
  version, and a link back to the source export manifest / forge training run).
* :class:`EvalReport` — per-image and aggregate scores, HITL ratings, structured
  reject reasons, a pass/fail verdict, and the provenance of every scorer.
* :class:`RejectArchive` — the **metadata-only** shape for rejected/flagged
  outputs: params + scores + reasons, with **no image or thumbnail paths
  retained**. A reject is keyed by ``(run_id, seed)`` — the seed plus its
  :class:`RunManifest` reconstructs the image without ever storing it.

Every top-level model carries :data:`PROOF_VERSION`; a consumer refuses an
incompatible major (see :func:`check_proof_version`) instead of misreading it.
"""

from __future__ import annotations

from typing import Annotated, Literal

from argus_cortex.wire import check_version, make_versioned_base, schema_major
from argus_cortex.wire import render_schema as _core_render_schema
from argus_cortex.wire import wire_schema as _core_wire_schema
from pydantic import BaseModel, Field, model_validator

# Version of the proof wire contract (RunManifest / EvalReport / RejectArchive).
# Bump the minor for backward-compatible additions (a new optional field, a new
# scorer metric); bump the major for a breaking change to an existing field's
# shape or meaning. Every top-level model stamps it so a consumer can refuse an
# incompatible major instead of misreading it.
# 1.0: initial contract — reproducible RunManifest, per-image + aggregate
#      EvalReport with structured reject reasons, image-free RejectArchive.
# 1.1: additive — ImageScores.hitl_rater records who rated an image, so a report
#      can be split by rater for inter-rater reliability (issue #10).
PROOF_VERSION = "1.1"

# Majors of the proof contract this build understands. Refuse anything else up
# front rather than deserialize it into the wrong shape.
SUPPORTED_PROOF_MAJORS: tuple[str, ...] = ("1",)

# Title of the emitted JSON Schema (schema/proof-wire.schema.json).
WIRE_TITLE = "argus-proof wire contract"


class ProofError(RuntimeError):
    """A user-facing failure: incompatible schema, bad input, malformed run."""


# The version machinery is shared suite-wide via argus-cortex; proof supplies its
# own field name, version, and error type so a version mismatch is a ProofError.
proof_major = schema_major


def check_proof_version(version: str) -> None:
    """Raise :class:`ProofError` if *version*'s major is not supported."""
    check_version(version, SUPPORTED_PROOF_MAJORS, label="proof_version", error=ProofError)


# A lowercase hex SHA256 digest. Carried on every model file so a RunManifest
# pins the exact weights it was generated with, not just a filename that can be
# swapped underneath it. The pattern surfaces in the emitted JSON schema.
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", description="lowercase hex SHA256 digest")]

# Base for the three top-level wire models: stamps ``proof_version`` and refuses
# an incompatible major (as a ProofError) on construction and model_validate.
_Versioned = make_versioned_base(
    "proof_version", PROOF_VERSION, SUPPORTED_PROOF_MAJORS, label="proof_version", error=ProofError
)


# ---------------------------------------------------------------------------
# RunManifest — a fully reproducible generation run
# ---------------------------------------------------------------------------


class ModelRef(BaseModel):
    """A model file pinned by content: how the engine names it + its SHA256.

    ``name`` is what the generation engine loads (e.g. a ComfyUI checkpoint
    filename); ``sha256`` is the real identity, so a manifest can't drift if the
    file behind the name changes.
    """

    name: str
    sha256: Sha256


class LoRARef(ModelRef):
    """A LoRA applied during generation, as recorded in a resolved run: a
    :class:`ModelRef` (name pinned by SHA256) plus its weight."""

    weight: float = 1.0


class LoRASpec(BaseModel):
    """A LoRA to apply, as *requested* — name + weight, before the file is hashed.

    The request-time counterpart of :class:`LoRARef`: a :class:`RunSpec` names a
    LoRA by filename; the backend resolves it to a file, hashes it, and records
    the resulting :class:`LoRARef` in the :class:`RunManifest`.
    """

    name: str
    weight: float = 1.0


class SamplingParams(BaseModel):
    """The sampler knobs that, with the seed-set, make a run reproducible."""

    sampler: str
    scheduler: str
    steps: int = Field(gt=0)
    cfg: float
    clip_skip: int = 1
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class RunManifest(_Versioned):
    """A fully reproducible generation run: everything needed to recreate it.

    ``seeds`` is the seed strategy made concrete — a single fixed seed for a
    within-checkpoint sweep is a one-element list; a fixed seed-set (so seed luck
    averages out across a cross-checkpoint comparison) is an N-element list.
    Each generated image is identified by ``(run_id, seed)``.

    ``source_manifest`` links back to the curator export ``manifest.jsonl`` the
    LoRA was trained from, and ``training_run_id`` (optional) is the forge
    ``training_run_id`` / ``RunEvent.run_id`` join key when the LoRA came through
    the suite rather than being supplied ad hoc.
    """

    run_id: str
    base_checkpoint: ModelRef
    vae: ModelRef | None = None
    loras: list[LoRARef] = Field(default_factory=list)
    sampling: SamplingParams
    prompt: str
    negative_prompt: str = ""
    seeds: list[int] = Field(min_length=1)
    engine: str
    engine_version: str
    source_manifest: str | None = None
    source_manifest_version: str | None = None
    training_run_id: str | None = None
    created_at: str | None = None


# ---------------------------------------------------------------------------
# Generation — the request a backend runs and what it produces
# ---------------------------------------------------------------------------


class RunSpec(BaseModel):
    """A generation request: what to generate, before a backend resolves it.

    The prompt-grid builder (issue #3) emits a list of these; a
    :class:`~argus_proof.backends.base.GenBackend` turns one into images plus the
    resolved :class:`RunManifest`. Models are named (not yet hashed): the backend
    resolves each name to a file and records its SHA256 in the manifest.

    ``seeds`` is the seed strategy made concrete — one fixed seed for a
    within-checkpoint sweep, or a fixed seed-set (so seed luck averages out) for
    a cross-checkpoint comparison. One image is produced per seed.
    """

    run_id: str
    base_checkpoint: str
    vae: str | None = None
    loras: list[LoRASpec] = Field(default_factory=list)
    sampling: SamplingParams
    prompt: str
    negative_prompt: str = ""
    seeds: list[int] = Field(min_length=1)
    source_manifest: str | None = None
    source_manifest_version: str | None = None
    training_run_id: str | None = None


class GeneratedImage(BaseModel):
    """One image a backend produced, ready to be scored.

    ``image_id`` is the opaque handle scores reference; ``(run_id, seed)`` is the
    reproducible key. ``pnginfo`` is the metadata read back from the PNG the
    engine embedded (ComfyUI PNGInfo), so the recorded params can be checked
    against what actually rendered rather than trusted blind.
    """

    image_id: str
    run_id: str
    seed: int
    path: str
    width: int
    height: int
    pnginfo: dict[str, str] = Field(default_factory=dict)


class BackendCapabilities(BaseModel):
    """What a generation backend can do — its capability descriptor.

    Lets a caller pick or validate a backend without hard-coding which one it
    is: swapping backends is a config change, not a code change.
    """

    name: str
    supports_seed_set: bool = True
    max_loras: int | None = None  # None = unbounded
    reads_pnginfo: bool = False
    streams_progress: bool = False


ProgressType = Literal["start", "progress", "image", "done", "error"]


class ProgressEvent(BaseModel):
    """One NDJSON progress line streamed during generation (suite convention).

    ``type`` selects which fields matter: ``start`` sets ``total`` (images to
    make); ``progress`` sets ``completed``/``total``; ``image`` sets ``seed`` +
    ``image_id`` as each finishes; ``done`` closes the run; ``error`` carries a
    failure ``message``.
    """

    run_id: str
    type: ProgressType
    message: str | None = None
    seed: int | None = None
    image_id: str | None = None
    completed: int | None = None
    total: int | None = None


# ---------------------------------------------------------------------------
# Prompt grid — the matrix of runs that systematically probes a LoRA
# ---------------------------------------------------------------------------


class GridConfig(BaseModel):
    """The axes that expand a source export into a grid of :class:`RunSpec`s.

    Base prompts come from the export's captions (the zeroshot caption variant,
    falling back to the training ``.txt`` sidecar). Those are multiplied across
    the axes below into a deterministic, reproducible set of runs; a control
    ``seeds`` set is shared by every run so seed luck can't skew a comparison.

    * ``lora_checkpoints`` — the LoRA(s) to sweep; supply saved epoch checkpoints
      here to find the under/overtrained sweet spot (cheap, no retrain).
    * ``lora_weights`` — the weight sweep (e.g. 0.6–1.0); optimal weight varies
      per checkpoint and interacts with overtraining.
    * ``token_axes`` — variable tokens (setting/wardrobe/pose …) combined into
      Monte-Carlo combos appended to each base prompt, capped by
      ``max_token_combos`` and sampled deterministically from ``combo_seed`` so
      token effects stay attributable.
    * ``flexibility_prompts`` — off-distribution prompts that require a novel
      attribute, to catch an overfit LoRA that only reproduces its training set.
    """

    base_checkpoint: str
    lora_checkpoints: list[str] = Field(min_length=1)
    lora_weights: list[float] = Field(default_factory=lambda: [1.0], min_length=1)
    sampling: SamplingParams
    negative_prompt: str = ""
    seeds: list[int] = Field(min_length=1)
    token_axes: dict[str, list[str]] = Field(default_factory=dict)
    max_token_combos: int | None = Field(default=None, ge=1)
    max_base_prompts: int | None = Field(default=None, ge=1)
    flexibility_prompts: list[str] = Field(default_factory=list)
    combo_seed: int = 0
    seconds_per_image: float = 6.0  # rough SDXL @ ~30 steps; basis for the GPU-hour estimate
    run_id_prefix: str = "proof"
    source_manifest: str | None = None
    source_manifest_version: str | None = None
    training_run_id: str | None = None


class GridEstimate(BaseModel):
    """The up-front count + GPU-hour estimate reported before any generation.

    ``axes`` breaks the grid down (how many checkpoints × weights × prompts ×
    seeds) so a caller sees why the count is what it is before committing GPU.
    """

    n_runs: int
    n_images: int
    seconds_per_image: float
    est_gpu_seconds: float
    est_gpu_hours: float
    axes: dict[str, int] = Field(default_factory=dict)


class GridPlan(BaseModel):
    """A fully enumerated grid: the runs to generate plus their cost estimate."""

    run_id_prefix: str
    estimate: GridEstimate
    specs: list[RunSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# EvalReport — per-image + aggregate scores, verdict, provenance
# ---------------------------------------------------------------------------

# The automated scoring axes proof computes (phases 2 & 4). Kept as a shared
# shape so a per-image row and the aggregate means line up field-for-field.
# All optional: a scorer that didn't run leaves its field None rather than
# reporting a fabricated zero.


class MetricScores(BaseModel):
    """The scoring axes for one image (or their means, in the aggregate).

    Every field is optional so an axis whose scorer did not run stays ``None``
    instead of a misleading ``0.0``. Semantics: ``identity`` similarity to the
    reference (higher = closer), ``clip_score`` prompt adherence, ``aesthetic``
    an IQA/aesthetic score, ``preference`` an ImageReward/HPS-style score, and
    ``safety`` a safety score.
    """

    identity: float | None = None
    clip_score: float | None = None
    aesthetic: float | None = None
    preference: float | None = None
    safety: float | None = None


# Structured reject reasons — a closed vocabulary so downstream stats can group
# rejects by cause instead of parsing free text. ``note`` carries the detail.
RejectReasonCode = Literal[
    "identity_mismatch",  # doesn't look like the subject
    "prompt_mismatch",  # ignored / contradicted the prompt
    "low_quality",  # blurry, low IQA, compression artifacts
    "anatomy",  # hands, faces, limbs, proportions
    "artifact",  # rendering artifacts / glitches
    "duplicate",  # near-duplicate of another output
    "overfit",  # reproduced training data, failed a flexibility prompt
    "unsafe",  # failed safety evaluation
    "other",  # see note
]


class RejectReason(BaseModel):
    """One structured reason an image was rejected or flagged."""

    code: RejectReasonCode
    note: str | None = None


class ImageScores(BaseModel):
    """Per-image scores for one generated sample, keyed by ``(run_id, seed)``.

    ``image_id`` is an opaque handle for the report to reference; the durable,
    reproducible key is ``seed`` (with the run's :class:`RunManifest`).
    ``hitl_rating`` is a 1–5 star human rating when review has happened;
    ``hitl_rater`` records who gave it (an opaque rater id) so a report can be
    split per rater for inter-rater reliability.
    """

    image_id: str
    seed: int
    metrics: MetricScores = Field(default_factory=MetricScores)
    hitl_rating: int | None = Field(default=None, ge=1, le=5)
    hitl_rater: str | None = None
    reject_reasons: list[RejectReason] = Field(default_factory=list)
    # None = undecided (routed to HITL by the gate); True/False = auto pass/fail.
    passed: bool | None = None
    # Near-duplicate group id; images sharing an id collapse to one unit for the
    # pass-rate math. None when no deduper ran (each image is its own group).
    duplicate_group: int | None = None


class AggregateScores(BaseModel):
    """Run-level roll-up: counts, pass rate, mean of each metric axis, diversity.

    ``pass_rate`` is computed over near-duplicate *groups*, not raw frames, so a
    cluster of Monte-Carlo near-dups counts once: ``n_passed / n_groups``.
    ``n_groups`` equals ``n_images`` when no deduper ran.
    """

    n_images: int = Field(ge=0)
    n_passed: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    means: MetricScores = Field(default_factory=MetricScores)
    # Number of near-dup groups the pass-rate is computed over; the scorer always
    # sets it (equal to n_images when no deduper ran). Optional only for callers
    # that build an AggregateScores by hand.
    n_groups: int | None = Field(default=None, ge=0)
    # Groups the gate routed to human review (composite in the middle band).
    n_needs_hitl: int = Field(default=0, ge=0)
    # Output variety in [0,1] (higher = more varied); None if not measured.
    diversity: float | None = None


class ScorerProvenance(BaseModel):
    """Where a score came from: the scorer, its version, and the model it used.

    Recorded for every scorer that contributed to an :class:`EvalReport` so a
    result is auditable and reproducible — a score is only as trustworthy as the
    model and version that produced it.
    """

    name: str
    metric: str
    version: str | None = None
    model: str | None = None


class Verdict(BaseModel):
    """The automated pass/fail decision for a run, with the reasons behind it.

    ``passed`` reflects the automated scoring alone (groups that auto-passed).
    ``pending`` is True when the run hasn't cleared the bar yet but un-reviewed
    HITL groups could still push it over once rated — so a caller can tell
    "failed" apart from "not judged yet" instead of reading a premature fail.
    """

    passed: bool
    pending: bool = False
    reasons: list[str] = Field(default_factory=list)


class GateConfig(BaseModel):
    """Thresholds that route each image to auto-pass / auto-fail / needs-HITL.

    An automated pre-pass so humans only rate the borderline band (HITL doesn't
    scale to every image). Scorers return a normalised score in ``[0, 1]``
    (higher = better); the gate takes a ``weights``-weighted mean of the metrics
    present (curator's weighted ``score_breakdown`` pattern) into a composite:
    ``composite >= auto_pass`` → pass, ``<= auto_fail`` → fail, between → HITL.
    ``hard_gates`` are absolute per-metric floors (e.g. a minimum identity) that
    fail an image regardless of composite. A run passes when its group pass-rate
    reaches ``run_pass_rate``.

    ``safety`` is in the default composite weights so an unsafe image is dragged
    down once safety scoring lands (Phase 4); for a true veto, add ``safety`` to
    ``hard_gates`` with your scorer's floor — the composite alone can let a
    high-quality-but-unsafe image through.
    """

    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "identity": 1.0,
            "clip_score": 1.0,
            "aesthetic": 1.0,
            "preference": 1.0,
            "safety": 1.0,
        }
    )
    auto_pass: float = Field(default=0.7, ge=0.0, le=1.0)
    auto_fail: float = Field(default=0.4, ge=0.0, le=1.0)
    hard_gates: dict[str, float] = Field(default_factory=dict)
    run_pass_rate: float = Field(default=0.75, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_band(self) -> GateConfig:
        if self.auto_fail > self.auto_pass:
            raise ValueError(f"auto_fail ({self.auto_fail}) must be <= auto_pass ({self.auto_pass})")
        return self


class EvalReport(_Versioned):
    """Scored evaluation of a generation run: per-image + aggregate + verdict.

    Links to its :class:`RunManifest` by ``run_id`` rather than embedding it, so
    the run dir stays the single source of truth for generation params.
    """

    run_id: str
    images: list[ImageScores] = Field(default_factory=list)
    aggregate: AggregateScores
    scorers: list[ScorerProvenance] = Field(default_factory=list)
    verdict: Verdict
    created_at: str | None = None


# ---------------------------------------------------------------------------
# Acceptance gate — the CI-consumable pass/fail verdict against thresholds
# ---------------------------------------------------------------------------


class AcceptanceThresholds(BaseModel):
    """Declared thresholds a run must clear to be accepted (the CI gate, #12).

    Turns "was this LoRA/dataset good enough?" into an automatable yes/no. Each
    check runs only when its threshold is set; ``min_pass_rate`` is **on by
    default** (set it ``None`` to skip), the rest default off. If no check is
    configured the gate rejects rather than accept on zero evidence.
    ``min_pass_rate_ci_lower`` is the stricter, small-N-safe version of
    ``min_pass_rate`` — the Wilson lower bound must clear it, so a lucky 3/3
    doesn't pass a bar that 300/400 would. ``max_unsafe_rate`` counts an image
    unsafe when its ``safety`` metric is below ``unsafe_safety_floor`` or it
    carries an ``unsafe`` reject reason.
    """

    min_pass_rate: float | None = Field(default=0.75, ge=0.0, le=1.0)
    min_pass_rate_ci_lower: float | None = Field(default=None, ge=0.0, le=1.0)
    min_identity_mean: float | None = Field(default=None, ge=0.0, le=1.0)
    max_unsafe_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    unsafe_safety_floor: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)


class GateCheck(BaseModel):
    """One threshold check's outcome, with the actual value vs the threshold."""

    name: str
    passed: bool
    actual: float | None = None
    threshold: float | None = None
    detail: str = ""


class GateResult(BaseModel):
    """The overall accept/reject decision — ``passed`` iff every check passed."""

    passed: bool
    checks: list[GateCheck] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# RejectArchive — metadata-only archival of rejected / flagged outputs
# ---------------------------------------------------------------------------


class RejectRecord(BaseModel):
    """One rejected/flagged output, described without retaining its image.

    Keyed by ``(run_id, seed)``: the seed plus the run's :class:`RunManifest`
    reconstructs the exact image on demand, so nothing about the pixels needs to
    be stored. There is deliberately no path, thumbnail, or image field here —
    that is the whole point of the archive.
    """

    run_id: str
    seed: int
    metrics: MetricScores = Field(default_factory=MetricScores)
    hitl_rating: int | None = Field(default=None, ge=1, le=5)
    reasons: list[RejectReason] = Field(default_factory=list)
    # Who flagged/rated it, when a human was in the loop (for inter-rater analysis).
    rater_id: str | None = None


class RejectArchive(_Versioned):
    """Metadata-only archive of rejected/flagged outputs — zero image references.

    ``manifests`` (keyed by ``run_id``) carries the full generation params for
    every run a record came from, so the archive is self-describing: you can see
    exactly what produced a bad output without keeping the output itself. Both
    :class:`RunManifest` and :class:`RejectRecord` are image-free by
    construction, so the archive retains no image or thumbnail paths.
    """

    manifests: dict[str, RunManifest] = Field(default_factory=dict)
    records: list[RejectRecord] = Field(default_factory=list)


# Models that make up the HTTP/CLI wire contract, in schema order. Consumers
# (argus-studio) codegen against the emitted schema/proof-wire.schema.json.
WIRE_MODELS: tuple[type[BaseModel], ...] = (
    ModelRef,
    LoRARef,
    LoRASpec,
    SamplingParams,
    RunManifest,
    RunSpec,
    GeneratedImage,
    BackendCapabilities,
    ProgressEvent,
    GridConfig,
    GridEstimate,
    GridPlan,
    MetricScores,
    RejectReason,
    ImageScores,
    AggregateScores,
    ScorerProvenance,
    Verdict,
    GateConfig,
    EvalReport,
    AcceptanceThresholds,
    GateCheck,
    GateResult,
    RejectRecord,
    RejectArchive,
)


def wire_schema() -> dict:
    """Combined JSON Schema for proof's wire contract (all WIRE_MODELS)."""
    return _core_wire_schema(WIRE_MODELS, title=WIRE_TITLE)


def render_wire_schema() -> str:
    """The canonical committed-schema string (sorted, indented, newline-terminated).

    Delegates to argus_cortex.wire.render_schema so proof's ``schema`` /
    ``schema --check`` formatting stays identical to the rest of the suite.
    """
    return _core_render_schema(WIRE_MODELS, title=WIRE_TITLE)
