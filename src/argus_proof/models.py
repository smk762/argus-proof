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

from pydantic import BaseModel, Field, model_validator

# Version of the proof wire contract (RunManifest / EvalReport / RejectArchive).
# Bump the minor for backward-compatible additions (a new optional field, a new
# scorer metric); bump the major for a breaking change to an existing field's
# shape or meaning. Every top-level model stamps it so a consumer can refuse an
# incompatible major instead of misreading it.
# 1.0: initial contract — reproducible RunManifest, per-image + aggregate
#      EvalReport with structured reject reasons, image-free RejectArchive.
PROOF_VERSION = "1.0"

# Majors of the proof contract this build understands. Refuse anything else up
# front rather than deserialize it into the wrong shape.
SUPPORTED_PROOF_MAJORS: tuple[str, ...] = ("1",)


def proof_major(version: str) -> str:
    """The major component of a ``proof_version`` string (``'1.4' -> '1'``)."""
    return version.split(".", 1)[0]


class ProofError(RuntimeError):
    """A user-facing failure: incompatible schema, bad input, malformed run."""


def check_proof_version(version: str) -> None:
    """Raise :class:`ProofError` if *version*'s major is not supported.

    The single gate every deserialization path runs through, so an incompatible
    manifest/report is refused with a clear message instead of being silently
    misread. Called from the :class:`_Versioned` validator, so it fires on both
    direct construction and ``model_validate``.
    """
    if proof_major(version) not in SUPPORTED_PROOF_MAJORS:
        understood = ", ".join(f"{m}.x" for m in SUPPORTED_PROOF_MAJORS)
        raise ProofError(
            f"proof_version {version} is not supported (this build understands {understood}) "
            "— upgrade argus-proof or regenerate the run"
        )


# A lowercase hex SHA256 digest. Carried on every model file so a RunManifest
# pins the exact weights it was generated with, not just a filename that can be
# swapped underneath it. The pattern surfaces in the emitted JSON schema.
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", description="lowercase hex SHA256 digest")]


class _Versioned(BaseModel):
    """Base for the three top-level wire models: stamps and checks the version.

    ``proof_version`` defaults to the running build's :data:`PROOF_VERSION`; the
    validator refuses an incompatible major however the object is built.
    """

    proof_version: str = PROOF_VERSION

    @model_validator(mode="after")
    def _check_version(self) -> _Versioned:
        check_proof_version(self.proof_version)
        return self


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
    """A LoRA applied during generation: a :class:`ModelRef` plus its weight."""

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
    ``hitl_rating`` is a 1–5 star human rating when review has happened.
    """

    image_id: str
    seed: int
    metrics: MetricScores = Field(default_factory=MetricScores)
    hitl_rating: int | None = Field(default=None, ge=1, le=5)
    reject_reasons: list[RejectReason] = Field(default_factory=list)
    passed: bool | None = None


class AggregateScores(BaseModel):
    """Run-level roll-up: counts, pass rate, and the mean of each metric axis."""

    n_images: int = Field(ge=0)
    n_passed: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    means: MetricScores = Field(default_factory=MetricScores)


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
    """The pass/fail decision for a run, with the reasons behind it."""

    passed: bool
    reasons: list[str] = Field(default_factory=list)


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
    SamplingParams,
    RunManifest,
    MetricScores,
    RejectReason,
    ImageScores,
    AggregateScores,
    ScorerProvenance,
    Verdict,
    EvalReport,
    RejectRecord,
    RejectArchive,
)


def wire_schema() -> dict:
    """Combined JSON Schema for proof's wire contract (all WIRE_MODELS)."""
    from pydantic.json_schema import models_json_schema

    _, schema = models_json_schema(
        [(m, "serialization") for m in WIRE_MODELS],
        title="argus-proof wire contract",
        ref_template="#/$defs/{model}",
    )
    return schema
