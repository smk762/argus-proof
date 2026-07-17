from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from argus_proof.models import (
    PROOF_VERSION,
    SUPPORTED_PROOF_MAJORS,
    AggregateScores,
    EvalReport,
    ImageScores,
    LoRARef,
    MetricScores,
    ModelRef,
    ProofError,
    RejectArchive,
    RejectReason,
    RejectRecord,
    RunManifest,
    SamplingParams,
    ScorerProvenance,
    Verdict,
    check_proof_version,
    proof_major,
    wire_schema,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def make_manifest(**overrides) -> RunManifest:
    base = dict(
        run_id="run-001",
        base_checkpoint=ModelRef(name="sdxl_base.safetensors", sha256=SHA_A),
        vae=ModelRef(name="sdxl_vae.safetensors", sha256=SHA_B),
        loras=[LoRARef(name="subject.safetensors", sha256=SHA_C, weight=0.8)],
        sampling=SamplingParams(
            sampler="dpmpp_2m", scheduler="karras", steps=30, cfg=7.0, clip_skip=2, width=1024, height=1024
        ),
        prompt="a photo of sks person, studio lighting",
        negative_prompt="blurry, lowres",
        seeds=[1, 2, 3],
        engine="comfyui",
        engine_version="0.3.0",
        source_manifest="/exports/run/manifest.jsonl",
        source_manifest_version="2.0",
        training_run_id="forge-abc123",
    )
    base.update(overrides)
    return RunManifest(**base)


# --------------------------------------------------------------------------
# version gate
# --------------------------------------------------------------------------


def test_proof_major_extracts_major() -> None:
    assert proof_major("1.0") == "1"
    assert proof_major("1.7") == "1"
    assert proof_major("2.3") == "2"


def test_current_version_major_is_supported() -> None:
    assert proof_major(PROOF_VERSION) in SUPPORTED_PROOF_MAJORS


def test_check_proof_version_accepts_current() -> None:
    check_proof_version(PROOF_VERSION)  # no raise


def test_check_proof_version_refuses_incompatible_major() -> None:
    with pytest.raises(ProofError, match="not supported"):
        check_proof_version("2.0")


def test_top_level_models_refuse_incompatible_major_on_validate() -> None:
    payload = make_manifest().model_dump()
    payload["proof_version"] = "2.0"
    with pytest.raises(ProofError, match="not supported"):
        RunManifest.model_validate(payload)


def test_top_level_models_default_to_current_version() -> None:
    assert make_manifest().proof_version == PROOF_VERSION


# --------------------------------------------------------------------------
# RunManifest — reproducibility / round-trip
# --------------------------------------------------------------------------


def test_run_manifest_round_trips() -> None:
    manifest = make_manifest()
    restored = RunManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest


def test_run_manifest_pins_files_by_sha256() -> None:
    manifest = make_manifest()
    assert manifest.base_checkpoint.sha256 == SHA_A
    assert manifest.loras[0].sha256 == SHA_C
    assert manifest.loras[0].weight == 0.8


def test_sha256_must_be_hex_digest() -> None:
    with pytest.raises(ValidationError):
        ModelRef(name="x.safetensors", sha256="not-a-hash")
    with pytest.raises(ValidationError):
        ModelRef(name="x.safetensors", sha256="A" * 64)  # uppercase rejected


def test_seeds_may_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        make_manifest(seeds=[])


def test_single_fixed_seed_is_a_one_element_list() -> None:
    assert make_manifest(seeds=[42]).seeds == [42]


# --------------------------------------------------------------------------
# EvalReport
# --------------------------------------------------------------------------


def make_report() -> EvalReport:
    return EvalReport(
        run_id="run-001",
        images=[
            ImageScores(
                image_id="run-001-1",
                seed=1,
                metrics=MetricScores(identity=0.82, clip_score=0.31, aesthetic=6.1),
                hitl_rating=4,
                passed=True,
            ),
            ImageScores(
                image_id="run-001-2",
                seed=2,
                metrics=MetricScores(identity=0.4),
                reject_reasons=[RejectReason(code="identity_mismatch", note="different face")],
                passed=False,
            ),
        ],
        aggregate=AggregateScores(n_images=2, n_passed=1, pass_rate=0.5, means=MetricScores(identity=0.61)),
        scorers=[ScorerProvenance(name="insightface", metric="identity", version="0.7.3", model="buffalo_l")],
        verdict=Verdict(passed=False, reasons=["pass_rate 0.50 < threshold 0.75"]),
    )


def test_eval_report_round_trips() -> None:
    report = make_report()
    assert EvalReport.model_validate_json(report.model_dump_json()) == report


def test_reject_reason_carries_optional_policy_category() -> None:
    # A moderation flag can attribute an unsafe reject to a taxonomy category (#41).
    r = RejectReason(code="unsafe", category="self_harm", note="depicted self-harm")
    assert r.category == "self_harm"
    assert RejectReason.model_validate_json(r.model_dump_json()) == r
    # backward-compatible: category defaults to None for a plain reject
    assert RejectReason(code="anatomy").category is None


def test_missing_scorer_leaves_metric_none_not_zero() -> None:
    scores = MetricScores(identity=0.82)
    assert scores.identity == 0.82
    assert scores.clip_score is None
    assert scores.safety is None


def test_hitl_rating_bounded_one_to_five() -> None:
    with pytest.raises(ValidationError):
        ImageScores(image_id="x", seed=1, hitl_rating=0)
    with pytest.raises(ValidationError):
        ImageScores(image_id="x", seed=1, hitl_rating=6)


# --------------------------------------------------------------------------
# RejectArchive — metadata only, zero image references
# --------------------------------------------------------------------------


def make_archive() -> RejectArchive:
    return RejectArchive(
        manifests={"run-001": make_manifest()},
        records=[
            RejectRecord(
                run_id="run-001",
                seed=2,
                metrics=MetricScores(identity=0.4),
                hitl_rating=1,
                reasons=[RejectReason(code="identity_mismatch")],
            )
        ],
    )


def test_reject_archive_round_trips() -> None:
    archive = make_archive()
    assert RejectArchive.model_validate_json(archive.model_dump_json()) == archive


def test_reject_archive_carries_zero_image_references() -> None:
    """Acceptance: a RejectArchive retains no image/thumbnail path anywhere."""
    blob = json.dumps(make_archive().model_dump(mode="json")).lower()
    for banned in ("image_id", "thumbnail", "thumb", ".png", ".jpg", ".jpeg", ".webp", "path"):
        # `path` in a RejectArchive would only appear via an image field — the
        # manifests it embeds reference models by name+sha, never a filesystem path.
        assert banned not in blob, f"reject archive leaked an image reference: {banned!r}"


def test_reject_record_keys_image_by_seed_not_path() -> None:
    fields = set(RejectRecord.model_fields)
    assert "seed" in fields
    assert not fields & {"image_id", "image_path", "path", "thumbnail"}


# --------------------------------------------------------------------------
# wire schema
# --------------------------------------------------------------------------


def test_wire_schema_includes_all_top_level_models() -> None:
    defs = wire_schema()["$defs"]
    for name in ("RunManifest", "EvalReport", "RejectArchive"):
        assert name in defs


def test_wire_schema_pins_sha256_pattern() -> None:
    props = wire_schema()["$defs"]["ModelRef"]["properties"]
    assert props["sha256"]["pattern"] == r"^[0-9a-f]{64}$"
