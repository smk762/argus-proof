"""The likeness scorer (#103): DINOv2 image-embedding similarity to held-out references.

Every test uses a stub embedder: no model download, no GPU, no torch.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from argus_cortex.taxonomy import TargetProfile
from test_evaluate import _export, _generated_run
from test_scorers_identity import FakeEmbedder, gen
from test_scoring import manifest

import argus_proof.evaluate as evaluate
from argus_proof.hashing import sha256_file
from argus_proof.models import GateConfig, ModelRef
from argus_proof.scoring import ScoreContext, score_run
from argus_proof.scoring.scorers import (
    DEFAULT_LIKENESS_MODEL,
    Dinov2Embedder,
    IdentityScorer,
    LikenessScorer,
)


def ctx(refs: list[str], category: str = "identity") -> ScoreContext:
    return ScoreContext(
        prompt="a chrome android in a corridor",
        profile=TargetProfile(target_category=category),  # type: ignore[arg-type]
        reference_images=[Path(f"{r}.png") for r in refs],
    )


class PinnedEmbedder(FakeEmbedder):
    """A stub that also pins its weights, as Dinov2Embedder does once loaded."""

    def model_ref(self) -> ModelRef:
        return ModelRef(name=self.name, sha256="ab" * 32)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def test_the_same_subject_scores_high_and_another_low() -> None:
    emb = FakeEmbedder({"ref": [1.0, 0.0, 0.0], "same": [1.0, 0.0, 0.0], "other": [0.0, 1.0, 0.0]})
    scorer = LikenessScorer(embedder=emb)
    assert scorer.metric == "likeness"
    assert scorer.score(Path("same.png"), ctx(["ref"])) == 1.0
    assert scorer.score(Path("other.png"), ctx(["ref"])) == 0.0


@pytest.mark.parametrize("category", ["identity", "wardrobe", "pose_composition", "setting", "concept"])
def test_it_scores_every_target_category(category: str) -> None:
    # Unlike ArcFace identity, which only runs on identity runs: a scene or a set is a
    # "setting" run, and its references still say what it should look like.
    emb = FakeEmbedder({"ref": [1.0, 0.0], "gen": [1.0, 0.0]})
    assert LikenessScorer(embedder=emb).score(Path("gen.png"), ctx(["ref"], category)) == 1.0


def test_no_references_returns_none_not_zero() -> None:
    emb = FakeEmbedder({"gen": [1.0, 0.0]})
    assert LikenessScorer(embedder=emb).score(Path("gen.png"), ctx([])) is None


def test_mean_aggregate_and_clamping() -> None:
    emb = FakeEmbedder({"refA": [1.0, 0.0], "refB": [0.0, 1.0], "gen": [0.0, 1.0], "opp": [-1.0, 0.0]})
    assert LikenessScorer(embedder=emb, aggregate="mean").score(Path("gen.png"), ctx(["refA", "refB"])) == 0.5
    assert LikenessScorer(embedder=emb).score(Path("opp.png"), ctx(["refA"])) == 0.0  # cosine -1 clamps to 0


def test_a_nan_embedding_is_passed_on_not_scored_perfect() -> None:
    emb = FakeEmbedder({"ref": [1.0, 0.0], "gen": [math.nan, 0.0]})
    assert math.isnan(LikenessScorer(embedder=emb).score(Path("gen.png"), ctx(["ref"])))


def test_a_masked_subject_gets_likeness_where_identity_finds_no_face() -> None:
    # The helmeted/chrome case: ArcFace sees no face (None -> identity 0.0), DINOv2 still matches.
    face = FakeEmbedder({"ref": [1.0, 0.0], "run-1-1": None})
    image = FakeEmbedder({"ref": [0.0, 1.0], "run-1-1": [0.0, 1.0]})
    report = score_run(
        manifest(),
        [gen("run-1-1")],
        scorers=[IdentityScorer(embedder=face), LikenessScorer(embedder=image)],  # two metrics, no clash
        context=ctx(["ref"]),
    )
    metrics = report.images[0].metrics
    assert (metrics.identity, metrics.likeness) == (0.0, 1.0)
    assert report.aggregate.means.likeness == 1.0
    assert [p.metric for p in report.scorers] == ["identity", "likeness"]


def test_a_likeness_hard_gate_fails_as_identity_mismatch() -> None:
    emb = FakeEmbedder({"ref": [1.0, 0.0], "run-1-1": [0.6, 0.8]})  # cosine 0.6
    report = score_run(
        manifest(),
        [gen("run-1-1")],
        scorers=[LikenessScorer(embedder=emb)],
        context=ctx(["ref"], "setting"),
        gate=GateConfig(hard_gates={"likeness": 0.7}),
    )
    [image] = report.images
    assert image.passed is False
    assert [(r.code, r.note) for r in image.reject_reasons] == [
        ("identity_mismatch", "likeness 0.600 < hard gate 0.700")
    ]


def test_likeness_is_out_of_the_default_composite() -> None:
    # Uncalibrated: it gates only through --min-likeness, a hard gate or an explicit weight.
    assert GateConfig().weights["likeness"] == 0.0
    emb = FakeEmbedder({"ref": [1.0, 0.0], "run-1-1": [0.0, 1.0]})  # likeness 0.0
    [image] = score_run(
        manifest(), [gen("run-1-1")], scorers=[LikenessScorer(embedder=emb)], context=ctx(["ref"])
    ).images
    assert image.metrics.likeness == 0.0
    assert image.passed is None  # nothing weighted was scored: HITL, not an auto-fail


# --------------------------------------------------------------------------
# provenance: the model and its weights, pinned by SHA256
# --------------------------------------------------------------------------


def test_provenance_pins_the_loaded_weights() -> None:
    prov = LikenessScorer(embedder=PinnedEmbedder({})).provenance()
    assert (prov.name, prov.metric, prov.model) == ("likeness", "likeness", "fake-embedder")
    assert prov.model_file == ModelRef(name="fake-embedder", sha256="ab" * 32)


def test_an_embedder_with_nothing_to_pin_records_none() -> None:
    assert LikenessScorer(embedder=FakeEmbedder({})).provenance().model_file is None


def test_dinov2_pins_nothing_before_it_loads(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    embedder = Dinov2Embedder(str(tmp_path))
    assert embedder.weights_file() == tmp_path / "model.safetensors"
    assert embedder.model_ref() is None  # a /scorers probe never hashes or downloads weights


def test_dinov2_pins_the_file_it_loaded_by_sha256(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"dinov2 weights")
    embedder = Dinov2Embedder(str(tmp_path))
    embedder._model = object()  # stands in for a loaded model: no torch here
    ref = embedder.model_ref()
    assert ref == ModelRef(name=f"dinov2:{tmp_path}", sha256=sha256_file(weights))


def test_dinov2_with_no_weights_file_pins_nothing(tmp_path: Path) -> None:
    embedder = Dinov2Embedder(str(tmp_path))  # a dir without model.safetensors
    embedder._model = object()
    assert embedder.weights_file() is None
    assert embedder.model_ref() is None


def test_dinov2_name_carries_the_revision() -> None:
    assert Dinov2Embedder().name == f"dinov2:{DEFAULT_LIKENESS_MODEL}"
    assert Dinov2Embedder("facebook/dinov2-small", revision="abc123").name == "dinov2:facebook/dinov2-small@abc123"


# --------------------------------------------------------------------------
# the default pipeline
# --------------------------------------------------------------------------


def test_default_scorers_include_likeness_from_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(evaluate.ENV_LIKENESS_MODEL, raising=False)
    [likeness] = [s for s in evaluate.default_scorers() if s.metric == "likeness"]
    assert likeness.embedder.model == DEFAULT_LIKENESS_MODEL
    monkeypatch.setenv(evaluate.ENV_LIKENESS_MODEL, "/models/dinov2-large")
    [likeness] = [s for s in evaluate.default_scorers() if s.metric == "likeness"]
    assert likeness.embedder.model == "/models/dinov2-large"
    assert "likeness" in {s.provenance().metric for s in evaluate.all_reporting_scorers()}


def _stub_likeness(monkeypatch: pytest.MonkeyPatch, emb: FakeEmbedder) -> None:
    monkeypatch.setattr(evaluate, "default_scorers", lambda: [LikenessScorer(embedder=emb)])


def test_score_images_reports_likeness_against_held_out_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = _export(tmp_path)
    run_dir = _generated_run(tmp_path)
    _stub_likeness(monkeypatch, FakeEmbedder({"held_out": [1.0, 0.0], "run-1-1": [1.0, 0.0], "run-1-2": [0.0, 1.0]}))
    report = evaluate.score_run_dir(run_dir, references=evaluate.reference_images(export / "references"))
    assert [img.metrics.likeness for img in report.images] == [1.0, 0.0]
    assert report.aggregate.means.likeness == 0.5


def test_likeness_references_that_are_training_images_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same held-out rule as identity: one reference set, one check, before any scorer runs.
    export = _export(tmp_path)
    run_dir = _generated_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    run = evaluate.load_manifest(run_dir).model_copy(update={"source_manifest": str(export / "manifest.jsonl")})
    manifest_path.write_text(run.model_dump_json(), encoding="utf-8")
    emb = FakeEmbedder({})
    _stub_likeness(monkeypatch, emb)
    with pytest.raises(evaluate.EvaluateError, match="training images"):
        evaluate.score_run_dir(run_dir, references=[export / "train" / "a.png"])
    assert emb.calls == []  # refused before anything was embedded
