from __future__ import annotations

import math
from pathlib import Path

import pytest
from argus_cortex.taxonomy import TargetProfile
from test_scoring import manifest

from argus_proof.models import GeneratedImage
from argus_proof.scoring import ScoreContext, score_run
from argus_proof.scoring.scorers import IdentityScorer
from argus_proof.scoring.scorers.identity import _ref_key, cosine


class FakeEmbedder:
    """Maps an image path's stem to a fixed vector (None = no face detected)."""

    name = "fake-embedder"

    def __init__(self, vectors: dict[str, list[float] | None], available: bool = True) -> None:
        self.vectors = vectors
        self._available = available
        self.calls: list[str] = []

    def is_available(self) -> bool:
        return self._available

    def embed(self, image_path: Path) -> list[float] | None:
        self.calls.append(Path(image_path).stem)
        return self.vectors.get(Path(image_path).stem)


def gen(image_id: str, seed: int = 1) -> GeneratedImage:
    return GeneratedImage(image_id=image_id, run_id="run-1", seed=seed, path=f"{image_id}.png", width=64, height=64)


def identity_ctx(refs: list[str]) -> ScoreContext:
    return ScoreContext(
        prompt="a photo of sks",
        profile=TargetProfile(target_category="identity"),
        reference_images=[Path(f"{r}.png") for r in refs],
    )


# --------------------------------------------------------------------------
# cosine
# --------------------------------------------------------------------------


def test_cosine_basic() -> None:
    assert cosine([1, 0, 0], [1, 0, 0]) == 1.0
    assert cosine([1, 0, 0], [0, 1, 0]) == 0.0
    assert cosine([1, 0], [0, 0]) == 0.0  # zero vector -> 0, no div-by-zero


# --------------------------------------------------------------------------
# IdentityScorer
# --------------------------------------------------------------------------


def test_matching_face_scores_high() -> None:
    emb = FakeEmbedder({"ref": [1.0, 0.0, 0.0], "gen": [1.0, 0.0, 0.0]})
    score = IdentityScorer(embedder=emb).score(Path("gen.png"), identity_ctx(["ref"]))
    assert score == 1.0


def test_different_face_scores_low() -> None:
    emb = FakeEmbedder({"ref": [1.0, 0.0, 0.0], "gen": [0.0, 1.0, 0.0]})
    assert IdentityScorer(embedder=emb).score(Path("gen.png"), identity_ctx(["ref"])) == 0.0


def test_aggregate_max_picks_best_reference() -> None:
    emb = FakeEmbedder({"refA": [1.0, 0.0], "refB": [0.0, 1.0], "gen": [0.0, 1.0]})
    # gen matches refB exactly, orthogonal to refA -> max = 1.0
    assert IdentityScorer(embedder=emb, aggregate="max").score(Path("gen.png"), identity_ctx(["refA", "refB"])) == 1.0


def test_aggregate_mean_averages_references() -> None:
    emb = FakeEmbedder({"refA": [1.0, 0.0], "refB": [0.0, 1.0], "gen": [0.0, 1.0]})
    # mean of cos(gen,refA)=0 and cos(gen,refB)=1 -> 0.5
    assert IdentityScorer(embedder=emb, aggregate="mean").score(Path("gen.png"), identity_ctx(["refA", "refB"])) == 0.5


def test_no_face_scores_zero() -> None:
    emb = FakeEmbedder({"ref": [1.0, 0.0], "gen": None})  # no face in the generated image
    assert IdentityScorer(embedder=emb).score(Path("gen.png"), identity_ctx(["ref"])) == 0.0


def test_references_without_a_face_return_none_not_zero() -> None:
    # #106: a helmeted or non-human subject's references have no face either, so identity
    # isn't measurable: None, never the 0.0 a lost face scores.
    emb = FakeEmbedder({"ref": None, "gen": None})
    assert IdentityScorer(embedder=emb).score(Path("gen.png"), identity_ctx(["ref"])) is None


def test_no_references_returns_none() -> None:
    emb = FakeEmbedder({"gen": [1.0, 0.0]})
    assert IdentityScorer(embedder=emb).score(Path("gen.png"), identity_ctx([])) is None


def test_non_identity_category_returns_none() -> None:
    emb = FakeEmbedder({"ref": [1.0, 0.0], "gen": [1.0, 0.0]})
    ctx = ScoreContext(
        prompt="x", profile=TargetProfile(target_category="wardrobe"), reference_images=[Path("ref.png")]
    )
    assert IdentityScorer(embedder=emb).score(Path("gen.png"), ctx) is None


def test_negative_cosine_clamped_to_zero() -> None:
    emb = FakeEmbedder({"ref": [1.0, 0.0], "gen": [-1.0, 0.0]})  # cosine -1
    assert IdentityScorer(embedder=emb).score(Path("gen.png"), identity_ctx(["ref"])) == 0.0


def test_nan_embedding_not_scored_as_perfect() -> None:
    # a degenerate (NaN) embedding must NOT clamp to a perfect 1.0; it passes
    # through as NaN so the orchestrator rejects it (the review's headline bug).
    emb = FakeEmbedder({"ref": [1.0, 0.0], "gen": [float("nan"), 0.0]})
    score = IdentityScorer(embedder=emb).score(Path("gen.png"), identity_ctx(["ref"]))
    assert math.isnan(score)


def test_ref_key_changes_with_file_content(tmp_path: Path) -> None:
    p = tmp_path / "ref.png"
    p.write_bytes(b"one")
    key1 = _ref_key(p)
    p.write_bytes(b"different content, new size")
    assert _ref_key(p) != key1  # same path, changed content -> different cache key


def test_ref_key_missing_file_is_stable() -> None:
    assert _ref_key(Path("does-not-exist.png")) == ("does-not-exist.png", -1, -1)


def test_reference_set_embedded_once_and_cached() -> None:
    emb = FakeEmbedder({"ref": [1.0, 0.0], "g1": [1.0, 0.0], "g2": [1.0, 0.0]})
    scorer = IdentityScorer(embedder=emb)
    ctx = identity_ctx(["ref"])
    scorer.score(Path("g1.png"), ctx)
    scorer.score(Path("g2.png"), ctx)
    assert emb.calls.count("ref") == 1  # reference embedded once, not per image


# --------------------------------------------------------------------------
# end-to-end
# --------------------------------------------------------------------------


def test_score_run_fills_identity_metric() -> None:
    emb = FakeEmbedder({"ref": [1.0, 0.0, 0.0], "run-1-1": [1.0, 0.0, 0.0], "run-1-2": [0.0, 1.0, 0.0]})
    report = score_run(
        manifest(),
        [gen("run-1-1"), gen("run-1-2")],
        scorers=[IdentityScorer(embedder=emb)],
        context=identity_ctx(["ref"]),
    )
    assert report.images[0].metrics.identity == 1.0
    assert report.images[1].metrics.identity == 0.0
    assert [p.name for p in report.scorers] == ["identity"]


def test_unavailable_embedder_is_skipped_by_orchestrator() -> None:
    emb = FakeEmbedder({"ref": [1.0, 0.0], "run-1-1": [1.0, 0.0]}, available=False)
    report = score_run(
        manifest(), [gen("run-1-1")], scorers=[IdentityScorer(embedder=emb)], context=identity_ctx(["ref"])
    )
    assert report.images[0].metrics.identity is None
    assert report.scorers == []


def test_score_returns_normalised_value_in_range() -> None:
    emb = FakeEmbedder({"ref": [2.0, 1.0], "gen": [1.0, 2.0]})
    score = IdentityScorer(embedder=emb).score(Path("gen.png"), identity_ctx(["ref"]))
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(cosine([2.0, 1.0], [1.0, 2.0]))


@pytest.mark.parametrize("aggregate", ["max", "mean"])
@pytest.mark.parametrize("refs", [["bad", "good"], ["good", "bad"]])
def test_a_nan_similarity_is_nan_whatever_the_reference_order(aggregate: str, refs: list[str]) -> None:
    emb = FakeEmbedder({"gen": [1.0, 0.0], "good": [1.0, 0.0], "bad": [math.nan, 0.0]})
    scorer = IdentityScorer(embedder=emb, aggregate=aggregate)  # type: ignore[arg-type]
    assert math.isnan(scorer.score(Path("gen.png"), identity_ctx(refs)))


def test_insightface_embedder_honours_exif_orientation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    from argus_proof.scoring.scorers.identity import InsightFaceEmbedder

    path = tmp_path / "sideways.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW to display
    Image.new("RGB", (40, 20)).save(path, exif=exif)
    shapes: list[tuple] = []

    class App:
        def get(self, bgr):  # noqa: ANN001, ANN202
            shapes.append(bgr.shape)
            return []

    embedder = InsightFaceEmbedder()
    monkeypatch.setattr(embedder, "_load", lambda: App())
    assert embedder.embed(path) is None
    assert shapes == [(40, 20, 3)]  # upright: stored 40x20, displayed 20 wide x 40 tall


def test_insightface_without_onnxruntime_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    # _load() imports onnxruntime, which the insightface package doesn't depend on: availability must check it too
    from argus_proof.scoring.scorers import identity

    checked: list[tuple[str, ...]] = []

    def fake(*names: str) -> bool:
        checked.append(names)
        return "onnxruntime" not in names

    monkeypatch.setattr(identity, "module_available", fake)
    assert identity.InsightFaceEmbedder().is_available() is False
    assert "onnxruntime" in checked[0]
