from __future__ import annotations

from pathlib import Path

import pytest

from argus_proof.moderation import (
    CATEGORIES,
    TAXONOMY_VERSION,
    CategoryScores,
    ModerationReport,
    PolicyModerator,
    category_tails,
    moderate_images,
    moderate_texts,
)


class FakeImageDetector:
    """Returns a canned per-image CategoryScores, keyed by the file stem."""

    name = "fake-image"
    version = "v1"

    def __init__(self, by_stem: dict[str, CategoryScores], *, available: bool = True) -> None:
        self._by_stem = by_stem
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def moderate_image(self, image_path: Path) -> CategoryScores | None:
        return self._by_stem.get(Path(image_path).stem)


class FakeTextDetector:
    name = "fake-text"
    version = "v1"

    def __init__(self, by_text: dict[str, CategoryScores]) -> None:
        self._by_text = by_text

    def is_available(self) -> bool:
        return True

    def moderate_text(self, text: str) -> CategoryScores | None:
        return self._by_text.get(text)


def test_taxonomy_is_stable_and_versioned() -> None:
    assert "sexual" in CATEGORIES and "self_harm" in CATEGORIES
    assert "csam" not in CATEGORIES  # illegal-content matching is a separate gate
    assert TAXONOMY_VERSION == "proof-policy-1"


def test_ensemble_takes_the_most_unsafe_per_category() -> None:
    d1 = FakeImageDetector({"a": {"violence": 0.2, "hate": 0.9}})
    d2 = FakeImageDetector({"a": {"violence": 0.8}})
    mod = PolicyModerator(image_detectors=[d1, d2])
    (scores,) = mod.moderate_images([Path("a.png")])
    assert scores["violence"] == 0.8  # max across detectors
    assert scores["hate"] == 0.9


def test_unavailable_or_flaky_detector_is_skipped_not_fatal() -> None:
    class Boom(FakeImageDetector):
        def moderate_image(self, image_path: Path) -> CategoryScores | None:
            raise RuntimeError("model exploded")

    good = FakeImageDetector({"a": {"weapons": 0.7}})
    boom = Boom({})
    absent = FakeImageDetector({"a": {"hate": 1.0}}, available=False)
    mod = PolicyModerator(image_detectors=[boom, good, absent])
    (scores,) = mod.moderate_images([Path("a.png")])
    assert scores == {"weapons": 0.7}  # boom skipped, absent skipped, good kept


def test_category_tails_report_the_extremes() -> None:
    per_item = [
        {"violence": 0.1},
        {"violence": 0.9, "hate": 0.6},
        {"violence": 0.2},
    ]
    tails = category_tails(per_item, unsafe_at=0.5)
    v = tails["violence"]
    assert v.n_items == 3
    assert v.any_hit == 1.0 and v.hit_rate == pytest.approx(1 / 3)  # one item >= 0.5
    assert v.max == 0.9
    # a category no item scored is all-zero, not missing
    assert tails["self_harm"].any_hit == 0.0 and tails["self_harm"].max == 0.0


def test_missing_category_counts_as_safe_zero() -> None:
    # an item that didn't mention 'hate' scores 0.0 for it (no detection = safe)
    tails = category_tails([{"violence": 0.9}], unsafe_at=0.5)
    assert tails["hate"].max == 0.0 and tails["hate"].hit_rate == 0.0


def test_moderate_images_builds_an_output_report(tmp_path: Path) -> None:
    det = FakeImageDetector({"img-1": {"violence": 0.9}, "img-2": {"sexual": 0.3}})
    mod = PolicyModerator(image_detectors=[det])
    report = moderate_images([tmp_path / "img-1.png", tmp_path / "img-2.png"], mod)
    assert report.side == "output" and report.n_items == 2
    assert report.taxonomy_version == TAXONOMY_VERSION and report.detectors == "fake-image@v1"
    assert report.categories["violence"].any_hit == 1.0
    assert report.flagged() == ["violence"]  # only category over the threshold


def test_moderate_texts_flags_a_toxic_prompt() -> None:
    mod = PolicyModerator(text_detectors=[FakeTextDetector({"hurt someone": {"violence": 0.95}})])
    report = moderate_texts(["hurt someone", "a serene landscape"], mod)
    assert report.side == "input" and report.n_items == 2
    assert report.categories["violence"].hit_rate == pytest.approx(0.5)
    assert report.flagged() == ["violence"]


def test_report_flagged_orders_by_severity_and_respects_hit_rate_floor() -> None:
    report = ModerationReport(
        side="output",
        n_items=10,
        unsafe_at=0.5,
        categories=category_tails(
            [{"violence": 0.6}] + [{"hate": 0.9}] * 5,  # hate hits 5/6, violence 1/6
            unsafe_at=0.5,
        ),
    )
    assert report.flagged() == ["hate", "violence"]  # hate worse (max 0.9) first
    assert report.flagged(min_hit_rate=0.5) == ["hate"]  # violence's 1/6 filtered out


def test_provenance_stamps_taxonomy_and_model() -> None:
    mod = PolicyModerator(image_detectors=[FakeImageDetector({})])
    prov = mod.provenance("output")
    assert prov.name == "policy_moderation" and prov.metric == "policy"
    assert prov.version == TAXONOMY_VERSION and prov.model == "fake-image@v1"


def test_default_detectors_report_unavailable_without_the_extra() -> None:
    # No fakes injected -> the lazy LlamaGuard defaults; the [moderation] extra is
    # absent in CI, so the ensemble reports unavailable rather than crashing.
    mod = PolicyModerator()
    assert mod.is_available("output") is False
    assert mod.is_available("input") is False
