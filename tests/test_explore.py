from __future__ import annotations

from argus_proof.explore import (
    image_paths_from_dir,
    ingest_tags,
    is_available,
    sample_fields,
    sample_tags,
)
from argus_proof.models import (
    AggregateScores,
    EvalReport,
    ImageScores,
    MetricScores,
    RejectReason,
    Verdict,
)


def _img(image_id: str, **kw) -> ImageScores:
    return ImageScores(image_id=image_id, seed=kw.pop("seed", 1), **kw)


def _report(images: list[ImageScores]) -> EvalReport:
    n = len(images)
    n_passed = sum(1 for i in images if i.passed)
    return EvalReport(
        run_id="run-1",
        images=images,
        aggregate=AggregateScores(n_images=n, n_groups=n, n_passed=n_passed, pass_rate=0.0, means=MetricScores()),
        verdict=Verdict(passed=False),
    )


def test_sample_fields_exports_set_metrics_and_omits_none() -> None:
    img = _img("a", metrics=MetricScores(identity=0.9, clip_score=0.4), passed=True, hitl_rating=4, duplicate_group=2)
    fields = sample_fields(img)
    assert fields["image_id"] == "a"
    assert fields["verdict"] == "passed"
    assert fields["identity"] == 0.9 and fields["clip_score"] == 0.4
    assert "aesthetic" not in fields and "safety" not in fields  # unset metrics omitted, not null
    assert fields["hitl_rating"] == 4 and fields["duplicate_group"] == 2


def test_sample_fields_verdict_tristate() -> None:
    assert sample_fields(_img("a", passed=True))["verdict"] == "passed"
    assert sample_fields(_img("b", passed=False))["verdict"] == "failed"
    assert sample_fields(_img("c", passed=None))["verdict"] == "needs_review"


def test_sample_fields_reject_reasons_as_codes() -> None:
    img = _img("a", passed=False, reject_reasons=[RejectReason(code="anatomy"), RejectReason(code="artifact")])
    assert sample_fields(img)["reject_reasons"] == ["anatomy", "artifact"]


def test_sample_tags_verdict_and_reject_codes() -> None:
    img = _img("a", passed=False, reject_reasons=[RejectReason(code="unsafe")])
    # reject:/rating: are the INPUT vocabulary — never seeded, else a round-trip
    # re-ingests the run's own auto-rejects as human decisions.
    assert sample_tags(img) == ["failed"]
    assert not any(t.startswith("reject:") for t in sample_tags(img))


def test_sample_fields_and_tags_export_refinement() -> None:
    from argus_proof.models import Refinement

    img = _img("a", passed=True, refinement=Refinement(rank=5, notes="cleanest"))
    fields = sample_fields(img)
    assert fields["refined_rank"] == 5 and fields["refined_notes"] == "cleanest"
    assert "refined" in sample_tags(img)
    # no refinement -> no refined fields/tag
    plain = _img("b", passed=True)
    assert "refined_rank" not in sample_fields(plain) and "refined" not in sample_tags(plain)


def test_ingest_rating_tag_sets_hitl_and_recomputes_verdict() -> None:
    report = _report([_img("a", passed=None)])
    out = ingest_tags(report, {"a": ["rating:5"]}, rater="alice")
    row = next(i for i in out.images if i.image_id == "a")
    assert row.hitl_rating == 5 and row.hitl_rater == "alice"
    assert row.passed is True  # 5 stars -> passes (via apply_hitl)


def test_ingest_reject_tag_fails_image() -> None:
    report = _report([_img("a", passed=True)])
    out = ingest_tags(report, {"a": ["reject:anatomy"]})
    row = next(i for i in out.images if i.image_id == "a")
    assert [r.code for r in row.reject_reasons] == ["anatomy"]
    assert row.passed is False


def test_ingest_is_authoritative_rating_unrejects() -> None:
    # A rating tag with no reject tag CLEARS a prior reject (un-reject), so tagging
    # a previously-failed image rating:5 in the App actually passes it.
    report = _report([_img("a", passed=False, reject_reasons=[RejectReason(code="anatomy")])])
    out = ingest_tags(report, {"a": ["rating:5"]})
    row = next(i for i in out.images if i.image_id == "a")
    assert row.reject_reasons == [] and row.hitl_rating == 5 and row.passed is True


def test_ingest_both_rating_and_reject_reject_wins() -> None:
    report = _report([_img("a", passed=True)])
    out = ingest_tags(report, {"a": ["rating:5", "reject:anatomy"]})
    row = next(i for i in out.images if i.image_id == "a")
    assert [r.code for r in row.reject_reasons] == ["anatomy"] and row.passed is False


def test_ingest_ignores_unknown_image_and_unknown_tags() -> None:
    report = _report([_img("a", passed=True, hitl_rating=5)])
    out = ingest_tags(report, {"ghost": ["rating:1"], "a": ["favourite", "reject:not_a_code"]})
    row = next(i for i in out.images if i.image_id == "a")
    # "a" had no *recognised* tag -> untouched; "ghost" isn't in the report -> skipped
    assert row.hitl_rating == 5 and row.reject_reasons == []


def test_ingest_last_valid_rating_tag_wins() -> None:
    report = _report([_img("a", passed=None)])
    # decreasing sequence so "last wins" is distinguishable from "max wins"
    out = ingest_tags(report, {"a": ["rating:4", "rating:9", "rating:2"]})  # 9 invalid, ignored
    assert next(i for i in out.images if i.image_id == "a").hitl_rating == 2


class _FakeSample:
    def __init__(self, image_id, tags) -> None:  # noqa: ANN001
        self._image_id = image_id
        self.tags = tags

    def get_field(self, name):  # noqa: ANN001, ANN202 - duck-types fiftyone.Sample
        return self._image_id if name == "image_id" else None


def test_dataset_tags_skips_samples_without_image_id() -> None:
    from argus_proof.explore import dataset_tags

    dataset = [_FakeSample("a", ["rating:5"]), _FakeSample(None, ["rating:1"])]  # second has no image_id
    assert dataset_tags(dataset) == {"a": ["rating:5"]}


def test_round_trip_via_dataset_tags() -> None:
    from argus_proof.explore import ingest_from_dataset

    report = _report([_img("a", passed=True), _img("b", passed=True)])
    dataset = [_FakeSample("a", ["reject:anatomy"]), _FakeSample("b", ["rating:5"])]
    out = ingest_from_dataset(dataset, report, rater="alice")
    a = next(i for i in out.images if i.image_id == "a")
    b = next(i for i in out.images if i.image_id == "b")
    assert a.passed is False and [r.code for r in a.reject_reasons] == ["anatomy"]
    assert b.passed is True and b.hitl_rating == 5 and b.hitl_rater == "alice"


def test_image_paths_from_dir_keys_by_stem_and_ignores_sidecars(tmp_path) -> None:  # noqa: ANN001
    (tmp_path / "img-1.png").write_bytes(b"x")
    (tmp_path / "img-1.json").write_text("{}")  # sidecar must NOT shadow the image
    (tmp_path / "img-2.jpg").write_bytes(b"x")
    (tmp_path / ".hidden.png").write_bytes(b"x")
    paths = image_paths_from_dir(tmp_path)
    assert set(paths) == {"img-1", "img-2"}  # hidden + non-image skipped
    assert paths["img-1"].endswith("img-1.png")  # the image, not the .json sidecar


def test_is_available_false_without_extra() -> None:
    import importlib.util

    if importlib.util.find_spec("fiftyone") is not None:  # pragma: no cover - dev has the extra
        import pytest

        pytest.skip("fiftyone installed")
    assert is_available() is False
