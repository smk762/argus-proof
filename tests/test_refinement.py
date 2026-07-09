from __future__ import annotations

import pytest

from argus_proof.models import (
    AggregateScores,
    EvalReport,
    ImageScores,
    MetricScores,
    RejectReason,
    Verdict,
)
from argus_proof.refinement import (
    RefinementError,
    RefinementImageUpdate,
    RefinementRequest,
    apply_refinement,
    passing_subset,
    refined_ranking,
)


def _img(image_id: str, *, passed: bool, hitl: int | None = None) -> ImageScores:
    return ImageScores(image_id=image_id, seed=1, passed=passed, hitl_rating=hitl)


def _report(images: list[ImageScores]) -> EvalReport:
    n = len(images)
    n_passed = sum(1 for i in images if i.passed)
    return EvalReport(
        run_id="run-1",
        images=images,
        aggregate=AggregateScores(
            n_images=n, n_groups=n, n_passed=n_passed, pass_rate=(n_passed / n if n else 0.0), means=MetricScores()
        ),
        verdict=Verdict(passed=n_passed == n),
    )


def _req(*updates: tuple[str, int], rater: str | None = None) -> RefinementRequest:
    return RefinementRequest(
        rater=rater, updates=[RefinementImageUpdate(image_id=iid, rank=rank) for iid, rank in updates]
    )


def test_passing_subset_is_only_passed_images() -> None:
    report = _report([_img("a", passed=True), _img("b", passed=False), _img("c", passed=None)])  # type: ignore[arg-type]
    assert [i.image_id for i in passing_subset(report)] == ["a"]


def test_apply_refinement_sets_layer_on_passing_image() -> None:
    report = _report([_img("a", passed=True), _img("b", passed=True)])
    out = apply_refinement(report, _req(("a", 5), rater="alice"))
    by_id = {i.image_id: i for i in out.images}
    assert by_id["a"].refinement.rank == 5
    assert by_id["a"].refinement.rater == "alice"
    assert by_id["b"].refinement is None  # untouched


def test_refinement_does_not_touch_first_pass_or_verdict() -> None:
    imgs = [
        ImageScores(image_id="a", seed=1, passed=True, hitl_rating=3, reject_reasons=[RejectReason(code="anatomy")]),
        ImageScores(image_id="b", seed=2, passed=True, hitl_rating=4),
    ]
    report = _report(imgs)
    out = apply_refinement(report, _req(("a", 1), ("b", 5)))
    a = next(i for i in out.images if i.image_id == "a")
    # first-pass values are retained, refinement is additive
    assert a.hitl_rating == 3
    assert [r.code for r in a.reject_reasons] == ["anatomy"]
    assert a.passed is True
    assert a.refinement.rank == 1
    # run-level rollup is unchanged (refinement is not a verdict input)
    assert out.aggregate == report.aggregate
    assert out.verdict == report.verdict


def test_apply_refinement_is_a_copy() -> None:
    report = _report([_img("a", passed=True)])
    apply_refinement(report, _req(("a", 4)))
    assert report.images[0].refinement is None  # original untouched


def test_refining_non_passing_image_raises() -> None:
    report = _report([_img("a", passed=True), _img("b", passed=False)])
    with pytest.raises(RefinementError, match="passing subset"):
        apply_refinement(report, _req(("b", 5)))


def test_refining_unknown_image_raises() -> None:
    report = _report([_img("a", passed=True)])
    with pytest.raises(RefinementError, match="passing subset"):
        apply_refinement(report, _req(("nope", 5)))


def test_refined_ranking_orders_refined_first_then_by_rank() -> None:
    report = _report([_img("a", passed=True, hitl=3), _img("b", passed=True, hitl=5), _img("c", passed=True, hitl=4)])
    out = apply_refinement(report, _req(("a", 5), ("c", 2)))
    ranked = [i.image_id for i in refined_ranking(out)]
    # refined images first (a=5 > c=2), then un-refined (b) last
    assert ranked == ["a", "c", "b"]


def test_refined_ranking_tie_breaks_on_hitl_then_id() -> None:
    report = _report([_img("a", passed=True, hitl=2), _img("b", passed=True, hitl=5)])
    out = apply_refinement(report, _req(("a", 4), ("b", 4)))  # equal refined rank
    ranked = [i.image_id for i in refined_ranking(out)]
    assert ranked == ["b", "a"]  # equal rank -> higher hitl_rating (b) first


def test_rank_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="less than or equal to 5"):
        RefinementImageUpdate(image_id="a", rank=6)


def test_store_refine_round_trips(tmp_path) -> None:  # noqa: ANN001
    from argus_proof.reports import ReportStore

    store = ReportStore(tmp_path)
    store.save(_report([_img("a", passed=True), _img("b", passed=True)]))
    updated = store.refine("run-1", _req(("a", 5), rater="bob"))
    assert next(i for i in updated.images if i.image_id == "a").refinement.rank == 5
    # persisted
    reloaded = store.get("run-1")
    assert next(i for i in reloaded.images if i.image_id == "a").refinement.rater == "bob"
