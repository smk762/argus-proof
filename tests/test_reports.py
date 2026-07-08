from __future__ import annotations

import pytest

from argus_proof.models import EvalReport, GateConfig, ImageScores, MetricScores, ProofError, RejectReason
from argus_proof.reports import HitlImageUpdate, HitlRequest, ReportStore, apply_hitl, summarise_report
from argus_proof.scoring.summary import summarise


def _row(image_id: str, seed: int, *, passed: bool | None = None, group: int | None = None) -> ImageScores:
    return ImageScores(
        image_id=image_id,
        seed=seed,
        metrics=MetricScores(identity=0.9 if passed else None),
        passed=passed,
        duplicate_group=group,
    )


def _report(run_id: str = "run-1", rows: list[ImageScores] | None = None, gate: GateConfig | None = None) -> EvalReport:
    rows = rows if rows is not None else []
    gate = gate or GateConfig()
    aggregate, verdict = summarise(rows, gate=gate, n_images=len(rows))
    return EvalReport(run_id=run_id, images=rows, aggregate=aggregate, verdict=verdict)


# --- ReportStore -----------------------------------------------------------


def test_store_roundtrip_and_list(tmp_path) -> None:
    store = ReportStore(tmp_path)
    report = _report(rows=[_row("a", 1, passed=True)])
    store.save(report)

    assert store.get("run-1").run_id == "run-1"
    summaries = store.list()
    assert [s.run_id for s in summaries] == ["run-1"]
    assert summaries[0].passed is True


def test_get_missing_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        ReportStore(tmp_path).get("nope")


def test_list_empty_when_dir_absent(tmp_path) -> None:
    assert ReportStore(tmp_path / "does-not-exist").list() == []


@pytest.mark.parametrize("bad", ["../evil", "a/b", "", ".", "..", "a\\b"])
def test_invalid_run_id_rejected(tmp_path, bad: str) -> None:
    with pytest.raises(ProofError):
        ReportStore(tmp_path)._path(bad)


def test_list_skips_unreadable(tmp_path) -> None:
    store = ReportStore(tmp_path)
    store.save(_report(rows=[_row("a", 1, passed=True)]))
    (tmp_path / "junk.json").write_text("{not valid json", encoding="utf-8")
    # The junk file is skipped, the real report still lists.
    assert [s.run_id for s in store.list()] == ["run-1"]


def test_summarise_report_digest() -> None:
    report = _report(rows=[_row("a", 1, passed=True), _row("b", 2, passed=None)])
    digest = summarise_report(report)
    assert digest.n_images == 2
    assert digest.n_needs_hitl == 1
    assert digest.pass_rate == pytest.approx(0.5)


# --- apply_hitl ------------------------------------------------------------


def test_hitl_rating_passes_and_records_rater() -> None:
    report = _report(rows=[_row("a", 1, passed=True), _row("b", 2, passed=None)])
    updated = apply_hitl(
        report,
        HitlRequest(rater="alice", updates=[HitlImageUpdate(image_id="b", hitl_rating=4)]),
    )
    row_b = next(r for r in updated.images if r.image_id == "b")
    assert row_b.passed is True
    assert row_b.hitl_rating == 4
    assert row_b.hitl_rater == "alice"
    # both groups now pass -> run clears the bar
    assert updated.aggregate.pass_rate == pytest.approx(1.0)
    assert updated.verdict.passed is True
    assert updated.verdict.pending is False


def test_hitl_low_rating_fails() -> None:
    report = _report(rows=[_row("a", 1, passed=None)])
    updated = apply_hitl(report, HitlRequest(updates=[HitlImageUpdate(image_id="a", hitl_rating=2)]))
    assert updated.images[0].passed is False


def test_hitl_reject_reason_fails_regardless_of_stars() -> None:
    report = _report(rows=[_row("a", 1, passed=None)])
    updated = apply_hitl(
        report,
        HitlRequest(
            updates=[HitlImageUpdate(image_id="a", hitl_rating=5, reject_reasons=[RejectReason(code="anatomy")])]
        ),
    )
    assert updated.images[0].passed is False
    assert [r.code for r in updated.images[0].reject_reasons] == ["anatomy"]


def test_hitl_rating_one_of_a_dup_group_passes_the_group() -> None:
    rows = [
        _row("a", 1, passed=True, group=0),
        _row("b", 2, passed=None, group=5),
        _row("c", 3, passed=None, group=5),  # near-dup of b
    ]
    report = _report(rows=rows)
    # two groups, one passing -> 0.5
    assert report.aggregate.pass_rate == pytest.approx(0.5)
    updated = apply_hitl(report, HitlRequest(updates=[HitlImageUpdate(image_id="b", hitl_rating=5)]))
    # rating one member passes the whole near-dup group -> both groups pass
    assert updated.aggregate.n_groups == 2
    assert updated.aggregate.pass_rate == pytest.approx(1.0)


def test_hitl_is_pure_does_not_mutate_original() -> None:
    report = _report(rows=[_row("a", 1, passed=None)])
    apply_hitl(report, HitlRequest(rater="bob", updates=[HitlImageUpdate(image_id="a", hitl_rating=5)]))
    assert report.images[0].passed is None
    assert report.images[0].hitl_rater is None


def test_hitl_can_retract_a_rating() -> None:
    # An update fully specifies the decision, so sending null/[] clears a prior
    # rating (the reviewer can un-rate) rather than being ignored.
    report = _report(rows=[_row("a", 1, passed=None)])
    rated = apply_hitl(report, HitlRequest(updates=[HitlImageUpdate(image_id="a", hitl_rating=5)]))
    assert rated.images[0].hitl_rating == 5
    cleared = apply_hitl(rated, HitlRequest(updates=[HitlImageUpdate(image_id="a", hitl_rating=None)]))
    assert cleared.images[0].hitl_rating is None  # retracted, not silently kept


def test_hitl_untouched_image_carried_through() -> None:
    report = _report(rows=[_row("a", 1, passed=None), _row("b", 2, passed=None)])
    updated = apply_hitl(report, HitlRequest(updates=[HitlImageUpdate(image_id="a", hitl_rating=5)]))
    assert next(r for r in updated.images if r.image_id == "b").passed is None
