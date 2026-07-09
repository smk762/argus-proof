from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from argus_proof import __version__
from argus_proof.models import EvalReport, GateConfig, ImageScores, MetricScores
from argus_proof.scoring.summary import summarise
from argus_proof.server import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(cors=True))


@pytest.fixture
def report_client(tmp_path) -> TestClient:
    return TestClient(create_app(reports_dir=str(tmp_path)))


def _report(run_id: str = "run-1") -> dict:
    rows = [
        ImageScores(image_id="a", seed=1, metrics=MetricScores(identity=0.9), passed=True, duplicate_group=0),
        ImageScores(image_id="b", seed=2, metrics=MetricScores(), passed=None, duplicate_group=1),
    ]
    aggregate, verdict = summarise(rows, gate=GateConfig(), n_images=len(rows))
    return EvalReport(run_id=run_id, images=rows, aggregate=aggregate, verdict=verdict).model_dump(mode="json")


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["service"] == "argus-proof"
    assert body["version"] == __version__


def test_cors_header(client: TestClient) -> None:
    resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
    # allow_credentials=True makes CORSMiddleware echo the origin rather than "*"
    assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_put_get_and_list_report(report_client: TestClient) -> None:
    put = report_client.put("/report/run-1", json=_report("run-1"))
    assert put.status_code == 200

    got = report_client.get("/report/run-1")
    assert got.status_code == 200
    assert got.json()["run_id"] == "run-1"

    listed = report_client.get("/reports").json()["reports"]
    assert [r["run_id"] for r in listed] == ["run-1"]
    assert listed[0]["n_needs_hitl"] == 1


def test_get_missing_report_404(report_client: TestClient) -> None:
    assert report_client.get("/report/nope").status_code == 404


def test_put_run_id_mismatch_400(report_client: TestClient) -> None:
    resp = report_client.put("/report/other", json=_report("run-1"))
    assert resp.status_code == 400


def test_hitl_recomputes_verdict(report_client: TestClient) -> None:
    report_client.put("/report/run-1", json=_report("run-1"))
    # image "b" was undecided; rating it 5 passes it -> both groups pass
    resp = report_client.post(
        "/report/run-1/hitl",
        json={"rater": "alice", "updates": [{"image_id": "b", "hitl_rating": 5}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"]["passed"] is True
    row_b = next(r for r in body["images"] if r["image_id"] == "b")
    assert row_b["hitl_rater"] == "alice"
    # persisted: a re-fetch reflects the review
    assert report_client.get("/report/run-1").json()["verdict"]["passed"] is True


def test_hitl_missing_report_404(report_client: TestClient) -> None:
    resp = report_client.post("/report/nope/hitl", json={"updates": []})
    assert resp.status_code == 404


def test_refine_adds_layer_without_touching_verdict(report_client: TestClient) -> None:
    report_client.put("/report/run-1", json=_report("run-1"))
    before = report_client.get("/report/run-1").json()["verdict"]
    resp = report_client.post(
        "/report/run-1/refine",
        json={"rater": "bob", "updates": [{"image_id": "a", "rank": 5, "notes": "crisp"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    row_a = next(r for r in body["images"] if r["image_id"] == "a")
    assert row_a["refinement"]["rank"] == 5 and row_a["refinement"]["rater"] == "bob"
    assert body["verdict"] == before  # refinement is additive, verdict unchanged
    # persisted
    assert report_client.get("/report/run-1").json()["images"][0]["refinement"]["notes"] == "crisp"


def test_refine_non_passing_image_400(report_client: TestClient) -> None:
    report_client.put("/report/run-1", json=_report("run-1"))
    # image "b" did not pass -> refining it is a 400
    resp = report_client.post("/report/run-1/refine", json={"updates": [{"image_id": "b", "rank": 3}]})
    assert resp.status_code == 400


def test_refine_missing_report_404(report_client: TestClient) -> None:
    resp = report_client.post("/report/nope/refine", json={"updates": []})
    assert resp.status_code == 404
