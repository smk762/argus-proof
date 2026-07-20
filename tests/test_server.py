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


@pytest.fixture
def read_only_client(tmp_path) -> TestClient:  # noqa: ANN001
    return TestClient(create_app(reports_dir=str(tmp_path), read_only=True))


@pytest.fixture(autouse=True)
def _isolate_read_only_env(monkeypatch) -> None:  # noqa: ANN001
    """Pin read-only OFF unless a test opts in, so an ambient ARGUS_PROOF_READ_ONLY
    (e.g. a demo-configured shell) can't flip the default-mode assertions. Tests
    that exercise env resolution re-set it via the same monkeypatch."""
    monkeypatch.delenv("ARGUS_PROOF_READ_ONLY", raising=False)


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
    assert body["read_only"] is False  # live host: eval available


def test_health_reports_read_only(read_only_client: TestClient) -> None:
    assert read_only_client.get("/health").json()["read_only"] is True


def test_read_only_mode_403s_writes_but_serves_reads(read_only_client: TestClient) -> None:
    """Replay/demo host: live eval + every mutation refused; reads still served.

    The guard is method-based middleware, so it fires before body validation —
    an invalid body still 403s (never a 422), and future write routes are covered.
    """
    # live GPU eval is the primary target ...
    assert (
        read_only_client.post("/run/stream", json={"lora": "l", "base_checkpoint": "c", "prompt": "p"}).status_code
        == 403
    )
    # ... and every other mutating route (bodies intentionally invalid: 403 wins over 422)
    assert read_only_client.put("/report/r", json={}).status_code == 403
    assert read_only_client.post("/report/r/hitl", json={}).status_code == 403
    assert read_only_client.post("/report/r/refine", json={}).status_code == 403
    # reads still work
    assert read_only_client.get("/reports").status_code == 200
    assert read_only_client.get("/scorers").status_code == 200
    assert read_only_client.get("/health").status_code == 200


def test_read_only_off_by_default_allows_writes(report_client: TestClient) -> None:
    """A normal (non-replay) host still accepts a report write — the guard is off."""
    assert report_client.put("/report/run-1", json=_report("run-1")).status_code == 200


def test_read_only_resolves_from_env(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """The demo host enables the mode via $ARGUS_PROOF_READ_ONLY, not a kwarg;
    an explicit kwarg (the --no-read-only path) still overrides the env."""
    monkeypatch.setenv("ARGUS_PROOF_READ_ONLY", "1")
    env_app = TestClient(create_app(reports_dir=str(tmp_path)))  # no kwarg -> reads env
    assert env_app.get("/health").json()["read_only"] is True
    assert env_app.post("/run/stream", json={"lora": "l", "base_checkpoint": "c", "prompt": "p"}).status_code == 403
    # explicit read_only=False wins over the truthy env var
    off_app = TestClient(create_app(reports_dir=str(tmp_path), read_only=False))
    assert off_app.get("/health").json()["read_only"] is False
    assert off_app.put("/report/run-1", json=_report("run-1")).status_code == 200


def test_read_only_403_keeps_cors_headers(tmp_path) -> None:  # noqa: ANN001
    """Cross-origin studio -> demo host: the refused write still carries CORS
    headers (guard sits inside CORS), and preflight OPTIONS is not blocked."""
    app = TestClient(create_app(reports_dir=str(tmp_path), read_only=True, cors=True))
    origin = "http://studio.local"
    refused = app.post("/run/stream", json={}, headers={"Origin": origin})
    assert refused.status_code == 403
    assert refused.headers["access-control-allow-origin"] == origin
    preflight = app.options("/run/stream", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
    assert preflight.status_code == 200


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


# ---------------------------------------------------------------------------
# refined ranking + retract (issue #35)
# ---------------------------------------------------------------------------


def test_refined_ranking_endpoint_orders_and_summarises(report_client: TestClient) -> None:
    report_client.put("/report/run-1", json=_report("run-1"))
    report_client.post("/report/run-1/refine", json={"updates": [{"image_id": "a", "rank": 4}]})

    body = report_client.get("/report/run-1/refined").json()
    assert [img["image_id"] for img in body["images"]] == ["a"]  # only the passing subset
    assert body["images"][0]["refinement"]["rank"] == 4

    listed = report_client.get("/reports").json()["reports"]
    assert listed[0]["n_refined"] == 1


def test_refine_retract_clears_the_layer(report_client: TestClient) -> None:
    report_client.put("/report/run-1", json=_report("run-1"))
    report_client.post("/report/run-1/refine", json={"updates": [{"image_id": "a", "rank": 4, "notes": "ok"}]})
    resp = report_client.post("/report/run-1/refine", json={"updates": [{"image_id": "a", "rank": None}]})
    assert resp.status_code == 200
    row_a = next(r for r in resp.json()["images"] if r["image_id"] == "a")
    assert row_a["refinement"] is None
    assert report_client.get("/reports").json()["reports"][0]["n_refined"] == 0


def test_refined_missing_report_404(report_client: TestClient) -> None:
    assert report_client.get("/report/nope/refined").status_code == 404


# ---------------------------------------------------------------------------
# exports + models + image serving + run trigger
# ---------------------------------------------------------------------------


@pytest.fixture
def suite_dirs(tmp_path):  # noqa: ANN201
    """A reports/runs/exports layout with one export containing a prompt."""
    from fakebackend import save_png

    exports = tmp_path / "exports"
    export = exports / "rufina"
    export.mkdir(parents=True)
    (export / "manifest.jsonl").write_text('{"rel_path": "img.png"}\n', encoding="utf-8")
    (export / "img.txt").write_text("a photo of sks person", encoding="utf-8")
    save_png(export / "references" / "ref.png")
    return {
        "reports": tmp_path / "reports",
        "runs": tmp_path / "runs",
        "exports": exports,
    }


@pytest.fixture
def suite_client(suite_dirs) -> TestClient:  # noqa: ANN001
    return TestClient(
        create_app(
            reports_dir=str(suite_dirs["reports"]),
            runs_dir=str(suite_dirs["runs"]),
            exports_dir=str(suite_dirs["exports"]),
        )
    )


def test_list_exports(suite_client: TestClient) -> None:
    body = suite_client.get("/exports").json()
    assert body["exports"] == [{"name": "rufina", "n_rows": 1, "has_references": True}]


def test_list_exports_unconfigured_is_empty(report_client: TestClient, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("ARGUS_PROOF_EXPORTS_DIR", raising=False)
    assert report_client.get("/exports").json() == {"exports": []}


def test_list_models(monkeypatch, tmp_path, report_client: TestClient) -> None:  # noqa: ANN001
    (tmp_path / "checkpoints" / "sdxl").mkdir(parents=True)
    (tmp_path / "checkpoints" / "sdxl" / "base.safetensors").write_bytes(b"x")
    (tmp_path / "loras").mkdir()
    (tmp_path / "loras" / "subject.safetensors").write_bytes(b"x")
    (tmp_path / "loras" / "notes.txt").write_bytes(b"x")
    monkeypatch.setenv("PROOF_MODELS_DIR", str(tmp_path))
    body = report_client.get("/models").json()
    assert body == {"checkpoints": ["sdxl/base.safetensors"], "loras": ["subject.safetensors"]}


def _fake_backend(monkeypatch):  # noqa: ANN001, ANN202
    from fakebackend import FakeBackend

    import argus_proof.evaluate as evaluate

    backend = FakeBackend()
    monkeypatch.setattr(evaluate, "backend_from_env", lambda *a, **k: backend)
    return backend


def _stream_frames(client: TestClient, payload: dict) -> list[dict]:
    import json

    resp = client.post("/run/stream", json=payload)
    assert resp.status_code == 200
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def test_run_stream_generates_scores_and_stores(suite_client: TestClient, suite_dirs, monkeypatch) -> None:  # noqa: ANN001
    _fake_backend(monkeypatch)
    frames = _stream_frames(
        suite_client,
        {"lora": "subject.safetensors", "base_checkpoint": "sdxl.safetensors", "export": "rufina", "seeds": [1, 2]},
    )
    types = [f["type"] for f in frames]
    assert types[0] == "start"
    assert "image" in types and "scoring" in types
    assert types[-1] == "complete"

    complete = frames[-1]
    run_id = complete["run_id"]
    assert complete["report"]["n_images"] == 2

    # report persisted and images servable by id
    assert suite_client.get(f"/report/{run_id}").status_code == 200
    image_id = f"{run_id}-1"
    img = suite_client.get(f"/report/{run_id}/image/{image_id}")
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"


def test_image_at_serves_by_index(suite_client: TestClient, monkeypatch) -> None:  # noqa: ANN001
    """The seed-free image address blind review uses: by position, not <run_id>-<seed>."""
    _fake_backend(monkeypatch)
    frames = _stream_frames(
        suite_client,
        {"lora": "subject.safetensors", "base_checkpoint": "sdxl.safetensors", "export": "rufina", "seeds": [1, 2]},
    )
    run_id = frames[-1]["run_id"]

    report = suite_client.get(f"/report/{run_id}").json()
    image_id = report["images"][0]["image_id"]

    img = suite_client.get(f"/report/{run_id}/image_at/0")
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"
    # index 0 resolves to the *same* bytes as the by-id route for that position —
    # proves the position->image_id mapping, not just that some image came back.
    assert img.content == suite_client.get(f"/report/{run_id}/image/{image_id}").content
    assert suite_client.get(f"/report/{run_id}/image_at/1").status_code == 200
    # out of range / negative / unknown run resolve to 404, never a 500
    assert suite_client.get(f"/report/{run_id}/image_at/9").status_code == 404
    assert suite_client.get(f"/report/{run_id}/image_at/-1").status_code == 404
    assert suite_client.get("/report/nope/image_at/0").status_code == 404

    # A stored report can't turn image_at into a path-traversal read: even though
    # PUT accepts an arbitrary image_id, _serve_run_image re-validates it (400).
    report["images"][0]["image_id"] = "../../../../etc/passwd"
    assert suite_client.put(f"/report/{run_id}", json=report).status_code == 200
    assert suite_client.get(f"/report/{run_id}/image_at/0").status_code == 400


def test_scorers_reports_availability(client: TestClient) -> None:
    """UI reads this to warn up-front when the learned scorers aren't installed."""
    scorers = client.get("/scorers").json()["scorers"]
    assert scorers and all({"metric", "name", "available"} <= set(s) for s in scorers)
    # every row names its metric and scorer — incl. the phash dedup/diversity
    # scorers, whose metric comes from provenance() (a bare getattr returned null).
    assert all(s["metric"] and s["name"] for s in scorers)
    assert {"duplicate", "diversity"} <= {s["metric"] for s in scorers}
    # phash dedup/diversity ship in the dev/test env, so at least one is available
    assert any(s["available"] for s in scorers)


def test_run_stream_explicit_prompt_needs_no_export(suite_client: TestClient, monkeypatch) -> None:  # noqa: ANN001
    backend = _fake_backend(monkeypatch)
    frames = _stream_frames(
        suite_client,
        {"lora": "l.safetensors", "base_checkpoint": "c.safetensors", "prompt": "hello", "seeds": [7]},
    )
    assert frames[-1]["type"] == "complete"
    assert backend.generated[0].prompt == "hello"


def test_run_stream_without_prompt_or_export_400(suite_client: TestClient) -> None:
    resp = suite_client.post("/run/stream", json={"lora": "l", "base_checkpoint": "c"})
    assert resp.status_code == 400


def test_run_stream_unknown_export_404(suite_client: TestClient) -> None:
    resp = suite_client.post("/run/stream", json={"lora": "l", "base_checkpoint": "c", "export": "ghost"})
    assert resp.status_code == 404


def test_run_stream_traversal_export_400(suite_client: TestClient) -> None:
    resp = suite_client.post("/run/stream", json={"lora": "l", "base_checkpoint": "c", "export": "../etc"})
    assert resp.status_code == 400


def test_run_stream_backend_failure_streams_error_frame(suite_client: TestClient, monkeypatch) -> None:  # noqa: ANN001
    import argus_proof.evaluate as evaluate

    def boom(*a, **k):  # noqa: ANN002, ANN003, ANN202
        from argus_proof.backends import BackendError

        raise BackendError("engine unreachable")

    monkeypatch.setattr(evaluate, "backend_from_env", boom)
    frames = _stream_frames(suite_client, {"lora": "l", "base_checkpoint": "c", "prompt": "x"})
    assert frames[-1]["type"] == "error"
    assert "engine unreachable" in frames[-1]["message"]


def test_get_image_rejects_traversal_ids(suite_client: TestClient) -> None:
    # Ids that fail the charset check are a 400 from the handler; an encoded
    # slash never even matches the route (404). Either way: rejected.
    assert suite_client.get("/report/run-1/image/.dotfile").status_code == 400
    assert suite_client.get("/report/.escape/image/img-1").status_code == 400
    assert suite_client.get("/report/run-1/image/..%2F..%2Fetc%2Fpasswd").status_code in (400, 404)


def test_get_image_missing_404(suite_client: TestClient) -> None:
    assert suite_client.get("/report/run-1/image/nope").status_code == 404
