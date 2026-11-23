from __future__ import annotations

import json

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
    monkeypatch.delenv("ARGUS_PROOF_DEVMODE", raising=False)


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
    assert "report_manifest" in body["capabilities"]  # GET /report/{run_id}/manifest
    assert body["devmode"] is False


def test_health_is_the_suite_service_health(client: TestClient, read_only_client: TestClient) -> None:
    from argus_cortex.service import ServiceHealth

    from argus_proof.models import PROOF_VERSION

    resp = client.get("/health")
    assert resp.status_code == 200
    health = ServiceHealth.model_validate(resp.json())
    assert (health.service, health.version, health.status) == ("argus-proof", __version__, "ok")
    assert health.wire == {"proof_version": PROOF_VERSION}
    assert health.capabilities == ("report_manifest", "score") and health.read_only is False
    assert health.checks == {} and health.started_at is not None
    # the shallow probe a sibling or a container healthcheck uses
    assert client.get("/health?checks=0").json()["status"] == "ok"
    replay = ServiceHealth.model_validate(read_only_client.get("/health").json())
    assert replay.read_only is True and replay.capabilities == ("report_manifest",)


def test_health_names_the_peer_and_build_it_was_told(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.setenv("ARGUS_PEER", "proof")
    monkeypatch.setenv("ARGUS_BUILD", "abc1234")
    body = TestClient(create_app(reports_dir=str(tmp_path))).get("/health").json()
    assert (body["peer"], body["build"]) == ("proof", "abc1234")


@pytest.mark.parametrize("name", ["PROOF_NUDENET_URL", "ARGUS_PUPPET_URL", "COMFYUI_BASE_URL", "ARGUS_TTS_URL"])
def test_a_malformed_peer_url_stops_the_server_at_startup(name: str, monkeypatch, tmp_path) -> None:  # noqa: ANN001
    from argus_cortex.peers import PeerRefused

    monkeypatch.setenv(name, "made-up-host:8000")
    with pytest.raises(PeerRefused, match=name) as refused:
        create_app(reports_dir=str(tmp_path))
    assert "made-up-host" not in str(refused.value)


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


def test_read_only_unrecognised_env_resolves_off(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """A set-but-unrecognised value (a typo) resolves the flag OFF rather than
    on — the fail-open direction is deliberate and pinned so a future change to
    the truthy set can't silently start treating typos as read-only (or vice
    versa). Recognised falsy spellings resolve off too."""
    monkeypatch.setenv("ARGUS_PROOF_READ_ONLY", "enabled")  # plausible typo, not in the truthy set
    typo_app = TestClient(create_app(reports_dir=str(tmp_path)))
    assert typo_app.get("/health").json()["read_only"] is False
    monkeypatch.setenv("ARGUS_PROOF_READ_ONLY", "off")  # explicit falsy spelling
    assert TestClient(create_app(reports_dir=str(tmp_path))).get("/health").json()["read_only"] is False


def test_read_only_403_keeps_cors_headers(tmp_path) -> None:  # noqa: ANN001
    """Cross-origin studio -> demo host: the refused write still carries CORS
    headers (guard sits inside CORS), and preflight OPTIONS is not blocked."""
    app = TestClient(create_app(reports_dir=str(tmp_path), read_only=True, cors=True))
    origin = "http://studio.local"
    refused = app.post("/run/stream", json={}, headers={"Origin": origin})
    assert refused.status_code == 403
    assert refused.headers["access-control-allow-origin"] == "*"
    preflight = app.options("/run/stream", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
    assert preflight.status_code == 200


def test_cors_header(client: TestClient) -> None:
    resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
    # A wildcard never comes with credentials, so the origin isn't reflected.
    assert resp.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in resp.headers


def test_wildcard_cors_does_not_let_any_site_send_credentialed_writes(tmp_path) -> None:  # noqa: ANN001
    app = TestClient(create_app(reports_dir=str(tmp_path), cors=True, devmode=True))
    headers = {"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
    preflight = app.options("/purge", headers=headers)
    assert preflight.headers.get("access-control-allow-credentials") != "true"


def test_explicit_cors_origins_echo_with_credentials(tmp_path) -> None:  # noqa: ANN001
    app = TestClient(create_app(reports_dir=str(tmp_path), cors=True, cors_origins=["http://studio.local"]))
    resp = app.get("/health", headers={"Origin": "http://studio.local"})
    assert resp.headers["access-control-allow-origin"] == "http://studio.local"
    assert resp.headers["access-control-allow-credentials"] == "true"
    other = app.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers


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


def test_report_manifest_serves_the_run_setup(suite_client: TestClient, suite_dirs, monkeypatch) -> None:  # noqa: ANN001
    """Studio's proof -> darkroom handoff reads the validated setup from here."""
    _fake_backend(monkeypatch)
    frames = _stream_frames(
        suite_client,
        {"lora": "subject.safetensors", "base_checkpoint": "sdxl.safetensors", "export": "rufina", "seeds": [1, 2]},
    )
    run_id = frames[-1]["run_id"]

    resp = suite_client.get(f"/report/{run_id}/manifest")
    assert resp.status_code == 200
    body = resp.json()
    on_disk = json.loads((suite_dirs["runs"] / run_id / "manifest.json").read_text(encoding="utf-8"))
    assert body == on_disk
    assert body["run_id"] == run_id
    assert body["seeds"] == [1, 2]
    # ModelRef / LoRARef / SamplingParams field names — what studio prefills from
    assert {"name", "sha256"} <= set(body["base_checkpoint"])
    assert body["loras"] and {"name", "sha256", "weight"} <= set(body["loras"][0])
    assert {"sampler", "scheduler", "steps", "cfg"} <= set(body["sampling"])


def test_report_manifest_missing_cases(suite_client: TestClient, suite_dirs, monkeypatch) -> None:  # noqa: ANN001
    import shutil

    _fake_backend(monkeypatch)
    frames = _stream_frames(
        suite_client,
        {"lora": "subject.safetensors", "base_checkpoint": "sdxl.safetensors", "export": "rufina", "seeds": [1]},
    )
    run_id = frames[-1]["run_id"]
    manifest_path = suite_dirs["runs"] / run_id / "manifest.json"

    # no report at all -> 404; unsafe run_id -> 400 (never joined under the runs root)
    assert suite_client.get("/report/nope/manifest").status_code == 404
    for unsafe in (".hidden", "a%20b", "bad%24id"):  # match the {run_id} segment, so the handler check runs
        assert suite_client.get(f"/report/{unsafe}/manifest").status_code == 400

    # a manifest belonging to another run is refused, not served under this name
    original = manifest_path.read_text(encoding="utf-8")
    data = json.loads(original)
    data["run_id"] = "someone-else"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    assert suite_client.get(f"/report/{run_id}/manifest").status_code == 400

    # corrupt manifest -> 400 (not a 500)
    manifest_path.write_text("{not json", encoding="utf-8")
    assert suite_client.get(f"/report/{run_id}/manifest").status_code == 400

    # report kept, run dir pruned -> 410 Gone (distinguishable from "no such run")
    shutil.rmtree(suite_dirs["runs"] / run_id)
    resp = suite_client.get(f"/report/{run_id}/manifest")
    assert resp.status_code == 410
    assert suite_client.get(f"/report/{run_id}").status_code == 200


def test_report_manifest_served_in_read_only_mode(tmp_path) -> None:  # noqa: ANN001
    """It's a read: the replay/demo host must still serve it."""
    app = TestClient(create_app(reports_dir=str(tmp_path / "reports"), runs_dir=str(tmp_path / "runs"), read_only=True))
    # no report -> 404, i.e. routed and not refused by the write guard
    assert app.get("/report/r/manifest").status_code == 404


def test_scorers_reports_availability(client: TestClient) -> None:
    """UI reads this to warn up-front when the learned scorers aren't installed."""
    scorers = client.get("/scorers").json()["scorers"]
    assert scorers and all({"metric", "name", "available"} <= set(s) for s in scorers)
    # every row names its metric and scorer — incl. the phash dedup/diversity
    # scorers, whose metric comes from provenance() (a bare getattr returned null).
    assert all(s["metric"] and s["name"] for s in scorers)
    assert {"duplicate", "diversity"} <= {s["metric"] for s in scorers}
    # identity for faces and likeness (DINOv2) for everything else are both listed
    assert {"identity", "likeness"} <= {s["metric"] for s in scorers}
    # the run-level pose scorer (argus-puppet), unavailable without PROOF_PUPPET_URL
    assert {"metric": "pose", "name": "argus-puppet", "available": False} in scorers
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


def test_run_stream_refuses_references_that_are_training_images(suite_client, suite_dirs, monkeypatch) -> None:  # noqa: ANN001
    import shutil

    backend = _fake_backend(monkeypatch)
    export = suite_dirs["exports"] / "rufina"
    shutil.copy(export / "references" / "ref.png", export / "img.png")  # the reference is also a training image
    resp = suite_client.post(
        "/run/stream", json={"lora": "l.safetensors", "base_checkpoint": "sdxl.safetensors", "export": "rufina"}
    )
    assert resp.status_code == 400
    assert "training images" in resp.json()["detail"]
    assert backend.generated == []  # refused before any GPU spend


def test_run_stream_success_clears_a_stale_failure_marker(suite_client, suite_dirs, monkeypatch) -> None:  # noqa: ANN001
    from argus_proof.backends.base import FAILURE_NAME, mark_failed

    _fake_backend(monkeypatch)
    run_dir = suite_dirs["runs"] / "retry-1"
    mark_failed(run_dir, "retry-1", "ComfyUI stalled")
    frames = _stream_frames(
        suite_client,
        {
            "lora": "l.safetensors",
            "base_checkpoint": "c.safetensors",
            "prompt": "hi",
            "seeds": [1],
            "run_id": "retry-1",
        },
    )
    assert frames[-1]["type"] == "complete"
    assert not (run_dir / FAILURE_NAME).exists()


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


def test_ids_with_a_trailing_newline_are_refused(suite_client: TestClient) -> None:
    # `$` matches before a final newline, so the id check must use fullmatch: "run-1\n" is not "run-1".
    assert suite_client.get("/report/run-1%0A/image/img-1").status_code == 400
    assert suite_client.get("/report/run-1/image/img-1%0A").status_code == 400


def test_get_image_missing_404(suite_client: TestClient) -> None:
    assert suite_client.get("/report/run-1/image/nope").status_code == 404


def test_run_request_takes_the_checkpoint_familys_sampling_defaults() -> None:
    from argus_proof.server.app import RunRequest

    sdxl = RunRequest(lora="l", base_checkpoint="sdxl/base.safetensors")
    assert (sdxl.cfg, sdxl.sampler, sdxl.scheduler, sdxl.guidance) == (7.0, "dpmpp_2m", "karras", None)
    flux = RunRequest(lora="l", base_checkpoint="flux/flux1-dev.safetensors")
    assert (flux.cfg, flux.sampler, flux.scheduler, flux.guidance) == (1.0, "euler", "simple", 3.5)
    # explicit values win; an explicit null guidance stays off
    custom = RunRequest(lora="l", base_checkpoint="flux/f.safetensors", cfg=2.0, guidance=None)
    assert (custom.cfg, custom.guidance) == (2.0, None)


def test_run_request_takes_the_checkpoint_familys_model_files() -> None:
    from argus_proof.server.app import RunRequest

    sdxl = RunRequest(lora="l", base_checkpoint="sdxl/base.safetensors")
    assert (sdxl.vae, sdxl.text_encoders) == (None, [])
    flux = RunRequest(lora="l", base_checkpoint="flux/flux1-dev.safetensors")
    assert (flux.vae, flux.text_encoders) == ("ae.safetensors", ["t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors"])
    # explicit values win; an explicit null VAE / null encoders stay off
    custom = RunRequest(lora="l", base_checkpoint="flux/f.safetensors", vae=None, text_encoders=None)
    assert (custom.vae, custom.text_encoders) == (None, [])
    fp16 = RunRequest(lora="l", base_checkpoint="flux/f.safetensors", text_encoders=["t5xxl_fp16.safetensors", "c"])
    assert fp16.text_encoders == ["t5xxl_fp16.safetensors", "c"] and fp16.vae == "ae.safetensors"


# --- devmode purge (issue #59) ------------------------------------------------

PURGE = {"confirm": "purge"}


@pytest.fixture
def dev_client(suite_dirs) -> TestClient:  # noqa: ANN001
    return TestClient(
        create_app(
            reports_dir=str(suite_dirs["reports"]),
            runs_dir=str(suite_dirs["runs"]),
            exports_dir=str(suite_dirs["exports"]),
            devmode=True,
        )
    )


def _make_run(client: TestClient, monkeypatch, run_id: str | None = None) -> str:  # noqa: ANN001
    _fake_backend(monkeypatch)
    payload = {"lora": "l.safetensors", "base_checkpoint": "c.safetensors", "prompt": "p", "seeds": [1]}
    if run_id:
        payload["run_id"] = run_id
    return _stream_frames(client, payload)[-1]["run_id"]


def test_purge_routes_absent_without_devmode(suite_client: TestClient, monkeypatch) -> None:  # noqa: ANN001
    run_id = _make_run(suite_client, monkeypatch)
    assert suite_client.post(f"/purge/{run_id}", json=PURGE).status_code == 404
    assert suite_client.post("/purge", json=PURGE).status_code == 404
    health = suite_client.get("/health").json()
    assert health["devmode"] is False and "purge" not in health["capabilities"]
    assert suite_client.get(f"/report/{run_id}").status_code == 200  # untouched


def test_devmode_resolves_from_env(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.setenv("ARGUS_PROOF_DEVMODE", "1")
    health = TestClient(create_app(reports_dir=str(tmp_path))).get("/health").json()
    assert health["devmode"] is True and "purge" in health["capabilities"]


def test_purge_one_run_removes_report_and_run_dir(dev_client: TestClient, suite_dirs, monkeypatch) -> None:  # noqa: ANN001
    keep = _make_run(dev_client, monkeypatch, "keep-me")
    run_id = _make_run(dev_client, monkeypatch, "drop-me")
    # a reviewed run: the HITL data lives in the report and goes with it
    review = {"rater": "r", "updates": [{"image_id": f"{run_id}-1", "hitl_rating": 5}]}
    assert dev_client.post(f"/report/{run_id}/hitl", json=review).status_code == 200

    # confirm body required
    assert dev_client.post(f"/purge/{run_id}").status_code == 400
    assert dev_client.post(f"/purge/{run_id}", json={"confirm": "yes"}).status_code == 400

    resp = dev_client.post(f"/purge/{run_id}", json=PURGE)
    assert resp.status_code == 200
    assert resp.json() == {"run_id": run_id, "report": True, "run_dir": True}
    assert dev_client.get(f"/report/{run_id}").status_code == 404
    assert not (suite_dirs["runs"] / run_id).exists()
    assert [r["run_id"] for r in dev_client.get("/reports").json()["reports"]] == [keep]
    assert (suite_dirs["runs"] / keep).is_dir()

    # gone -> 404; unsafe id -> 400
    assert dev_client.post(f"/purge/{run_id}", json=PURGE).status_code == 404
    assert dev_client.post("/purge/.hidden", json=PURGE).status_code == 400


def test_purge_a_run_dir_without_a_report(dev_client: TestClient, suite_dirs) -> None:  # noqa: ANN001
    (suite_dirs["runs"] / "failed-run").mkdir(parents=True)
    resp = dev_client.post("/purge/failed-run", json=PURGE)
    assert resp.json() == {"run_id": "failed-run", "report": False, "run_dir": True}


def test_purge_unlinks_a_symlinked_run_dir_without_following_it(dev_client: TestClient, suite_dirs, tmp_path) -> None:  # noqa: ANN001
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.png").write_bytes(b"x")
    suite_dirs["runs"].mkdir(parents=True, exist_ok=True)
    (suite_dirs["runs"] / "linked").symlink_to(outside, target_is_directory=True)
    assert dev_client.post("/purge/linked", json=PURGE).status_code == 200
    assert (outside / "precious.png").exists()
    assert not (suite_dirs["runs"] / "linked").exists()


def test_purge_refuses_a_run_in_progress(dev_client: TestClient, monkeypatch) -> None:  # noqa: ANN001
    """A purge must not pull a run dir out from under its generation."""
    import threading

    from fakebackend import FakeBackend

    import argus_proof.evaluate as evaluate

    started, release = threading.Event(), threading.Event()

    class SlowBackend(FakeBackend):
        def generate(self, spec, out_dir, progress=None):  # noqa: ANN001, ANN201
            started.set()
            release.wait(5)
            return super().generate(spec, out_dir, progress=progress)

    monkeypatch.setattr(evaluate, "backend_from_env", lambda *a, **k: SlowBackend())
    payload = {"lora": "l", "base_checkpoint": "c", "prompt": "p", "seeds": [1], "run_id": "busy"}
    done: dict = {}
    worker = threading.Thread(target=lambda: done.setdefault("resp", dev_client.post("/run/stream", json=payload)))
    worker.start()
    try:
        assert started.wait(5)
        assert dev_client.post("/purge/busy", json=PURGE).status_code == 409
        assert dev_client.post("/purge", json=PURGE).status_code == 409
        # and a second stream can't write into the same run dir
        assert dev_client.post("/run/stream", json=payload).status_code == 409
    finally:
        release.set()
        worker.join(10)
    assert done["resp"].status_code == 200
    # once the stream has ended the run is purgeable
    assert dev_client.post("/purge/busy", json=PURGE).status_code == 200


def test_purge_all(dev_client: TestClient, suite_dirs, monkeypatch) -> None:  # noqa: ANN001
    _make_run(dev_client, monkeypatch, "one")
    _make_run(dev_client, monkeypatch, "two")
    (suite_dirs["runs"] / "orphan").mkdir()
    assert dev_client.post("/purge").status_code == 400
    resp = dev_client.post("/purge", json=PURGE)
    assert resp.json() == {"reports": 2, "run_dirs": 3}
    assert dev_client.get("/reports").json() == {"reports": []}
    assert list(suite_dirs["runs"].iterdir()) == []
    # idempotent on an empty store
    assert dev_client.post("/purge", json=PURGE).json() == {"reports": 0, "run_dirs": 0}


def test_purge_refused_in_read_only_mode(tmp_path) -> None:  # noqa: ANN001
    client = TestClient(
        create_app(reports_dir=str(tmp_path), runs_dir=str(tmp_path / "runs"), read_only=True, devmode=True)
    )
    assert client.post("/purge/r", json=PURGE).status_code == 403
    assert client.post("/purge", json=PURGE).status_code == 403
    health = client.get("/health").json()
    # devmode is on but the button would not work, so it isn't advertised
    assert health["devmode"] is True and "purge" not in health["capabilities"]


@pytest.mark.parametrize(
    "override",
    [{"seeds": list(range(65))}, {"steps": 201}, {"width": 4096}, {"height": 100_000}],
)
def test_run_stream_refuses_an_oversized_request(suite_client: TestClient, monkeypatch, override) -> None:  # noqa: ANN001
    _fake_backend(monkeypatch)
    payload = {"lora": "subject.safetensors", "base_checkpoint": "sdxl.safetensors", "prompt": "x", **override}
    assert suite_client.post("/run/stream", json=payload).status_code == 422


def test_an_unexpected_worker_failure_ends_the_stream_with_an_error_frame(
    suite_client: TestClient, monkeypatch
) -> None:  # noqa: ANN001
    backend = _fake_backend(monkeypatch)

    def oom(*a, **kw):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(backend, "generate", oom)
    frames = _stream_frames(
        suite_client,
        {"lora": "subject.safetensors", "base_checkpoint": "sdxl.safetensors", "prompt": "x", "seeds": [1]},
    )
    assert frames[-1]["type"] == "error"
    assert "CUDA out of memory" in frames[-1]["message"]


# --------------------------------------------------------------------------
# per-run scoring choices on /run/stream (#110, #106)
# --------------------------------------------------------------------------


def test_run_stream_passes_its_scoring_choices_and_training_run_id(suite_client, monkeypatch) -> None:  # noqa: ANN001
    import argus_proof.evaluate as evaluate

    _fake_backend(monkeypatch)
    seen: dict = {}
    real = evaluate.score_images

    def recording(manifest, images, **kwargs):  # noqa: ANN001, ANN003, ANN202
        seen.update(kwargs, training_run_id=manifest.training_run_id)
        return real(manifest, images, **kwargs)

    monkeypatch.setattr(evaluate, "score_images", recording)
    gate = {"weights": {"likeness": 1.0}, "hard_gates": {"likeness": 0.6}}
    frames = _stream_frames(
        suite_client,
        {
            "lora": "subject.safetensors",
            "base_checkpoint": "sdxl.safetensors",
            "prompt": "a chrome android",
            "seeds": [1],
            "training_run_id": "forge-run-7",
            "scorers": ["likeness", "aesthetic"],
            "gate": gate,
            "target_category": "concept",
        },
    )
    assert frames[-1]["type"] == "complete", frames[-1]
    assert seen["training_run_id"] == "forge-run-7"  # recorded in the run's manifest
    assert seen["scorers"] == ["likeness", "aesthetic"]
    assert (seen["gate"].weights["likeness"], seen["gate"].hard_gates["likeness"]) == (1.0, 0.6)
    assert seen["target_category"] == "concept"
    report = suite_client.get(f"/report/{frames[-1]['run_id']}").json()
    assert report["gate"]["hard_gates"] == {"likeness": 0.6}  # the run is judged under its own gate


def test_run_stream_defaults_score_everything_under_the_default_gate(suite_client, monkeypatch) -> None:  # noqa: ANN001
    import argus_proof.evaluate as evaluate

    _fake_backend(monkeypatch)
    seen: dict = {}
    real = evaluate.score_images
    monkeypatch.setattr(evaluate, "score_images", lambda m, i, **kw: seen.update(kw) or real(m, i, **kw))
    frames = _stream_frames(
        suite_client,
        {"lora": "subject.safetensors", "base_checkpoint": "sdxl.safetensors", "prompt": "p", "seeds": [1]},
    )
    assert frames[-1]["type"] == "complete"
    assert (seen["scorers"], seen["gate"], seen["target_category"]) == (None, None, "identity")


@pytest.mark.parametrize(
    "bad",
    [
        {"scorers": ["vibes"]},
        {"scorers": []},
        {"scorers": ["likeness", "likeness"]},
        {"target_category": "vibes"},
        {"training_run_id": ""},
        {"gate": {"weights": {"vibes": 1.0}}},
    ],
)
def test_run_stream_refuses_a_bad_scoring_choice(suite_client: TestClient, bad: dict) -> None:
    payload = {"lora": "l", "base_checkpoint": "c", "prompt": "p", **bad}
    assert suite_client.post("/run/stream", json=payload).status_code == 422
