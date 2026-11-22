"""The wire schema says what the server really sends (issue #58).

Two guards: the set of models that list their defaulted fields as required is
exactly the response-only set, and real payloads carry every field the schema
requires.
"""

from __future__ import annotations

import json
import typing

import pytest
from pydantic import BaseModel, RootModel

from argus_proof import models
from argus_proof.models import RunStreamFrame, wire_schema


def _reachable(*roots: type) -> set[type]:
    seen: set[type] = set()
    todo = list(roots)
    while todo:
        model = todo.pop()
        if model in seen:
            continue
        seen.add(model)
        for field in model.model_fields.values():
            if field.exclude:  # never on the wire, so it has no schema to flag
                continue
            stack = [field.annotation]
            while stack:
                tp = stack.pop()
                if isinstance(tp, type) and issubclass(tp, BaseModel):
                    todo.append(tp)
                stack.extend(typing.get_args(tp))
    return seen


def test_only_models_no_client_authors_list_their_defaults_as_required() -> None:
    pytest.importorskip("fastapi")
    from argus_proof.refinement import RefinementRequest
    from argus_proof.reports import HitlRequest
    from argus_proof.server.app import RunRequest

    # What a client (studio, a user's JSON file, the remote backend's progress
    # stream) builds and proof receives. EvalReport is deliberately absent: PUT
    # /report only stores a report some proof produced (see RESPONSE_ONLY).
    received = _reachable(
        models.RunSpec,
        models.GridConfig,
        models.GateConfig,
        models.AcceptanceThresholds,
        models.ProgressEvent,
        HitlRequest,
        RefinementRequest,
        RunRequest,
        models.ScoreRequest,
        models.SpeechRunSpec,
    )
    sent = _reachable(
        models.EvalReport,
        models.RunManifest,
        models.ReportSummary,
        models.RunStreamFrame,
        models.GateResult,
        models.RejectArchive,
        models.GridPlan,
        models.GridEstimate,
        models.ScoreResult,
        models.SpeechRunManifest,
    )
    # A RootModel (the frame union) has no fields of its own to require.
    ours = {m for m in sent - received if m.__module__ == models.__name__ and not issubclass(m, RootModel)}
    flagged = {
        m for m in _reachable(*models.WIRE_MODELS) if m.model_config.get("json_schema_serialization_defaults_required")
    }

    assert ours and flagged == ours
    # the shapes studio wraps in Filled<> today are among them
    assert {models.ImageScores, models.AggregateScores, models.EvalReport, models.ReportSummary} <= flagged
    # shared with a request: a client may omit their defaults
    assert not {models.SamplingParams, models.RejectReason, models.ProgressEvent} & flagged


def _kinds(schema: dict, defs: dict) -> set[str] | None:
    """The JSON types a (sub)schema accepts, or None when it doesn't say (accepts anything)."""
    if "$ref" in schema:
        schema = defs[schema["$ref"].rsplit("/", 1)[-1]]
    declared = schema.get("type")
    if declared is not None:
        return set(declared) if isinstance(declared, list) else {declared}
    if "properties" in schema or "required" in schema:
        return {"object"}
    if "items" in schema:
        return {"array"}
    return None


def _fits(value: object, branch: dict, defs: dict) -> bool:
    """Whether *value*'s JSON type is one *branch* accepts."""
    kinds = _kinds(branch, defs)
    if kinds is None:
        return True
    if isinstance(value, dict):
        return "object" in kinds
    if isinstance(value, list):
        return "array" in kinds
    return not kinds <= {"object", "array"}


def _missing_required(value: object, schema: dict, defs: dict, path: str = "$") -> list[str]:
    """Paths where *value* lacks a key its schema requires (a small subset of JSON Schema)."""
    if "$ref" in schema:
        return _missing_required(value, defs[schema["$ref"].rsplit("/", 1)[-1]], defs, path)
    for key in ("anyOf", "oneOf"):
        if key in schema:  # among the branches whose type matches the value, the best fit decides
            branches = [b for b in schema[key] if _fits(value, b, defs)]
            if not branches:
                return [f"{path} (matches no {key} branch)"]
            return min((_missing_required(value, b, defs, path) for b in branches), key=len)

    out: list[str] = []
    if isinstance(value, dict):
        out += [f"{path}.{name}" for name in schema.get("required", []) if name not in value]
        for name, sub in schema.get("properties", {}).items():
            if name in value:
                out += _missing_required(value[name], sub, defs, f"{path}.{name}")
        extra = schema.get("additionalProperties")
        if isinstance(extra, dict):
            for name, item in value.items():
                if name not in schema.get("properties", {}):
                    out += _missing_required(item, extra, defs, f"{path}[{name!r}]")
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            out += _missing_required(item, schema["items"], defs, f"{path}[{i}]")
    return out


def test_what_the_server_sends_has_every_field_the_schema_requires(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    pytest.importorskip("fastapi")
    from fakebackend import FakeBackend
    from fastapi.testclient import TestClient

    import argus_proof.evaluate as evaluate
    from argus_proof.server import create_app

    monkeypatch.setattr(evaluate, "backend_from_env", lambda *a, **k: FakeBackend())
    client = TestClient(create_app(reports_dir=str(tmp_path / "reports"), runs_dir=str(tmp_path / "runs")))
    defs = wire_schema()["$defs"]

    def check(model: str, payload: object) -> None:
        assert _missing_required(payload, defs[model], defs) == [], model

    resp = client.post(
        "/run/stream",
        json={"lora": "subject.safetensors", "base_checkpoint": "sdxl.safetensors", "prompt": "p", "seeds": [1, 2]},
    )
    assert resp.status_code == 200, resp.text
    frames = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    types = [f["type"] for f in frames]
    assert "scoring" in types and types[-1] == "complete"
    for frame in frames:
        RunStreamFrame.model_validate(frame)  # every line is a member of the union
        check("RunStreamFrame", frame)
    run_id = frames[-1]["run_id"]

    check("EvalReport", client.get(f"/report/{run_id}").json())
    check("RunManifest", client.get(f"/report/{run_id}/manifest").json())
    reports = client.get("/reports").json()["reports"]
    assert reports
    for summary in reports:
        check("ReportSummary", summary)


def test_a_report_with_a_pose_block_has_every_field_the_schema_requires() -> None:
    from argus_proof.models import (
        AggregateScores,
        EvalReport,
        PoseClosePair,
        PoseComparison,
        PoseCoverage,
        PoseDistanceQuantiles,
        PoseReport,
        Verdict,
    )

    q = PoseDistanceQuantiles(n=3, p01=0.01, p10=0.02, p50=0.05, p90=0.1)
    pose = PoseReport(
        count=2,
        posed=2,
        usable=2,
        diversity=0.4,
        coverage=PoseCoverage(stance={"standing": 2}),
        comparison=PoseComparison(
            training_frames=3,
            output_count=2,
            matched_count=2,
            reference_count=3,
            close_eligible=2,
            baseline_eligible=3,
            threshold=0.04,
            output_nn_distance=q,
            close_pairs=[PoseClosePair(image_id="a", training_frame="t.png", training_sha256="0" * 64, distance=0.01)],
        ),
    )
    report = EvalReport(
        run_id="r", aggregate=AggregateScores(n_images=2, n_passed=0, pass_rate=0.0), verdict=Verdict(passed=False)
    )
    defs = wire_schema()["$defs"]
    payload = report.model_copy(update={"pose": pose}).model_dump(mode="json")
    assert _missing_required(payload, defs["EvalReport"], defs) == []


def test_the_error_frame_is_a_stream_frame(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import argus_proof.evaluate as evaluate
    from argus_proof.server import create_app

    def boom(*a, **k):  # noqa: ANN002, ANN003, ANN202
        raise evaluate.EvaluateError("no backend")

    monkeypatch.setattr(evaluate, "backend_from_env", boom)
    client = TestClient(create_app(reports_dir=str(tmp_path / "reports"), runs_dir=str(tmp_path / "runs")))
    resp = client.post("/run/stream", json={"lora": "l", "base_checkpoint": "c", "prompt": "p"})
    frames = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert frames[-1]["type"] == "error"
    assert RunStreamFrame.model_validate(frames[-1]).root.message == "no backend"


# --- a wire change bumps PROOF_VERSION (CLAUDE.md must-never, #81) ------------------------------------------------

# The wire schema's shape, descriptions stripped (prose may change freely), per PROOF_VERSION. When this test fails
# you changed the wire: bump PROOF_VERSION in models.py (minor for an additive change, major for a breaking one),
# note it in the version history there, regenerate the schema, and add the new version's digest below. Never edit
# an existing entry: a shipped version's shape is fixed.
SCHEMA_DIGESTS = {
    "1.5": "9c349411243316294ca466bb8a267918e882b681193f369fd90983eaeb5aed2a",
    "1.6": "305e949056d00055426bf6bcc007cdaac6ff502861ae645038cedcde65b191fb",
    "1.7": "76cd564e21a838fe0d02327a78224d5d4f6dbe7e473035bf38b319801dc00a94",
    "1.8": "0a6043b7b1b2514d47f202cb36e38255ff8426ff21223fdb4d24b53036deb659",
    "1.9": "9dfbf632a0ec24c01ffd68831d88a8d49da2db9d7c1647c4ac4bc888050f238e",
    "1.10": "869af0fef7715587772af42c176f6321be9487c7989f0415c2b6e9d0c6e7a673",
    "1.11": "5e7b0363319b7aec626e8c9700490755740d7ccf8135b7d26122098a81f88838",
    "1.12": "ce5eaaea3115c6e00c20119bedda9ce5e02c9f6bf29303440ea9490b7fb90a13",
    "1.13": "29f1c8f1fc68a7ea0ed07c15aad42a270e2a074ec2ef7912f052f04851ba3255",
    "1.14": "5748f8c6bb8d258bbec99ab4f7c753a8c7b500b654824b8868998c7ad01cc125",
    "1.15": "e59b46dba545ca052debdfe7552c0e247290587e4ff2246fbf462475f1488371",
}


def _shape_digest() -> str:
    import hashlib

    def strip(node: object) -> object:
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k != "description"}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    return hashlib.sha256(json.dumps(strip(wire_schema()), sort_keys=True).encode()).hexdigest()


def test_a_wire_change_bumps_proof_version() -> None:
    expected = SCHEMA_DIGESTS.get(models.PROOF_VERSION)
    assert expected is not None, f"PROOF_VERSION {models.PROOF_VERSION} has no digest: add {_shape_digest()!r}"
    assert _shape_digest() == expected, (
        f"the wire schema changed but PROOF_VERSION is still {models.PROOF_VERSION}: bump it, then record "
        f"{_shape_digest()!r} under the new version"
    )


def test_the_required_key_walker_only_tries_branches_of_the_values_type() -> None:
    defs = {"Model": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}}
    union = {"anyOf": [{"$ref": "#/$defs/Model"}, {"type": "string"}, {"type": "null"}]}
    assert _missing_required({}, union, defs) == ["$.a"]  # the string branch can't excuse a dict
    assert _missing_required("text", union, defs) == []
    assert _missing_required(None, union, defs) == []
    assert _missing_required([1], union, defs) == ["$ (matches no anyOf branch)"]
