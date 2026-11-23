"""``POST /run/stream`` with ``modality: speech`` (#118): references checked first, a speech grid generated through
argus-echo (:mod:`fakeecho`), ending with a ``generated`` frame."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fakeecho import FakeEcho
from fastapi.testclient import TestClient
from test_speech_export import _export, _row

from argus_proof.backends import echo as echo_mod
from argus_proof.backends.base import MANIFEST_NAME
from argus_proof.models import RunStreamFrame, SpeechRunManifest
from argus_proof.server.app import create_app


@pytest.fixture
def speech(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    exports = tmp_path / "exports"
    _export(
        exports / "voice-train",
        [
            _row("a1.wav", split="train", speaker="ana", cluster="spk_1"),
            _row("a2.wav", split="train", speaker="ana", cluster="spk_1"),
        ],
    )
    _export(
        exports / "voice-held",
        [
            _row("c1.wav", split="held_out", speaker="carla", cluster="spk_4"),
            _row("d1.wav", split="held_out", speaker="davi", cluster="spk_5"),
        ],
    )
    _export(exports / "overlapping", [_row("a9.wav", split="held_out", speaker="ana", cluster="spk_9")])
    fake = FakeEcho()
    monkeypatch.setattr(
        echo_mod.EchoClient, "from_env", classmethod(lambda cls, engine="xtts", **kw: fake.client(**kw))
    )
    client = TestClient(
        create_app(reports_dir=str(tmp_path / "reports"), runs_dir=str(tmp_path / "runs"), exports_dir=str(exports))
    )
    return {"client": client, "echo": fake, "runs": tmp_path / "runs"}


BODY = {
    "modality": "speech",
    "engine": "xtts",
    "checkpoint": "forge/run 7",
    "export": "voice-train",
    "references": "voice-held",
    "prompts": ["Bom dia.", "Uma frase mais longa."],
    "seeds": [42],
    "run_id": "speech-1",
    "training_run_id": "forge-run-7",
}


def _frames(client: TestClient, body: dict) -> list[dict]:
    resp = client.post("/run/stream", json=body)
    assert resp.status_code == 200, resp.text
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def test_a_speech_run_generates_the_grid_and_ends_generated(speech: dict) -> None:
    frames = _frames(speech["client"], BODY)
    for frame in frames:
        RunStreamFrame.model_validate(frame)  # every line is a wire frame
    types = [f["type"] for f in frames]
    assert types[0] == "start" and types.count("image") == 4 and types[-2:] == ["done", "generated"]
    last = frames[-1]
    assert (last["n_clips"], last["engine"], last["checkpoint"]) == (4, "argus-echo-xtts", "forge/run 7")
    assert last["speaker_check"] == "speaker_label+speaker_cluster+sha256"

    run_dir = speech["runs"] / "speech-1"
    manifest = SpeechRunManifest.model_validate_json((run_dir / MANIFEST_NAME).read_text())
    assert manifest.weights.fingerprint == last["weights_fingerprint"]
    assert manifest.language == "pt"  # the references' shared label
    assert manifest.training_run_id == "forge-run-7"
    assert manifest.source_manifest.endswith("voice-train/speech_manifest.jsonl")
    assert manifest.references_manifest.endswith("voice-held/speech_manifest.jsonl")
    assert sorted(p.name for p in run_dir.glob("*.wav")) == sorted(f"{c.item_id}.wav" for c in manifest.clips)
    assert speech["echo"].clones == {}
    # No report: nothing is scored until #119.
    assert speech["client"].get("/report/speech-1").status_code == 404


def test_a_reference_whose_speaker_trained_is_refused_before_echo_is_touched(speech: dict) -> None:
    resp = speech["client"].post("/run/stream", json={**BODY, "references": "overlapping"})
    assert resp.status_code == 400
    assert "speaker 'ana' is in training" in resp.json()["detail"]
    assert speech["echo"].requests == []


def test_a_training_export_as_references_is_refused(speech: dict) -> None:
    resp = speech["client"].post("/run/stream", json={**BODY, "references": "voice-train"})
    assert resp.status_code == 400 and "not a held-out export" in resp.json()["detail"]


def test_an_unconfigured_echo_is_a_503(speech: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    def unset(cls, **kw):  # noqa: ANN001, ANN003, ANN202
        raise echo_mod.EchoError("set PROOF_ECHO_URL to the argus-echo engine to generate speech with")

    monkeypatch.setattr(echo_mod.EchoClient, "from_env", classmethod(unset))
    resp = speech["client"].post("/run/stream", json=BODY)
    assert resp.status_code == 503 and "PROOF_ECHO_URL" in resp.json()["detail"]


def test_mixed_reference_languages_need_an_explicit_language(speech: dict, tmp_path: Path) -> None:
    _export(
        tmp_path / "exports" / "mixed",
        [
            _row("e1.wav", split="held_out", speaker="eva", language="pt"),
            _row("f1.wav", split="held_out", speaker="fabio", language="es"),
        ],
    )
    resp = speech["client"].post("/run/stream", json={**BODY, "references": "mixed"})
    assert resp.status_code == 400 and "supply `language`" in resp.json()["detail"]
    frames = _frames(speech["client"], {**BODY, "references": "mixed", "language": "pt", "run_id": "speech-2"})
    assert frames[-1]["type"] == "generated"


def test_an_engine_failure_ends_with_an_error_frame_and_cleans_up(speech: dict) -> None:
    import httpx

    speech["echo"].script["/synthesise/stream"] = [httpx.Response(500, json={"error": "boom", "request_id": "r1"})]
    frames = _frames(speech["client"], BODY)
    assert frames[-1]["type"] == "error" and "boom" in frames[-1]["message"]
    assert speech["echo"].clones == {} and speech["echo"].selected is None
    assert (speech["runs"] / "speech-1" / "failed.json").is_file()


@pytest.mark.parametrize(
    "bad",
    [
        {"engine": "miso"},
        {"prompts": []},
        {"export": "../etc"},
        {"seeds": [2**32]},
        {"references": None},
    ],
)
def test_a_bad_speech_body_is_refused(speech: dict, bad: dict) -> None:
    status = speech["client"].post("/run/stream", json={**BODY, **bad}).status_code
    assert status in (400, 422)
    assert speech["echo"].requests == []


def test_an_image_body_without_modality_is_still_an_image_run(speech: dict) -> None:
    # Every client before #118 sends no modality: it must keep validating as an image RunRequest.
    resp = speech["client"].post("/run/stream", json={"lora": "l"})
    assert resp.status_code == 422
    assert any("base_checkpoint" in str(e.get("loc")) for e in resp.json()["detail"])


def test_the_read_only_mode_refuses_a_speech_run(tmp_path: Path) -> None:
    client = TestClient(create_app(reports_dir=str(tmp_path), read_only=True))
    assert client.post("/run/stream", json=BODY).status_code == 403
