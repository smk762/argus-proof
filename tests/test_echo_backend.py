"""The argus-echo speech backend (#118): clones per run, seeded utterances, weights pinned per clip.

Every test runs against :mod:`fakeecho` (an ``httpx.MockTransport``): no network, no engine.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import httpx
import pytest
from fakeecho import URL, FakeEcho, resolve
from test_gpu import Recording

from argus_proof import gpu
from argus_proof.backends.base import FAILURE_NAME, MANIFEST_NAME, BackendError
from argus_proof.backends.echo import (
    EchoBackend,
    EchoClient,
    EchoError,
    clone_id,
    multipart,
    wav_float32,
    weights_fingerprint,
)
from argus_proof.models import SpeechReferenceSpec, SpeechRunManifest, SpeechRunSpec


def _refs(tmp_path: Path, n: int = 2) -> list[SpeechReferenceSpec]:
    refs = []
    for i in range(n):
        clip = tmp_path / "held-out" / f"spk{i}" / f"clip{i}.wav"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(wav_float32(struct.pack("<4f", 0.1 * i, 0.2, 0.3, 0.4), 24000))
        refs.append(
            SpeechReferenceSpec(
                ref_id=f"r{i}",
                path=str(clip),
                rel_path=f"spk{i}/clip{i}.wav",
                transcript=f"reference {i}",
                speaker_label=f"spk{i}",
                accent="carioca",
            )
        )
    return refs


def _spec(tmp_path: Path, **kw) -> SpeechRunSpec:  # noqa: ANN003
    base = {
        "run_id": "speech-run-1",
        "engine": "xtts",
        "checkpoint": "forge/run 7",
        "language": "pt",
        "prompts": ["Bom dia.", "Uma frase bem mais longa, para ver a voz se manter até o fim."],
        "references": _refs(tmp_path),
        "seeds": [42, 43],
    }
    return SpeechRunSpec(**{**base, **kw})


def _backend(echo: FakeEcho, coordinator: Recording | None = None) -> EchoBackend:
    return EchoBackend(
        echo.client(), image="ghcr.io/smk762/argus-echo-xtts:0.2.0", coordinator=coordinator or Recording()
    )


# --------------------------------------------------------------------------
# the wire helpers
# --------------------------------------------------------------------------


def test_the_fingerprint_matches_echos_documented_examples() -> None:
    # echo docs/api.md, "Weights fingerprint": the sha256 values are those of the strings weights, config and base.
    w, c, b = (hashlib.sha256(s.encode()).hexdigest() for s in ("weights", "config", "base"))
    files = [("custom_voice/model.safetensors", w), ("custom_voice/config.json", c), ("base/model.safetensors", b)]
    assert weights_fingerprint(files[:2]) == "b955ea711653a31092cf4acad596e71ac31cb8f85c6feabe209f32ada86fecfb"
    assert weights_fingerprint(files[2:]) == "8763f2a5235402fc83d026fd46b563e7fa6c7b0c9d12a3cad770d4856148eebc"
    assert weights_fingerprint(files) == "e205d97735019470b091f339590c5f050f068b0f7817439400ed4337db73c89c"
    assert (
        weights_fingerprint([("config.json", c), ("model.pth", w)])
        == "311b8ed7abed4ae45274fedccda241ff6d0a129db13ea34b25b9fe0431de2649"
    )


def test_the_wav_holds_echos_float32_samples_unchanged() -> None:
    pcm = struct.pack("<3f", 0.5, -0.25, 1.0)
    wav = wav_float32(pcm, 24000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert struct.unpack("<HHI", wav[20:28]) == (3, 1, 24000)  # IEEE float, mono, 24 kHz
    assert wav.endswith(pcm)
    with pytest.raises(EchoError, match="whole float32 frames"):
        wav_float32(b"\0\0\0", 24000)


def test_a_clone_id_fits_echos_pattern_whatever_the_run_id() -> None:
    import re

    cid = clone_id("a/very long run id " * 20, "r12")
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cid) and cid.endswith("-r12")
    assert clone_id("run-a", "r0") != clone_id("run-b", "r0")


# --------------------------------------------------------------------------
# a run
# --------------------------------------------------------------------------


def test_a_run_speaks_every_prompt_in_every_voice_at_every_seed(tmp_path: Path) -> None:
    echo = FakeEcho()
    spec = _spec(tmp_path)
    events: list = []
    result = _backend(echo).generate(spec, tmp_path / "run", progress=events.append)
    m = result.manifest

    assert [(c.prompt_index, c.ref_id, c.seed) for c in m.clips] == [
        (p, r, s) for p in (0, 1) for r in ("r0", "r1") for s in (42, 43)
    ]
    for clip in m.clips:
        data = Path(clip.path).read_bytes()
        assert Path(clip.path).name == f"{clip.item_id}.wav"
        assert hashlib.sha256(data).hexdigest() == clip.sha256
        assert (clip.sample_rate, clip.device) == (24000, "cuda:0")
    # Every utterance went out seeded, in its reference's clone, in the run's language.
    sent = echo.bodies("/synthesise/stream")
    assert [(b["seed"], b["language"]) for b in sent][:2] == [(42, "pt"), (43, "pt")]
    # Spoken by the voice_clone_id echo answered /clone-voice with, not the bare companion_id.
    assert {b["voice_clone_id"] for b in sent} == {
        f"voice_clones/{clone_id(spec.run_id, r)}/prompt.pt" for r in ("r0", "r1")
    }
    assert [e.type for e in events][0] == "start" and events[-1].type == "done"
    assert sum(e.type == "image" for e in events) == 8


def test_the_manifest_pins_the_weights_image_references_and_prompts(tmp_path: Path) -> None:
    echo = FakeEcho()
    spec = _spec(tmp_path, source_manifest="/exports/train/speech_manifest.jsonl", training_run_id="forge-7")
    m = _backend(echo).generate(spec, tmp_path / "run").manifest

    assert m.weights.checkpoint == "forge/run 7"  # percent-decoded from X-Echo-Checkpoint
    tuned = echo.checkpoints["forge/run 7"]  # the run restored the stock weights afterwards
    assert m.weights.fingerprint == weights_fingerprint([(f["name"], f["sha256"]) for f in tuned])
    assert [(f.name, f.sha256) for f in m.weights.files] == [(f["name"], f["sha256"]) for f in tuned]
    assert (m.image.tag, m.image.version, m.image.revision) == (
        "ghcr.io/smk762/argus-echo-xtts:0.2.0",
        "0.3.0",
        "abc1234",
    )
    assert m.engine == "argus-echo-xtts" and m.route == "/synthesise/stream"
    assert [r.sha256 for r in m.references] == [
        hashlib.sha256(Path(r.path).read_bytes()).hexdigest() for r in spec.references
    ]
    assert [r.rel_path for r in m.references] == ["spk0/clip0.wav", "spk1/clip1.wav"]
    assert [p.text for p in m.prompts] == spec.prompts
    assert (m.source_manifest, m.training_run_id) == ("/exports/train/speech_manifest.jsonl", "forge-7")
    # Written to disk, and it reads back as the same manifest.
    stored = SpeechRunManifest.model_validate_json((tmp_path / "run" / MANIFEST_NAME).read_text())
    assert stored == m


def test_the_same_seed_gives_the_same_clip_and_another_seed_a_different_one(tmp_path: Path) -> None:
    echo = FakeEcho()
    first = _backend(echo).generate(_spec(tmp_path), tmp_path / "a").manifest
    second = _backend(echo).generate(_spec(tmp_path), tmp_path / "b").manifest
    assert [c.sha256 for c in first.clips] == [c.sha256 for c in second.clips]
    by_seed = {(c.prompt_index, c.ref_id, c.seed): c.sha256 for c in first.clips}
    assert by_seed[(0, "r0", 42)] != by_seed[(0, "r0", 43)]


def test_the_checkpoint_is_loaded_then_the_previous_one_restored(tmp_path: Path) -> None:
    echo = FakeEcho()
    _backend(echo).generate(_spec(tmp_path), tmp_path / "run")
    assert [b["checkpoint"] for b in echo.bodies("/admin/load")] == ["forge/run 7", None]
    assert echo.selected is None


@pytest.mark.parametrize("weights", [None, {}])
def test_an_unreported_checkpoint_is_not_restored_to_the_stock_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, weights: dict | None
) -> None:
    echo = FakeEcho()
    health = echo.health
    answers = iter([health() | {"weights": weights}])  # only the first read: pinning the clips needs the real weights
    monkeypatch.setattr(echo, "health", lambda: next(answers, None) or health())
    _backend(echo).generate(_spec(tmp_path), tmp_path / "run")
    assert [b["checkpoint"] for b in echo.bodies("/admin/load")] == ["forge/run 7"]
    assert (tmp_path / "run" / MANIFEST_NAME).is_file()


@pytest.mark.parametrize("weights", [None, {}])
def test_an_unreported_selection_still_loads_the_stock_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, weights: dict | None
) -> None:
    """With /health silent on the selection, a run asking for the stock weights loads them rather than speak with
    whatever another run left selected (audit pack-argus-proof-aa3e6af18e)."""
    echo = FakeEcho()
    echo.selected = "forge/run 7"  # left over from another client
    health = echo.health
    answers = iter([health() | {"weights": weights}])
    monkeypatch.setattr(echo, "health", lambda: next(answers, None) or health())
    manifest = _backend(echo).generate(_spec(tmp_path, checkpoint=None), tmp_path / "run").manifest
    assert [b["checkpoint"] for b in echo.bodies("/admin/load")] == [None]
    assert manifest.weights.checkpoint is None


def test_clips_spoken_by_another_checkpoint_than_requested_fail_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/health claims the requested checkpoint but echo speaks with another: the run fails instead of recording the
    wrong weights as the requested run (audit pack-argus-proof-aa3e6af18e)."""
    echo = FakeEcho()  # stock weights selected
    health = echo.health
    answers = iter([health() | {"weights": {"checkpoint": "forge/run 7"}}])  # a stale answer
    monkeypatch.setattr(echo, "health", lambda: next(answers, None) or health())
    with pytest.raises(EchoError, match="not the requested forge/run 7"):
        _backend(echo).generate(_spec(tmp_path), tmp_path / "run")
    assert not (tmp_path / "run" / MANIFEST_NAME).exists()


def test_a_run_on_the_selected_checkpoint_loads_nothing(tmp_path: Path) -> None:
    echo = FakeEcho()
    _backend(echo).generate(_spec(tmp_path, checkpoint=None), tmp_path / "run")
    assert echo.bodies("/admin/load") == []


def test_every_clone_is_deleted_when_the_run_ends(tmp_path: Path) -> None:
    echo = FakeEcho()
    spec = _spec(tmp_path)
    _backend(echo).generate(spec, tmp_path / "run")
    assert echo.clones == {}
    assert sorted(echo.deleted) == sorted(clone_id(spec.run_id, r) for r in ("r0", "r1"))


def test_a_failed_run_still_deletes_its_clones_and_restores_the_checkpoint(tmp_path: Path) -> None:
    echo = FakeEcho()
    echo.script["/synthesise/stream"] = [httpx.Response(200, content=b"x" * 8)]  # no format headers
    with pytest.raises(EchoError, match="not pcm_f32le"):
        _backend(echo).generate(_spec(tmp_path), tmp_path / "run")
    assert echo.clones == {} and echo.selected is None
    assert (tmp_path / "run" / FAILURE_NAME).is_file()
    assert not (tmp_path / "run" / MANIFEST_NAME).exists()


def test_a_leftover_clone_from_an_earlier_run_is_replaced(tmp_path: Path) -> None:
    echo = FakeEcho()
    spec = _spec(tmp_path)
    echo.clones[clone_id(spec.run_id, "r0")] = {"bytes": 1}  # a crashed run with this run_id left it
    _backend(echo).generate(spec, tmp_path / "run")
    assert echo.clones == {}


def test_weights_that_change_mid_run_fail_it(tmp_path: Path) -> None:
    # Another client loads a different checkpoint between two of proof's utterances (A -> B).
    echo = FakeEcho()
    echo.swap_after, echo.swap_to = 3, None
    with pytest.raises(EchoError, match="spoken by other weights"):
        _backend(echo).generate(_spec(tmp_path), tmp_path / "run")
    assert echo.clones == {}


def test_qwen_pins_only_its_base_slot(tmp_path: Path) -> None:
    echo = FakeEcho("qwen")
    m = _backend(echo).generate(_spec(tmp_path, engine="qwen"), tmp_path / "run").manifest
    assert m.weights.model == "base"
    assert [f.name for f in m.weights.files] == ["base/model.pth"]
    assert m.clips[0].device is None


def test_qwen_speaking_a_clone_with_another_slot_fails(tmp_path: Path) -> None:
    echo = FakeEcho("qwen")
    real = echo.speak

    def custom_voice(req: dict) -> httpx.Response:
        resp = real(req)
        resp.headers["X-Echo-Model"] = "custom_voice"
        return resp

    echo.speak = custom_voice  # type: ignore[method-assign]
    with pytest.raises(EchoError, match="custom_voice model, not base"):
        _backend(echo).generate(_spec(tmp_path, engine="qwen"), tmp_path / "run")


def test_the_wrong_engine_is_refused_before_anything_is_cloned(tmp_path: Path) -> None:
    echo = FakeEcho("qwen")
    with pytest.raises(EchoError, match="qwen-tts engine, not xtts-v2"):
        _backend(echo).generate(_spec(tmp_path), tmp_path / "run")
    assert echo.clones == {} and echo.bodies("/admin/load") == []


def test_an_echo_without_weight_headers_is_refused(tmp_path: Path) -> None:
    echo = FakeEcho()
    real = echo.speak

    def old_echo(req: dict) -> httpx.Response:
        resp = real(req)
        del resp.headers["X-Echo-Weights"]
        return resp

    echo.speak = old_echo  # type: ignore[method-assign]
    with pytest.raises(EchoError, match="0.2.0 or later"):
        _backend(echo).generate(_spec(tmp_path), tmp_path / "run")


def test_a_missing_reference_clip_fails_before_echo_is_touched(tmp_path: Path) -> None:
    echo = FakeEcho()
    spec = _spec(tmp_path)
    Path(spec.references[1].path).unlink()
    with pytest.raises(EchoError, match="r1 .* not readable"):
        _backend(echo).generate(spec, tmp_path / "run")
    assert echo.paths() == ["GET /health"]


def test_the_run_holds_one_batch_lease_on_the_engine(tmp_path: Path) -> None:
    coordinator = Recording()
    _backend(FakeEcho(), coordinator).generate(_spec(tmp_path), tmp_path / "run")
    [call] = coordinator.calls
    assert (call["kind"], call["target"], call["priority"], call["preemptible"]) == (
        "speak-xtts",
        "xtts",
        "batch",
        True,
    )
    assert (call["min_vram_mb"], call["model_mb"]) == (gpu.SPEAK_XTTS.vram_mb, gpu.SPEAK_XTTS.model_mb)


def test_a_refused_lease_fails_the_run_and_cleans_up(tmp_path: Path) -> None:
    from argus_cortex.gpu import GpuLeaseRefused

    echo = FakeEcho()
    with pytest.raises(BackendError, match="lease"):
        _backend(echo, Recording(raises=GpuLeaseRefused("no room"))).generate(_spec(tmp_path), tmp_path / "run")
    assert echo.clones == {} and echo.selected is None


# --------------------------------------------------------------------------
# the client
# --------------------------------------------------------------------------


def test_a_busy_engine_is_retried_honouring_retry_after(tmp_path: Path) -> None:
    echo = FakeEcho()
    echo.script["/health"] = [httpx.Response(503, json={"error": "busy"}, headers={"Retry-After": "2"})]
    assert echo.client().health()["engine"] == "xtts-v2"
    assert echo.sleeps == [2.0]


def test_health_does_not_retry_a_permanent_gpu_error() -> None:
    echo = FakeEcho()
    echo.script["/health"] = [httpx.Response(503, json={"error": "gpu_error"})]
    with pytest.raises(EchoError, match="gpu_error"):
        echo.client().health()
    assert echo.sleeps == []
    assert len(echo.requests) == 1


def test_multipart_escapes_quotes_and_rejects_header_injection() -> None:
    with pytest.raises(EchoError, match="control character"):
        multipart({}, [("audio", "clip\r\nX-Injected: 1.wav", b"pcm")])
    body, _ctype = multipart({'a"b': "value"}, [("audio", 'say "hi".wav', b"pcm")])
    assert b'name="a\\"b"' in body
    assert b'filename="say \\"hi\\".wav"' in body
    body, _ctype = multipart({}, [("audio", "voz café.wav", b"pcm")])
    assert b"filename*=UTF-8''" in body
    assert "voz caf" in body.decode()


def test_a_filesystem_error_marks_the_run_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unreadable(_path: Path) -> str:
        raise OSError("permission denied")

    monkeypatch.setattr("argus_proof.backends.echo.sha256_cached", unreadable)
    events: list = []
    out = tmp_path / "run"
    with pytest.raises(BackendError, match="permission denied"):
        _backend(FakeEcho()).generate(_spec(tmp_path), out, progress=events.append)
    assert (out / FAILURE_NAME).is_file()
    assert events[-1].type == "error"
    assert not (out / MANIFEST_NAME).exists()


def test_a_non_oom_cuda_error_is_not_retried() -> None:
    echo = FakeEcho()
    echo.script["/health"] = [httpx.Response(500, json={"error": "CUDA error", "request_id": "abc"})]
    with pytest.raises(EchoError, match=r"HTTP 500: CUDA error \(request_id abc\)"):
        echo.client().health()
    assert echo.sleeps == []


def test_admin_load_needs_the_token() -> None:
    echo = FakeEcho()
    with pytest.raises(EchoError, match="PROOF_ECHO_ADMIN_TOKEN"):
        echo.client(admin_token=None).load("forge/run 7")
    assert echo.requests == []  # refused before anything was sent


def test_a_redirect_is_not_followed() -> None:
    echo = FakeEcho()
    echo.script["/health"] = [httpx.Response(302, headers={"Location": "http://elsewhere.test/health"})]
    with pytest.raises(EchoError, match="redirect"):
        echo.client().health()
    assert len(echo.requests) == 1  # the Location was never requested


def test_another_host_is_refused_by_the_fetch_policy() -> None:
    client = EchoClient(URL, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})), resolve=resolve)
    client.base = "http://other.test:8141"  # as if a path were steered elsewhere
    with pytest.raises(EchoError, match="could not be reached"):
        client.health()


@pytest.mark.parametrize("url", ["ftp://echo:8141", "http://user:pw@echo:8141", "http://echo:8141/?x=1", "echo"])
def test_only_a_plain_http_origin_is_accepted(url: str) -> None:
    with pytest.raises(ValueError):
        EchoClient(url)


def test_from_env_needs_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROOF_ECHO_URL", raising=False)
    with pytest.raises(EchoError, match="PROOF_ECHO_URL"):
        EchoClient.from_env()
    monkeypatch.setenv("PROOF_ECHO_URL", URL)
    monkeypatch.setenv("PROOF_ECHO_ADMIN_TOKEN", "t")
    client = EchoClient.from_env()
    assert (client.base, client.admin_token) == (URL, "t")


def test_from_env_falls_back_to_the_engines_suite_wide_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARGUS_XTTS_URL", "http://xtts.test:8141")
    monkeypatch.setenv("ARGUS_TTS_URL", "http://qwen.test:8142")
    assert EchoClient.from_env().base == "http://xtts.test:8141"  # xtts is the default engine
    assert EchoClient.from_env(engine="xtts").base == "http://xtts.test:8141"
    assert EchoClient.from_env(engine="qwen").base == "http://qwen.test:8142"
    monkeypatch.setenv("PROOF_ECHO_URL", URL)  # proof's own variable wins, for either engine
    assert EchoClient.from_env(engine="qwen").base == URL


def test_from_env_refuses_a_malformed_url_by_variable_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARGUS_XTTS_URL", "made-up-host:8141")
    with pytest.raises(EchoError, match="ARGUS_XTTS_URL") as refused:
        EchoClient.from_env()
    assert "made-up-host" not in str(refused.value)


# --------------------------------------------------------------------------
# the spec
# --------------------------------------------------------------------------


def test_the_spec_caps_the_grid_and_refuses_repeats(tmp_path: Path) -> None:
    refs = _refs(tmp_path)
    with pytest.raises(ValueError, match="over the 512 cap"):
        SpeechRunSpec(run_id="r", engine="xtts", prompts=["a"] * 64, references=refs, seeds=list(range(5)))
    with pytest.raises(ValueError, match="repeat a ref_id"):
        SpeechRunSpec(run_id="r", engine="xtts", prompts=["a"], references=[refs[0], refs[0]], seeds=[1])
    with pytest.raises(ValueError):
        SpeechRunSpec(run_id="r", engine="xtts", prompts=["a"], references=refs, seeds=[2**32])
    with pytest.raises(ValueError, match="prompt is empty"):
        SpeechRunSpec(run_id="r", engine="xtts", prompts=["  "], references=refs, seeds=[1])


def test_the_manifest_records_echos_runtime_libraries(tmp_path: Path) -> None:
    echo = FakeEcho()
    m = _backend(echo).generate(_spec(tmp_path), tmp_path / "run").manifest
    assert m.image.runtime == {"python": "3.11.11", "torch": "2.6.0+cu124", "cuda": "12.4", "coqui-tts": "0.28.0"}


@pytest.mark.parametrize(
    ("runtime", "recorded"),
    [
        (None, None),  # echo before 0.3.0 has no runtime block
        ({"torch": "2.6.0", "moshi": None}, {"torch": "2.6.0", "moshi": None}),  # None: not installed
        ({"torch": 2.6, "ok": "1.0"}, {"ok": "1.0"}),  # a malformed entry is dropped, not fatal
        ("2.6.0", None),
    ],
)
def test_a_missing_or_malformed_runtime_never_fails_a_run(tmp_path: Path, runtime: object, recorded: object) -> None:
    echo = FakeEcho()
    echo.runtime = runtime  # type: ignore[assignment]
    m = _backend(echo).generate(_spec(tmp_path), tmp_path / "run").manifest
    assert m.image.runtime == recorded
