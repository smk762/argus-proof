"""Speech sample grids through argus-echo (#118): every prompt in every held-out reference voice at every seed.

Proof speaks echo's HTTP API (echo's docs/api.md) and never imports the ``argus-echo`` package, whose argus-cortex pin
would come with it. Only the configured ``PROOF_ECHO_URL`` is called: argus-cortex's fetch policy allows that one
host, and redirects are never followed.

One run, on one engine (XTTS-v2 or Qwen3-TTS):

1. ``GET /health``: the engine, its image identity and the selected checkpoint.
2. ``POST /admin/load`` (``PROOF_ECHO_ADMIN_TOKEN``) when the run's checkpoint isn't the selected one; the
   previous selection is restored when the run ends.
3. ``POST /clone-voice`` once per reference clip, under an id derived from the run (``proof-<hash>-r<n>``), speaking
   it by the ``voice_clone_id`` echo answers with; every clone is deleted when the run ends, failed runs included,
   so no reference voice stays stored in echo.
4. ``POST /synthesise/stream`` per utterance, with the run's ``seed``. The stream is raw PCM from echo itself (the
   non-streaming route hands back a storage URL on another host). Each clip is written as ``<item_id>.wav``.

Every response names the weights that spoke it (``X-Echo-Weights``, echo's weights fingerprint). The first one is
pinned in the manifest with the files ``/health`` lists for it, recomputed here; any later clip spoken by other
weights fails the run. The run holds one preemptible GPU lease on the engine (:data:`argus_proof.gpu.SPEAK`) and
yields between utterances, staying on the node it started on.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import struct
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import structlog

from argus_proof import gpu
from argus_proof.backends.base import (
    BackendError,
    ProgressSink,
    clear_failed,
    mark_failed,
    write_manifest,
)
from argus_proof.hashing import sha256_cached
from argus_proof.models import (
    EchoImage,
    GeneratedClip,
    ModelRef,
    ProgressEvent,
    SpeechPrompt,
    SpeechReference,
    SpeechRunManifest,
    SpeechRunSpec,
    SpeechWeights,
)
from argus_proof.service_origin import service_origin

logger = structlog.get_logger()

ENV_ECHO_URL = "PROOF_ECHO_URL"
ENV_ECHO_ADMIN_TOKEN = "PROOF_ECHO_ADMIN_TOKEN"
ENV_ECHO_IMAGE = "PROOF_ECHO_IMAGE"
ENV_ECHO_TIMEOUT = "PROOF_ECHO_TIMEOUT"

DEFAULT_TIMEOUT = 300.0  # one utterance, long ones included
MAX_RETRIES = 3  # a busy or out-of-memory 503, honouring Retry-After
MAX_REPLY_BYTES = 64 * 1024 * 1024  # 64 MiB of float32 PCM is about 11 minutes at 24 kHz
MAX_CLIP_BYTES = 32 * 1024 * 1024  # one reference clip sent to /clone-voice
STREAM_FORMAT = "pcm_f32le"
STREAM_ROUTE = "/synthesise/stream"
# What each engine's /health reports as ``engine`` (echo's engines/*/ servers), by the name a SpeechRunSpec uses.
ECHO_ENGINE_NAMES: dict[str, str] = {"xtts": "xtts-v2", "qwen": "qwen-tts"}


class EchoError(BackendError):
    """echo failed, refused, or answered something proof can't use. ``status`` is the HTTP status, when there was one."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# wire helpers
# ---------------------------------------------------------------------------


def weights_fingerprint(files: list[tuple[str, str]]) -> str:
    """echo's weights fingerprint over ``(name, sha256)`` pairs (echo's docs/api.md, "Weights fingerprint").

    One ``<name>:<sha256>\\n`` line per file (names exactly as ``/health`` lists them, Qwen's slot prefix kept),
    sorted byte-wise as UTF-8, concatenated, sha256 in lowercase hex.
    """
    lines = sorted(f"{name}:{sha}\n".encode() for name, sha in files)
    return hashlib.sha256(b"".join(lines)).hexdigest()


def wav_float32(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """A WAV file (IEEE float, format 3) holding little-endian float32 *pcm*, byte for byte as echo sent it."""
    if len(pcm) % (4 * channels):
        raise EchoError(f"echo's stream is {len(pcm)} bytes, not whole float32 frames of {channels} channel(s)")
    block = 4 * channels
    fmt = struct.pack("<HHIIHHH", 3, channels, sample_rate, sample_rate * block, block, 32, 0)
    fact = struct.pack("<I", len(pcm) // block)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"fact" + struct.pack("<I", 4) + fact
    body += b"data" + struct.pack("<I", len(pcm)) + pcm + (b"\0" if len(pcm) % 2 else b"")
    return b"RIFF" + struct.pack("<I", len(body)) + body


def multipart(fields: Mapping[str, str], files: list[tuple[str, str, bytes]]) -> tuple[bytes, str]:
    """A ``multipart/form-data`` body: text *fields*, then ``(field, filename, data)`` *files*. Returns body, type."""
    boundary = f"argus-proof-{secrets.token_hex(12)}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(value.encode() + b"\r\n")
    for name, filename, data in files:
        head = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        )
        parts.append(head.encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def clone_id(run_id: str, ref_id: str) -> str:
    """The echo clone id of reference *ref_id* in run *run_id*: ``proof-<12 hex>-<ref_id>``, inside echo's
    ``^[A-Za-z0-9_-]{1,64}$`` whatever the run id."""
    return f"proof-{hashlib.sha256(run_id.encode()).hexdigest()[:12]}-{ref_id}"


def item_id(prompt_index: int, ref_id: str, seed: int) -> str:
    """A clip's handle within its run, and its WAV's stem."""
    return f"p{prompt_index}-{ref_id}-s{seed}"


def _error_text(body: bytes) -> str:
    """echo's error body, short: ``error`` / ``detail`` (a 422's first message), plus a ``request_id`` when given."""
    try:
        data = json.loads(body)
    except ValueError:
        return ""
    if not isinstance(data, dict):
        return ""
    detail = data.get("error") or data.get("detail") or data.get("status") or ""
    if isinstance(detail, list) and detail and isinstance(detail[0], dict):
        detail = detail[0].get("msg", "")
    text = str(detail)[:200] if isinstance(detail, str | int | float) else ""
    request_id = data.get("request_id")
    return f"{text} (request_id {request_id})".strip() if request_id else text


def _runtime(value: Any) -> dict[str, str | None] | None:
    """echo's ``image.runtime`` (library -> version, ``None`` when not installed), or ``None`` before echo 0.3.0.

    Anything that isn't a short string version is dropped rather than recorded, so a malformed block can't fail a run
    or bloat the manifest.
    """
    if not isinstance(value, dict):
        return None
    return {
        str(k)[:64]: (v[:128] if isinstance(v, str) else None)
        for k, v in list(value.items())[:64]
        if isinstance(k, str) and (v is None or isinstance(v, str))
    }


def _lower(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in dict(headers).items()}


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


# ---------------------------------------------------------------------------
# the client
# ---------------------------------------------------------------------------


class EchoClient:
    """echo's HTTP API at *base_url* (the configured one only).

    *admin_token* goes to ``/admin/*`` as a bearer token; *priority* goes along as ``X-Argus-Priority``.
    *transport* and *resolve* go to :func:`argus_cortex.net.fetch` (tests use an ``httpx.MockTransport``);
    *sleep* waits between retries.
    """

    def __init__(
        self,
        base_url: str,
        *,
        admin_token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        priority: str | None = None,
        retries: int = MAX_RETRIES,
        transport: Any = None,
        resolve: Callable[[str, int], list[str]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base, allowed = service_origin(base_url, ENV_ECHO_URL)
        self._allow = (allowed,)
        self.admin_token = admin_token or None
        self.timeout = timeout
        self.priority = priority
        self.retries = retries
        self._transport = transport
        self._resolve = resolve
        self._sleep = sleep

    @classmethod
    def from_env(cls, **kwargs: Any) -> EchoClient:
        """The client for ``$PROOF_ECHO_URL`` (:class:`EchoError` when unset)."""
        url = os.environ.get(ENV_ECHO_URL, "").strip()
        if not url:
            raise EchoError(f"set {ENV_ECHO_URL} to the argus-echo engine to generate speech with")
        kwargs.setdefault("admin_token", os.environ.get(ENV_ECHO_ADMIN_TOKEN, "").strip() or None)
        kwargs.setdefault("timeout", _float_env(ENV_ECHO_TIMEOUT, DEFAULT_TIMEOUT))
        try:
            return cls(url, **kwargs)
        except ValueError as exc:
            raise EchoError(str(exc)) from exc

    def _call(
        self,
        method: str,
        path: str,
        *,
        content: bytes | None = None,
        content_type: str | None = None,
        admin: bool = False,
        accept: str = "application/json",
    ):  # noqa: ANN202 - argus_cortex.net.Reply
        """One API call → its reply (2xx only). Retries a busy or out-of-memory 503; raises :class:`EchoError`."""
        from argus_cortex.net import UrlRefused, fetch

        headers = {"Accept": accept}
        if content_type:
            headers["Content-Type"] = content_type
        if self.priority:
            headers["X-Argus-Priority"] = self.priority
        if admin:
            if not self.admin_token:
                raise EchoError(f"{path} needs echo's admin token: set {ENV_ECHO_ADMIN_TOKEN}")
            headers["Authorization"] = f"Bearer {self.admin_token}"
        for attempt in range(self.retries + 1):
            try:
                reply = fetch(
                    self.base + path,
                    method=method,
                    content=content,
                    headers=headers,
                    allow_hosts=self._allow,
                    timeout=self.timeout,
                    deadline=self.timeout * 2,
                    max_bytes=MAX_REPLY_BYTES,
                    max_redirects=0,
                    resolve=self._resolve,
                    transport=self._transport,
                )
            except (UrlRefused, OSError) as exc:
                raise EchoError(f"echo could not be reached: {exc}") from exc
            # Only a 503 is retried: busy, loading, or a real out-of-memory. Any other CUDA error is a 500, after
            # which XTTS's /health answers gpu_error until a restart, so retrying it would only fail again.
            if reply.status == 503 and attempt < self.retries:
                self._sleep(self._retry_after(reply.headers, attempt))
                continue
            break
        if 300 <= reply.status < 400:
            raise EchoError(f"echo answered a redirect (HTTP {reply.status}); it is not followed", status=reply.status)
        if reply.status >= 400:
            detail = _error_text(reply.body)
            raise EchoError(
                f"echo {method} {path} answered HTTP {reply.status}{f': {detail}' if detail else ''}",
                status=reply.status,
            )
        return reply

    def _json(self, reply) -> Any:  # noqa: ANN001
        try:
            return json.loads(reply.body)
        except ValueError as exc:
            raise EchoError("echo's answer is not JSON", status=reply.status) from exc

    @staticmethod
    def _retry_after(headers: Mapping[str, str], attempt: int) -> float:
        value = _lower(headers).get("retry-after", "").strip()
        try:
            wait = float(value)
        except ValueError:
            wait = float(2**attempt)
        if not math.isfinite(wait) or wait < 0:
            wait = float(2**attempt)
        return min(wait, 60.0)

    def health(self) -> dict:
        """``GET /health`` (a ``503 gpu_error`` is an :class:`EchoError`: the engine needs a restart)."""
        data = self._json(self._call("GET", "/health"))
        if not isinstance(data, dict):
            raise EchoError("echo's /health is not an object")
        return data

    def load(self, checkpoint: str | None) -> dict:
        """``POST /admin/load``: select *checkpoint* (``None``: the stock weights)."""
        body = json.dumps({"checkpoint": checkpoint}).encode()
        return self._json(self._call("POST", "/admin/load", content=body, content_type="application/json", admin=True))

    def clone(self, companion_id: str, audio: bytes, filename: str, reference_text: str) -> str:
        """``POST /clone-voice``: store *audio* as clone *companion_id*; returns the ``voice_clone_id`` to speak it with.

        Echo answers with the clone's ``voice_clone_id`` (``voice_clones/<companion_id>/prompt.pt`` on 0.2.0), which is
        what ``/synthesise`` takes; the bare ``companion_id`` is only how the clone is named and deleted.
        """
        body, ctype = multipart(
            {"companion_id": companion_id, "reference_text": reference_text}, [("audio", filename, audio)]
        )
        reply = self._json(self._call("POST", "/clone-voice", content=body, content_type=ctype))
        voice = reply.get("voice_clone_id") if isinstance(reply, dict) else None
        if not isinstance(voice, str) or not voice:
            raise EchoError(f"echo's /clone-voice didn't say how to speak clone {companion_id} (no voice_clone_id)")
        return voice

    def delete_clone(self, companion_id: str) -> None:
        """``DELETE /clones/{id}``; a clone that is already gone is fine."""
        try:
            self._call("DELETE", f"/clones/{quote(companion_id, safe='')}")
        except EchoError as exc:
            if exc.status != 404:
                raise

    def speak(self, request: Mapping[str, Any]) -> tuple[bytes, dict[str, str]]:
        """``POST /synthesise/stream`` → ``(pcm, lower-cased headers)``."""
        reply = self._call(
            "POST",
            STREAM_ROUTE,
            content=json.dumps(dict(request)).encode(),
            content_type="application/json",
            accept="application/octet-stream",
        )
        return reply.body, _lower(reply.headers)


# ---------------------------------------------------------------------------
# the backend
# ---------------------------------------------------------------------------


class SpeechResult:
    """A completed speech run: its manifest (with every clip) and the run dir it was written to."""

    def __init__(self, manifest: SpeechRunManifest, out_dir: Path) -> None:
        self.manifest = manifest
        self.out_dir = out_dir

    @property
    def clips(self) -> list[GeneratedClip]:
        return self.manifest.clips


class EchoBackend:
    """Generates a :class:`~argus_proof.models.SpeechRunSpec` through one argus-echo engine.

    *image* is the engine image reference to record (default ``$PROOF_ECHO_IMAGE``); *coordinator* the GPU lease
    coordinator (default the ``ARGUS_GPU_*`` environment).
    """

    def __init__(
        self,
        client: EchoClient,
        *,
        image: str | None = None,
        coordinator: Any = None,
        priority: gpu.PriorityClass = gpu.DEFAULT_PRIORITY,
    ) -> None:
        self.client = client
        self.image = image if image is not None else (os.environ.get(ENV_ECHO_IMAGE, "").strip() or None)
        self.coordinator = coordinator
        self.priority = priority

    def generate(self, spec: SpeechRunSpec, out_dir: Path, progress: ProgressSink | None = None) -> SpeechResult:
        """Run *spec*, writing ``<item_id>.wav`` clips and ``manifest.json`` under *out_dir*.

        Raises :class:`~argus_proof.backends.base.BackendError` on any failure, after leaving ``failed.json``
        in *out_dir*; the clones are deleted and the engine's previous checkpoint restored either way.
        """
        emit = progress or (lambda _event: None)
        total = len(spec.prompts) * len(spec.references) * len(spec.seeds)
        out_dir.mkdir(parents=True, exist_ok=True)
        emit(ProgressEvent(run_id=spec.run_id, type="start", total=total))
        try:
            manifest = self._run(spec, out_dir, emit, total)
            write_manifest(out_dir, manifest)
        except BackendError as exc:
            mark_failed(out_dir, spec.run_id, str(exc))
            emit(ProgressEvent(run_id=spec.run_id, type="error", message=str(exc)))
            raise
        clear_failed(out_dir)
        emit(ProgressEvent(run_id=spec.run_id, type="done", completed=total, total=total))
        return SpeechResult(manifest, out_dir)

    def _run(self, spec: SpeechRunSpec, out_dir: Path, emit: ProgressSink, total: int) -> SpeechRunManifest:
        health = self.client.health()
        engine = str(health.get("engine") or "")
        if engine != ECHO_ENGINE_NAMES[spec.engine]:
            raise EchoError(
                f"{ENV_ECHO_URL} is echo's {engine or 'unknown'} engine, not {ECHO_ENGINE_NAMES[spec.engine]}"
            )
        if "tts" not in (health.get("capabilities") or ["tts"]):
            raise EchoError(f"echo's {engine} engine doesn't serve text to speech")
        image = health.get("image") if isinstance(health.get("image"), dict) else {}
        previous = (health.get("weights") or {}).get("checkpoint") if isinstance(health.get("weights"), dict) else None
        references = [self._reference(ref) for ref in spec.references]

        cloned: list[str] = []
        voices: dict[str, str] = {}  # ref_id -> the voice_clone_id echo answered with
        loaded = False
        try:
            if previous != spec.checkpoint:
                self.client.load(spec.checkpoint)
                loaded = True
            for ref, pinned in zip(spec.references, references, strict=True):
                cid = clone_id(spec.run_id, ref.ref_id)
                audio = Path(ref.path).read_bytes()
                try:
                    voice = self.client.clone(cid, audio, Path(ref.path).name, ref.transcript)
                except EchoError as exc:
                    if exc.status != 409:  # left over from an earlier run with this run_id
                        raise
                    self.client.delete_clone(cid)
                    voice = self.client.clone(cid, audio, Path(ref.path).name, ref.transcript)
                cloned.append(cid)
                voices[ref.ref_id] = voice
                logger.info("echo.cloned", run_id=spec.run_id, ref_id=ref.ref_id, sha256=pinned.sha256)
            clips, weights = self._speak_all(spec, voices, out_dir, emit, total)
        finally:
            for cid in cloned:
                try:
                    self.client.delete_clone(cid)
                except EchoError as exc:
                    logger.warning("echo.clone_not_deleted", run_id=spec.run_id, clone=cid, error=str(exc))
            if loaded:
                try:
                    self.client.load(previous)
                except EchoError as exc:
                    logger.warning("echo.checkpoint_not_restored", run_id=spec.run_id, error=str(exc))

        return SpeechRunManifest(
            run_id=spec.run_id,
            engine=f"argus-echo-{spec.engine}",
            image=EchoImage(
                tag=self.image,
                version=image.get("version"),
                revision=image.get("revision"),
                runtime=_runtime(image.get("runtime")),
            ),
            weights=weights,
            language=spec.language,
            prompts=[SpeechPrompt(index=i, text=t) for i, t in enumerate(spec.prompts)],
            references=references,
            seeds=list(spec.seeds),
            clips=clips,
            source_manifest=spec.source_manifest,
            references_manifest=spec.references_manifest,
            training_run_id=spec.training_run_id,
            created_at=datetime.now(UTC).isoformat(),
        )

    @staticmethod
    def _reference(ref) -> SpeechReference:  # noqa: ANN001 - SpeechReferenceSpec
        path = Path(ref.path)
        if not path.is_file():
            raise EchoError(f"reference {ref.ref_id} ({ref.rel_path}) is not readable here")
        if path.stat().st_size > MAX_CLIP_BYTES:
            raise EchoError(f"reference {ref.ref_id} ({ref.rel_path}) is over {MAX_CLIP_BYTES // (1024 * 1024)} MiB")
        return SpeechReference(
            ref_id=ref.ref_id,
            rel_path=ref.rel_path,
            sha256=sha256_cached(path),
            transcript=ref.transcript,
            speaker_label=ref.speaker_label,
            speaker_cluster=ref.speaker_cluster,
            language=ref.language,
            accent=ref.accent,
        )

    def _speak_all(
        self, spec: SpeechRunSpec, voices: Mapping[str, str], out_dir: Path, emit: ProgressSink, total: int
    ) -> tuple[list[GeneratedClip], SpeechWeights]:
        clips: list[GeneratedClip] = []
        weights: SpeechWeights | None = None
        footprint = gpu.SPEAK[spec.engine]
        coordinator = self.coordinator or gpu.default_coordinator()
        try:
            with gpu.SeedLeases(coordinator, footprint, caller=spec.run_id, priority=self.priority) as leases:
                for pi in range(len(spec.prompts)):
                    for ref in spec.references:
                        for seed in spec.seeds:
                            leases.current()  # yields between utterances when the broker asks
                            clip, fingerprint, headers = self._speak(
                                spec, pi, ref.ref_id, voices[ref.ref_id], seed, out_dir
                            )
                            if weights is None:
                                weights = self._pin(fingerprint, headers)
                            elif fingerprint != weights.fingerprint:
                                raise EchoError(
                                    f"clip {clip.item_id} was spoken by other weights ({fingerprint[:12]}…, not "
                                    f"{weights.fingerprint[:12]}…): echo's checkpoint changed during the run"
                                )
                            clips.append(clip)
                            emit(ProgressEvent(run_id=spec.run_id, type="image", seed=seed, image_id=clip.item_id))
                            emit(ProgressEvent(run_id=spec.run_id, type="progress", completed=len(clips), total=total))
        except gpu.LEASE_ERRORS as exc:
            raise BackendError(gpu.lease_failure(exc)) from exc
        assert weights is not None  # the spec has at least one utterance
        return clips, weights

    def _speak(
        self, spec: SpeechRunSpec, prompt_index: int, ref_id: str, voice: str, seed: int, out_dir: Path
    ) -> tuple[GeneratedClip, str, dict[str, str]]:
        request = {
            "text": spec.prompts[prompt_index],
            "voice_clone_id": voice,
            "language": spec.language,
            "seed": seed,
        }
        pcm, headers = self.client.speak(request)
        if headers.get("x-audio-format") != STREAM_FORMAT:
            raise EchoError(
                f"echo streamed {headers.get('x-audio-format') or 'an unnamed format'}, not {STREAM_FORMAT}"
            )
        try:
            rate, channels = int(headers["x-sample-rate"]), int(headers.get("x-channels", "1"))
        except (KeyError, ValueError) as exc:
            raise EchoError("echo's stream has no usable X-Sample-Rate / X-Channels") from exc
        if rate <= 0 or channels != 1:
            raise EchoError(f"echo streamed {channels} channel(s) at {rate} Hz; proof expects mono")
        fingerprint = headers.get("x-echo-weights", "")
        if len(fingerprint) != 64:
            raise EchoError("echo's stream doesn't name its weights (X-Echo-Weights): echo 0.2.0 or later is needed")
        if spec.engine == "qwen" and headers.get("x-echo-model") not in (None, "base"):
            raise EchoError(f"Qwen spoke a clone with its {headers.get('x-echo-model')} model, not base")
        iid = item_id(prompt_index, ref_id, seed)
        data = wav_float32(pcm, rate, channels)
        path = out_dir / f"{iid}.wav"
        path.write_bytes(data)
        clip = GeneratedClip(
            item_id=iid,
            run_id=spec.run_id,
            prompt_index=prompt_index,
            ref_id=ref_id,
            seed=seed,
            path=str(path),
            sha256=hashlib.sha256(data).hexdigest(),
            sample_rate=rate,
            duration_s=round(len(pcm) / (4 * channels * rate), 3),
            device=headers.get("x-xtts-device"),
        )
        return clip, fingerprint, headers

    def _pin(self, fingerprint: str, headers: Mapping[str, str]) -> SpeechWeights:
        """The weights behind *fingerprint*, from ``/health``'s file list, recomputed so the pin can't be wrong."""
        weights = self.client.health().get("weights") or {}
        files = weights.get("files") or []
        model = headers.get("x-echo-model")
        if model:  # Qwen: the slot's own files, prefix kept
            files = [f for f in files if str(f.get("name", "")).startswith(f"{model}/")]
        pairs = [(str(f.get("name", "")), str(f.get("sha256", ""))) for f in files]
        if not pairs or weights_fingerprint(pairs) != fingerprint:
            raise EchoError(
                "echo's /health doesn't list the weights that spoke the first clip (its fingerprint doesn't match "
                "the files): the checkpoint changed between the clip and the check"
            )
        checkpoint = headers.get("x-echo-checkpoint", "")
        return SpeechWeights(
            checkpoint=unquote(checkpoint) or None,
            model=model or None,
            fingerprint=fingerprint,
            files=[ModelRef(name=name, sha256=sha) for name, sha in pairs],
        )


__all__ = [
    "ECHO_ENGINE_NAMES",
    "ENV_ECHO_ADMIN_TOKEN",
    "ENV_ECHO_IMAGE",
    "ENV_ECHO_URL",
    "EchoBackend",
    "EchoClient",
    "EchoError",
    "SpeechResult",
    "clone_id",
    "item_id",
    "multipart",
    "wav_float32",
    "weights_fingerprint",
]
