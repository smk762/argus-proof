"""An in-memory argus-echo for the speech backend's tests: an ``httpx.MockTransport`` that keeps the engine's state
(selected checkpoint, weight files, stored clones) and answers ``/health``, ``/admin/load``, ``/clone-voice``,
``DELETE /clones/{id}`` and ``/synthesise/stream`` as echo's docs/api.md describes. No network, no engine."""

from __future__ import annotations

import hashlib
import json
import struct
from urllib.parse import quote

import httpx

from argus_proof.backends.echo import EchoClient, weights_fingerprint

HOST, PORT = "echo.test", 8141
URL = f"http://{HOST}:{PORT}"
TOKEN = "admin-secret"


def resolve(host: str, port: int) -> list[str]:
    return ["127.0.0.1"]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class FakeEcho:
    """One echo engine (``xtts`` or ``qwen``). ``checkpoints`` maps a name (``None`` = stock) to its weight files;
    ``swap_after`` makes another client load ``swap_to`` after that many utterances; ``script[path]`` holds
    responses served first, in order."""

    def __init__(self, engine: str = "xtts") -> None:
        self.engine = engine
        prefix = "base/" if engine == "qwen" else ""
        self.checkpoints: dict[str | None, list[dict]] = {
            None: [{"name": f"{prefix}model.pth", "sha256": _sha("stock")}],
            "forge/run 7": [{"name": f"{prefix}model.pth", "sha256": _sha("fine-tune")}],
        }
        if engine == "qwen":  # the other slot, always stock
            for files in self.checkpoints.values():
                files.append({"name": "custom_voice/model.safetensors", "sha256": _sha("custom")})
        self.selected: str | None = None
        self.loaded = False  # files are null until the first synthesis (lazy load)
        self.clones: dict[str, dict] = {}
        self.deleted: list[str] = []
        self.spoken = 0
        self.swap_after: int | None = None
        self.swap_to: str | None = "forge/run 7"
        self.script: dict[str, list[httpx.Response]] = {}
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []

    # -- helpers ---------------------------------------------------------
    def files(self) -> list[dict]:
        return self.checkpoints[self.selected]

    def slot_files(self) -> list[dict]:
        files = self.files()
        return [f for f in files if f["name"].startswith("base/")] if self.engine == "qwen" else files

    def fingerprint(self) -> str:
        return weights_fingerprint([(f["name"], f["sha256"]) for f in self.slot_files()])

    def bodies(self, path: str) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]

    def paths(self) -> list[str]:
        return [f"{r.method} {r.url.path}" for r in self.requests]

    def client(self, **kwargs) -> EchoClient:  # noqa: ANN003
        kwargs.setdefault("admin_token", TOKEN)
        kwargs.setdefault("sleep", self.sleeps.append)
        return EchoClient(URL, transport=httpx.MockTransport(self.handler), resolve=resolve, **kwargs)

    @staticmethod
    def pcm(text: str, clone: str, seed: int, fingerprint: str) -> bytes:
        """Deterministic float32 samples for one utterance: same inputs, same bytes."""
        digest = hashlib.sha256(f"{text}|{clone}|{seed}|{fingerprint}".encode()).digest()
        return b"".join(struct.pack("<f", (b - 128) / 128) for b in digest * 4)

    # -- the API ---------------------------------------------------------
    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if self.script.get(path):
            return self.script[path].pop(0)
        if path == "/health" and request.method == "GET":
            return httpx.Response(200, json=self.health())
        if path == "/admin/load" and request.method == "POST":
            if request.headers.get("authorization") != f"Bearer {TOKEN}":
                return httpx.Response(401, json={"error": "unauthorized"})
            name = json.loads(request.content)["checkpoint"]
            if name not in self.checkpoints:
                return httpx.Response(404, json={"error": f"no checkpoint {name}"})
            self.selected = name
            return httpx.Response(200, json={"weights": {"checkpoint": name}})
        if path == "/clone-voice" and request.method == "POST":
            body = request.content
            cid = body.split(b'name="companion_id"\r\n\r\n', 1)[1].split(b"\r\n", 1)[0].decode()
            if cid in self.clones:
                return httpx.Response(409, json={"error": "exists"})
            self.clones[cid] = {"bytes": len(body)}
            # echo 0.2.0 answers with the storage key, which is what /synthesise takes (not the bare companion_id)
            return httpx.Response(200, json={"voice_clone_id": f"voice_clones/{cid}/prompt.pt", "companion_id": cid})
        if path.startswith("/clones/") and request.method == "DELETE":
            cid = path.removeprefix("/clones/")
            if cid not in self.clones:
                return httpx.Response(404, json={"error": "no such clone"})
            del self.clones[cid]
            self.deleted.append(cid)
            return httpx.Response(200, json={})
        if path == "/synthesise/stream" and request.method == "POST":
            return self.speak(json.loads(request.content))
        return httpx.Response(404, json={"error": "not found"})

    def health(self) -> dict:
        weights = {
            "checkpoint": self.selected,
            "fingerprint": weights_fingerprint([(f["name"], f["sha256"]) for f in self.files()])
            if self.loaded
            else None,
            "files": self.files() if self.loaded else None,
        }
        return {
            "status": "ok",
            "engine": {"xtts": "xtts-v2", "qwen": "qwen-tts"}[self.engine],  # what echo's servers report
            "model": "xtts_v2" if self.engine == "xtts" else "1.7b",
            "device": "cuda:0",
            "capabilities": ["tts"],
            "image": {"version": "0.2.0", "revision": "abc1234"},
            "weights": weights,
        }

    def speak(self, req: dict) -> httpx.Response:
        voice = str(req.get("voice_clone_id") or "")
        clone = voice.removeprefix("voice_clones/").removesuffix("/prompt.pt")
        if not voice.startswith("voice_clones/") or clone not in self.clones:
            return httpx.Response(404, json={"error": f"no clone {clone}"})
        if self.swap_after is not None and self.spoken == self.swap_after:
            self.selected = self.swap_to  # another client's /admin/load, between two of proof's utterances
        self.loaded = True
        self.spoken += 1
        fp = self.fingerprint()
        headers = {
            "X-Sample-Rate": "24000",
            "X-Audio-Format": "pcm_f32le",
            "X-Channels": "1",
            "X-Echo-Weights": fp,
            "X-Echo-Checkpoint": quote(self.selected or "", safe=""),
        }
        if self.engine == "xtts":
            headers["X-XTTS-Device"] = "cuda:0"
        else:
            headers["X-Echo-Model"] = "base"
        return httpx.Response(200, content=self.pcm(req["text"], clone, req["seed"], fp), headers=headers)
