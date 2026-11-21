"""An in-memory argus-puppet for the pose scorer's tests: an ``httpx.MockTransport`` answering from small payloads
hand-built from puppet's wire schema (``tests/fixtures/puppet/``). No network, no images, no puppet package."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import httpx

from argus_proof.scoring.scorers.pose import PuppetClient

FIXTURES = Path(__file__).parent / "fixtures" / "puppet"
HOST, PORT = "puppet.test", 8220
URL = f"http://{HOST}:{PORT}"


def payload(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def resolve(host: str, port: int) -> list[str]:
    return ["127.0.0.1"]


class FakePuppet:
    """Answers ``/health``, ``/pose``, ``/stats`` and ``/compare``. ``script[path]`` holds responses served first, in
    order; ``fail_pose`` names images ``/pose`` answers 400 for; ``down`` refuses every connection."""

    def __init__(self) -> None:
        self.health = payload("health")
        self.pose = payload("pose")
        self.stats = payload("stats")
        self.training_stats: dict | None = None
        self.compare = payload("compare")
        self.script: dict[str, list[httpx.Response]] = {}
        self.fail_pose: set[str] = set()
        self.down = False
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []

    def bodies(self, path: str) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        self.requests.append(request)
        path = request.url.path
        if self.script.get(path):
            return self.script[path].pop(0)
        if path == "/health":
            return httpx.Response(200, json=self.health)
        if path == "/pose":
            name = request.url.params["name"]
            if name in self.fail_pose:
                return httpx.Response(400, json={"detail": "the image could not be decoded"})
            doc = copy.deepcopy(self.pose)
            doc["image"] = name
            return httpx.Response(200, json=doc)
        if path == "/stats":
            images = [r["image"] for r in json.loads(request.content)["results"]]
            training = images and all(i.startswith("ref-") for i in images)
            return httpx.Response(200, json=self.training_stats if training and self.training_stats else self.stats)
        if path == "/compare":
            return httpx.Response(200, json=self.compare)
        return httpx.Response(404, json={"detail": "Not Found"})

    def client(self, **kwargs) -> PuppetClient:  # noqa: ANN003
        return PuppetClient(
            URL, transport=httpx.MockTransport(self.handler), resolve=resolve, sleep=self.sleeps.append, **kwargs
        )
