"""Shared HTTP transport for the HTTP-based generation backends.

ComfyUI, A1111/SD.Next, and the remote/cloud backend all speak HTTP; this is the
tiny injectable surface they share, so each adapter is unit-testable without a
live server (tests pass a fake :class:`Transport`). The default
:class:`UrllibTransport` uses the stdlib, so the HTTP backends need no runtime
dependency, and wraps transport-level failures as :class:`BackendError` so a
caller sees one error type whether the server is down or a request was malformed.
"""

from __future__ import annotations

import base64
import binascii
import json
import urllib.error
import urllib.request
from typing import Protocol

from argus_proof.backends.base import BackendError


def decode_base64_image(encoded: str, *, context: str) -> bytes:
    """Decode a base64 image (tolerating a ``data:...;base64,`` prefix), or raise
    :class:`BackendError` — the shared decoder for the base64-over-HTTP backends."""
    if "," in encoded and encoded.lstrip().startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BackendError(f"{context}: undecodable base64 image: {exc}") from exc


class Transport(Protocol):
    """The small HTTP surface the adapters need, so tests can fake it."""

    def post_json(self, path: str, payload: dict) -> dict: ...
    def get_json(self, path: str) -> dict: ...
    def get_bytes(self, path: str) -> bytes: ...


class UrllibTransport:
    """Default :class:`Transport` over stdlib ``urllib`` — no runtime deps.

    ``headers`` are sent on every request (e.g. an ``Authorization`` bearer token
    for a cloud endpoint); ``label`` names the backend in error messages.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        label: str = "HTTP",
    ) -> None:
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = dict(headers or {})
        self.label = label

    def _open(self, req: urllib.request.Request) -> bytes:
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError) as exc:
            raise BackendError(f"{self.label} request to {req.full_url} failed: {exc}") from exc

    def _request(
        self, path: str, *, data: bytes | None = None, method: str = "GET", content_type: str | None = None
    ) -> bytes:
        headers = dict(self.headers)
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        return self._open(req)

    def _loads(self, raw: bytes) -> dict:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise BackendError(f"{self.label} returned a non-JSON response: {exc}") from exc

    def post_json(self, path: str, payload: dict) -> dict:
        raw = self._request(
            path, data=json.dumps(payload).encode("utf-8"), method="POST", content_type="application/json"
        )
        return self._loads(raw)

    def get_json(self, path: str) -> dict:
        return self._loads(self._request(path))

    def get_bytes(self, path: str) -> bytes:
        return self._request(path)
