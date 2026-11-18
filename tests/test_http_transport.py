"""The stdlib HTTP transport: credential headers on redirects, protocol errors as BackendError."""

from __future__ import annotations

import http.client
import io
import urllib.request

import pytest

from argus_proof.backends import http as http_mod
from argus_proof.backends.base import BackendError
from argus_proof.backends.http import UrllibTransport


def _redirect(from_url: str, to_url: str) -> urllib.request.Request:
    req = urllib.request.Request(from_url, headers={"Authorization": "Bearer s3cret", "Accept": "application/json"})
    handler = http_mod._SameOriginAuthRedirectHandler()
    new = handler.redirect_request(req, io.BytesIO(), 302, "Found", {}, to_url)
    assert new is not None
    return new


def test_a_cross_origin_redirect_drops_the_bearer_token() -> None:
    new = _redirect("https://api.example/generate", "https://evil.example/collect")
    assert "Authorization" not in new.headers and "authorization" not in {k.lower() for k in new.headers}
    assert new.get_header("Accept") == "application/json"


@pytest.mark.parametrize("to_url", ["http://api.example/x", "https://api.example:8443/x"])
def test_a_scheme_or_port_change_counts_as_another_origin(to_url: str) -> None:
    new = _redirect("https://api.example/generate", to_url)
    assert "authorization" not in {k.lower() for k in new.headers}


def test_a_same_origin_redirect_keeps_the_bearer_token() -> None:
    new = _redirect("https://api.example/generate", "https://api.example:443/v2/generate")
    assert new.get_header("Authorization") == "Bearer s3cret"


@pytest.mark.parametrize("exc", [http.client.IncompleteRead(b"x", 10), http.client.BadStatusLine("garbage")])
def test_an_http_protocol_error_is_a_backend_error(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    def boom(req, timeout):  # noqa: ANN001, ANN202
        raise exc

    monkeypatch.setattr(http_mod._OPENER, "open", boom)
    with pytest.raises(BackendError, match="request to http://svc/x failed"):
        UrllibTransport("http://svc").get_bytes("/x")
