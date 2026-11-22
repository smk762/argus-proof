"""The one origin proof may call for a suite service it reaches over HTTP (argus-puppet, argus-echo).

A configured service URL must be a plain http(s) origin, optionally with a path prefix: no credentials, query
or fragment. Its ``host:port`` is the only host proof lets argus-cortex's fetch policy reach for that service,
and redirects are never followed, so a service can't send proof elsewhere.
"""

from __future__ import annotations

from urllib.parse import urlsplit


def service_origin(url: str, env_name: str) -> tuple[str, str]:
    """``(base, host:port)`` of *url*, configured as *env_name*; :class:`ValueError` for anything but a plain origin."""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"{env_name} must be an http(s) URL, not {url!r}")
    if "@" in parts.netloc or parts.query or parts.fragment:
        raise ValueError(f"{env_name} must be a plain origin (no credentials, query or fragment)")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as exc:
        raise ValueError(f"{env_name} has an invalid port") from exc
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    base = f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"
    return base, f"{host}:{port}"
