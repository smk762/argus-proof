"""FastAPI micro-server for argus-proof (peer to argus-forge on :8104).

Routes:

    GET /health -> {status, service, version}

Phase 0 scaffold — eval routes (/run, /score, /report) land with the epic
phases: https://github.com/smk762/argus-studio/issues/6
"""

from __future__ import annotations

try:
    from fastapi import FastAPI
except ImportError as exc:  # pragma: no cover
    raise ImportError("Server requires: pip install argus-proof[server]") from exc

from argus_proof import __version__


def create_app(cors: bool = False, cors_origins: list[str] | None = None) -> FastAPI:
    """Create the proof FastAPI application."""
    app = FastAPI(
        title="Argus Proof",
        description="Post-training LoRA evaluation: generated samples in, scored verdicts out.",
        version=__version__,
    )

    if cors:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "argus-proof", "version": __version__}

    return app
