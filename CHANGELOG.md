# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-10-26

### Added
- `CHANGELOG.md`, with sections for every published version (#52).
- `release.yml` creates a **GitHub Release** for each `v*` tag once the version is on PyPI (#52).
  Release notes come from that version's section in this file, and a missing section fails the job.
  The built wheel and sdist are attached. Re-running the job updates the existing Release instead
  of failing. Tags containing `-`, `rc`, `a<n>` or `b<n>` become pre-releases.

### Security
- The Docker image no longer trusts **emSign ECC Root CA - G3** after the eMudhra/emSign
  certificate incident
  ([details](https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576)).
  uv is pinned to 0.13.5 instead of `:latest`, because a cached `:latest` can still be a
  pre-0.13.4 digest. The system CA bundle, which urllib callers such as the ComfyUI backend use,
  disables emSign ECC Root CA - G3 and the Comodo AAA cross-sign root. The image installs pip 26.2.2
  (hash-pinned) and `certifi>=2026.10.19`.
- `certifi>=2026.10.19` is pinned in the `dev` extra.

## [0.2.1] - 2026-07-22

### Fixed
- Release workflow: a TestPyPI failure no longer blocks the real PyPI publish. `publish-pypi` still
  runs after the TestPyPI rehearsal but only requires the build and install-test jobs, and a
  TestPyPI failure still fails the run. Before this, v0.1.0 and v0.2.0 shipped container images
  with no wheel on PyPI.
- `httpx2` added to the `dev` extra. Starlette's `TestClient` prefers it and raises when neither
  `httpx2` nor `httpx` is installed.

## [0.2.0] - 2026-07-21

### Added
- **Replay / read-only mode** (#45). `ARGUS_PROOF_READ_ONLY=1` (or `serve --read-only`) serves
  stored reports but refuses live GPU evaluation and other writes with 403.
  `/health` reports `read_only`. Off by default.
- `GET /scorers` lists the scorers and whether each one is available.
  `GET /report/{run_id}/image_at/{index}` serves a run's image by position, a seed-free URL for
  blind review.
- **Policy-taxonomy moderation** (#41). `argus_proof.moderation` scores images and input prompts
  against a category taxonomy (sexual, violence, hate, self-harm, weapons, shocking, illegal) with
  per-category tail stats. A Llama Guard adapter is available behind the new `[moderation]` extra.
- `RejectReason.category` attributes a HITL reject to a policy category. This bumps
  `PROOF_VERSION` to 1.3 (additive).
- Cross-run stats can slice by `step_config` and by experiment labels (`label:<key>`) (#36).

### Changed
- `GridConfig` and `ExperimentMatrix` share a common `GridAxes` base instead of duplicating its
  fields (#37).
- The server uses the shared `argus_cortex.server` write guard (#47) instead of a local
  `BaseHTTPMiddleware`, which buffered the `/run/stream` NDJSON stream. Requires
  `argus-cortex>=0.2.0`.

### Security
- `image_at` validates the image id it gets from a stored report, which closes a path traversal
  that allowed arbitrary image reads. A 404 no longer echoes the image id.

## [0.1.0] - 2026-07-17

First release: the MVP evaluation pipeline for trained LoRAs.

### Added
- Versioned wire contract (`PROOF_VERSION` 1.2) with a generated JSON Schema
  (`schema/proof-wire.schema.json`).
- Generation backends behind one protocol: ComfyUI, in-process diffusers, AUTOMATIC1111, and a
  remote endpoint that speaks the proof wire. Every run records a model manifest with hashes.
- Prompt-grid builder that reads curator export captions.
- Scoring spine (composite gate with auto-pass, auto-fail and HITL bands, plus near-duplicate
  collapse) and the scorers: phash dedup and diversity, ArcFace identity against a held-out
  reference set, CLIPScore / pyiqa / ImageReward quality, and NudeNet safety.
- HITL review and an optional refinement re-rank of the passing subset.
- Reject archival.
- CI acceptance gate with Wilson intervals (`argus-proof gate`, exit 0/1/2).
- Cross-run stats store (parquet) with pooled pass-rate intervals per slice and Krippendorff's alpha.
- Routed recommendations (`argus-proof recommend`).
- A/B experiment matrix with a GPU-hour budget guardrail and optional Optuna search.
- FiftyOne exploration (`argus-proof explore`) behind the `[fiftyone]` extra.
- CLI verbs `run`, `score`, `report`, `inspect`, plus a FastAPI server (`argus-proof serve`) with
  `/run/stream`, report/HITL/refine endpoints, and model and export listings.
- Container image `ghcr.io/smk762/argus-proof`.

[Unreleased]: https://github.com/smk762/argus-proof/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/smk762/argus-proof/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/smk762/argus-proof/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/smk762/argus-proof/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/smk762/argus-proof/releases/tag/v0.1.0
