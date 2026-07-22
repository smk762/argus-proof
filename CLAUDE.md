# CLAUDE.md — argus-proof

Guidance for AI agents working in this repo. Human-facing usage lives in [README.md](README.md); this file is the orientation an agent needs to change code safely.

## What this is

The **post-training evaluation/optimisation** stage — it closes the suite loop: generate a sample grid from a trained LoRA, score it against the curated dataset it was trained from, and emit a pass/fail verdict that feeds back to curation, captioning, and training.

```
argus-quarry -> argus-curator -> argus-lens -> argus-forge -> your trainer -> argus-proof
  acquire        curate/export    caption       configs        LoRA           evaluate + optimise
```

**The forge→proof handoff is a soft join, not an embed.** A `RunManifest` links back to the curator export via `source_manifest` (the `manifest.jsonl`) and, when the LoRA came through the suite, to forge's `training_run_id` (== forge `RunEvent.run_id`). Both are optional — a LoRA can be supplied ad hoc. Proof owns its own `run_id` (the eval run), distinct from the forge training run. README.md is the source of truth for behaviour; it is unusually detailed.

## Layout

`src/argus_proof/`:

- `models.py` — Pydantic v2 wire types, the three versioned top-level shapes (`RunManifest`, `EvalReport`, `RejectArchive`) plus `RunSpec`/grid/gate/acceptance models. **This is the API contract**; `wire_schema()`/`render_wire_schema()` back `argus-proof schema` and the committed `schema/proof-wire.schema.json`.
- `grid.py` / `experiment.py` — the prompt-grid builder (axes × prompts × seeds → `RunSpec`s) and the A/B `ExperimentMatrix` that varies base-checkpoint × step-config as outer factors.
- `backends/` — pluggable generation engines over `base.GenBackend`: `comfyui` (shipped), `diffusers`, `a1111`, `remote`; `workflow.py`/`pnginfo.py`/`http.py` are shared helpers. `get_backend(name, ...)` selects by config (`_BACKENDS` registry).
- `scoring/` — `base.py` (the `ImageScorer`/`Deduper`/`DiversityScorer` protocols + `METRIC_FIELDS`), `orchestrator.py` (`score_run`), `gate.py`, `summary.py`; concrete scorers under `scoring/scorers/` (phash, identity, quality, safety) behind the `[score]` extra.
- `reports.py` (`ReportStore`, `apply_hitl`), `refinement.py`, `acceptance.py`, `recommend.py`, `crossrun.py`, `archive.py`, `stats.py`, `moderation.py`, `explore.py` — the report store + the optimisation layers on top of a scored run.
- `cli.py` — Typer app (`inspect`, `run`, `score`, `report`, `gate`, `recommend`, `experiment`, `explore`, `schema`, `serve`).
- `server/app.py` — FastAPI micro-server on **:8104** (peer to lens :8100, curator :8101, quarry :8102, forge :8103); backs the argus-studio `/proof` view. Optional `[server]` extra.
- `templates/comfyui_sdxl_lora.json` — the shipped parametric ComfyUI workflow (`$placeholder` graph).

## Commands

```bash
make install   # uv venv + editable install with [dev,server,cli]
make test      # uv run --no-sync pytest --tb=short -q
make lint      # ruff 0.15.16 check + format --check
make format    # ruff format + check --fix
```

Run a single test: `uv run --no-sync pytest tests/test_scoring.py::test_name -q`.

## Conventions & gotchas

- **The wire schema is checked in CI** (`argus-proof schema --check` runs post-test). If you touch `WIRE_MODELS`/types in `models.py`, regenerate: `argus-proof schema > schema/proof-wire.schema.json`. Bump `PROOF_VERSION` (minor = additive field/metric, major = breaking) and mind `SUPPORTED_PROOF_MAJORS`.
- **Pass-rate is computed over near-duplicate *groups*, not raw frames** (`n_passed / n_groups`), so a Monte-Carlo cluster counts once. Never revert this to per-image — it's the core anti-inflation invariant. `n_groups == n_images` when no deduper ran.
- **Scorers return normalised `[0,1]` (higher = better) or `None`** — never a fabricated `0.0` for an axis that didn't run. At most one scorer per metric (`METRIC_FIELDS`); the orchestrator raises if two target the same one. The `[0,1]` quality-scorer normalisation `lo`/`hi` defaults are **placeholders** — calibrate against real generations.
- **Every model file is pinned by SHA256** in the `RunManifest`, not by filename — a run must reconstruct exactly. Backends resolve a requested name to a file and hash it; don't record a name without its digest. `RejectArchive`/`RejectRecord` are **image-free by construction** (keyed by `(run_id, seed)` — seed + manifest reconstructs the pixels); never add a path/thumbnail field there.
- **Refinement is a separate layer.** `ImageScores.refinement` never overwrites the first-pass `hitl_rating`/`reject_reasons` or the verdict; refining an image not in the passing subset is refused.
- **Backend swap is a config change, not a code change** — scoring/report code is engine-agnostic (`PROOF_BACKEND` selects). Keep new engines behind `GenBackend` + the `_BACKENDS` registry; heavy adapters stay lazy-imported and `is_available()`-guarded (same for optional scorers/moderation/explore).
- **Server safety — don't loosen either boundary:**
  - Images are addressed by `(run_id, image_id)`, both validated against a strict charset (`_SAFE_ID`) — **no client-supplied filesystem path is ever resolved**. Keep it that way; there is no traversal surface by design.
  - Read-only/replay mode (`ARGUS_PROOF_READ_ONLY`, `serve --read-only`): every mutating request — `POST /run/stream` (live GPU eval) above all — is refused with `403` via the shared `argus_cortex.server.WriteGuard`. `/health` echoes the flag. It fails *safe*.
  - Concurrent report writes are serialised by a per-run `flock` sidecar (`reports.py`), so a streaming run's write and a reviewer's HITL save can't drop updates. On non-POSIX it degrades to a no-op — don't rely on it for multi-host.
- Shared suite code (taxonomy, wire-schema tooling, the server write-guard/env-flag helpers, `RemoteBackend`) lives in **argus-cortex** (`argus-cortex[server]`) — reuse it, don't re-implement.
- Versioning is git-tag-derived (`hatch-vcs`); `_version.py` is generated (gitignored). `structlog` logging, Pydantic v2, async server (`asyncio_mode = auto`), ruff line-length 120. Scaffolded from `argus-pkg-template` — `copier update` pulls tooling changes.
