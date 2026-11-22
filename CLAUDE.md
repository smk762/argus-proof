# CLAUDE.md — argus-proof

Guidance for AI agents working in this repo. Human-facing usage lives in [README.md](README.md); this file is the orientation an agent needs to change code safely.

## What this is

The **post-training evaluation/optimisation** stage — it closes the suite loop: generate a sample grid from a trained LoRA, score it against the curated dataset it was trained from, and emit a pass/fail verdict that feeds back to curation, captioning, and training.

```
argus-quarry -> argus-curator -> argus-lens -> argus-forge -> your trainer -> argus-proof
  acquire        curate/export    caption       configs        LoRA           evaluate + optimise
```

**The forge→proof handoff is a soft join, not an embed.** A `RunManifest` links back to the curator export via `source_manifest` (the `manifest.jsonl`) and, when the LoRA came through the suite, to forge's `training_run_id` (== forge `RunEvent.run_id`). Both are optional — a LoRA can be supplied ad hoc. Proof owns its own `run_id` (the eval run), distinct from the forge training run. README.md is the source of truth for behaviour; it is unusually detailed.

## Argus suite
<!-- argus-suite: keep this block; argus-ci checks for it -->
This repo is part of the Argus suite. The suite map below lists every sibling and what it owns; check it before
building something a sibling may already do. **On every release, update this repo's row and capabilities in
argus-cortex's SUITE.md** (a PR to argus-cortex in the same release).

@../argus-core/SUITE.md

Not checked out? Read https://github.com/smk762/argus-cortex/blob/main/SUITE.md

## Must never happen

Each row names what enforces it. An **UNENFORCED** row is a known gap: when one bites, add a check, not more prose. A rule about code shape (a banned import or API) can live in argus-ci's suite-rules catalogue instead of a test here.

| Never | Enforced by |
|---|---|
| `nudenet` or `ultralytics` imported in-process (AGPL-3.0). NudeNet is reached over HTTP via argus-cortex's `NudeNetClient` | `test_must_never::test_no_agpl_package_is_imported_in_process`; argus-ci suite rule `agpl-import` once `v1` carries it |
| A `models.py` wire change without regenerating `schema/proof-wire.schema.json` | CI post-test `argus-proof schema --check`; `test_schema.py` |
| A wire change without a `PROOF_VERSION` bump (minor additive, major breaking; mind `SUPPORTED_PROOF_MAJORS`) | `test_schema::test_a_wire_change_bumps_proof_version` (record the new version's digest there) |
| The pass rate counted per image instead of per near-duplicate group (`n_passed / n_groups`; `n_groups == n_images` when no deduper ran) | `test_scoring::test_pass_rate_counts_groups_not_frames` |
| A scorer reports a made-up `0.0` for an axis that didn't run, instead of `None` | Partial: `test_scoring::test_unavailable_scorer_is_skipped`, `test_scorers_identity::test_no_references_returns_none`, `test_scorers_identity::test_references_without_a_face_return_none_not_zero`, `test_evaluate::test_a_non_face_run_leaves_arcface_identity_none`, `test_scorers_likeness::test_no_references_returns_none_not_zero`, `test_scorers_safety::test_no_detector_can_score_returns_none`, `test_scorers_pose::test_pose_never_fabricates_a_zero` (puppet's 0-by-convention values read `None`), `test_scorers_pose::test_puppet_down_never_fails_the_run` |
| A pose gate (`--min-pose-diversity`) that passes a run whose pose block is missing or not measurable (usable < 10, usable/count < 0.5, comparable share < 0.8) | `test_acceptance::test_a_pose_gate_never_passes_when_not_measurable`, `test_cli::test_gate_fails_an_unmeasurable_pose_block` |
| `pose_copying` rejecting a run (a pose-conditioned generator reproduces training poses by design; it is a flag for review only) | `test_acceptance::test_pose_copying_is_flagged_for_review_never_rejected`, `test_cli::test_gate_flags_pose_copying_without_rejecting` |
| The pose scorer failing a run because argus-puppet is unset, down, busy or erroring | `test_scorers_pose::test_without_a_url_pose_is_unavailable_and_the_report_has_no_pose_block`, `test_scorers_pose::test_puppet_down_never_fails_the_run`, `test_scorers_pose::test_still_busy_after_the_retries_stops_the_pass_without_failing_the_run` |
| A request to any host but the configured `PROOF_PUPPET_URL` (no redirects followed; only that host allowed past argus-cortex's fetch policy) | `test_scorers_pose::test_the_puppet_url_is_only_the_configured_one`, `test_scorers_pose::test_a_redirect_is_not_followed`, `test_scorers_pose::test_only_a_plain_http_origin_is_accepted`, `test_scorers_pose::test_another_host_is_refused_by_the_fetch_policy` |
| The `argus-puppet` package imported or depended on (proof speaks its wire only: its cortex pin and onnxruntime would come with it) | `test_must_never::test_argus_puppet_is_reached_over_http_only` |
| Two scorers on one metric (`METRIC_FIELDS`) | `test_scorer_util::test_duplicate_metric_scorers_rejected` |
| A model file recorded by name without its SHA256 in the `RunManifest` | `test_backends::test_manifest_pins_files_by_sha256_and_reconstructs_run`, `test_backends::test_a_flux_manifest_pins_its_text_encoders_and_vae` |
| A scorer's loaded weights left unpinned in its provenance. Likeness (DINOv2): `ScorerProvenance.model_file` | `test_scorers_likeness::test_dinov2_pins_the_file_it_loaded_by_sha256`. InsightFace, CLIPScore and pyiqa: **UNENFORCED** (they record a model name only; #105) |
| A workflow template that loads a model file by literal name, unpinned | `test_workflow::test_a_literal_model_file_in_a_template_is_refused` |
| Identity references that overlap the training set (they measure memorisation) | Exact copies: `test_evaluate::test_a_reference_that_is_a_training_image_is_refused`, `test_evaluate::test_scoring_refuses_overlapping_references`, `test_server::test_run_stream_refuses_references_that_are_training_images`, `test_scorers_likeness::test_likeness_references_that_are_training_images_are_refused`, `test_score_endpoint::test_references_that_are_training_images_are_refused_before_scoring` (identity and likeness share the reference set and the check). Resized or re-encoded copies (pHash, `[score]`): `test_evaluate::test_a_resized_or_reencoded_training_image_is_refused`. A crop of a training image: **UNENFORCED** (pHash can't tell a crop from an unrelated photo; the report's `held-out-references` row says which check ran) |
| An image path or thumbnail in `RejectArchive`/`RejectRecord` (keyed by `(run_id, seed)`) | `test_models::test_reject_archive_carries_zero_image_references`, `test_models::test_reject_record_keys_image_by_seed_not_path`, `test_archive::test_reject_archive_has_zero_image_references` |
| Refinement overwrites the first pass (`hitl_rating`, `reject_reasons`) or the verdict, or refines an image outside the passing subset | `test_refinement::test_refinement_does_not_touch_first_pass_or_verdict`, `test_refinement::test_refining_non_passing_image_raises` |
| Moderation that fails reads as safe | `test_moderation::test_not_implemented_detector_fails_loud_not_silent_safe`, `test_moderation::test_unreadable_guard_output_fails_loud` |
| The server resolves a client-supplied filesystem path (images are `(run_id, image_id)` under `_SAFE_ID`; `POST /score` writes uploads to a temp dir as `<index>.png`, never by the client's filename) | `test_server::test_get_image_rejects_traversal_ids`, `test_score_endpoint::test_an_upload_filename_is_echoed_never_used_as_a_path` |
| Read-only mode lets a write through (`POST /run/stream` above all) | `test_server::test_read_only_mode_403s_writes_but_serves_reads`, `test_score_endpoint::test_read_only_refuses_score` |
| A concurrent report write drops an update (per-run `flock`; a no-op off POSIX, so not multi-host) | `test_reports::test_concurrent_reviews_do_not_lose_updates` |
| An optional heavy dependency (torch, diffusers, insightface, fiftyone, polars, …) imported at module load | `test_must_never::test_importing_every_module_loads_no_heavy_dependency` |
| Model weights (Llama Guard, NudeNet, checkpoints) committed, or fetched into the image | `test_must_never::test_no_model_weights_are_committed`, `test_must_never::test_the_image_fetches_no_model_weights` |
| GPU scoring without a `score` lease, or a scorer model still loaded when the lease ends | `test_gpu::test_scoring_holds_one_interactive_score_lease_and_frees_the_models_inside_it`, `test_gpu::test_a_failed_scoring_pass_frees_the_models_and_reports_the_lease_failed`, `test_gpu::test_run_stream_scores_under_a_batch_lease` |
| A failed run reported to the GPU broker as `ok`, or a yielded run re-queued on another node | `test_gpu::test_a_failed_seed_is_reported_to_the_broker_as_failed`, `test_gpu::test_a_yield_releases_and_requeues_on_the_same_node` |

## Facts the code doesn't show

- `models.py` is the API contract. `argus-proof schema` writes the schema file itself: don't redirect its stdout into the file, because that appends a status line.
- Backends sit behind `GenBackend` and the `_BACKENDS` registry, so swapping one is a config change (`PROOF_BACKEND`). Optional adapters, scorers, moderation and explore are `is_available()`-guarded.
- The quality scorers' `[0,1]` `lo`/`hi` bounds are placeholders, not calibrated. The safety cutoff is set for NudeNet 640m: an exposed score of at least 0.2 is unsafe (`safety.UNSAFE_SAFETY_FLOOR` = 0.8, inclusive via `is_unsafe`). It is conservative on purpose (#84).
- GPU work leases through argus-cortex's coordinator (`gpu.py`, `gpu-tenant.yaml`; `ARGUS_GPU_*`), with one `batch` lease per run for generation, and one `score` lease (interactive; `batch` inside `/run/stream`) around each scoring pass. The scorers free their models before that lease ends: proof keeps nothing resident (`residents: []`).
- Pose (`scoring/scorers/pose.py`, #122) is run-level and comes from argus-puppet over HTTP (`PROOF_PUPPET_URL`): `PuppetClient` mirrors the few puppet wire fields proof reads (unknown fields ignored, `puppet_version` major 1 only) and `PoseScorer` fills the top-level `EvalReport.pose` after the scoring lease (`uses_gpu = False`: puppet leases its own). It is top-level, not in `AggregateScores`, because the HITL re-roll rebuilds the aggregate with `summarise()`. Run-level gate codes (`pose_collapse`, `pose_copying`) live in `GateCheck.code`, never in the per-image `RejectReasonCode`. Training frames are compared only when the export is readable here (`evaluate.training_export`, the held-out check's rule), sampled to `MAX_TRAINING_FRAMES`, and not cached between runs.
- Reuse argus-cortex for shared code (taxonomy, wire-schema tooling, the write guard, env flags, the NudeNet client, GPU leases) instead of re-implementing it.
- The server listens on **:8104** (peers: lens :8100, curator :8101, quarry :8102, forge :8103) and backs argus-studio's `/proof` view.
- The version comes from git tags (`hatch-vcs`; `_version.py` is generated). The repo uses structlog, Pydantic v2, `asyncio_mode = auto`, ruff 0.15.16 at line length 120, and `make install|test|lint|format`. It was scaffolded from `argus-pkg-template`; `copier update` pulls tooling changes.

## Branches

`dev` is the default branch and the base for every PR. Cut feature branches from `origin/dev` and bring them up to date by merging `origin/dev`, never `main`. Never open a PR against `main` or push work to it directly: `main` must never be ahead of `dev`. A release fast-forwards `main` to a `dev` commit that passed CI (`git push origin origin/dev:main`), then tags it. If `main` is protected and needs a PR, merge the release PR, then immediately merge `main` back into `dev` so the merge commit is not left on `main` alone.
