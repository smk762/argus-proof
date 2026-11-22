# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- argus-cortex `>=0.13,<0.14` (was `>=0.9,<0.10`), for the speech taxonomy (`TargetProfile.modality`,
  `SpeechCategory`) and `limits.open_audio` that speech evaluation (#118) builds on. Nothing proof uses changed.

## [0.6.0] - 2026-11-22

### Added

- **Every default scorer pins its loaded weights by SHA256** (#105): identity pins the ArcFace recogniser
  (`model_file`) and the face detector (the new `ScorerProvenance.extra_files`) and loads only those two models;
  CLIPScore pins its CLIP checkpoint's weights file; pyiqa pins every weights file it fetched or found cached
  (CLIP-IQA's RN50, CLIP-IQA+'s prompts). Pins are taken only after a load. Wire `PROOF_VERSION` 1.13 (additive).
- **Per-run scoring choices on `/run/stream`** (#110): `RunRequest.scorers` (a subset of `identity`, `likeness`,
  `clip_score`, `aesthetic`, `safety`, `pose`; phash dedup/diversity always run), `gate` (a `GateConfig` the run is
  judged and stored under), `target_category`, and `training_run_id` (forge's run, recorded in the `RunManifest`).
  `argus-proof score` gains `--scorers` and `--target-category`.

- **Run-level pose diversity and pose-copying evidence from argus-puppet** (#122, item 3): `PoseScorer` and
  `PuppetClient` (`scoring/scorers/pose.py`) post each output to argus-puppet's `/pose` over HTTP
  (`PROOF_PUPPET_URL`, `PROOF_PUPPET_TIMEOUT`; that host only, no redirects, a per-call deadline, `Retry-After`
  honoured on a busy 503, proof's priority as `X-Argus-Priority`) and fill the new top-level `EvalReport.pose` from
  `/stats`, plus `/compare` against the training frames when the export is readable here. Proof speaks puppet's wire
  (major 1, unknown fields ignored) and never depends on the package. Unset, it is reported unavailable (`GET
  /scorers`, the `score` warning, `--require-scorers pose`); down or erroring, the run still scores and the pose block
  says what wasn't measured (`None`, never `0.0`).
- `argus-proof gate --min-pose-diversity` / `--max-pose-cluster-share` (`AcceptanceThresholds.min_pose_diversity`,
  off by default; `max_pose_cluster_share` 0.75): an unmeasurable pose block fails it, a collapse fails with
  `pose_collapse`, and pose copying is only ever a `pose_copying` flag for review. Thresholds are calibrated on
  synthetic poses only. `GateCheck.code` carries those run-level codes.
- `recommend` routes a pose collapse to **curator** (narrow training poses, or outputs matching their mix) or the
  **grid** (`RecommendConfig.min_pose_diversity`, `argus-proof recommend --min-pose-diversity`; off by default).
- Wire `PROOF_VERSION` 1.10 (additive: `EvalReport.pose` and its models), then 1.11 (additive: the two pose
  thresholds and `GateCheck.code`), then 1.12 (additive: `PoseComparison.close_excess` accepts `[-1, 1]`).

### Fixed

- A non-face subject (a scene, a helmeted or non-human character) no longer scores ArcFace `identity` 0.0 on every
  output: a run that declares its `target_category` leaves `identity` `None` (#106). An `identity` run with face
  references still scores an output with no face 0.0.
- `PoseComparison.close_excess` is a signed difference: a negative value (outputs closer to training frames less
  often than the frames are to each other, the healthy case) no longer fails to parse.
- Safety fails loud when a detector crashes or the NudeNet check fails (timeout, 5xx, unreadable image): the image
  reads NaN and is rejected by a `safety` hard gate, instead of `None` ("never ran"). A finite reading from another
  detector still wins.
- Re-archiving a run reconciles `passing_images.zip` with the current verdicts: an image rejected after it was
  zipped leaves the zip, and one sent back to pending is restored to the run directory.
- A HITL batch's own gate decides only that batch's images; the run's aggregate and verdict stay under the stored
  gate, so the stored verdict matches the stored gate.
- The A1111 backend always pins `sd_vae` (`"None"` without a VAE), so a run can't inherit an earlier run's VAE.
- Only a failed GPU lease acquisition reads as "GPU unavailable"; errors from the scoring work propagate as raised.
- ComfyUI asks each host its version once per run, so a host switch can't add a spurious `unknown`.
- The remote backend writes images only after the whole response validates: a failed run leaves no orphan files.
- Krippendorff's alpha ignores unpairable ratings: perfect agreement no longer reads `nan`, and a unit with one real
  rating no longer counts as scorable.
- InsightFace reports itself unavailable without `onnxruntime`.
- A pose pass that can't reach puppet no longer reports the previous pass's puppet models in provenance.
- Demo: the tape builder applies EXIF orientation, composites transparency onto white, checks seeds and scorers
  across runs, handles empty runs/reports/diversity, and swaps in a new tape only once it is fully built. The
  grid's column count follows the tape's prompt count, and the score chart handles empty and out-of-`[0, 1]` series.

## [0.5.2] - 2026-11-19

### Fixed

- **The `-score` image has `cv2` again**: 0.5.1's swap uninstalled `opencv-python`, which also deleted the `cv2/`
  directory that `opencv-python-headless` (already installed by another `[score]` dependency) shares, so `import
  pyiqa` failed with `No module named 'cv2'` and the image again wasn't published. The headless build is now
  reinstalled after the swap.

## [0.5.1] - 2026-11-19

### Fixed

- **The `-score` image imports `pyiqa`**: `pyiqa` pulls `opencv-python`, whose `cv2` needs `libxcb`, which
  `python:3.11-slim` doesn't ship, so the 0.5.0 score image failed its smoke test and wasn't published. The image now
  swaps in `opencv-python-headless` at the same version.

## [0.5.0] - 2026-11-19

### Added

- **A `likeness` metric for subjects ArcFace can't see** (#103): `LikenessScorer`, DINOv2 whole-image embedding
  cosine against the held-out reference set, for helmeted, chrome or non-human characters and for scenes. It runs on
  every target category when there are references, is in `default_scorers()` and `GET /scorers`, and the
  reference/training-set overlap refusal covers it. `PROOF_LIKENESS_MODEL` picks the checkpoint (default
  `facebook/dinov2-base`). `argus-proof gate --min-likeness` (`AcceptanceThresholds.min_likeness_mean`) floors its
  mean, like `--min-identity`; `RecommendConfig.min_likeness` routes a low one to forge (off by default, as the
  cosine isn't calibrated yet). `[score]` now declares `transformers>=4.38`, which it already pulled.
- `ScorerProvenance.model_file` pins the weights a scorer loaded by SHA256 (the likeness scorer's
  `model.safetensors`), the way a `RunManifest` pins the generation models. Wire `PROOF_VERSION` 1.8 (additive:
  `MetricScores.likeness`, `likeness` in `GateConfig`'s default weights at **0**, `min_likeness_mean`, `model_file`).
  The weight is 0 because the cosine is uncalibrated (its threshold comes from the mint pilot), so existing runs'
  composites don't change; likeness acts only through `--min-likeness`, a hard gate or an explicit weight.

- **`POST /score`** (argus-mint#51): score uploaded images (multipart `images`, 1-16) against an export's held-out
  `references/` with `likeness` and/or `identity`, an optional crop box per image (so a teaser's character is scored,
  not its background), and the target category. Returns a `ScoreResult` (per-image `MetricScores`, means, scorer
  provenance with pinned weights, the held-out check) and stores nothing. The held-out refusal applies (400), an
  unavailable scorer or a refused GPU lease is 503, read-only refuses it, and it runs under one interactive `score`
  lease. `/health` advertises `score`. Wire `PROOF_VERSION` 1.9 (additive: `ScoreRequest`, `CropBox`, `ScoredImage`,
  `ScoreResult`).
- **Scoring takes a GPU lease** (`score`, in-process, interactive, not preemptible, 3500 MiB, `model_mb` 0), declared
  in `gpu-tenant.yaml`. `score_images` holds it around the scoring pass when an available scorer runs on the GPU, so
  `argus-proof score` and `/run/stream` (which leases at `batch`, as part of a batch run) no longer use the GPU
  unannounced. The scorers free their models (`unload()`, `release_models`) before the lease ends, so proof still
  keeps nothing resident; each pass reloads them. The figure is the torch scorers' measured peak on an RTX 3090 plus
  an estimate for InsightFace on CUDA (see `gpu-tenant.yaml`). A refused lease is an `EvaluateError`.

- **Hosted demo** for proof.dragonhound.dev (`web/`): a static replay of one recorded evaluation. A synthetic-identity
  SDXL LoRA was trained for 1600 steps; proof scored each 320-step checkpoint and a no-LoRA baseline (8 prompts x 3
  seeds each) and the page shows where it peaks: score chart, the image grid with per-image scores and verdicts,
  proof's per-run reasons, the dataset and its pipeline. `scripts/demo_tape.py` builds the tape (metrics, verdicts,
  WebP images with no metadata, no host or container paths); `make pages` exports it with a hash CSP on every page.

- **A `-score` image** (`:<version>-score`, `:latest-score`) with the `[score]` stack, CUDA torch and onnxruntime-gpu
  (#99). The thin image scores phash only. The release smoke-tests every package-backed scorer in it before pushing.
- `argus-proof score` warns on stderr when a default scorer can't run here, so its metric will be empty.
  `--require-scorers identity,clip_score,…` exits 2 instead (#99).
- `argus-proof run --fail-fast`, to keep the old stop-at-the-first-failure behaviour (#100).
- The Dockerfile takes a `VERSION` build arg, so a context with no `.git` (a `git archive` export) builds (#99).
- CI's `score-imports` job installs the `[score]` extra and checks every package-backed scorer is available (#98).

### Changed

- **`argus-proof run` carries on past a failed run** (#100). The failed run's dir gets a `failed.json` (run_id,
  error) in place of a manifest, and any stale `manifest.json` from an earlier run of that id is removed. The command
  exits 1 at the end, listing each failed run. `score` on such a dir says it failed to generate. `/run/stream` writes
  the same marker.
- **ComfyUI retries a timed-out seed once** (#100). The stalled prompt is removed from the queue and interrupted,
  then resubmitted. `ComfyUIBackend(seed_retries=…)` sets the count. Only timeouts (the new `BackendTimeout`, a
  `BackendError`) retry.
- **`[score]` needs `pyiqa>=0.1.16`; ImageReward moved to its own `[preference]` extra** (#98). pyiqa 0.1.12 with
  timm 0.6 didn't import on numpy 2, so the aesthetic scorer silently dropped out. ImageReward pins
  `timm==0.6.13`, which pyiqa 0.1.16 can't use, so the two extras are declared conflicting.
- Run, image and export ids, and `--run-prefix`, are checked by argus-cortex's `fs.safe_name` instead of a local
  pattern (the same characters, whole string, at most 128 characters). argus-cortex is pinned `>=0.9,<0.10`
  (was `>=0.7,<0.8`).

### Fixed
- **Id and value checks match the whole string** (argus-cortex's `fullmatch` codemod,
  ai-code-auditor#56). With `re.match`, an anchored check like `^[A-Za-z0-9][A-Za-z0-9._-]*$`
  also accepted the value followed by a newline, because `$` matches before a final `\n`. The
  server's run and image ids, `--run-prefix`, the `$lora`/`$text_encoder` placeholders and the
  image-name match in `evaluate` now use `fullmatch`. The Llama Guard verdict parser and the
  literal-model-name refusal keep `match`.
- **Archiving a run again no longer loses its earlier passing images.** `archive_run` rewrote
  `passing_images.zip` from the images still on disk, but `blank_slate` had already deleted the ones
  zipped last time, so archiving after reviewing the pending images dropped them from both. It now
  appends to an existing zip and skips entries it already holds.

## [0.4.0] - 2026-11-16

Wire schema 1.6 (from 1.4): new fields and one changed default, no breaking shape change.
Changes an upgrade may notice: a template that names a model file literally is refused; the
safety cutoff moves from 0.5 to an exposed score of at least 0.2, so more images count as unsafe;
identity references that are copies of training images are refused; and argus-cortex moves to 0.7.

### Added
- **Identity references that are resized or re-encoded copies of training images are refused**
  (#87). With the `[score]` extra, the held-out check adds a whole-image pHash within 6 bits to
  the SHA256 match. Measured on real photos, a resize plus JPEG re-encode lands at 0–2 bits. A crop
  is still not caught. The report's `scorers` gains a `held-out-references` row naming the check
  that ran (`sha256+phash<=6`, `sha256` without the extra, or why none could).
- **GPU leases from the argus-iris broker** (#79). The `comfyui` and `diffusers` backends hold
  one `batch` lease per run (kinds `generate` on `comfyui` and `generate-local` in-process),
  through argus-cortex's coordinator and the host-wide `ARGUS_GPU_*` variables; unset, nothing
  is gated, as before. A ComfyUI run goes to the URL the grant names. Between seeds a run gives
  the lease back when `should_yield()` is set and queues again on the same node, since the
  manifest hashed that node's models. A refused or timed-out lease, or a broker error, fails the
  run with a message that says so, and a failed run is reported to the broker as failed.
  `gpu-tenant.yaml` declares both kinds; `.env.example` and the README list the variables.
- **The report names the NudeNet detector** (#80). The safety scorer asks the server
  (`NudeNetClient.info()`, cortex 0.7) and records `nudenet-640m`, `nudenet-320n`, or `nudenet`
  when the server doesn't say as its `model` in the report's `scorers`. Cross-run stats gain a
  `safety_detector` column, sliceable, so 320n and 640m safety figures aren't pooled.

- **Flux's text encoders and VAE are pinned by SHA256** (#77; wire 1.5). `RunSpec.text_encoders`
  and `RunManifest.text_encoders` (a `ModelRef` each) fill the Flux template's new
  `$text_encoder` / `$text_encoder_2` slots, and the template's VAE is `$vae`, so the manifest
  records the T5, CLIP-L and VAE a run loaded. A Flux run that names none gets
  `t5xxl_fp8_e4m3fn` + `clip_l` and `ae` (`workflow.model_defaults`), on `/run/stream`,
  `argus-proof run` and experiment cells. The slots and the list must match one for one.
  `/run/stream` takes `vae` and `text_encoders`; `argus-proof run` takes `--text-encoder`.
  `GridConfig` carries `vae` and `text_encoders`. The remote backend compares the field.

- **Identity references that are training images are refused** (#81). `evaluate.check_held_out`
  compares each reference's SHA256 with the export's training images (the export's files outside
  `references/`, plus the `manifest.jsonl` rows' `exported_path`/`abs_path`). `/run/stream`
  answers `400` before generating, and `argus-proof score` exits 1 when the run's `source_manifest`
  is readable. Exact copies only; near-duplicates are #87.
- CLAUDE.md has a "Must never happen" table naming the check behind each rule (#81). New guards:
  no in-process AGPL import, no heavy optional dependency at module load, no committed model weights
  and none fetched by the Dockerfile, a wire change bumps `PROOF_VERSION` (a schema digest per
  version in `test_schema.py`), and every test the table cites exists.

### Fixed
- `argus-proof run --vae` reaches the run. It was accepted and dropped.

### Changed
- **The safety cutoff is set for NudeNet 640m** (#84; wire 1.6): an exposed score of at least
  0.2 is unsafe. `AcceptanceThresholds.unsafe_safety_floor` and `RecommendConfig.unsafe_safety_floor`
  default to 0.8 (were 0.5), as does `safety_tail_aggregate(unsafe_below=)`, and an image **at or
  below** the floor counts as unsafe (was strictly below), so an exposed score of exactly 0.2 is
  caught. 0.2 is NudeNet's reporting floor, so any exposed detection now counts, as in curator and
  cull. Conservative by design; it may flag the odd false positive. `safety.EXPOSED_UNSAFE_AT`,
  `UNSAFE_SAFETY_FLOOR` and `is_unsafe()` are the single definition.
- **A template that loads a model file by name is refused** (#77). A loader input such as
  `vae_name`, `clip_name1`, `unet_name` or `ckpt_name` must be a placeholder, since a literal file
  would load without a SHA256 in the manifest. A user template with a hard-coded file needs a slot.
- The `diffusers` and `a1111` backends refuse `text_encoders`, as they refuse `guidance`.
- **The suggested NudeNet server is the suite's shared 640m** (#80): argus-retina's
  `ghcr.io/smk762/argus-retina-nudenet` on morpheus :8130, in `.env.example`, the README and the
  safety scorer's docstring, in place of the upstream 320n image.
- **argus-cortex `>=0.7,<0.8`**, with the `[remote,yaml]` extras for the broker client and
  tenant-manifest loading (was `>=0.4,<0.5`).

## [0.3.0] - 2026-11-15

### Added
- **A new argus-cortex rebuilds the published image.** `release.yml` also runs on
  the `cortex-released` dispatch argus-cortex's release sends: only the image job runs,
  rebuilding the newest final `v*` tag without cache and republishing its tags, so the
  image picks up the cortex its pyproject range now resolves to.
- **`GET /report/{run_id}/manifest`** (#57) serves the `RunManifest` behind a
  report (checkpoint, VAE, LoRAs, sampling, prompt, seeds), so studio can prefill
  `/darkroom` from a passed run (argus-studio#125). It is `404` when there is no report,
  `410 Gone` when the report was kept but its run dir was pruned, and `400` for an
  unreadable manifest or one belonging to another run. `GET /health` gains a
  `capabilities` list (`["report_manifest"]`) so a client can detect the route.
- **Wire schema 1.4** (#58): `ReportSummary` and the `/run/stream` frames join
  `schema/proof-wire.schema.json`. New `ScoringFrame` and `CompleteFrame` models, plus the
  `RunStreamFrame` union discriminated by `type`, describe every NDJSON line, and the server
  builds its frames from them. `ReportSummary` moved to `argus_proof.models` and is still
  importable from `argus_proof.reports`.
- **Flux LoRAs** (#60): `SamplingParams.guidance` (wire 1.4; same field as darkroom's)
  and a `$guidance` template slot. A packaged `comfyui_flux_lora.json` is picked when
  the base checkpoint sits under `checkpoints/flux/`, with SDXL's template for every
  other family, unless `PROOF_WORKFLOW_TEMPLATE` overrides both. `/run/stream` and
  `argus-proof run` take `guidance`, and leaving out `cfg`/`sampler`/`scheduler`/`guidance`
  gets the family's defaults (Flux: cfg 1, `euler`/`simple`, guidance 3.5).
- **Devmode purge** (#59): `POST /purge/{run_id}` deletes one run's report and run dir,
  and `POST /purge` deletes all of them. They're registered only with `ARGUS_PROOF_DEVMODE=1`
  (`serve --devmode`) and need a `{"confirm": "purge"}` body. A run still generating gets
  `409`, and read-only mode refuses them. The HITL review and refinement go with the report.
  `/health` gains `devmode`, and lists `purge` in `capabilities` when the routes work.
  `/run/stream` now also refuses (`409`) a `run_id` that is already generating.
- **Llama Guard 3 moderation adapters** (#67): `LlamaGuardImageDetector`
  (Llama-Guard-3-11B-Vision) and `LlamaGuardTextDetector` (Llama-Guard-3-8B) now run
  the model instead of raising `NotImplementedError`. They lazy-load, place the model
  through cortex's `resolve_device`, and score each flagged category with P(unsafe) at
  the verdict token. `parse_guard_output` reads the verdict. The checkpoints are gated
  (`HF_TOKEN`); the README covers the licenses. `[moderation]` now needs
  `transformers>=4.45`.

### Changed
- **The wire schema lists defaulted fields as required where proof always sends them**
  (#58), like curator's `RESPONSE_ONLY`: the `EvalReport` tree, `RunManifest`, `ReportSummary`,
  the stream frames, `GateResult`, `RejectArchive` and the grid plan. Models a client
  builds (`RunSpec`, the configs, `SamplingParams`, `RejectReason`, `ProgressEvent`) are
  unchanged. Payloads and validation are the same; generated client types get stricter.

### Fixed
- **A run is refused instead of silently dropping a value** (#60) when the template
  has no slot for it: a non-empty negative prompt (`$negative`, which Flux templates
  don't have), `guidance` (`$guidance`), or `clip_skip` above 2 (`$clip_skip`). The
  `diffusers` and `a1111` backends refuse `guidance`. ComfyUI renders validate before
  the first seed is queued.
- **`clip_skip` 1 no longer selects CLIP's last layer** (#60, as darkroom #19): ComfyUI
  gets `-max(clip_skip, 2)`, so 1 and 2 both mean the penultimate layer SDXL uses.
  `-1` blanked some SDXL checkpoints. A rerun of an older `clip_skip: 1` manifest
  now renders at layer -2.
- **Moderation no longer reads as safe when a Guard model can't load** (#67).
  `PolicyModerator` re-raises `ModerationError` (a gated checkpoint with no token, an
  unreadable verdict) instead of skipping the detector as flaky. Other per-item errors
  are still skipped.
- README drift (#65): the policy-moderation section lists the `sexual` category;
  the route list adds `/health`, `/scorers` and `/report/{run_id}/image_at/{index}`;
  the CLI section documents `argus-proof schema [--check]`; the `gate` example is
  copy-pasteable and mentions `--confidence`; the experiment sample cell id is
  corrected; the diffusers limitations and the ImageReward-less default scorer set
  are noted.

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

[Unreleased]: https://github.com/smk762/argus-proof/compare/v0.5.2...HEAD
[0.5.2]: https://github.com/smk762/argus-proof/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/smk762/argus-proof/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/smk762/argus-proof/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/smk762/argus-proof/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/smk762/argus-proof/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/smk762/argus-proof/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/smk762/argus-proof/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/smk762/argus-proof/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/smk762/argus-proof/releases/tag/v0.1.0
