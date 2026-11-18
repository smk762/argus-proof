<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/smk762/argus-cortex/dev/brand/packages/proof/argus-proof-lockup-dark.svg">
    <img alt="argus-proof" src="https://raw.githubusercontent.com/smk762/argus-cortex/dev/brand/packages/proof/argus-proof-lockup-light.svg" height="52">
  </picture>
</h1>

Post-training LoRA evaluation and optimisation: generate samples from a trained LoRA and score them against the curated dataset it was trained from.

Part of the [Argus suite](https://github.com/smk762?tab=repositories&q=argus) — the stage that closes the loop after
[argus-forge](https://github.com/smk762/argus-forge) emits a training config and you train a LoRA:

```
argus-quarry -> argus-curator -> argus-lens -> argus-forge -> your trainer -> argus-proof
  acquire         curate/export     caption       configs        LoRA           evaluate + optimise
```

> **Status: functional end-to-end.** The executing verbs (`run`, `score`, `report`,
> `inspect`) are live, the server triggers generation+scoring runs and serves the
> resulting images to the argus-studio `/proof` review view. Roadmap:
> [argus-proof epic](https://github.com/smk762/argus-studio/issues/6).

**Demo:** [proof.dragonhound.dev](https://proof.dragonhound.dev) replays one recorded evaluation: a synthetic-identity
SDXL LoRA trained for 1600 steps, every 320-step checkpoint and a no-LoRA baseline scored by proof (144 images), and
where it peaks. It is a static page with no backend; the recording is `web/public/tape/`, built by
`scripts/demo_tape.py <recording-dir>`, and `make pages` exports the site to `web/out/`.

## Install

```bash
uv pip install "argus-proof[cli]"          # CLI
uv pip install "argus-proof[cli,server]"   # + HTTP server for argus-studio
uv pip install "argus-proof[cli,score]"    # + the full scorer stack (torch, insightface, …)
```

`[preference]` (ImageReward) can't share an environment with `[score]`: it pins `timm==0.6.13`, which pyiqa 0.1.16+
can't use. ImageReward isn't in the default scorer set, so install it on its own when you want the `preference` metric.

Images are on GHCR. `ghcr.io/smk762/argus-proof:<version>` (and `latest`) is the thin server/CLI and scores **phash
only**: identity, likeness, CLIP score and aesthetic come back empty. To score for real, use `:<version>-score` (`latest-score`),
which adds the `[score]` stack, with CUDA torch and onnxruntime-gpu (they fall back to CPU without a GPU). To build
from a context without `.git` (a `git archive` export), pass `--build-arg VERSION=<x.y.z>`.

## CLI

```bash
# Generate a sample grid from a trained LoRA (prompts come from the curator
# export's captions; backend/engine/model dirs from the environment, see
# .env.example — PROOF_BACKEND, COMFYUI_BASE_URL, PROOF_MODELS_DIR):
argus-proof run subject.safetensors ./curated_export \
  --checkpoint sdxl_base.safetensors --seed 1 --seed 2 --seed 3 --out runs
# A run that fails (ComfyUI retries a timed-out seed once first) leaves failed.json in its dir
# instead of a manifest, and the rest of the grid still runs; the command then exits 1 listing
# the failed run_ids. --fail-fast stops at the first failure.

# Score a generated run into a stored EvalReport (identity and likeness need --references,
# a held-out image dir that must NOT overlap the training set). A default scorer
# this install can't run is warned about on stderr; --require-scorers makes it exit 2:
argus-proof score runs/proof-l00-w00-p000 --references ./holdout \
  [--require-scorers identity,likeness,clip_score,aesthetic,safety]

# Browse stored reports / print one run's digest:
argus-proof report
argus-proof report proof-l00-w00-p000 [--json]

# Summarise an export dir (prompt sources) or a run dir (manifest + images):
argus-proof inspect ./curated_export
argus-proof inspect runs/proof-l00-w00-p000

# Emit the wire-contract JSON Schema consumers codegen against (default
# schema/proof-wire.schema.json; --check exits non-zero if the committed copy is stale):
argus-proof schema [--output PATH] [--check]
```

`gate`, `recommend`, `experiment` and `explore` are covered in their own sections below.

## Serve

```bash
argus-proof serve --port 8104 --cors   # peer to lens :8100, curator :8101, quarry :8102, forge :8103
argus-proof serve --read-only          # replay/demo mode: serve stored reports, 403 live eval + writes
curl -s localhost:8104/health          # {"status":"ok","service":"argus-proof","version":"...","read_only":false,"capabilities":["report_manifest"]}
```

Routes (the backend of the argus-studio `/proof` view):

```
GET  /health                            liveness + version + read_only flag + capabilities
GET  /exports                           curator export dirs available to evaluate against
GET  /models                            checkpoints + LoRAs under $PROOF_MODELS_DIR
GET  /scorers                           scorers a run applies + whether each is installed
POST /run/stream                        generate + score + store one run (NDJSON RunStreamFrame lines)
GET  /reports                           stored report digests (run browser)
GET  /report/{run_id}                   full EvalReport            PUT to store one
GET  /report/{run_id}/manifest          the RunManifest behind a report (410 if its run dir is gone)
GET  /report/{run_id}/refined           passing subset, refined ranks first
GET  /report/{run_id}/image/{image_id}  a generated sample (ids only — no path input)
GET  /report/{run_id}/image_at/{index}  sample by position — seed-free URL for blind review
POST /report/{run_id}/hitl              apply a review; recomputes pass-rate + verdict
POST /report/{run_id}/refine            second-pass re-rank (rank: null retracts)
POST /purge/{run_id}                    devmode only: delete one run's report + run dir
POST /purge                             devmode only: delete every report + run dir
```

`/report/{run_id}/manifest` serves the SHA256-pinned setup (checkpoint, VAE,
LoRAs, sampling, prompt, seeds) so a client can carry a passed run forward, e.g.
studio's proof → darkroom handoff. It is `404` when there is no report and `410
Gone` when the report was kept but the run dir was pruned. `/health` lists
`report_manifest` in `capabilities`, so a client can tell an older proof (route
absent) from a run with no manifest.

The `/purge` routes are a dev tool, since reports are the evaluation record. They exist
only with `ARGUS_PROOF_DEVMODE=1` (or `serve --devmode`), need a `{"confirm": "purge"}`
body, return `409` while the run (or, for `POST /purge`, any run) is still generating,
and read-only mode refuses them. A run's HITL review and refinement are stored in its
report, so they go with it. `/health` echoes `devmode` and lists `purge` in
`capabilities` only when the routes will work.

Concurrent reviews of the same run are serialised with a per-run file lock, so
a streaming run's report write and a reviewer's HITL save can't drop updates.

## Generation backend

Generation is a pluggable backend (`argus_proof.backends`) so swapping the engine
is a config change, not a code change — the **ComfyUI** adapter ships first:

```python
from pathlib import Path
from argus_proof.backends import get_backend
from argus_proof.backends.base import make_dir_resolver
from argus_proof.backends.workflow import example_template
from argus_proof.models import RunSpec, LoRASpec, SamplingParams

backend = get_backend(
    "comfyui",
    workflow_template=example_template(),          # or workflow.load_template(path)
    resolve_model=make_dir_resolver(Path("~/ComfyUI/models")),
    base_url="http://127.0.0.1:8188",
)
spec = RunSpec(
    run_id="run-1",
    base_checkpoint="sdxl_base.safetensors",
    loras=[LoRASpec(name="subject.safetensors", weight=0.8)],
    sampling=SamplingParams(sampler="dpmpp_2m", scheduler="karras", steps=30,
                            cfg=7.0, clip_skip=2, width=1024, height=1024),
    prompt="a photo of sks person",
    seeds=[1, 2, 3],                               # seed-set: one image per seed
)
result = backend.generate(spec, Path("out/run-1"))  # writes images + manifest.json
```

The ComfyUI adapter drives a **parametric workflow template** (an API-format graph
with `$placeholder` values — `$base_checkpoint`, `$positive`, `$seed`, `$steps`,
`$lora` / `$lora_weight`, `$clip_skip`, …), polls for completion, reads back each
image's embedded **PNGInfo**, and emits a `RunManifest` that pins every
checkpoint/LoRA by **SHA256** so the run reconstructs exactly. See
[`templates/comfyui_sdxl_lora.json`](src/argus_proof/templates/comfyui_sdxl_lora.json)
for the shipped example.

**SDXL and Flux.** The template is chosen by the base checkpoint's *family*, its
top-level folder under `checkpoints/`: `flux/…` renders
[`templates/comfyui_flux_lora.json`](src/argus_proof/templates/comfyui_flux_lora.json)
and every other family renders the SDXL template, so one server evaluates both.
`PROOF_WORKFLOW_TEMPLATE` replaces both for every checkpoint. The Flux checkpoints
are bare transformers, so the Flux template also loads two text encoders
(`DualCLIPLoader`, type `flux`: the `$text_encoder` and `$text_encoder_2` slots) and
a VAE (`VAELoader`: `$vae`). They come from the run (`RunSpec.text_encoders`, `vae`)
and the manifest pins each by SHA256, like the checkpoint and LoRA, so swapping the
T5 for another build changes the manifest. A run that names none gets the Flux
defaults, `t5xxl_fp8_e4m3fn.safetensors` + `clip_l.safetensors` and `ae.safetensors`
(`/run/stream`'s `text_encoders`/`vae`, `argus-proof run --text-encoder/--vae`); they
must be in ComfyUI's `clip/` and `vae/` folders and under `PROOF_MODELS_DIR`. The
template's encoder slots and the run's list must match one for one. **A template
whose loader names a model file literally** (`"vae_name": "ae.safetensors"`) is
refused: that file would load unrecorded. Flux takes no negative prompt (the template zeroes the positive) and runs
at `cfg` 1 with `sampling.guidance` (the `$guidance` slot, a `FluxGuidance` node).
A request that leaves `cfg`/`sampler`/`scheduler`/`guidance` out gets its family's
defaults: SDXL cfg 7, `dpmpp_2m`/`karras`; Flux cfg 1, `euler`/`simple`, guidance 3.5.

Nothing a run sets is silently dropped. A run is refused when it sets a value
the template has no slot for: a VAE, a non-empty negative prompt, `guidance`,
or a `clip_skip` above 2. The `diffusers` and `a1111` backends refuse `guidance`
and `text_encoders` outright. `clip_skip` is sent to ComfyUI as `-max(clip_skip, 2)`: SDXL uses the
penultimate layer, and `-1` blanks some checkpoints.

**More backends.** `get_backend(name, ...)` selects the engine by config
(`PROOF_BACKEND` in [`.env.example`](.env.example)); scoring/report code is
unchanged regardless of which produced the run, and the `RunManifest` records the
engine + version:

- **`diffusers`** — in-process diffusers SDXL pipeline: deterministic, no external
  service, weights hashed from disk (`pip install "argus-proof[diffusers]"`).
  Limitations: `clip_skip` is not applied, and only a handful of samplers map to a
  diffusers scheduler (`euler`, `euler_a`, `ddim`, `dpmpp_2m`, `unipc`; unknown
  names fall back to the pipeline default). The manifest still records the
  *requested* values, so compare across engines with care.
- **`a1111`** — an AUTOMATIC1111 / SD.Next `/sdapi` server (checkpoint via
  `override_settings`, LoRAs via `<lora:…>` prompt syntax; models hashed from disk).
- **`remote`** — a hosted/cloud endpoint that speaks the proof wire (`POST /generate`
  → `RunManifest` + base64 images). The weights live remotely, so the **service**
  supplies the manifest and it's validated at the boundary; a bearer `api_key`
  authenticates. Point it at a self-hosted proof-gen service or a thin wrapper in
  front of Replicate / fal.

The `a1111` / `remote` adapters need no extra (stdlib HTTP); all three reuse the
shared manifest + transport helpers and are unit-tested with fakes.

**GPU leases (argus-iris).** The `comfyui` and `diffusers` backends hold a GPU lease
for each run from the suite's coordinator, set host-wide by `ARGUS_GPU_*` (unset:
no gating, as before). With `ARGUS_GPU_COORDINATOR=broker`, a run asks
[argus-iris](https://github.com/smk762/argus-iris) for a `batch` lease, on
`comfyui` (kind `generate`) or `in-process` (kind `generate-local`), declared in
[`gpu-tenant.yaml`](gpu-tenant.yaml). The work goes to the ComfyUI the grant
names (else `COMFYUI_BASE_URL`). Between seeds proof checks `should_yield()`:
when the broker wants the GPU back, it releases the lease and queues again,
pinned to the node the run started on, because the manifest hashed one set of
model files. A refused or timed-out lease fails the run with a message that says
so. When iris can't be reached, the lease comes from `ARGUS_GPU_BROKER_FALLBACK`
(`lease`, the host-wide `flock`, by default). The `a1111` and `remote` backends
take no lease.

Scoring leases too. `score_images` (so `argus-proof score` and `/run/stream`) holds
one `score` lease (`in-process`, not preemptible, 3500 MiB) for the scoring pass
whenever an available scorer runs on the GPU (InsightFace, CLIP-IQA, CLIPScore; not
safety, which is a NudeNet service, and not phash). It is `interactive`, or `batch`
when it is part of a `/run/stream` run. The scorers load their models inside the
lease and free them before it ends, so proof keeps no model resident between leases
(`residents: []`): each pass reloads its models, a few seconds' cost. A refused or
timed-out lease fails scoring with a message that says so. The 3500 MiB is the torch
scorers' measured peak on an RTX 3090 (1399 MiB with the CUDA context) plus an
estimated 1500 MiB for InsightFace on CUDA, rounded up; `gpu-tenant.yaml` has the
breakdown.

| Variable | Value |
|---|---|
| `ARGUS_GPU_COORDINATOR` | `broker` (or `lease`, `none`) |
| `ARGUS_GPU_BROKER_URL` | `http://<GPU_HOST>:8120` |
| `ARGUS_GPU_BROKER_TOKEN` | the shared caller token (secret: the host's `.env`, never a repo) |
| `ARGUS_GPU_TENANT_MANIFEST` | `gpu-tenant.yaml` (the image ships it at `/app/gpu-tenant.yaml`) |
| `ARGUS_GPU_BROKER_FALLBACK` | `lease` (default) or `none` |

## Scoring

Generated images are scored into an `EvalReport` by a pluggable framework
(`argus_proof.scoring`). Per-image `ImageScorer`s each fill one normalised
`[0,1]` metric (identity / likeness / clip_score / aesthetic / preference / safety); a
`Deduper` collapses Monte-Carlo near-duplicates so a cluster counts **once**
toward the pass rate; a `DiversityScorer` rewards variety. A `GateConfig` routes
each image to **auto-pass / auto-fail / needs-HITL** on a weighted composite,
so humans only rate the borderline band:

```python
from argus_proof.scoring import score_run, ScoreContext
report = score_run(manifest, images, scorers=[...], deduper=..., diversity=...)
report.aggregate.pass_rate   # computed over near-dup groups, not raw frames
report.verdict.passed        # run pass/fail vs GateConfig.run_pass_rate
```

Concrete scorers live in `argus_proof.scoring.scorers`, behind the `[score]`
extra, and are lazy-imported (each reports `is_available()` so the orchestrator
skips it when the extra is absent). Shipped:

- **dedup + diversity** — `PhashDeduper`, `PhashDiversityScorer` (perceptual hash, CPU-only)
- **identity** — `IdentityScorer` (InsightFace ArcFace cosine vs a held-out reference set).
  References must be held out from training, and a run refuses ones that aren't: before generating (`/run/stream`) and before scoring, each reference is compared with the export's training images (its files outside `references/`, plus the manifest rows' sources). An exact copy is caught by SHA256; with the `[score]` extra, a resized or re-encoded copy is caught by a whole-image pHash within 6 bits. **A crop is not caught.** The report's `scorers` gains a `held-out-references` row saying which check ran (`sha256+phash<=6`, `sha256`, or why none could).
- **likeness** — `LikenessScorer` (DINOv2 whole-image embedding cosine vs the same held-out reference set, #103). The identity score for subjects ArcFace can't see: helmeted, masked, chrome-skinned or non-human characters, and scenes or sets, where `identity` reads `0.0` (no face). It runs on every target category whenever there are references (`None` without), and the held-out refusal above covers it, since both scorers read the one reference set. DINOv2 rather than CLIP image embeddings: CLIP is trained to match captions, so two different chrome robots embed close together because both are "a chrome robot"; DINOv2 is self-supervised on images and keeps instance-level appearance. It loads through `transformers` in the `[score]` extra, which CLIPScore and pyiqa already bring. `PROOF_LIKENESS_MODEL` picks the checkpoint (default `facebook/dinov2-base`; a Hub id or a local `save_pretrained` dir for an offline host). Once loaded, its `model.safetensors` is pinned by SHA256 in the report's scorer provenance (`ScorerProvenance.model_file`), as a `RunManifest` pins the generation models. The raw cosine is clamped, not calibrated: unrelated images still sit around 0.1–0.4 (and simple synthetic shapes higher), so take `--min-likeness` from a real run. `likeness` has weight **0** in the default composite, because it is uncalibrated and its threshold comes from the mint pilot run: it acts only through `--min-likeness`, a `likeness` hard gate (reject code `identity_mismatch`) or an explicit `GateConfig.weights["likeness"]`.
- **quality / adherence** — `clip_score_scorer()` (CLIPScore), `pyiqa_scorer()` (CLIP-IQA), `image_reward_scorer()` (ImageReward), each normalising its raw score to `[0,1]`
- **safety** — `SafetyScorer` (NudeNet ensemble, `1 - unsafe`); set a `safety` hard gate to auto-fail unsafe images. `safety_tail_aggregate()` surfaces the any-hit/min/percentile tail. NudeNet runs as a **server over HTTP** (`PROOF_NUDENET_URL`, via argus-cortex's `NudeNetClient`), never in-process (it is AGPL-3.0); without a URL the scorer is skipped. Point it at the suite's shared **640m** server, argus-retina's `ghcr.io/smk762/argus-retina-nudenet` (`http://<GPU_HOST>:8130` on the LAN); the upstream `notai-tech/nudenet` image is **320n**. The two score the same image differently, so the report records which one ran (the safety scorer's `model`: `nudenet-640m`, `nudenet-320n`, or `nudenet` when the server doesn't say, as the upstream image doesn't), and the cross-run store keeps it as a sliceable `safety_detector` column. The cutoff is tuned for 640m: an exposed score of **at least 0.2** is unsafe, so `unsafe_safety_floor` defaults to 0.8 and an image at or below it counts (the acceptance gate's `max_unsafe_rate`, `recommend`, and `safety_tail_aggregate()`). That is NudeNet's reporting floor, so any exposed detection counts, as in curator and cull. It is deliberately conservative and may flag the odd false positive; raise the floor's exposed score if that becomes a problem. "Unsafe" is the suite's exposed-nudity set (exposed buttocks, breasts, genitalia, anus), the same as curator's

The default scorer set used by `argus-proof run` / `score` and the server
(`evaluate.default_scorers()`, plus phash dedup/diversity) is identity, likeness, CLIPScore,
CLIP-IQA and safety — it does **not** include ImageReward, so the `preference`
metric stays empty unless you pass `image_reward_scorer()` via the library API (in a `[preference]` environment).
`GET /scorers` reports which of the default set this install can actually run.

```python
from argus_proof.scoring import score_run
from argus_proof.scoring.scorers import (
    PhashDeduper, PhashDiversityScorer, IdentityScorer, clip_score_scorer, pyiqa_scorer,
)
report = score_run(
    manifest, images,
    scorers=[IdentityScorer(), clip_score_scorer(), pyiqa_scorer()],
    deduper=PhashDeduper(), diversity=PhashDiversityScorer(),
)
```

> The quality scorers' default `[0,1]` normalization ranges are **placeholders** —
> calibrate `lo`/`hi` (e.g. `clip_score_scorer(lo=…, hi=…)`) against real
> generations. Heavy backends need `pip install "argus-proof[score]"`; remote/hosted
> variants build on `argus_cortex.backends.RemoteBackend` (point at a service by IP/port).
> The spine itself is dependency-free and fully tested with fakes.

## HITL review & refinement

Reports are stored per-run (`argus_proof.reports.ReportStore`, a directory of
`<run_id>.json`) and reviewed over the server (peer to the argus-studio `/proof`
view):

```
POST /report/{run_id}/hitl     # 5-star ratings + reject reasons; recomputes pass-rate + verdict
POST /report/{run_id}/refine   # optional second pass: re-rank the passing subset 1-5 + notes
```

The **refinement** stage (`argus_proof.refinement`) is a finer re-rank of just
the images that already passed — a **separate layer** (`ImageScores.refinement`)
that never overwrites the first-pass `hitl_rating`/`reject_reasons` or the run's
verdict, so both the original decision and the refined ordering are kept.
`refined_ranking(report)` surfaces the passing subset best-first; refining an
image that isn't in the passing subset is refused.

```python
from argus_proof.refinement import RefinementRequest, RefinementImageUpdate, apply_refinement, refined_ranking

refined = apply_refinement(report, RefinementRequest(
    rater="alice", updates=[RefinementImageUpdate(image_id="img-3", rank=5, notes="cleanest hands")],
))
best_first = refined_ranking(refined)   # passing subset, refined re-ranks on top
```

## Policy moderation (optional)

The Phase-2 `safety` scorer catches **nudity**; `argus_proof.moderation` extends
it to a **Guard-class policy taxonomy** (sexual / violence / hate / self-harm /
weapons / illegal) — over both the **generated images** and the **input prompts /
captions**, so a toxic prompt is flagged even when its output is clean.

The default detectors are **Llama Guard 3**: `LlamaGuardImageDetector`
(`meta-llama/Llama-Guard-3-11B-Vision`, about 22 GB in bf16) for outputs and
`LlamaGuardTextDetector` (`meta-llama/Llama-Guard-3-8B`, about 16 GB) for prompts and
captions. Install `argus-proof[moderation]`. Both checkpoints are **gated** on Hugging
Face: accept the license on each model page and set `HF_TOKEN`. The weights download
at first use; nothing is bundled. Device placement goes through argus-cortex's
`resolve_device` (`device="auto"` picks cuda, then mps, then cpu).

Each verdict (`safe`, or `unsafe` plus hazard codes `S1`–`S14`) maps onto the taxonomy.
A flagged category scores P(unsafe) at the verdict token rather than a hard 1.0.
Moderation fails loud rather than reading as safe. A model that can't load (no
token, license not accepted) or an unreadable verdict raises `ModerationError`, and an
`unsafe` verdict naming no code flags every category. An image is sent with the
neutral prompt `"Describe this image."`, so a flag comes from the image.

> **Licenses.** Llama-Guard-3-8B is under the Llama 3.1 Community License, and
> Llama-Guard-3-11B-Vision under the Llama 3.2 Community License. The 3.2 license
> does **not** grant its rights for multimodal models to individuals domiciled in,
> or companies with their principal place of business in, the European Union
> (end users of a product built on them are exempt). Both carry Meta's acceptable
> use policy. argus-proof only downloads the weights at runtime and doesn't
> redistribute them. Check the terms against your use, and pass your own detectors
> if they don't fit.

Any object with `name`, `version`, `is_available()` and `moderate_image(path)` /
`moderate_text(text)` returning per-category `[0,1]` scores can replace them:

```python
from argus_proof.moderation import PolicyModerator, moderate_images, moderate_texts

mod = PolicyModerator()                          # Llama Guard 3 Vision + text
# mod = PolicyModerator(image_detectors=[MyImageGuard()], text_detectors=[MyTextGuard()])
out = moderate_images(image_paths, mod)         # per-category tails over the outputs
inp = moderate_texts(prompt_grid_variants, mod) # ...and over the inputs
out.flagged()                                   # e.g. ["violence", "hate"], worst first
report.scorers.append(mod.provenance("output")) # version-stamp the Guard model + taxonomy
```

Each category gets a **tail** view (any-hit / max / 95th percentile — the extremes
that matter, not the mean, same rule as `safety`), combined conservatively across
an ensemble (most-unsafe detector wins). Detectors are pluggable and injectable, so
the taxonomy/ensemble/tail logic is dependency-free and unit-tested. Of the Llama
Guard adapters, only the model call needs the `[moderation]` extra; verdict parsing
and mapping are unit-tested. A reviewer's HITL flag attributes to a
category via `RejectReason.category`. CSAM matching stays a **separate** policy gate
(Thorn Safer / PhotoDNA), not an ML metric here.

## CI acceptance gate

Turn "was this LoRA/dataset good enough?" into an automatable yes/no. `argus-proof
gate` evaluates a scored `EvalReport` against declared thresholds and **exits
non-zero when rejected**, so it drops straight into CI:

```bash
# --min-pass-rate-ci-lower: Wilson lower bound — a lucky 3/3 won't pass
# --confidence: confidence level for that bound (default 0.95)
# exit 0 = accepted, 1 = rejected, 2 = unreadable report / invalid threshold
argus-proof gate eval_report.json \
  --min-pass-rate 0.75 \
  --min-pass-rate-ci-lower 0.7 \
  --confidence 0.95 \
  --min-identity 0.6 \
  --min-likeness 0.7 \
  --max-unsafe-rate 0.0
```

`--min-identity` floors the mean ArcFace identity (human faces); `--min-likeness`
floors the mean DINOv2 likeness (any subject, `AcceptanceThresholds.min_likeness_mean`).
Use whichever measures the subject, or both.

The pass-rate lower bound uses a Wilson score interval (`argus_proof.stats`, no
scipy), so acceptance is statistically defensible at small N. A configured metric
that wasn't measured fails its check rather than passing silently.

## Cross-run stats

Per-run reports accumulate into a queryable store so "which checkpoint / LoRA
weight / token wins?" is answered with evidence, not vibes (`argus_proof.crossrun`,
`pip install "argus-proof[stats]"`):

```python
from argus_proof.crossrun import CrossRunStore, run_stats, krippendorff_alpha

store = CrossRunStore("proof_stats.parquet")
store.append(run_stats(manifest, report))          # one tidy row per run (re-append updates)
for cell in store.slice_pass_rate("base_checkpoint"):
    print(cell.value, cell.pass_rate, (cell.ci_low, cell.ci_high))   # pooled pass-rate + Wilson CI

# Comparing A/B experiment arms: attribute each run to its cell, then slice by the arm.
for arm in plan.cells:                             # an ExperimentCell (see the matrix section)
    store.append(run_stats(manifest, report, step_config=arm.step_config, labels=arm.labels))
store.slice_pass_rate("step_config")               # fast vs quality
store.slice_pass_rate("label:caption_strategy")    # florence vs wd14 (an upstream factor)

alpha = krippendorff_alpha([{"alice": 5, "bob": 4}, ...])   # inter-rater reliability
```

Pass-rate slices carry a **Wilson confidence interval**, so a lucky 3/3 cell reads
as far less certain than 300/400; the store is parquet, keyed by `run_id` + versions.

## Recommendations

The gate says *did it pass?*; `argus_proof.recommend` says *what to change, and
where* — mapping weak metrics to the suite stage that owns the fix:

```bash
argus-proof recommend eval_report.json --store proof_stats.parquet
# [lens]  unsafe outputs [unsafe_rate 0.04 vs 0.00]: filter/re-caption training data…   (safety first)
# [forge] identity didn't transfer [identity 0.41 vs 0.60]: add/curate more identity images…
# [lens]  prompt adherence low [clip_score 0.32 vs 0.50]: revisit the captioning strategy…
# [grid]  prompt adherence low [clip_score 0.32 vs 0.50]: try different prompt/token combinations…
# [checkpoint] base_checkpoint outcome varies across runs: prefer base_checkpoint='sdxl_v2'…
```

```python
from argus_proof.recommend import RecommendConfig, recommend
# keep the floors in lock-step with the CI gate so the two can't disagree:
cfg = RecommendConfig.from_acceptance(thresholds)
for rec in recommend(report, config=cfg, store=cross_run_store):   # store optional
    print(rec.stage, rec.metric, rec.value, rec.action)
```

Safety first, then: low identity/likeness/aesthetic → **forge** (training), low adherence →
**lens** + **grid**, low diversity → **grid**, borderline → **refine** (HITL). With a
cross-run store it also surfaces the best checkpoint / LoRA weight — but only when
the evidence separates a clear winner (non-overlapping CIs), never on a tie.

## A/B experiment matrix

Compare LoRAs across more than one axis at once. An `ExperimentMatrix`
(`argus_proof.experiment`) declares factors × levels and expands to a cell per
`base_checkpoint × step_config`, each a full grid (LoRA checkpoint × weight ×
prompt × seed). Cost is aggregated across cells and estimated **before launch**,
with a `--max-gpu-hours` guardrail that refuses an intractable matrix:

```bash
argus-proof experiment matrix.json --export ./curated_export --max-gpu-hours 40
# experiment exp: 4 cells, 32 runs, 96 images
#   [exp-c00-sdxl-a-fast] 24 images (8 runs)
#   ...
# est. 0.2 GPU-hours @ 6s/image
```

```python
from argus_proof.experiment import ExperimentMatrix, StepConfig, expand_experiment
from argus_proof.grid import read_export_prompts
from argus_proof.models import SamplingParams

matrix = ExperimentMatrix(
    base_checkpoints=["sdxl_a.safetensors", "sdxl_b.safetensors"],
    step_configs=[StepConfig(name="quality", sampling=SamplingParams(...))],
    lora_checkpoints=["e10.safetensors", "e20.safetensors"],  # epoch sweep
    lora_weights=[0.8, 1.0],
    seeds=[1, 2, 3],
    labels={"caption_strategy": "florence"},  # upstream factor, carried for cross-run slicing
)
plan = expand_experiment(matrix, read_export_prompts(export_dir), max_gpu_hours=40)
for cell in plan.cells:
    ...  # generate + score each cell.plan; cell.labels feed the cross-run store
```

**Upstream factors** (caption strategy, source-image variation) are trained *into*
a LoRA, so proof can't vary them — it lists them in `labels`, which ride on every
cell. Feed a cell's `step_config`/`labels` into `run_stats(...)` and the cross-run
store compares the arms directly (`slice_pass_rate("step_config")`,
`slice_pass_rate("label:caption_strategy")`). For a matrix too large to
brute-force, `optuna_search()` (optional `[opt]` extra) does sample-efficient
search over the same factor levels.

## FiftyOne exploration (optional)

A power-user surface over a scored run, complementing the `/proof` HITL view.
`argus_proof.explore` turns an `EvalReport` into a [FiftyOne](https://docs.voxel51.com)
dataset — every computed field attached to its image — so you can visualise
embeddings (UMAP/t-SNE) to spot mode collapse / clusters / outliers, run the
uniqueness/near-dup brain, and triage by tag (`pip install "argus-proof[fiftyone]"`):

```bash
# open the App, then fold the tags you added back into a new report on close
argus-proof explore eval_report.json --images ./run-1/images --umap \
  --ingest reviewed.json --rater alice
```

```python
from argus_proof.explore import to_fiftyone_dataset, compute_visualization, ingest_from_dataset

ds = to_fiftyone_dataset(report, {"img-1": "run-1/images/img-1.png", ...})
compute_visualization(ds)                       # UMAP embedding viz (needs umap-learn)
report = ingest_from_dataset(ds, report)         # round-trip: fold tags back as ratings/reasons
```

The **round-trip** is tag-driven: in the App you add `rating:<1-5>` / `reject:<code>`
tags (these are the *input* channel — exported samples carry the scores as fields
and only a `verdict`/`refined` display tag, so a round-trip never re-ingests the
run's own auto-computed rejects). Ingest is **authoritative** — an image's tags are
its full decision, so a `rating:5` with no `reject:` tag *un-rejects* it — and folds
through the same `apply_hitl` path a review uses (the verdict recomputes identically;
pass the original `gate` to keep non-default thresholds). The mapping
(`sample_fields`/`sample_tags`/`ingest_tags`) is dependency-free and unit-tested;
only the dataset/brain/App adapters need the extra, and `explore.is_available()`
guards them.

## Develop

```bash
make install   # venv + editable install with the "dev,server,cli" extras
make test
make lint
make pages     # the hosted demo's static export (web/, needs Node 22)
```

## CI / Release

- **CI** runs via the shared [`argus-ci`](https://github.com/smk762/argus-ci) reusable workflow.
- **Release** publishes to PyPI (OIDC trusted publishing) and GHCR on `v*` tags: the thin image and the `-score`
  variant. The `-score` job runs `scripts/check_scorers.py` inside the image before pushing it. CI has no GPU, so
  that proves the scorers import, not that they run on CUDA.
- CI's `score-imports` job installs the `[score]` extra (CPU torch) and runs the same check, so a pin that resolves but
  won't import fails CI instead of quietly dropping a metric.
- Once PyPI has the version, a **GitHub Release** is created with that version's
  `CHANGELOG.md` section as its notes. Add the section before you tag; the job fails without it.
- Versioning is derived from git tags via `hatch-vcs` — tag `vX.Y.Z` to cut a release.

This repo was scaffolded from [`argus-pkg-template`](https://github.com/smk762/argus-pkg-template).
Run `copier update` to pull template changes (CI, release, tooling).
