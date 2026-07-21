# argus-proof

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

## Install

```bash
uv pip install "argus-proof[cli]"          # CLI
uv pip install "argus-proof[cli,server]"   # + HTTP server for argus-studio
uv pip install "argus-proof[cli,score]"    # + the full scorer stack (torch, insightface, …)
```

## CLI

```bash
# Generate a sample grid from a trained LoRA (prompts come from the curator
# export's captions; backend/engine/model dirs from the environment, see
# .env.example — PROOF_BACKEND, COMFYUI_BASE_URL, PROOF_MODELS_DIR):
argus-proof run subject.safetensors ./curated_export \
  --checkpoint sdxl_base.safetensors --seed 1 --seed 2 --seed 3 --out runs

# Score a generated run into a stored EvalReport (identity needs --references,
# a held-out image dir that must NOT overlap the training set):
argus-proof score runs/proof-l00-w00-p000 --references ./holdout

# Browse stored reports / print one run's digest:
argus-proof report
argus-proof report proof-l00-w00-p000 [--json]

# Summarise an export dir (prompt sources) or a run dir (manifest + images):
argus-proof inspect ./curated_export
argus-proof inspect runs/proof-l00-w00-p000
```

## Serve

```bash
argus-proof serve --port 8104 --cors   # peer to lens :8100, curator :8101, quarry :8102, forge :8103
argus-proof serve --read-only          # replay/demo mode: serve stored reports, 403 live eval + writes
curl -s localhost:8104/health          # {"status":"ok","service":"argus-proof","version":"...","read_only":false}
```

Routes (the backend of the argus-studio `/proof` view):

```
GET  /exports                           curator export dirs available to evaluate against
GET  /models                            checkpoints + LoRAs under $PROOF_MODELS_DIR
POST /run/stream                        generate + score + store one run (NDJSON progress)
GET  /reports                           stored report digests (run browser)
GET  /report/{run_id}                   full EvalReport            PUT to store one
GET  /report/{run_id}/refined           passing subset, refined ranks first
GET  /report/{run_id}/image/{image_id}  a generated sample (ids only — no path input)
POST /report/{run_id}/hitl              apply a review; recomputes pass-rate + verdict
POST /report/{run_id}/refine            second-pass re-rank (rank: null retracts)
```

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

**More backends.** `get_backend(name, ...)` selects the engine by config
(`PROOF_BACKEND` in [`.env.example`](.env.example)); scoring/report code is
unchanged regardless of which produced the run, and the `RunManifest` records the
engine + version:

- **`diffusers`** — in-process diffusers SDXL pipeline: deterministic, no external
  service, weights hashed from disk (`pip install "argus-proof[diffusers]"`).
- **`a1111`** — an AUTOMATIC1111 / SD.Next `/sdapi` server (checkpoint via
  `override_settings`, LoRAs via `<lora:…>` prompt syntax; models hashed from disk).
- **`remote`** — a hosted/cloud endpoint that speaks the proof wire (`POST /generate`
  → `RunManifest` + base64 images). The weights live remotely, so the **service**
  supplies the manifest and it's validated at the boundary; a bearer `api_key`
  authenticates. Point it at a self-hosted proof-gen service or a thin wrapper in
  front of Replicate / fal.

The `a1111` / `remote` adapters need no extra (stdlib HTTP); all three reuse the
shared manifest + transport helpers and are unit-tested with fakes.

## Scoring

Generated images are scored into an `EvalReport` by a pluggable framework
(`argus_proof.scoring`). Per-image `ImageScorer`s each fill one normalised
`[0,1]` metric (identity / clip_score / aesthetic / preference / safety); a
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
- **identity** — `IdentityScorer` (InsightFace ArcFace cosine vs a held-out reference set)
- **quality / adherence** — `clip_score_scorer()` (CLIPScore), `pyiqa_scorer()` (CLIP-IQA), `image_reward_scorer()` (ImageReward), each normalising its raw score to `[0,1]`
- **safety** — `SafetyScorer` (NudeNet ensemble, `1 - unsafe`); set a `safety` hard gate to auto-fail unsafe images. `safety_tail_aggregate()` surfaces the any-hit/min/percentile tail

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
it to a **Guard-class policy taxonomy** (violence / hate / self-harm / weapons /
illegal) — over both the **generated images** and the **input prompts /
captions**, so a toxic prompt is flagged even when its output is clean
(`pip install "argus-proof[moderation]"`, Llama Guard 3 Vision):

```python
from argus_proof.moderation import PolicyModerator, moderate_images, moderate_texts

mod = PolicyModerator()                         # default: Llama Guard 3 (lazy, [moderation])
out = moderate_images(image_paths, mod)         # per-category tails over the outputs
inp = moderate_texts(prompt_grid_variants, mod) # ...and over the inputs
out.flagged()                                   # e.g. ["violence", "hate"], worst first
report.scorers.append(mod.provenance("output")) # version-stamp the Guard model + taxonomy
```

Each category gets a **tail** view (any-hit / max / 95th percentile — the extremes
that matter, not the mean, same rule as `safety`), combined conservatively across
an ensemble (most-unsafe detector wins). Detectors are pluggable and injectable, so
the taxonomy/ensemble/tail logic is dependency-free and unit-tested; only the real
Llama Guard adapter needs the extra. A reviewer's HITL flag attributes to a
category via `RejectReason.category`. CSAM matching stays a **separate** policy gate
(Thorn Safer / PhotoDNA), not an ML metric here.

## CI acceptance gate

Turn "was this LoRA/dataset good enough?" into an automatable yes/no. `argus-proof
gate` evaluates a scored `EvalReport` against declared thresholds and **exits
non-zero when rejected**, so it drops straight into CI:

```bash
argus-proof gate eval_report.json \
  --min-pass-rate 0.75 \
  --min-pass-rate-ci-lower 0.7 \   # Wilson lower bound — a lucky 3/3 won't pass
  --min-identity 0.6 \
  --max-unsafe-rate 0.0            # exit 0 = accepted, 1 = rejected, 2 = unreadable
```

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

Safety first, then: low identity/aesthetic → **forge** (training), low adherence →
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
#   [exp-c00sdxl-a-fast] 24 images (8 runs)
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
```

## CI / Release

- **CI** runs via the shared [`argus-ci`](https://github.com/smk762/argus-ci) reusable workflow.
- **Release** publishes to PyPI (OIDC trusted publishing) and GHCR on `v*` tags.
- Versioning is derived from git tags via `hatch-vcs` — tag `vX.Y.Z` to cut a release.

This repo was scaffolded from [`argus-pkg-template`](https://github.com/smk762/argus-pkg-template).
Run `copier update` to pull template changes (CI, release, tooling).
