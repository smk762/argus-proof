# argus-proof

Post-training LoRA evaluation and optimisation: generate samples from a trained LoRA and score them against the curated dataset it was trained from.

Part of the [Argus suite](https://github.com/smk762?tab=repositories&q=argus) — the stage that closes the loop after
[argus-forge](https://github.com/smk762/argus-forge) emits a training config and you train a LoRA:

```
argus-quarry -> argus-curator -> argus-lens -> argus-forge -> your trainer -> argus-proof
  acquire         curate/export     caption       configs        LoRA           evaluate + optimise
```

> **Status: Phase 0 scaffold.** The service boots (`GET /health` on :8104) and the CLI exposes the eventual
> verbs (`inspect`, `run`, `score`, `report`) as stubs. Roadmap:
> [argus-proof epic](https://github.com/smk762/argus-studio/issues/6) (phases 0–7).

## Install

```bash
uv pip install "argus-proof[cli]"          # CLI
uv pip install "argus-proof[cli,server]"   # + HTTP server for argus-studio
```

## Serve

```bash
argus-proof serve --port 8104 --cors   # peer to lens :8100, curator :8101, quarry :8102, forge :8103
curl -s localhost:8104/health          # {"status":"ok","service":"argus-proof","version":"..."}
```

## Generation backend (Phase 1)

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

## Scoring (Phase 2)

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
