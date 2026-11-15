# `argus-proof`

Post-training LoRA evaluation: generate samples and score them against the curated dataset.

**Usage**:

```console
$ argus-proof [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `-v, --verbose`: Show info/debug logs
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `inspect`: Summarise a proof run dir or a curator...
* `run`: Generate a sample grid from a trained LoRA...
* `score`: Score a generated run into an EvalReport:...
* `report`: Show a stored EvalReport (or list all of...
* `gate`: Gate a scored EvalReport against...
* `recommend`: Suggest routed fixes (which suite stage to...
* `experiment`: Expand an A/B experiment matrix and report...
* `explore`: Open a scored run in FiftyOne with metrics...
* `schema`: Emit the wire-contract JSON Schema...
* `serve`: Start the argus-proof micro-server...

## `argus-proof inspect`

Summarise a proof run dir or a curator export dir.

**Usage**:

```console
$ argus-proof inspect [OPTIONS] {path}
```

**Arguments**:

* `path <path>`: A proof run dir (manifest.json) or a curator export dir (manifest.jsonl)  [required]

**Options**:

* `--help`: Show this message and exit.

## `argus-proof run`

Generate a sample grid from a trained LoRA via the configured backend.

**Usage**:

```console
$ argus-proof run [OPTIONS] {lora} {export}
```

**Arguments**:

* `lora <str>`: Trained LoRA, named as the engine loads it (e.g. subject.safetensors)  [required]
* `export <path>`: Curator export dir the LoRA was trained from (prompt source)  [required]

**Options**:

* `-c, --checkpoint <str>`: Base checkpoint, named as the engine loads it  [required]
* `-o, --out <path>`: Runs root (default $ARGUS_PROOF_RUNS_DIR or ./runs)
* `-w, --weight <list[float]>`: LoRA weight(s) to sweep (repeatable)  [default: 1.0]
* `-s, --seed <list[int]>`: Control seed-set, one image per seed (repeatable)  [default: 1, 2, 3]
* `--prompt <str>`: Explicit prompt (skips the export&#x27;s captions)
* `--negative <str>`: Negative prompt
* `--max-prompts <int range>`: Base prompts to take from the export  [default: 1; x&gt;=1]
* `--steps <int range>`: [default: 25; x&gt;=1]
* `--cfg <float>`: CFG scale (default: the checkpoint family&#x27;s — SDXL 7, Flux 1)
* `--sampler <str>`: Sampler (default: SDXL dpmpp_2m, Flux euler)
* `--scheduler <str>`: Scheduler (default: SDXL karras, Flux simple)
* `--guidance <float range>`: Flux guidance, the template&#x27;s $guidance slot (default: Flux 3.5)  [x&gt;=0]
* `--no-guidance`: Send no guidance, even for a Flux checkpoint (a template with no $guidance slot)
* `--width <int range>`: [default: 1024; x&gt;=64]
* `--height <int range>`: [default: 1024; x&gt;=64]
* `--clip-skip <int>`: [default: 1]
* `--vae <str>`: Explicit VAE (template needs a $vae slot)
* `--backend <str>`: Override $PROOF_BACKEND (comfyui/diffusers/a1111/remote)
* `--run-prefix <str>`: run_id prefix for the generated runs  [default: proof]
* `--help`: Show this message and exit.

## `argus-proof score`

Score a generated run into an EvalReport: identity, quality, diversity, safety.

**Usage**:

```console
$ argus-proof score [OPTIONS] {run_dir}
```

**Arguments**:

* `run_dir <path>`: Proof run dir containing manifest.json + generated samples  [required]

**Options**:

* `--references <path>`: Held-out reference image dir for identity scoring (must not overlap training)
* `--reports-dir <path>`: Report store (default $ARGUS_PROOF_REPORTS_DIR)
* `--save / --no-save`: Persist the report into the report store  [default: save]
* `--json`: Print the full EvalReport JSON to stdout
* `--help`: Show this message and exit.

## `argus-proof report`

Show a stored EvalReport (or list all of them) with its pass/fail verdict.

**Usage**:

```console
$ argus-proof report [OPTIONS] [run_id]
```

**Arguments**:

* `run_id <str>`: Run to show; omit to list every stored report

**Options**:

* `--reports-dir <path>`: Report store (default $ARGUS_PROOF_REPORTS_DIR)
* `--json`: Print JSON instead of the human digest
* `--help`: Show this message and exit.

## `argus-proof gate`

Gate a scored EvalReport against acceptance thresholds; exit non-zero if rejected (for CI).

**Usage**:

```console
$ argus-proof gate [OPTIONS] {report_json}
```

**Arguments**:

* `report_json <path>`: Scored EvalReport JSON to gate  [required]

**Options**:

* `--min-pass-rate <float>`: Minimum aggregate pass-rate  [default: 0.75]
* `--min-pass-rate-ci-lower <float>`: Minimum Wilson lower bound on the pass-rate
* `--min-identity <float>`: Minimum mean identity score
* `--max-unsafe-rate <float>`: Maximum fraction of images flagged unsafe
* `--confidence <float>`: Confidence level for the CI lower bound  [default: 0.95]
* `--help`: Show this message and exit.

## `argus-proof recommend`

Suggest routed fixes (which suite stage to act on) from a scored EvalReport.

**Usage**:

```console
$ argus-proof recommend [OPTIONS] {report_json}
```

**Arguments**:

* `report_json <path>`: Scored EvalReport JSON to analyse  [required]

**Options**:

* `--store <path>`: CrossRunStore parquet: also recommend the best checkpoint / LoRA weight (needs )
* `--help`: Show this message and exit.

## `argus-proof experiment`

Expand an A/B experiment matrix and report the up-front per-cell cost estimate.

**Usage**:

```console
$ argus-proof experiment [OPTIONS] {matrix_json}
```

**Arguments**:

* `matrix_json <path>`: ExperimentMatrix JSON (factors × levels)  [required]

**Options**:

* `--export <path>`: Curator export dir to source base prompts from  [required]
* `--max-gpu-hours <float>`: Refuse to expand if the matrix exceeds this GPU-hour budget
* `--help`: Show this message and exit.

## `argus-proof explore`

Open a scored run in FiftyOne with metrics attached, for embedding viz + tag triage.

**Usage**:

```console
$ argus-proof explore [OPTIONS] {report_json}
```

**Arguments**:

* `report_json <path>`: Scored EvalReport JSON to open in FiftyOne  [required]

**Options**:

* `--images <path>`: Dir of generated images named &lt;image_id&gt;.&lt;ext&gt;  [required]
* `--name <str>`: FiftyOne dataset name (default: auto)
* `--umap`: Compute a UMAP embedding visualisation (needs umap-learn)
* `--launch / --no-launch`: Open the FiftyOne App (blocks until closed)  [default: launch]
* `--ingest <path>`: After the App closes, fold rating:/reject: tags back and write the report here
* `--rater <str>`: Rater id stamped on ingested tags
* `--help`: Show this message and exit.

## `argus-proof schema`

Emit the wire-contract JSON Schema consumers codegen against.

**Usage**:

```console
$ argus-proof schema [OPTIONS]
```

**Options**:

* `-o, --output <path>`: Where to write the JSON Schema  [default: schema/proof-wire.schema.json]
* `--check`: Exit non-zero if the committed schema is stale (for CI)
* `--help`: Show this message and exit.

## `argus-proof serve`

Start the argus-proof micro-server (FastAPI) on :8104.

**Usage**:

```console
$ argus-proof serve [OPTIONS]
```

**Options**:

* `-p, --port <int>`: Port to listen on  [default: 8104]
* `--host <str>`: Host to bind to  [default: 0.0.0.0]
* `--cors`: Enable CORS (allow all origins)
* `--read-only / --no-read-only`: Replay/demo mode: serve stored reports but 403 all live eval + writes (default: $ARGUS_PROOF_READ_ONLY).
* `--devmode / --no-devmode`: Register the /purge routes that delete runs and their reports (default: $ARGUS_PROOF_DEVMODE).
* `--help`: Show this message and exit.
