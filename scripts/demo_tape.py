"""Build the hosted demo's tape (web/public/tape/) from a recorded argus-proof evaluation.

proof.dragonhound.dev replays one recorded run: a synthetic-identity SDXL LoRA, a checkpoint every 320 steps to
1600, each scored by argus-proof against held-out references, plus a no-LoRA baseline. The tape is the reports'
numbers and verdicts, the prompts, and WebP copies of the images. Its rules:

- Runs map to (variant, prompt) by the manifest's prompt text, never the run id (some runs were retried under
  other prefixes).
- Every image the evaluation scored is on the tape, with proof's own metrics and verdict: nothing is dropped,
  re-rolled or re-scored. ``preference`` never ran, so it is left out rather than shown as a number.
- No absolute or container path, private address or email address: manifests carry ``/data/...`` paths, so
  only whitelisted fields are copied, and anything that still looks like a path fails the build.
- Images are re-encoded to WebP (a tile and a lightbox size) with no metadata: ComfyUI's PNG text chunks (the
  whole workflow) are not carried over.

    python scripts/demo_tape.py <recording-dir> [--out web/public/tape]

<recording-dir> holds eval/{runs,reports,prompts8.json,token-probe} and dataset-v3/{train,refs}.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

FORMAT = "argus-proof-tape"
VERSION = 1

TILE = 512
FULL = 1024
PROBE = 384
QUALITY_TILE = 80
QUALITY_FULL = 82

METRICS = ("identity", "clip_score", "aesthetic", "safety")
STEPS = (320, 640, 960, 1280, 1600)
VARIANTS = ("baseline", *(str(s) for s in STEPS))

LEAKS = [
    ("home directory path", re.compile(r"(?:/home/|/Users/|/root/|~/)[\w.-]+")),
    ("container or server path", re.compile(r"/(?:data|srv|mnt|media|models|var/www|tmp)/[\w.-]+")),
    ("private IPv4 address", re.compile(r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b")),
    ("email address", re.compile(r"\b[\w.%+-]+@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}\b", re.I)),
]

#: The kohya sd-scripts settings the page shows (argus-forge's config). Paths are never copied.
TRAINING_KEYS = (
    "max_train_steps",
    "save_every_n_steps",
    "train_batch_size",
    "network_dim",
    "network_alpha",
    "learning_rate",
    "text_encoder_lr",
    "optimizer_type",
    "lr_scheduler",
    "lr_warmup_steps",
    "min_snr_gamma",
    "mixed_precision",
    "seed",
)


class TapeError(ValueError):
    """The recording cannot become a tape (a rule above would be broken)."""


def leaks(text: str) -> list[str]:
    """Every path, private address or email address left in ``text``."""
    return [f"{kind}: {m.group(0)}" for kind, pattern in LEAKS for m in pattern.finditer(text)]


def variant_of(run_id: str) -> str:
    """``baseline`` for a weight-0 run, else the checkpoint's step as a string (``synth-s00000320-...`` -> ``320``)."""
    if run_id.startswith("baseline"):
        return "baseline"
    m = re.match(r"synth-s0*(\d+)-", run_id)
    if not m or int(m.group(1)) not in STEPS:
        raise TapeError(f"run {run_id!r} is neither the baseline nor a known checkpoint")
    return m.group(1)


def basename(name: str) -> str:
    """A model file's name without its folder (``sdxl/proof/x.safetensors`` -> ``x.safetensors``)."""
    return name.rsplit("/", 1)[-1]


def image_entry(image: dict, variant: str, prompt: int) -> dict:
    """One scored image: proof's metrics (minus the never-run preference), verdict and reasons, and its files."""
    stem = f"{variant}/p{prompt}-s{image['seed']}"
    return {
        "image_id": image["image_id"],
        "seed": image["seed"],
        "metrics": {k: image["metrics"].get(k) for k in METRICS},
        "passed": image["passed"],
        "reject_reasons": list(image["reject_reasons"]),
        "duplicate_group": image.get("duplicate_group"),
        "tile": f"eval/{stem}-{TILE}.webp",
        "full": f"eval/{stem}-{FULL}.webp",
    }


def run_entry(manifest: dict, report: dict, prompts: dict[str, int]) -> tuple[str, dict]:
    """(variant, run) for one evaluation run, keyed to its prompt by the manifest's text."""
    if manifest["run_id"] != report["run_id"]:
        raise TapeError(f"manifest {manifest['run_id']} paired with report {report['run_id']}")
    if manifest["prompt"] not in prompts:
        raise TapeError(f"run {manifest['run_id']}: prompt {manifest['prompt']!r} is not one of the eight")
    variant = variant_of(manifest["run_id"])
    prompt = prompts[manifest["prompt"]]
    loras = [lo for lo in manifest["loras"] if lo["weight"]]
    if (variant == "baseline") != (not loras):
        raise TapeError(f"run {manifest['run_id']}: variant {variant} but LoRAs {manifest['loras']}")
    agg = report["aggregate"]
    run = {
        "run_id": manifest["run_id"],
        "prompt": prompt,
        "seeds": manifest["seeds"],
        "lora": {"name": basename(loras[0]["name"]), "sha256": loras[0]["sha256"], "weight": loras[0]["weight"]}
        if loras
        else None,
        "verdict": {k: report["verdict"][k] for k in ("passed", "pending", "reasons")},
        "pass_rate": agg["pass_rate"],
        "diversity": agg.get("diversity"),
        "n_groups": agg["n_groups"],
        "n_passed": agg["n_passed"],
        "n_needs_hitl": agg["n_needs_hitl"],
        "images": [image_entry(im, variant, prompt) for im in report["images"]],
    }
    return variant, run


def summarise(runs: list[dict]) -> dict:
    """Per-variant means, as scripts/eval_summary.py prints them: image means per metric, run means for pass rate."""
    images = [im for run in runs for im in run["images"]]
    means = {}
    for k in METRICS:
        values = [im["metrics"][k] for im in images if im["metrics"][k] is not None]
        means[k] = round(statistics.mean(values), 4) if values else None
    verdicts = [im["passed"] for im in images]
    return {
        "n_images": len(images),
        "means": means,
        "pass_rate": round(statistics.mean(r["pass_rate"] for r in runs), 4),
        "diversity": round(statistics.mean(r["diversity"] for r in runs if r["diversity"] is not None), 4),
        "passed": verdicts.count(True),
        "review": verdicts.count(None),
        "failed": verdicts.count(False),
        "runs_passed": sum(1 for r in runs if r["verdict"]["passed"]),
    }


def build(runs_by_variant: dict[str, list[dict]], prompts: list[str], extra: dict) -> dict:
    """The tape: one entry per variant (baseline first), each with all eight prompts' runs in prompt order."""
    variants = []
    for v in VARIANTS:
        runs = sorted(runs_by_variant.get(v, []), key=lambda r: r["prompt"])
        if [r["prompt"] for r in runs] != list(range(len(prompts))):
            raise TapeError(f"variant {v}: prompts {[r['prompt'] for r in runs]}, want one run for each of 0..7")
        variants.append(
            {
                "id": v,
                "label": "No LoRA" if v == "baseline" else f"Step {v}",
                "step": None if v == "baseline" else int(v),
                "lora": runs[0]["lora"],
                "summary": summarise(runs),
                "runs": runs,
            }
        )
    tape = {"format": FORMAT, "version": VERSION, "prompts": prompts, "variants": variants, **extra}
    found = leaks(json.dumps(tape))
    if found:
        raise TapeError("the tape would publish " + "; ".join(found))
    return tape


# ── I/O ──────────────────────────────────────────────────────────────────────────────────────────────────────────


def webp(src: Path, dest: Path, size: int, quality: int) -> dict:
    """Re-encode ``src`` to fit ``size``x``size`` as WebP, with no EXIF, XMP, ICC or text chunks."""
    from PIL import Image

    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        im.save(dest, "WEBP", quality=quality, method=6)
        return {"w": im.width, "h": im.height}


def load_runs(root: Path, prompts: dict[str, int]) -> dict[str, list[dict]]:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for mpath in sorted((root / "eval" / "runs").glob("*/manifest.json")):
        manifest = json.loads(mpath.read_text())
        report = json.loads((root / "eval" / "reports" / f"{manifest['run_id']}.json").read_text())
        variant, run = run_entry(manifest, report, prompts)
        if any(r["prompt"] == run["prompt"] for r in by_variant[variant]):
            raise TapeError(f"variant {variant} prompt {run['prompt']}: two runs")
        by_variant[variant].append(run)
    return by_variant


def recording(root: Path) -> dict:
    """What the evaluation ran with, from one manifest and one report (identical across runs; checked)."""
    manifests = [json.loads(p.read_text()) for p in sorted((root / "eval" / "runs").glob("*/manifest.json"))]
    first = manifests[0]
    for m in manifests:
        if m["sampling"] != first["sampling"] or m["base_checkpoint"] != first["base_checkpoint"]:
            raise TapeError(f"run {m['run_id']} sampled differently from {first['run_id']}")
    report = json.loads(next((root / "eval" / "reports").glob("*.json")).read_text())
    return {
        "proof_version": "0.4.0",
        "wire_version": first["proof_version"],
        "engine": f"{first['engine']} {first['engine_version']}",
        "base_checkpoint": {
            "name": basename(first["base_checkpoint"]["name"]),
            "sha256": first["base_checkpoint"]["sha256"],
        },
        "sampling": {k: first["sampling"][k] for k in ("sampler", "scheduler", "steps", "cfg", "width", "height")},
        "seeds": first["seeds"],
        "pass_threshold": 0.75,
        "scorers": [{"metric": s["metric"], "name": s["name"], "model": s["model"]} for s in report["scorers"]],
    }


def training(root: Path) -> dict:
    with (root / "dataset-v3" / "train" / "forge" / "kohya" / "config.toml").open("rb") as f:
        cfg = tomllib.load(f)
    out = {k: cfg[k] for k in TRAINING_KEYS}
    out["base_checkpoint"] = basename(cfg["pretrained_model_name_or_path"])
    out["trainer"] = "kohya sd-scripts (config from argus-forge)"
    out["duration_minutes"] = 41
    out["gpu"] = "1x RTX 3090"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "web" / "public" / "tape")
    args = ap.parse_args(argv)
    root, out = args.recording, args.out

    prompt_list = [
        v
        for _, v in sorted(json.loads((root / "eval" / "prompts8.json").read_text()).items(), key=lambda kv: int(kv[0]))
    ]
    by_text = {p: i for i, p in enumerate(prompt_list)}
    runs = load_runs(root, by_text)

    if out.exists():
        shutil.rmtree(out)
    for v, vruns in runs.items():
        for run in vruns:
            for im in run["images"]:
                src = root / "eval" / "runs" / run["run_id"] / f"{im['image_id']}.png"
                webp(src, out / im["tile"], TILE, QUALITY_TILE)
                webp(src, out / im["full"], FULL, QUALITY_FULL)
        print(f"  {v}: {len(vruns)} runs", file=sys.stderr)

    refs = []
    for src in sorted((root / "dataset-v3" / "refs").glob("*.png")):
        tile = webp(src, out / "refs" / f"{src.stem}-{TILE}.webp", TILE, QUALITY_TILE)
        webp(src, out / "refs" / f"{src.stem}-{FULL}.webp", FULL, QUALITY_FULL)
        refs.append(
            {"id": src.stem, "tile": f"refs/{src.stem}-{TILE}.webp", "full": f"refs/{src.stem}-{FULL}.webp", **tile}
        )

    train = []
    for src in sorted((root / "dataset-v3" / "train").glob("*.png")):
        tile = webp(src, out / "train" / f"{src.stem}-{TILE}.webp", TILE, QUALITY_TILE)
        caption = src.with_suffix(".txt").read_text().strip()
        train.append({"id": src.stem, "caption": caption, "tile": f"train/{src.stem}-{TILE}.webp", **tile})

    # Trigger-token probe: base SDXL (no LoRA), seed 1, each prompt with "ohwx woman" and with plain "woman".
    probe = []
    for i, text in enumerate(prompt_list):
        row = {"prompt": i, "plain_prompt": text.replace("ohwx woman", "woman")}
        for key, tok in (("ohwx", "ohwx"), ("plain", "NONE")):
            src = root / "eval" / "token-probe" / f"tok-{tok}-p{i}-l00-w00-p000" / f"tok-{tok}-p{i}-l00-w00-p000-1.png"
            webp(src, out / "probe" / f"{key}-p{i}-{PROBE}.webp", PROBE, QUALITY_TILE)
            row[key] = f"probe/{key}-p{i}-{PROBE}.webp"
        probe.append(row)

    extra = {
        "recording": recording(root),
        "training": training(root),
        "refs": refs,
        "train": train,
        "token_probe": probe,
    }
    tape = build(runs, prompt_list, extra)
    (out / "tape.json").write_text(json.dumps(tape, indent=1) + "\n")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(
        f"tape: {sum(len(r) for r in runs.values())} runs, {len(refs)} refs, {len(train)} train, {total / 2**20:.1f} MiB",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
