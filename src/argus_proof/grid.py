"""Prompt-grid builder — expand a source export into a grid of runs.

Turns "evaluate this LoRA against the dataset it was trained on" into a
systematic matrix instead of ad-hoc prompting: base prompts (from the export's
captions) multiplied across the LoRA-checkpoint, LoRA-weight, and token-combo
axes, plus off-distribution flexibility prompts, every run sharing one control
seed-set. The expansion is pure and deterministic — the same config + captions
always produce the same :class:`~argus_proof.models.RunSpec`s — and the plan
carries an up-front count + GPU-hour estimate so cost is known before launch.

Base prompts are sourced from the export by :func:`read_export_prompts`: the
lens **zeroshot** caption variant when a lens captions JSON is present (the
natural generation prompt), falling back to the training ``.txt`` sidecar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from random import Random

import structlog

from argus_proof.models import (
    GridConfig,
    GridEstimate,
    GridPlan,
    LoRASpec,
    ProofError,
    RunSpec,
)

logger = structlog.get_logger()


class GridError(ProofError):
    """A grid could not be built: no prompts to generate, unreadable export."""


# Caption variants preferred as generation prompts, best first. The zeroshot
# variant is assembled by argus-lens specifically for generation without a LoRA;
# the training variant is the next best signal if zeroshot wasn't exported.
VARIANT_PREFERENCE: tuple[str, ...] = ("zeroshot", "training")


# ---------------------------------------------------------------------------
# prompt sourcing
# ---------------------------------------------------------------------------


def read_export_prompts(export_dir: Path) -> list[str]:
    """Base prompts for *export_dir*: zeroshot captions if present, else ``.txt``.

    Prefers a lens captions JSON (whose entries carry ``caption_variants``);
    falls back to the ``.txt`` sidecars (the training caption) inside the
    export, then to the sidecars next to the dataset images the export's
    ``manifest.jsonl`` references by ``abs_path`` — lens writes training
    captions beside the *source* images, so a manifest-only export still
    yields prompts when the dataset volume is mounted at the recorded path.
    Returns a de-duplicated, order-stable list.
    """
    prompts = _prompts_from_captions_json(export_dir)
    if prompts:
        logger.debug("grid.prompts", source="captions_json", count=len(prompts))
        return prompts
    prompts = _prompts_from_txt_sidecars(export_dir)
    if prompts:
        logger.debug("grid.prompts", source="txt_sidecars", count=len(prompts))
        return prompts
    prompts = _prompts_from_manifest_rows(export_dir)
    logger.debug("grid.prompts", source="manifest_abs_path_sidecars", count=len(prompts))
    return prompts


def _prompts_from_captions_json(export_dir: Path) -> list[str]:
    for path in sorted(export_dir.rglob("*.json")) + sorted(export_dir.rglob("*.jsonl")):
        if _is_hidden(path, export_dir):
            continue
        entries = _load_lens_entries(path)
        picked = [p for e in entries if (p := _pick_variant(e))]
        if picked:
            return _dedup(picked)
    return []


def _load_lens_entries(path: Path) -> list[dict]:
    """Parse a lens captions export (``.json`` array or ``.jsonl``) to entries.

    Returns ``[]`` for anything that isn't a list of caption dicts, so an
    unrelated JSON file in the export is skipped rather than misread.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []  # unreadable or non-UTF-8 file in the export -> skip, don't crash
    entries: list[dict] = []
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue  # skip a malformed line, keep the rest of the file's captions
            if isinstance(obj, dict):
                entries.append(obj)
    else:
        try:
            data = json.loads(text)
        except ValueError:
            return []
        if not isinstance(data, list):
            return []
        entries = [e for e in data if isinstance(e, dict)]
    # Only treat it as a captions export if entries actually look like captions.
    if not any("caption_variants" in e or "final_caption" in e for e in entries):
        return []
    return entries


def _pick_variant(entry: dict) -> str | None:
    variants = entry.get("caption_variants") or {}
    if isinstance(variants, dict):
        for key in VARIANT_PREFERENCE:
            value = variants.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    final = entry.get("final_caption")
    return final.strip() if isinstance(final, str) and final.strip() else None


def _prompts_from_txt_sidecars(export_dir: Path) -> list[str]:
    prompts: list[str] = []
    for path in sorted(export_dir.rglob("*.txt")):
        if _is_hidden(path, export_dir):
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue  # skip an unreadable or non-UTF-8 sidecar rather than crash
        if text:
            prompts.append(text)
    return _dedup(prompts)


def _prompts_from_manifest_rows(export_dir: Path) -> list[str]:
    """Training-caption sidecars next to the dataset images a curator
    ``manifest.jsonl`` references (``<abs_path stem>.txt``)."""
    manifest = export_dir / "manifest.jsonl"
    if not manifest.is_file():
        return []
    prompts: list[str] = []
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        abs_path = row.get("abs_path") if isinstance(row, dict) else None
        if not isinstance(abs_path, str) or not abs_path:
            continue
        sidecar = Path(abs_path).with_suffix(".txt")
        try:
            text = sidecar.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue  # image not captioned (or volume not mounted) -> skip the row
        if text:
            prompts.append(text)
    return _dedup(prompts)


def _is_hidden(path: Path, root: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def _dedup(items: list[str]) -> list[str]:
    """De-duplicate, preserving first-seen order (stable across runs)."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# grid expansion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PromptItem:
    text: str
    flexibility: bool


def _token_combos(config: GridConfig) -> list[str]:
    """The token combos to append to base prompts (deterministic, capped).

    Full cartesian product of the (non-empty) token axes, joined per combo; if
    that exceeds ``max_token_combos``, a deterministic sample keyed by
    ``combo_seed`` is taken so token effects stay attributable across re-runs.
    """
    axes = {k: v for k, v in config.token_axes.items() if v}
    if not axes:
        return []
    keys = sorted(axes)  # sorted so combo strings are ordered deterministically
    combos = [", ".join(values) for values in product(*(axes[k] for k in keys))]
    if config.max_token_combos is not None and len(combos) > config.max_token_combos:
        sampled = Random(config.combo_seed).sample(combos, config.max_token_combos)
        combos = sorted(sampled)
    return combos


def _expand_prompts(base_prompts: list[str], config: GridConfig) -> list[_PromptItem]:
    base = _dedup([p.strip() for p in base_prompts if p.strip()])
    if config.max_base_prompts is not None:
        base = base[: config.max_base_prompts]

    combos = _token_combos(config)
    items: list[_PromptItem] = []
    for prompt in base:
        if combos:
            items.extend(_PromptItem(f"{prompt}, {combo}", False) for combo in combos)
        else:
            items.append(_PromptItem(prompt, False))

    items.extend(_PromptItem(p.strip(), True) for p in config.flexibility_prompts if p.strip())
    return items


def count_prompt_items(config: GridConfig, base_prompts: list[str]) -> int:
    """How many prompt items :func:`build_grid` would expand to — computed
    **arithmetically**, without materializing the token cartesian.

    For a pre-flight cost estimate (e.g. :mod:`argus_proof.experiment`'s budget
    guardrail) that must not build the very expansion it is trying to size up.
    Kept in lock-step with :func:`_expand_prompts` (a test asserts they agree).
    """
    base = _dedup([p.strip() for p in base_prompts if p.strip()])
    if config.max_base_prompts is not None:
        base = base[: config.max_base_prompts]

    axes = {k: v for k, v in config.token_axes.items() if v}
    n_combos = 1
    for values in axes.values():
        n_combos *= len(values)
    if axes and config.max_token_combos is not None:
        n_combos = min(n_combos, config.max_token_combos)

    n_flexibility = sum(1 for p in config.flexibility_prompts if p.strip())
    return len(base) * n_combos + n_flexibility


def _estimate(config: GridConfig, n_runs: int, n_prompts: int) -> GridEstimate:
    n_images = n_runs * len(config.seeds)
    gpu_seconds = n_images * config.seconds_per_image
    return GridEstimate(
        n_runs=n_runs,
        n_images=n_images,
        seconds_per_image=config.seconds_per_image,
        est_gpu_seconds=gpu_seconds,
        est_gpu_hours=gpu_seconds / 3600.0,
        axes={
            "lora_checkpoints": len(config.lora_checkpoints),
            "weights": len(config.lora_weights),
            "prompts": n_prompts,
            "seeds": len(config.seeds),
        },
    )


def build_grid(config: GridConfig, base_prompts: list[str]) -> GridPlan:
    """Expand *config* over *base_prompts* into a deterministic :class:`GridPlan`.

    One :class:`RunSpec` per (LoRA checkpoint × weight × prompt) combination,
    each with the shared control seed-set. ``run_id`` encodes the axis indices
    (``l``/``w`` + ``p`` for base prompts, ``f`` for flexibility prompts) so it
    is unique, stable, and traceable back to the grid. Raises :class:`GridError`
    if the expansion yields no prompts.
    """
    items = _expand_prompts(base_prompts, config)
    if not items:
        raise GridError("no prompts to generate — the export had no captions and no flexibility_prompts were given")

    specs: list[RunSpec] = []
    for li, checkpoint in enumerate(config.lora_checkpoints):
        for wi, weight in enumerate(config.lora_weights):
            for pi, item in enumerate(items):
                marker = "f" if item.flexibility else "p"
                run_id = f"{config.run_id_prefix}-l{li:02d}-w{wi:02d}-{marker}{pi:03d}"
                specs.append(
                    RunSpec(
                        run_id=run_id,
                        base_checkpoint=config.base_checkpoint,
                        loras=[LoRASpec(name=checkpoint, weight=weight)],
                        sampling=config.sampling,
                        prompt=item.text,
                        negative_prompt=config.negative_prompt,
                        seeds=list(config.seeds),
                        source_manifest=config.source_manifest,
                        source_manifest_version=config.source_manifest_version,
                        training_run_id=config.training_run_id,
                    )
                )

    estimate = _estimate(config, n_runs=len(specs), n_prompts=len(items))
    logger.info("grid.built", n_runs=estimate.n_runs, n_images=estimate.n_images, est_gpu_hours=estimate.est_gpu_hours)
    return GridPlan(run_id_prefix=config.run_id_prefix, estimate=estimate, specs=specs)


def plan_from_export(export_dir: Path, config: GridConfig) -> GridPlan:
    """Read *export_dir*'s captions and expand *config* into a :class:`GridPlan`."""
    return build_grid(config, read_export_prompts(export_dir))
