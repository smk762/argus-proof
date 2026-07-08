"""Render a parametric ComfyUI workflow template for one seed of a RunSpec.

A template is a ComfyUI **API-format** graph — ``{node_id: {"class_type": str,
"inputs": {name: value}}}`` — with placeholder *string* values that this module
substitutes with typed params from a :class:`~argus_proof.models.RunSpec`. Only
values that are exactly a placeholder are replaced (no substring interpolation),
so injected ints/floats stay typed and node-link lists (``[id, slot]``) are left
alone.

Placeholders:

* scalars — ``$base_checkpoint`` ``$vae`` ``$positive`` ``$negative`` ``$seed``
  ``$steps`` ``$cfg`` ``$sampler`` ``$scheduler`` ``$width`` ``$height``
  ``$clip_skip`` (emitted as ComfyUI's negative ``stop_at_clip_layer``);
* LoRAs — ``$lora`` / ``$lora_weight`` for the first LoRA, ``$lora_2`` /
  ``$lora_2_weight`` for the second, and so on. The template's LoRA slots and
  the spec's LoRA list must match exactly, so a run can't silently drop a LoRA.
"""

from __future__ import annotations

import copy
import importlib.resources
import json
import re
from pathlib import Path

from argus_proof.backends.base import BackendError
from argus_proof.models import RunSpec

_LORA_RE = re.compile(r"^\$lora(?:_(\d+))?(_weight)?$")

# The example SDXL+LoRA template shipped with the package (single LoRA, clip
# skip, checkpoint's built-in VAE). A starting point users adapt to their graph.
EXAMPLE_TEMPLATE = "comfyui_sdxl_lora.json"


def load_template(path: Path) -> dict:
    """Load and lightly validate a ComfyUI API-format workflow template."""
    try:
        template = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BackendError(f"cannot read workflow template {path}: {exc}") from exc
    if not isinstance(template, dict) or not template:
        raise BackendError(f"workflow template {path} is not a non-empty ComfyUI API graph")
    return template


def example_template() -> dict:
    """The packaged example SDXL+LoRA workflow template, as a graph dict."""
    resource = importlib.resources.files("argus_proof").joinpath("templates", EXAMPLE_TEMPLATE)
    return json.loads(resource.read_text(encoding="utf-8"))


def _scalar_values(spec: RunSpec, seed: int) -> dict[str, object]:
    """Placeholder -> typed value for the scalar knobs (``$vae`` may be None)."""
    s = spec.sampling
    return {
        "$base_checkpoint": spec.base_checkpoint,
        "$vae": spec.vae,  # None if the checkpoint's own VAE is used
        "$positive": spec.prompt,
        "$negative": spec.negative_prompt,
        "$seed": seed,
        "$steps": s.steps,
        "$cfg": s.cfg,
        "$sampler": s.sampler,
        "$scheduler": s.scheduler,
        "$width": s.width,
        "$height": s.height,
        "$clip_skip": -s.clip_skip,  # ComfyUI CLIPSetLastLayer wants a negative index
    }


def render_workflow(template: dict, spec: RunSpec, seed: int) -> dict:
    """Return a concrete ComfyUI graph for *spec* at *seed*.

    Raises :class:`BackendError` if a used placeholder has no value (e.g. a
    ``$vae`` slot but ``spec.vae is None``) or if the template's LoRA slots and
    ``spec.loras`` don't line up one-for-one.
    """
    scalars = _scalar_values(spec, seed)
    lora_by_index = {i: lora for i, lora in enumerate(spec.loras, start=1)}
    seen_scalars: set[str] = set()
    # Track name and weight placeholders per LoRA index SEPARATELY: a template
    # that injects $lora but hard-codes the strength (no $lora_weight) would
    # otherwise silently drop the spec's weight while the manifest records it.
    seen_lora_names: set[int] = set()
    seen_lora_weights: set[int] = set()

    def resolve(value: object) -> object:
        if not isinstance(value, str) or not value.startswith("$"):
            return value

        if value in scalars:
            resolved = scalars[value]
            if resolved is None:
                raise BackendError(f"workflow template uses {value} but the run spec has no value for it")
            seen_scalars.add(value)
            return resolved

        m = _LORA_RE.match(value)
        if m:
            index = int(m.group(1)) if m.group(1) else 1
            is_weight = bool(m.group(2))
            lora = lora_by_index.get(index)
            if lora is None:
                raise BackendError(
                    f"workflow template references {value} but the run spec has no LoRA #{index} "
                    f"({len(spec.loras)} LoRA(s) supplied)"
                )
            (seen_lora_weights if is_weight else seen_lora_names).add(index)
            return lora.weight if is_weight else lora.name

        # An unknown "$..." string: leave it untouched (it may be a literal).
        return value

    rendered = copy.deepcopy(template)
    for node in rendered.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if isinstance(inputs, dict):
            for key, val in inputs.items():
                inputs[key] = resolve(val)

    def _slot(index: int, weight: bool) -> str:
        suffix = "_weight" if weight else ""
        return f"$lora{suffix}" if index == 1 else f"$lora_{index}{suffix}"

    missing = [_slot(i, False) for i in lora_by_index if i not in seen_lora_names]
    missing += [_slot(i, True) for i in lora_by_index if i not in seen_lora_weights]
    if missing:
        raise BackendError(
            f"workflow template is missing LoRA placeholder(s) {sorted(missing)} — "
            f"the spec supplies {len(spec.loras)} LoRA(s); every LoRA needs both its name and weight slot "
            "so a LoRA (or its weight) can't be silently dropped"
        )

    # A configured VAE that the template never consumes would be silently ignored
    # yet still recorded in the manifest — refuse it rather than misrepresent the run.
    if spec.vae is not None and "$vae" not in seen_scalars:
        raise BackendError(
            "run spec sets a vae but the workflow template has no $vae slot — it would be ignored at "
            "generation yet recorded in the manifest; add a $vae slot or clear spec.vae"
        )
    return rendered
