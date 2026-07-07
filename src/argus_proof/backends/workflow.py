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
    seen_lora_indices: set[int] = set()

    def resolve(value: object) -> object:
        if not isinstance(value, str) or not value.startswith("$"):
            return value

        if value in scalars:
            resolved = scalars[value]
            if resolved is None:
                raise BackendError(f"workflow template uses {value} but the run spec has no value for it")
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
            seen_lora_indices.add(index)
            return lora.weight if is_weight else lora.name

        # An unknown "$..." string: leave it untouched (it may be a literal).
        return value

    rendered = copy.deepcopy(template)
    for node in rendered.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if isinstance(inputs, dict):
            for key, val in inputs.items():
                inputs[key] = resolve(val)

    missing = set(lora_by_index) - seen_lora_indices
    if missing:
        raise BackendError(
            f"workflow template has no slot for LoRA(s) {sorted(missing)} — "
            f"the spec supplies {len(spec.loras)} LoRA(s); add $lora_N placeholders or trim the spec"
        )
    return rendered
