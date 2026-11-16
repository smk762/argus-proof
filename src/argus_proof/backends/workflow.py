"""Render a parametric ComfyUI workflow template for one seed of a RunSpec.

A template is a ComfyUI **API-format** graph — ``{node_id: {"class_type": str,
"inputs": {name: value}}}`` — with placeholder *string* values that this module
substitutes with typed params from a :class:`~argus_proof.models.RunSpec`. Only
values that are exactly a placeholder are replaced (no substring interpolation),
so injected ints/floats stay typed and node-link lists (``[id, slot]``) are left
alone.

Placeholders:

* scalars — ``$base_checkpoint`` ``$vae`` ``$positive`` ``$negative`` ``$seed``
  ``$steps`` ``$cfg`` ``$guidance`` ``$sampler`` ``$scheduler`` ``$width``
  ``$height`` ``$clip_skip`` (emitted as ComfyUI's negative
  ``stop_at_clip_layer``, see :func:`clip_layer`);
* LoRAs — ``$lora`` / ``$lora_weight`` for the first LoRA, ``$lora_2`` /
  ``$lora_2_weight`` for the second, and so on. The template's LoRA slots and
  the spec's LoRA list must match exactly, so a run can't silently drop a LoRA.
* text encoders — ``$text_encoder``, ``$text_encoder_2``, … take
  ``spec.text_encoders`` in order (Flux: T5, then CLIP-L), matched one for one
  like LoRAs.

Every model file a graph loads comes from the run, so the manifest can pin it by
SHA256: a template whose loader names a file literally (``"vae_name":
"ae.safetensors"``) is refused, since that file would load unrecorded and could
change on the ComfyUI host while the manifest stays the same.

No value is silently dropped: a run that sets something the template has no
slot for — a VAE, a non-empty negative prompt (Flux takes none), ``guidance``,
or a ``clip_skip`` above 2 — is refused rather than recorded in the manifest
as used. The placeholders match argus-darkroom's.

Templates are per checkpoint *family* — the checkpoint's top-level folder under
``checkpoints/`` (``flux/flux1-dev.safetensors`` -> ``flux``), as in
argus-darkroom. :func:`packaged_templates` ships SDXL (the default for any
other family) and Flux.
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
_TEXT_ENCODER_RE = re.compile(r"^\$text_encoder(?:_(\d+))?$")
# Loader inputs that name a model file (ComfyUI's core loaders: ckpt_name, vae_name, lora_name,
# clip_name / clip_name1…, unet_name, control_net_name, style_model_name, …). sampler_name is a
# sampler, not a file.
_MODEL_INPUT_RE = re.compile(
    r"^(ckpt|vae|lora|clip|unet|control_net|style_model|clip_vision|upscale_model|gligen|hypernetwork|model)"
    r"_name\d*$"
)

# The example SDXL+LoRA template shipped with the package (single LoRA, clip
# skip, checkpoint's built-in VAE). A starting point users adapt to their graph.
EXAMPLE_TEMPLATE = "comfyui_sdxl_lora.json"

# Packaged templates for families other than the SDXL default. Flux.1: the
# checkpoint is a bare transformer, so the template loads the T5/CLIP-L text
# encoders ($text_encoder, $text_encoder_2) and the ae VAE ($vae) the run names
# (FAMILY_MODEL_DEFAULTS by default); no negative (ConditioningZeroOut), guidance
# through FluxGuidance.
FAMILY_TEMPLATES = {"flux": "comfyui_flux_lora.json"}


def load_template(path: Path) -> dict:
    """Load and lightly validate a ComfyUI API-format workflow template."""
    try:
        template = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BackendError(f"cannot read workflow template {path}: {exc}") from exc
    if not isinstance(template, dict) or not template:
        raise BackendError(f"workflow template {path} is not a non-empty ComfyUI API graph")
    return template


def _packaged(name: str) -> dict:
    resource = importlib.resources.files("argus_proof").joinpath("templates", name)
    return json.loads(resource.read_text(encoding="utf-8"))


def example_template() -> dict:
    """The packaged example SDXL+LoRA workflow template, as a graph dict."""
    return _packaged(EXAMPLE_TEMPLATE)


def packaged_templates() -> dict[str, dict]:
    """The packaged per-family templates (family -> graph), SDXL excluded: it
    is the default for every family not listed here."""
    return {family: _packaged(name) for family, name in FAMILY_TEMPLATES.items()}


def checkpoint_family(checkpoint: str) -> str | None:
    """The checkpoint's family: its top-level folder (``flux/x.safetensors`` ->
    ``flux``), lower-cased. None for a checkpoint at the top level."""
    parts = checkpoint.replace("\\", "/").split("/")
    return parts[0].lower() if len(parts) > 1 and parts[0] else None


# Sampling defaults per family, for a request that leaves a knob out. SDXL's are
# the long-standing ones (and the default for any family not listed); Flux.1-dev
# runs at cfg 1 (its negative is zeroed out) with distilled guidance instead.
_SDXL_DEFAULTS: dict[str, object] = {"cfg": 7.0, "sampler": "dpmpp_2m", "scheduler": "karras", "guidance": None}
FAMILY_SAMPLING_DEFAULTS: dict[str, dict[str, object]] = {
    "flux": {"cfg": 1.0, "sampler": "euler", "scheduler": "simple", "guidance": 3.5},
}


# Model files per family, for a request that names none: Flux.1 checkpoints are bare
# transformers, so the run loads the T5 and CLIP-L text encoders and the ae VAE. SDXL
# (and any family not listed) uses the checkpoint's own.
_SDXL_MODELS: dict[str, object] = {"vae": None, "text_encoders": []}
FAMILY_MODEL_DEFAULTS: dict[str, dict[str, object]] = {
    "flux": {"vae": "ae.safetensors", "text_encoders": ["t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors"]},
}


def model_defaults(checkpoint: str) -> dict[str, object]:
    """``vae`` / ``text_encoders`` defaults for *checkpoint*'s family (SDXL's, none, for a family without its own)."""
    defaults = FAMILY_MODEL_DEFAULTS.get(checkpoint_family(checkpoint) or "", _SDXL_MODELS)
    return {"vae": defaults["vae"], "text_encoders": list(defaults["text_encoders"])}  # type: ignore[call-overload]


def sampling_defaults(checkpoint: str) -> dict[str, object]:
    """``cfg`` / ``sampler`` / ``scheduler`` / ``guidance`` defaults for
    *checkpoint*'s family (SDXL's for a family without its own)."""
    return dict(FAMILY_SAMPLING_DEFAULTS.get(checkpoint_family(checkpoint) or "", _SDXL_DEFAULTS))


def clip_layer(clip_skip: int) -> int:
    """ComfyUI's ``stop_at_clip_layer`` for *clip_skip*, read as A1111 reads it
    for SDXL: 1 and 2 are both -2.

    SDXL conditions on the penultimate text-encoder layer, ComfyUI's own default.
    -1 would run the last layer, which SDXL never uses, and some checkpoints ship
    it as all NaN, so the render comes back blank (argus-darkroom#19).
    """
    return -max(clip_skip, 2)


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
        "$guidance": s.guidance,  # None unless the run sets it (Flux)
        "$sampler": s.sampler,
        "$scheduler": s.scheduler,
        "$width": s.width,
        "$height": s.height,
        "$clip_skip": clip_layer(s.clip_skip),  # ComfyUI CLIPSetLastLayer wants a negative index
    }


def render_workflow(template: dict, spec: RunSpec, seed: int) -> dict:
    """Return a concrete ComfyUI graph for *spec* at *seed*.

    Raises :class:`BackendError` if a used placeholder has no value (e.g. a
    ``$vae`` slot but ``spec.vae is None``), if the template's LoRA or text-encoder
    slots and the spec's lists don't line up one-for-one, if a loader names a model
    file literally, or if the run sets a value the template would ignore (a VAE, a
    negative prompt, ``guidance``, a ``clip_skip`` above 2).
    """
    scalars = _scalar_values(spec, seed)
    lora_by_index = {i: lora for i, lora in enumerate(spec.loras, start=1)}
    seen_scalars: set[str] = set()
    # Track name and weight placeholders per LoRA index SEPARATELY: a template
    # that injects $lora but hard-codes the strength (no $lora_weight) would
    # otherwise silently drop the spec's weight while the manifest records it.
    seen_lora_names: set[int] = set()
    seen_lora_weights: set[int] = set()
    encoder_by_index = {i: name for i, name in enumerate(spec.text_encoders, start=1)}
    seen_encoders: set[int] = set()

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

        m = _TEXT_ENCODER_RE.match(value)
        if m:
            index = int(m.group(1)) if m.group(1) else 1
            if index not in encoder_by_index:
                raise BackendError(
                    f"workflow template references {value} but the run spec has no text encoder #{index} "
                    f"({len(spec.text_encoders)} text encoder(s) supplied)"
                )
            seen_encoders.add(index)
            return encoder_by_index[index]

        # An unknown "$..." string: leave it untouched (it may be a literal).
        return value

    rendered = copy.deepcopy(template)
    for node_id, node in rendered.items():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if isinstance(inputs, dict):
            for key, val in inputs.items():
                if _MODEL_INPUT_RE.match(key) and isinstance(val, str) and not val.startswith("$"):
                    raise BackendError(
                        f"workflow template node {node_id} ({node.get('class_type')}) loads {key}={val!r} by name, "
                        "so the manifest can't pin it by SHA256 — use a placeholder ($base_checkpoint, $vae, "
                        "$text_encoder, $lora …) and name the file in the run"
                    )
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

    unused = [f"$text_encoder{'' if i == 1 else f'_{i}'}" for i in encoder_by_index if i not in seen_encoders]
    if unused:
        raise BackendError(
            f"workflow template is missing text encoder placeholder(s) {unused} — the spec supplies "
            f"{len(spec.text_encoders)} text encoder(s), and one the template doesn't load would be recorded in "
            "the manifest as used; add the slots or drop them from the spec"
        )

    # A configured VAE that the template never consumes would be silently ignored
    # yet still recorded in the manifest — refuse it rather than misrepresent the run.
    if spec.vae is not None and "$vae" not in seen_scalars:
        raise BackendError(
            "run spec sets a vae but the workflow template has no $vae slot — it would be ignored at "
            "generation yet recorded in the manifest; add a $vae slot or clear spec.vae"
        )
    # Same rule for the other knobs a template may not have a slot for.
    if spec.negative_prompt.strip() and "$negative" not in seen_scalars:
        raise BackendError(
            "run spec has a negative prompt but the workflow template has no $negative slot "
            "(Flux takes no negative prompt) — clear negative_prompt"
        )
    if spec.sampling.guidance is not None and "$guidance" not in seen_scalars:
        raise BackendError(
            "run spec sets guidance but the workflow template has no $guidance slot "
            "(guidance is Flux's; SDXL has none) — clear sampling.guidance"
        )
    if clip_layer(spec.sampling.clip_skip) != clip_layer(1) and "$clip_skip" not in seen_scalars:
        raise BackendError("run spec sets clip_skip above 2 but the workflow template has no $clip_skip slot")
    return rendered
