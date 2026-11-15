from __future__ import annotations

import pytest

from argus_proof.backends.base import BackendError
from argus_proof.backends.workflow import example_template, render_workflow
from argus_proof.models import LoRASpec, RunSpec, SamplingParams


def make_spec(**overrides) -> RunSpec:
    base = dict(
        run_id="run-1",
        base_checkpoint="base.safetensors",
        loras=[LoRASpec(name="subject.safetensors", weight=0.8)],
        sampling=SamplingParams(
            sampler="euler", scheduler="normal", steps=20, cfg=6.5, clip_skip=2, width=832, height=1216
        ),
        prompt="a photo of sks",
        negative_prompt="lowres",
        seeds=[11],
    )
    base.update(overrides)
    return RunSpec(**base)


def _find(graph: dict, class_type: str) -> dict:
    return next(n["inputs"] for n in graph.values() if n["class_type"] == class_type)


def test_render_substitutes_scalars_with_typed_values() -> None:
    graph = render_workflow(example_template(), make_spec(), seed=99)
    ksampler = _find(graph, "KSampler")
    assert ksampler["seed"] == 99 and isinstance(ksampler["seed"], int)
    assert ksampler["steps"] == 20
    assert ksampler["cfg"] == 6.5 and isinstance(ksampler["cfg"], float)
    assert ksampler["sampler_name"] == "euler"
    assert _find(graph, "CheckpointLoaderSimple")["ckpt_name"] == "base.safetensors"
    assert _find(graph, "EmptyLatentImage")["width"] == 832


def test_clip_skip_rendered_as_negative_index() -> None:
    graph = render_workflow(example_template(), make_spec(), seed=1)
    assert _find(graph, "CLIPSetLastLayer")["stop_at_clip_layer"] == -2


def test_lora_name_and_weight_injected() -> None:
    loader = _find(render_workflow(example_template(), make_spec(), seed=1), "LoraLoader")
    assert loader["lora_name"] == "subject.safetensors"
    assert loader["strength_model"] == 0.8


def test_node_links_left_untouched() -> None:
    graph = render_workflow(example_template(), make_spec(), seed=1)
    assert _find(graph, "KSampler")["model"] == ["10", 0]


def test_does_not_mutate_the_template() -> None:
    template = example_template()
    render_workflow(template, make_spec(), seed=1)
    assert _find(template, "KSampler")["seed"] == "$seed"


def test_more_loras_than_template_slots_raises() -> None:
    spec = make_spec(loras=[LoRASpec(name="a.safetensors"), LoRASpec(name="b.safetensors")])
    with pytest.raises(BackendError, match="missing LoRA placeholder"):
        render_workflow(example_template(), spec, seed=1)


def test_template_lora_slot_without_a_spec_lora_raises() -> None:
    spec = make_spec(loras=[])
    with pytest.raises(BackendError, match="has no LoRA #1"):
        render_workflow(example_template(), spec, seed=1)


def test_used_placeholder_without_value_raises() -> None:
    template = {"1": {"class_type": "VAELoader", "inputs": {"vae_name": "$vae"}}}
    with pytest.raises(BackendError, match=r"\$vae"):
        render_workflow(template, make_spec(vae=None), seed=1)


def test_lora_name_slot_without_weight_slot_raises() -> None:
    # $lora present but the strength is hard-coded (no $lora_weight): the spec's
    # weight would be silently dropped while the manifest records it.
    template = {
        "10": {"class_type": "LoraLoader", "inputs": {"lora_name": "$lora", "strength_model": 1.0, "clip": ["4", 1]}}
    }
    with pytest.raises(BackendError, match=r"missing LoRA placeholder"):
        render_workflow(template, make_spec(), seed=1)


def test_lora_weight_slot_without_name_slot_raises() -> None:
    template = {"10": {"class_type": "LoraLoader", "inputs": {"lora_name": "fixed.safetensors", "w": "$lora_weight"}}}
    with pytest.raises(BackendError, match=r"missing LoRA placeholder"):
        render_workflow(template, make_spec(), seed=1)


def test_vae_set_but_template_has_no_vae_slot_raises() -> None:
    # spec.vae would be ignored at generation yet recorded in the manifest.
    with pytest.raises(BackendError, match="no .vae slot"):
        render_workflow(example_template(), make_spec(vae="sdxl_vae.safetensors"), seed=1)


# --- Flux / per-family templates (issue #60) ---------------------------------

from argus_proof.backends.workflow import (  # noqa: E402
    checkpoint_family,
    clip_layer,
    packaged_templates,
    sampling_defaults,
)


def _flux_spec(**overrides) -> RunSpec:
    base = dict(
        base_checkpoint="flux/flux1-dev.safetensors",
        sampling=SamplingParams(
            sampler="euler", scheduler="simple", steps=20, cfg=1.0, guidance=3.5, width=1024, height=1024
        ),
        negative_prompt="",
    )
    base.update(overrides)
    return make_spec(**base)


def test_flux_template_renders_guidance_and_zeroed_negative() -> None:
    graph = render_workflow(packaged_templates()["flux"], _flux_spec(), seed=7)
    assert _find(graph, "FluxGuidance")["guidance"] == 3.5
    assert _find(graph, "KSampler")["cfg"] == 1.0
    assert _find(graph, "KSampler")["seed"] == 7
    assert _find(graph, "CheckpointLoaderSimple")["ckpt_name"] == "flux/flux1-dev.safetensors"
    assert _find(graph, "LoraLoader")["lora_name"] == "subject.safetensors"
    # no $negative slot: the negative is the positive zeroed out
    assert _find(graph, "ConditioningZeroOut")["conditioning"] == ["6", 0]
    assert _find(graph, "DualCLIPLoader")["type"] == "flux"
    assert not any(isinstance(v, str) and v.startswith("$") for n in graph.values() for v in n["inputs"].values())


def test_negative_prompt_without_a_slot_is_refused() -> None:
    with pytest.raises(BackendError, match=r"no \$negative slot"):
        render_workflow(packaged_templates()["flux"], _flux_spec(negative_prompt="lowres"), seed=1)
    # whitespace-only is not a negative prompt
    render_workflow(packaged_templates()["flux"], _flux_spec(negative_prompt="  "), seed=1)


def test_guidance_without_a_slot_is_refused() -> None:
    spec = make_spec(
        sampling=SamplingParams(sampler="euler", scheduler="normal", steps=20, cfg=6.5, guidance=3.5, width=8, height=8)
    )
    with pytest.raises(BackendError, match=r"no \$guidance slot"):
        render_workflow(example_template(), spec, seed=1)


def test_flux_template_needs_guidance() -> None:
    no_guidance = SamplingParams(sampler="euler", scheduler="simple", steps=20, cfg=1.0, width=8, height=8)
    with pytest.raises(BackendError, match=r"\$guidance"):
        render_workflow(packaged_templates()["flux"], _flux_spec(sampling=no_guidance), seed=1)


def test_clip_skip_above_two_without_a_slot_is_refused() -> None:
    spec = _flux_spec(
        sampling=SamplingParams(
            sampler="euler", scheduler="simple", steps=20, cfg=1.0, guidance=3.5, clip_skip=3, width=8, height=8
        )
    )
    with pytest.raises(BackendError, match=r"clip_skip above 2"):
        render_workflow(packaged_templates()["flux"], spec, seed=1)
    # 1 and 2 are the same layer, so a template without the slot accepts them
    ok = spec.sampling.model_copy(update={"clip_skip": 2})
    render_workflow(packaged_templates()["flux"], _flux_spec(sampling=ok), seed=1)


def test_clip_layer_never_selects_the_last_layer() -> None:
    """clip_skip 1 sent -1 and blanked some SDXL checkpoints (darkroom #19)."""
    assert [clip_layer(n) for n in (1, 2, 3, 12)] == [-2, -2, -3, -12]
    sdxl = make_spec(
        sampling=SamplingParams(sampler="euler", scheduler="normal", steps=20, cfg=6.5, clip_skip=1, width=8, height=8)
    )
    assert _find(render_workflow(example_template(), sdxl, seed=1), "CLIPSetLastLayer")["stop_at_clip_layer"] == -2


def test_checkpoint_family_and_sampling_defaults() -> None:
    assert checkpoint_family("flux/flux1-dev.safetensors") == "flux"
    assert checkpoint_family("Flux\\a.safetensors") == "flux"
    assert checkpoint_family("sdxl/base.safetensors") == "sdxl"
    assert checkpoint_family("base.safetensors") is None
    assert sampling_defaults("flux/x.safetensors") == {
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "guidance": 3.5,
    }
    assert sampling_defaults("base.safetensors")["guidance"] is None
    assert sampling_defaults("sd15/x.safetensors")["cfg"] == 7.0
