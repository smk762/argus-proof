from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus_proof.grid import GridError, build_grid, plan_from_export, read_export_prompts
from argus_proof.models import GridConfig, SamplingParams


def make_config(**overrides) -> GridConfig:
    base = dict(
        base_checkpoint="sdxl_base.safetensors",
        lora_checkpoints=["subject-e4.safetensors", "subject-e6.safetensors"],
        lora_weights=[0.7, 1.0],
        sampling=SamplingParams(
            sampler="dpmpp_2m", scheduler="karras", steps=30, cfg=7.0, clip_skip=2, width=1024, height=1024
        ),
        seeds=[1, 2, 3, 4],
    )
    base.update(overrides)
    return GridConfig(**base)


# --------------------------------------------------------------------------
# prompt sourcing
# --------------------------------------------------------------------------


def write_captions_json(export: Path, entries: list[dict], name: str = "captions.json") -> None:
    (export / name).write_text(json.dumps(entries), encoding="utf-8")


def test_prefers_zeroshot_variant_from_captions_json(tmp_path: Path) -> None:
    write_captions_json(
        tmp_path,
        [
            {"name": "a.png", "final_caption": "sks, tag1", "caption_variants": {"zeroshot": "a photo of sks person"}},
            {"name": "b.png", "final_caption": "sks, tag2", "caption_variants": {"zeroshot": "sks person outdoors"}},
        ],
    )
    assert read_export_prompts(tmp_path) == ["a photo of sks person", "sks person outdoors"]


def test_falls_back_to_training_variant_then_final_caption(tmp_path: Path) -> None:
    write_captions_json(
        tmp_path,
        [
            {"name": "a.png", "final_caption": "final-a", "caption_variants": {"training": "training-a"}},
            {"name": "b.png", "final_caption": "final-b", "caption_variants": {}},
        ],
    )
    assert read_export_prompts(tmp_path) == ["training-a", "final-b"]


def test_falls_back_to_txt_sidecars_when_no_captions_json(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("caption a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("caption b", encoding="utf-8")
    assert read_export_prompts(tmp_path) == ["caption a", "caption b"]


def test_captions_json_wins_over_txt(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("txt caption", encoding="utf-8")
    write_captions_json(tmp_path, [{"name": "a.png", "caption_variants": {"zeroshot": "json caption"}}])
    assert read_export_prompts(tmp_path) == ["json caption"]


def test_unrelated_json_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"unrelated": True}), encoding="utf-8")
    (tmp_path / "a.txt").write_text("txt caption", encoding="utf-8")
    assert read_export_prompts(tmp_path) == ["txt caption"]


def test_prompts_deduplicated_in_order(tmp_path: Path) -> None:
    for name, text in [("a.txt", "same"), ("b.txt", "same"), ("c.txt", "other")]:
        (tmp_path / name).write_text(text, encoding="utf-8")
    assert read_export_prompts(tmp_path) == ["same", "other"]


def test_non_utf8_txt_sidecar_is_skipped_not_crashed(tmp_path: Path) -> None:
    (tmp_path / "good.txt").write_text("a photo of sks", encoding="utf-8")
    (tmp_path / "latin1.txt").write_bytes(b"a photo of caf\xe9")  # invalid UTF-8
    assert read_export_prompts(tmp_path) == ["a photo of sks"]


def test_non_utf8_json_is_skipped_not_crashed(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_bytes(b"\xff\xfe not utf-8")
    (tmp_path / "a.txt").write_text("txt caption", encoding="utf-8")
    assert read_export_prompts(tmp_path) == ["txt caption"]


def test_malformed_jsonl_line_skips_line_not_whole_file(tmp_path: Path) -> None:
    lines = [
        json.dumps({"name": "a.png", "caption_variants": {"zeroshot": "prompt a"}}),
        "{ this is not valid json",
        json.dumps({"name": "b.png", "caption_variants": {"zeroshot": "prompt b"}}),
    ]
    (tmp_path / "captions.jsonl").write_text("\n".join(lines), encoding="utf-8")
    assert read_export_prompts(tmp_path) == ["prompt a", "prompt b"]


def test_negative_max_token_combos_rejected() -> None:
    with pytest.raises(ValueError):
        make_config(max_token_combos=-1)


# --------------------------------------------------------------------------
# grid expansion
# --------------------------------------------------------------------------


def test_axes_multiply_into_runs() -> None:
    plan = build_grid(make_config(), ["p1", "p2", "p3"])
    # 2 checkpoints x 2 weights x 3 prompts = 12 runs; x 4 seeds = 48 images
    assert plan.estimate.n_runs == 12
    assert plan.estimate.n_images == 48
    assert plan.estimate.axes == {"lora_checkpoints": 2, "weights": 2, "prompts": 3, "seeds": 4}
    assert len(plan.specs) == 12


def test_run_specs_carry_axis_values() -> None:
    plan = build_grid(make_config(lora_checkpoints=["ckpt-e4.safetensors"], lora_weights=[0.8]), ["a prompt"])
    spec = plan.specs[0]
    assert spec.base_checkpoint == "sdxl_base.safetensors"
    assert spec.loras[0].name == "ckpt-e4.safetensors"
    assert spec.loras[0].weight == 0.8
    assert spec.prompt == "a prompt"
    assert spec.seeds == [1, 2, 3, 4]


def test_run_ids_unique_and_deterministic() -> None:
    plan = build_grid(make_config(), ["p1", "p2"])
    ids = [s.run_id for s in plan.specs]
    assert len(ids) == len(set(ids))
    # identical inputs reproduce identical run_ids
    again = build_grid(make_config(), ["p1", "p2"])
    assert [s.run_id for s in again.specs] == ids


def test_grid_is_fully_reproducible() -> None:
    cfg = make_config(token_axes={"setting": ["a park", "a studio"]}, max_token_combos=1, combo_seed=7)
    a = build_grid(cfg, ["p1", "p2"])
    b = build_grid(cfg, ["p1", "p2"])
    assert a.model_dump() == b.model_dump()


def test_token_combos_appended_to_base_prompts() -> None:
    cfg = make_config(
        lora_checkpoints=["c.safetensors"], lora_weights=[1.0], token_axes={"setting": ["a park", "a studio"]}
    )
    plan = build_grid(cfg, ["sks person"])
    prompts = sorted(s.prompt for s in plan.specs)
    assert prompts == ["sks person, a park", "sks person, a studio"]


def test_max_token_combos_caps_deterministically() -> None:
    cfg = make_config(
        lora_checkpoints=["c.safetensors"],
        lora_weights=[1.0],
        token_axes={"setting": ["s1", "s2", "s3"], "pose": ["p1", "p2"]},  # 6 combos
        max_token_combos=2,
        combo_seed=1,
    )
    plan = build_grid(cfg, ["base"])
    assert len(plan.specs) == 2  # 1 prompt x 2 sampled combos
    # same seed -> same sampled combos
    again = build_grid(cfg, ["base"])
    assert [s.prompt for s in plan.specs] == [s.prompt for s in again.specs]


def test_flexibility_prompts_included_and_marked() -> None:
    cfg = make_config(
        lora_checkpoints=["c.safetensors"], lora_weights=[1.0], flexibility_prompts=["sks person as an astronaut"]
    )
    plan = build_grid(cfg, ["sks person"])
    flex = [s for s in plan.specs if "-f" in s.run_id]
    assert len(flex) == 1
    assert flex[0].prompt == "sks person as an astronaut"


def test_max_base_prompts_caps() -> None:
    cfg = make_config(lora_checkpoints=["c.safetensors"], lora_weights=[1.0], max_base_prompts=2)
    plan = build_grid(cfg, ["p1", "p2", "p3", "p4"])
    assert len(plan.specs) == 2


def test_no_prompts_raises() -> None:
    with pytest.raises(GridError, match="no prompts to generate"):
        build_grid(make_config(), [])


def test_estimate_gpu_hours() -> None:
    cfg = make_config(lora_checkpoints=["c.safetensors"], lora_weights=[1.0], seeds=[1, 2], seconds_per_image=10.0)
    plan = build_grid(cfg, ["p1", "p2"])  # 2 runs x 2 seeds = 4 images
    assert plan.estimate.n_images == 4
    assert plan.estimate.est_gpu_seconds == 40.0
    assert plan.estimate.est_gpu_hours == pytest.approx(40.0 / 3600.0)


def test_plan_from_export_end_to_end(tmp_path: Path) -> None:
    write_captions_json(tmp_path, [{"name": "a.png", "caption_variants": {"zeroshot": "a photo of sks person"}}])
    plan = plan_from_export(tmp_path, make_config())
    assert plan.estimate.n_runs == 4  # 2 checkpoints x 2 weights x 1 prompt
    assert all(s.prompt == "a photo of sks person" for s in plan.specs)
