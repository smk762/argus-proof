"""The hosted demo's tape builder (scripts/demo_tape.py) and the committed tape (web/public/tape/)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TAPE_DIR = ROOT / "web" / "public" / "tape"

_spec = importlib.util.spec_from_file_location("demo_tape", ROOT / "scripts" / "demo_tape.py")
assert _spec and _spec.loader
demo_tape = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo_tape)

PROMPTS = {f"ohwx woman, prompt {i}": i for i in range(8)}


def _manifest(run_id: str, prompt: int, lora: bool = True) -> dict:
    return {
        "run_id": run_id,
        "prompt": f"ohwx woman, prompt {prompt}",
        "seeds": [1, 2, 3],
        "loras": [
            {"name": "sdxl/proof/x-step00000960.safetensors", "sha256": "ab" * 32, "weight": 1.0 if lora else 0.0}
        ],
        "source_manifest": "/data/out/train/manifest.jsonl",
    }


def _report(run_id: str) -> dict:
    image = {
        "image_id": f"{run_id}-1",
        "seed": 1,
        "metrics": {"identity": 0.4, "clip_score": 0.8, "aesthetic": 0.7, "preference": None, "safety": 1.0},
        "passed": None,
        "reject_reasons": [],
        "duplicate_group": 0,
    }
    return {
        "run_id": run_id,
        "images": [image],
        "aggregate": {"pass_rate": 0.0, "diversity": 0.5, "n_groups": 1, "n_passed": 0, "n_needs_hitl": 1},
        "verdict": {"passed": False, "pending": True, "reasons": ["pass_rate 0.00 vs threshold 0.75"]},
    }


def test_a_retried_run_maps_by_prompt_text_not_run_id() -> None:
    rid = "synth-s00000960-retry-p007-l00-w00-p000"
    variant, run = demo_tape.run_entry(_manifest(rid, 7), _report(rid), PROMPTS)
    assert (variant, run["prompt"]) == ("960", 7)
    rid = "baseline-rest-p004-l00-w00-p000"
    variant, run = demo_tape.run_entry(_manifest(rid, 4, lora=False), _report(rid), PROMPTS)
    assert (variant, run["prompt"], run["lora"]) == ("baseline", 4, None)


def test_a_prompt_outside_the_eight_is_refused() -> None:
    m = _manifest("synth-s00000320-l00-w00-p000", 0) | {"prompt": "something else"}
    with pytest.raises(demo_tape.TapeError):
        demo_tape.run_entry(m, _report(m["run_id"]), PROMPTS)


def test_a_baseline_with_a_lora_is_refused() -> None:
    rid = "baseline-l00-w00-p000"
    with pytest.raises(demo_tape.TapeError):
        demo_tape.run_entry(_manifest(rid, 0, lora=True), _report(rid), PROMPTS)


def test_the_run_carries_no_container_path_and_no_preference() -> None:
    rid = "synth-s00000960-l00-w00-p000"
    _, run = demo_tape.run_entry(_manifest(rid, 0), _report(rid), PROMPTS)
    text = json.dumps(run)
    assert "/data/" not in text and "source_manifest" not in text
    assert "preference" not in run["images"][0]["metrics"]
    assert run["lora"]["name"] == "x-step00000960.safetensors"


def test_a_leaked_path_fails_the_build() -> None:
    assert demo_tape.leaks('{"x": "/data/out/train"}')
    # Built at runtime so the repo's own opsec scans don't flag the fixtures.
    assert demo_tape.leaks("/" + "home/someone/x")
    assert demo_tape.leaks(".".join(["192", "168", "1", "20"]))
    assert not demo_tape.leaks("sd_xl_base_1.0.safetensors")


# ── the committed tape ──


@pytest.fixture(scope="module")
def tape() -> dict:
    return json.loads((TAPE_DIR / "tape.json").read_text())


def test_the_committed_tape_has_every_variant_prompt_and_seed(tape: dict) -> None:
    assert [v["id"] for v in tape["variants"]] == list(demo_tape.VARIANTS)
    for v in tape["variants"]:
        assert [r["prompt"] for r in v["runs"]] == list(range(8))
        assert all(sorted(im["seed"] for im in r["images"]) == [1, 2, 3] for r in v["runs"])


def test_the_committed_tape_leaks_nothing(tape: dict) -> None:
    assert demo_tape.leaks(json.dumps(tape)) == []


def test_every_listed_image_is_a_metadata_free_webp_and_nothing_else_is_shipped(tape: dict) -> None:
    from PIL import Image

    listed = {"tape.json"}
    for v in tape["variants"]:
        for r in v["runs"]:
            for im in r["images"]:
                listed |= {im["tile"], im["full"]}
    listed |= {p for r in tape["refs"] for p in (r["tile"], r["full"])}
    listed |= {t["tile"] for t in tape["train"]}
    listed |= {p for t in tape["token_probe"] for p in (t["ohwx"], t["plain"])}
    on_disk = {p.relative_to(TAPE_DIR).as_posix() for p in TAPE_DIR.rglob("*") if p.is_file()}
    assert on_disk == listed
    for rel in sorted(listed - {"tape.json"}):
        with Image.open(TAPE_DIR / rel) as img:
            assert img.format == "WEBP", rel
            assert not ({"exif", "xmp", "icc_profile"} & set(img.info)), rel
