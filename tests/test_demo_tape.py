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
    # the run ids name a different prompt index (p000, p002) than the manifests' text: the text must win
    rid = "synth-s00000960-retry-p000-l00-w00-p000"
    variant, run = demo_tape.run_entry(_manifest(rid, 7), _report(rid), PROMPTS)
    assert (variant, run["prompt"]) == ("960", 7)
    rid = "baseline-rest-p002-l00-w00-p000"
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


def test_summarise_without_any_diversity_or_runs_reads_none() -> None:
    rid = "synth-s00000960-l00-w00-p000"
    _, run = demo_tape.run_entry(_manifest(rid, 0), _report(rid), PROMPTS)
    run["diversity"] = None
    assert demo_tape.summarise([run])["diversity"] is None
    empty = demo_tape.summarise([])
    assert (empty["pass_rate"], empty["diversity"], empty["n_images"]) == (None, None, 0)


def _recording_dir(tmp_path: Path, manifests: list[dict], reports: list[dict]) -> Path:
    for m in manifests:
        run_dir = tmp_path / "eval" / "runs" / m["run_id"]
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(json.dumps(m))
    (tmp_path / "eval" / "reports").mkdir(parents=True, exist_ok=True)
    for r in reports:
        (tmp_path / "eval" / "reports" / f"{r['run_id']}.json").write_text(json.dumps(r))
    return tmp_path


def _recorded(run_id: str, seeds: list[int], scorer: str = "insightface") -> tuple[dict, dict]:
    manifest = _manifest(run_id, 0) | {
        "seeds": seeds,
        "sampling": {"sampler": "euler", "scheduler": "normal", "steps": 30, "cfg": 6, "width": 1024, "height": 1024},
        "base_checkpoint": {"name": "sd_xl_base_1.0.safetensors", "sha256": "cd" * 32},
        "proof_version": "1.0",
        "engine": "comfyui",
        "engine_version": "0.3",
    }
    report = _report(run_id) | {"scorers": [{"metric": "identity", "name": scorer, "model": None}]}
    return manifest, report


def test_recording_with_no_runs_or_no_reports_is_a_tape_error(tmp_path: Path) -> None:
    with pytest.raises(demo_tape.TapeError, match="no run manifest"):
        demo_tape.recording(_recording_dir(tmp_path / "a", [], []))
    m, _ = _recorded("synth-s00000320-l00-w00-p000", [1, 2, 3])
    with pytest.raises(demo_tape.TapeError, match="no report"):
        demo_tape.recording(_recording_dir(tmp_path / "b", [m], []))


def test_recording_checks_seeds_and_scorers_across_runs(tmp_path: Path) -> None:
    a = _recorded("synth-s00000320-l00-w00-p000", [1, 2, 3])
    b = _recorded("synth-s00000640-l00-w00-p000", [1, 2, 4])
    with pytest.raises(demo_tape.TapeError, match="used seeds"):
        demo_tape.recording(_recording_dir(tmp_path / "seeds", [a[0], b[0]], [a[1], b[1]]))
    b = _recorded("synth-s00000640-l00-w00-p000", [1, 2, 3], scorer="arcface")
    with pytest.raises(demo_tape.TapeError, match="scored differently"):
        demo_tape.recording(_recording_dir(tmp_path / "scorers", [a[0], b[0]], [a[1], b[1]]))
    b = _recorded("synth-s00000640-l00-w00-p000", [1, 2, 3])
    rec = demo_tape.recording(_recording_dir(tmp_path / "ok", [a[0], b[0]], [a[1], b[1]]))
    assert rec["seeds"] == [1, 2, 3] and rec["scorers"][0]["name"] == "insightface"


def test_recording_refuses_a_different_engine_or_proof_version(tmp_path: Path) -> None:
    a = _recorded("synth-s00000320-l00-w00-p000", [1, 2, 3])
    b = _recorded("synth-s00000640-l00-w00-p000", [1, 2, 3])
    b[0]["engine_version"] = "9.9.9"
    with pytest.raises(demo_tape.TapeError, match="used engine"):
        demo_tape.recording(_recording_dir(tmp_path / "engine", [a[0], b[0]], [a[1], b[1]]))
    b = _recorded("synth-s00000640-l00-w00-p000", [1, 2, 3])
    b[0]["engine"] = "a1111"
    with pytest.raises(demo_tape.TapeError, match="used engine"):
        demo_tape.recording(_recording_dir(tmp_path / "name", [a[0], b[0]], [a[1], b[1]]))
    b = _recorded("synth-s00000640-l00-w00-p000", [1, 2, 3])
    b[0]["proof_version"] = "9.9"
    with pytest.raises(demo_tape.TapeError, match="proof_version"):
        demo_tape.recording(_recording_dir(tmp_path / "wire", [a[0], b[0]], [a[1], b[1]]))


def test_recording_still_runs_when_the_distribution_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The auditor runs tests from PYTHONPATH and does not install the distribution."""
    import importlib.metadata as metadata

    from argus_proof import __version__

    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    a = _recorded("synth-s00000320-l00-w00-p000", [1, 2, 3])
    b = _recorded("synth-s00000640-l00-w00-p000", [1, 2, 3])
    rec = demo_tape.recording(_recording_dir(tmp_path / "ok", [a[0], b[0]], [a[1], b[1]]))
    assert rec["proof_version"] == __version__


def test_recording_reads_the_threshold_and_the_package_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(demo_tape, "package_version", lambda: "9.9.9")
    a = _recorded("synth-s00000320-l00-w00-p000", [1, 2, 3])
    b = _recorded("synth-s00000640-l00-w00-p000", [1, 2, 3])
    a[1]["gate"] = {"run_pass_rate": 0.6}
    b[1]["gate"] = {"run_pass_rate": 0.6}
    rec = demo_tape.recording(_recording_dir(tmp_path / "ok", [a[0], b[0]], [a[1], b[1]]))
    assert rec["pass_threshold"] == 0.6
    assert rec["proof_version"] == "9.9.9"
    assert rec["wire_version"] == "1.0"
    b[1]["gate"] = {"run_pass_rate": 0.5}
    with pytest.raises(demo_tape.TapeError, match="pass thresholds"):
        demo_tape.recording(_recording_dir(tmp_path / "mix", [a[0], b[0]], [a[1], b[1]]))


def test_webp_applies_exif_orientation(tmp_path: Path) -> None:
    from PIL import Image

    src = tmp_path / "portrait.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # stored landscape, displayed rotated 90 degrees: a portrait
    Image.new("RGB", (40, 20), (200, 10, 10)).save(src, exif=exif)
    dims = demo_tape.webp(src, tmp_path / "out.webp", 512, 80)
    assert (dims["w"], dims["h"]) == (20, 40)
    with Image.open(tmp_path / "out.webp") as out:
        assert out.size == (20, 40) and "exif" not in out.info


def test_webp_composites_transparency_onto_white(tmp_path: Path) -> None:
    from PIL import Image

    src = tmp_path / "cutout.png"
    Image.new("RGBA", (16, 16), (0, 0, 0, 0)).save(src)  # fully transparent black
    demo_tape.webp(src, tmp_path / "out.webp", 512, 100)
    with Image.open(tmp_path / "out.webp") as out:
        assert out.mode == "RGB"
        assert all(c > 240 for c in out.convert("RGB").getpixel((8, 8)))


def test_a_failed_build_keeps_the_published_tape(tmp_path: Path) -> None:
    root = tmp_path / "rec"
    (root / "eval").mkdir(parents=True)
    (root / "eval" / "prompts8.json").write_text(json.dumps({str(i): t for t, i in PROMPTS.items()}))
    rid = "synth-s00000320-l00-w00-p000"
    m, r = _recorded(rid, [1])
    _recording_dir(root, [m], [r])  # no run image on disk: the build fails converting it
    out = tmp_path / "tape"
    out.mkdir()
    (out / "tape.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        demo_tape.main([str(root), "--out", str(out)])
    assert (out / "tape.json").read_text() == "{}"
    assert not (tmp_path / "tape.tmp").exists()


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
