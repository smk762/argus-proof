"""Curator speech exports (#118): reading the held-out references, and refusing ones whose speaker trained.

The exports are hand-built ``speech_manifest.jsonl`` files with curator's ``SpeechManifestRow`` fields
(argus-curator ``speech_manifest_version`` 1.0).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus_proof.speech import (
    SPEAKER_CHECK,
    SPEECH_MANIFEST_NAME,
    SpeakerOverlap,
    SpeechExportError,
    check_speakers_disjoint,
    held_out_references,
    read_speech_export,
)


def _row(path: str, *, split: str | None, speaker: str | None, cluster: str | None = None, **extra) -> dict:  # noqa: ANN003
    row = {
        "speech_manifest_version": "1.0",
        "modality": "speech",
        "rel_path": f"corpus/{path}",
        "abs_path": f"/somewhere/else/{path}",
        "exported_path": path,
        "exported_abs_path": f"/where/curator/ran/{path}",
        "transcript": f"the words of {path}",
        "duration_s": 3.2,
        "sample_rate": 24000,
        "channels": 1,
        "speaker_id": speaker or cluster,
        "speaker_label": speaker,
        "speaker_cluster": cluster,
        "speaker_mismatch": False,
        "accent": "carioca",
        "language": "pt",
        "split": split,
        "corpus": "public-domain-corpus",
        "license": "CC0-1.0",
        "provenance": {"kind": "stock", "source": "public-domain-corpus", "license": "CC0-1.0"},
        "a_field_from_a_newer_curator": True,
    }
    return {**row, **extra}


def _export(root: Path, rows: list[dict], *, audio: dict[str, bytes] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        clip = root / row["exported_path"]
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(
            (audio or {}).get(row["exported_path"], f"audio of {root.name}/{row['exported_path']}".encode())
        )
    (root / SPEECH_MANIFEST_NAME).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return root


def _train(tmp_path: Path, **kw) -> Path:  # noqa: ANN003
    rows = [
        _row("a1.wav", split="train", speaker="ana", cluster="spk_1"),
        _row("b1.wav", split="train", speaker="bruno", cluster="spk_2"),
        _row("x1.wav", split="train", speaker=None, cluster="spk_3"),  # unlabelled: its cluster is its speaker
    ]
    return _export(tmp_path / "train", rows, **kw)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def test_held_out_rows_become_references_in_manifest_order(tmp_path: Path) -> None:
    held = _export(
        tmp_path / "held",
        [
            _row("c/c1.wav", split="held_out", speaker="carla", cluster="spk_4"),
            _row("d1.wav", split="held_out", speaker="davi", cluster="spk_5"),
        ],
    )
    refs = held_out_references(read_speech_export(held))
    assert [r.ref_id for r in refs] == ["r0", "r1"]
    assert [r.rel_path for r in refs] == ["c/c1.wav", "d1.wav"]
    # The clip is read from the export, not from wherever curator ran.
    assert refs[0].path == str((held / "c" / "c1.wav").resolve())
    assert (refs[0].transcript, refs[0].speaker_label, refs[0].accent, refs[0].language) == (
        "the words of c/c1.wav",
        "carla",
        "carioca",
        "pt",
    )


def test_a_training_row_in_the_reference_export_is_refused(tmp_path: Path) -> None:
    held = _export(
        tmp_path / "held",
        [_row("c1.wav", split="held_out", speaker="carla"), _row("a1.wav", split="train", speaker="ana")],
    )
    with pytest.raises(SpeechExportError, match="not a held-out export: 1 clip"):
        held_out_references(read_speech_export(held))


def test_too_many_references_are_refused(tmp_path: Path) -> None:
    held = _export(tmp_path / "held", [_row(f"c{i}.wav", split="held_out", speaker=f"s{i}") for i in range(33)])
    with pytest.raises(SpeechExportError, match="at most 32"):
        held_out_references(read_speech_export(held))


def test_a_directory_without_a_speech_manifest_is_not_an_export(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "manifest.jsonl").write_text("{}\n")
    with pytest.raises(SpeechExportError, match="not a curator speech export"):
        read_speech_export(tmp_path / "images")


def test_an_unknown_manifest_major_is_refused(tmp_path: Path) -> None:
    root = _export(tmp_path / "held", [_row("c1.wav", split="held_out", speaker="c", speech_manifest_version="2.0")])
    with pytest.raises(SpeechExportError, match="proof reads 1.x"):
        read_speech_export(root)


@pytest.mark.parametrize("path", ["../outside.wav", "/etc/passwd"])
def test_a_clip_outside_the_export_is_refused(tmp_path: Path, path: str) -> None:
    root = tmp_path / "held"
    root.mkdir()
    (tmp_path / "outside.wav").write_bytes(b"x")
    (root / SPEECH_MANIFEST_NAME).write_text(json.dumps(_row(path, split="held_out", speaker="c")) + "\n")
    with pytest.raises(SpeechExportError, match="exported_path"):
        read_speech_export(root)


def test_a_missing_clip_is_refused(tmp_path: Path) -> None:
    root = _export(tmp_path / "held", [_row("c1.wav", split="held_out", speaker="c")])
    (root / "c1.wav").unlink()
    with pytest.raises(SpeechExportError, match="c1.wav is not in the export"):
        read_speech_export(root)


# --------------------------------------------------------------------------
# the speaker-overlap refusal
# --------------------------------------------------------------------------


def test_disjoint_speakers_pass_and_say_what_was_checked(tmp_path: Path) -> None:
    training = read_speech_export(_train(tmp_path))
    held = _export(tmp_path / "held", [_row("c1.wav", split="held_out", speaker="carla", cluster="spk_4")])
    assert check_speakers_disjoint(held_out_references(read_speech_export(held)), training) == SPEAKER_CHECK


def test_a_reference_whose_speaker_trained_is_refused(tmp_path: Path) -> None:
    training = read_speech_export(_train(tmp_path))
    held = _export(tmp_path / "held", [_row("a9.wav", split="held_out", speaker="ana", cluster="spk_9")])
    with pytest.raises(SpeakerOverlap, match=r"r0 \(a9.wav\): speaker 'ana' is in training"):
        check_speakers_disjoint(held_out_references(read_speech_export(held)), training)


def test_a_reference_whose_cluster_trained_is_refused_with_the_scan_hint(tmp_path: Path) -> None:
    training = read_speech_export(_train(tmp_path))
    held = _export(tmp_path / "held", [_row("x9.wav", split="held_out", speaker=None, cluster="spk_3")])
    with pytest.raises(SpeakerOverlap, match="speaker cluster 'spk_3' is in training") as err:
        check_speakers_disjoint(held_out_references(read_speech_export(held)), training)
    assert "same scan" in str(err.value)


def test_a_reference_that_is_a_training_clip_under_another_name_is_refused(tmp_path: Path) -> None:
    same = b"the exact same recording"
    training = read_speech_export(_train(tmp_path, audio={"b1.wav": same}))
    held = _export(
        tmp_path / "held",
        [_row("renamed.wav", split="held_out", speaker="nobody", cluster="spk_8")],
        audio={"renamed.wav": same},
    )
    with pytest.raises(SpeakerOverlap, match="it is training clip b1.wav"):
        check_speakers_disjoint(held_out_references(read_speech_export(held)), training)


def test_every_overlap_is_named_at_once(tmp_path: Path) -> None:
    training = read_speech_export(_train(tmp_path))
    held = _export(
        tmp_path / "held",
        [
            _row("ok.wav", split="held_out", speaker="carla", cluster="spk_4"),
            _row("a9.wav", split="held_out", speaker="ana", cluster="spk_1"),
            _row("b9.wav", split="held_out", speaker="bruno", cluster="spk_7"),
        ],
    )
    with pytest.raises(SpeakerOverlap) as err:
        check_speakers_disjoint(held_out_references(read_speech_export(held)), training)
    message = str(err.value)
    assert "r1 (a9.wav): speaker 'ana' is in training; speaker cluster 'spk_1' is in training" in message
    assert "r2 (b9.wav): speaker 'bruno'" in message
    assert "ok.wav" not in message
