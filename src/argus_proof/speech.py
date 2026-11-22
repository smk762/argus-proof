"""Curator speech exports for speech evaluation (#118): the held-out references, and the speaker-overlap refusal.

argus-curator writes a speech export as clips (with ``<stem>.txt`` transcripts) and a ``speech_manifest.jsonl``
(``SpeechManifestRow``, ``speech_manifest_version`` 1.x), one split part per export: the training part is what the
voice trained on, the held-out part is where a speech run's reference voices come from. Proof reads only the row
fields below; unknown fields are ignored and a major other than 1 is refused.

A reference whose speaker is in the training set measures memorisation, not how the fine-tune carries a voice it
never heard (the speech counterpart of identity references that overlap the training images).
:func:`check_speakers_disjoint` refuses one when:

* its corpus ``speaker_label`` is a training clip's label;
* its ``speaker_cluster`` is a training clip's cluster. Cluster ids (``spk_1`` …) are numbered per curator scan, so
  they compare only between the two parts of one scan, which is curator's documented way to export them (the same
  filters and split, one ``dest`` per part). Exports from different scans can collide on an id and be refused
  falsely: re-export both parts from one scan;
* its audio is a training clip, byte for byte (SHA256), under any name.

Speaker-embedding similarity (a re-recorded clip of a training speaker with no label or cluster in common) is not
checked yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from argus_proof.hashing import sha256_cached
from argus_proof.models import MAX_SPEECH_REFERENCES, ProofError, SpeechReferenceSpec

SPEECH_MANIFEST_NAME = "speech_manifest.jsonl"
SUPPORTED_SPEECH_MANIFEST_MAJORS = ("1",)
# What the overlap check ran, for a run's records.
SPEAKER_CHECK = "speaker_label+speaker_cluster+sha256"


class SpeechExportError(ProofError):
    """A speech export that can't be read, or a reference set proof refuses."""


class SpeakerOverlap(SpeechExportError):
    """A held-out reference whose speaker (or clip) is in the training export."""


class SpeechExportRow(BaseModel):
    """The fields proof reads from one curator ``SpeechManifestRow``; the rest are ignored."""

    model_config = ConfigDict(extra="ignore")

    speech_manifest_version: str
    modality: Literal["speech"]
    rel_path: str
    exported_path: str
    transcript: str = ""
    speaker_label: str | None = None
    speaker_cluster: str | None = None
    language: str | None = None
    accent: str | None = None
    split: Literal["train", "held_out"] | None = None
    duration_s: float | None = None


class SpeechExport:
    """A curator speech export at *root*: its rows, with each clip resolved inside *root*."""

    def __init__(self, root: Path, rows: list[SpeechExportRow], clips: list[Path]) -> None:
        self.root = root
        self.rows = rows
        self.clips = clips

    @property
    def manifest(self) -> Path:
        return self.root / SPEECH_MANIFEST_NAME


def read_speech_export(root: Path) -> SpeechExport:
    """Read the speech export at *root* (its ``speech_manifest.jsonl``); :class:`SpeechExportError` if it can't be.

    Every clip must resolve inside *root* (``exported_path`` is relative to the export; an ``exported_abs_path``
    from wherever curator ran is not trusted), and exist.
    """
    from argus_cortex.fs import PathRefused, safe_join

    root = Path(root)
    path = root / SPEECH_MANIFEST_NAME
    if not path.is_file():
        raise SpeechExportError(f"{root.name!r} is not a curator speech export (no {SPEECH_MANIFEST_NAME})")
    rows: list[SpeechExportRow] = []
    clips: list[Path] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except ValueError as exc:
            raise SpeechExportError(f"{SPEECH_MANIFEST_NAME} line {n} is not JSON") from exc
        major = str(raw.get("speech_manifest_version", "")).split(".")[0] if isinstance(raw, dict) else ""
        if major not in SUPPORTED_SPEECH_MANIFEST_MAJORS:
            raise SpeechExportError(
                f"{SPEECH_MANIFEST_NAME} line {n} is speech_manifest_version "
                f"{raw.get('speech_manifest_version') if isinstance(raw, dict) else None!r}; proof reads 1.x"
            )
        try:
            row = SpeechExportRow.model_validate(raw)
        except ValidationError as exc:
            err = exc.errors()[0]
            where = ".".join(str(p) for p in err.get("loc", ()))
            raise SpeechExportError(f"{SPEECH_MANIFEST_NAME} line {n}: {where}: {err.get('msg', '')}") from exc
        try:
            clip = safe_join(root, row.exported_path, label="exported_path")
        except PathRefused as exc:
            raise SpeechExportError(f"{SPEECH_MANIFEST_NAME} line {n}: {exc}") from exc
        if not clip.is_file():
            raise SpeechExportError(f"{SPEECH_MANIFEST_NAME} line {n}: {row.exported_path} is not in the export")
        rows.append(row)
        clips.append(clip)
    if not rows:
        raise SpeechExportError(f"{root.name!r} has no clips in its {SPEECH_MANIFEST_NAME}")
    return SpeechExport(root, rows, clips)


def held_out_references(export: SpeechExport) -> list[SpeechReferenceSpec]:
    """The reference voices of a held-out speech export, ``r0`` … in manifest order.

    Every row must be the ``held_out`` part: a training row in the reference set would be refused by the overlap
    check anyway, but this says what is wrong. At most :data:`~argus_proof.models.MAX_SPEECH_REFERENCES`.
    """
    wrong = [row.exported_path for row in export.rows if row.split != "held_out"]
    if wrong:
        raise SpeechExportError(
            f"{export.root.name!r} is not a held-out export: {len(wrong)} clip(s) are not split 'held_out' "
            f"(first: {wrong[0]}). Export the held-out part to its own dest."
        )
    if len(export.rows) > MAX_SPEECH_REFERENCES:
        raise SpeechExportError(
            f"{export.root.name!r} has {len(export.rows)} reference clips; a speech run takes at most "
            f"{MAX_SPEECH_REFERENCES}"
        )
    return [
        SpeechReferenceSpec(
            ref_id=f"r{i}",
            path=str(clip),
            rel_path=row.exported_path,
            transcript=row.transcript,
            speaker_label=row.speaker_label,
            speaker_cluster=row.speaker_cluster,
            language=row.language,
            accent=row.accent,
        )
        for i, (row, clip) in enumerate(zip(export.rows, export.clips, strict=True))
    ]


def check_speakers_disjoint(references: list[SpeechReferenceSpec], training: SpeechExport) -> str:
    """Refuse *references* whose speaker or clip is in the *training* export; return what was checked.

    Raises :class:`SpeakerOverlap` naming every overlapping reference and why. See the module docstring for what
    each comparison catches and its limits.
    """
    labels = {row.speaker_label for row in training.rows if row.speaker_label}
    clusters = {row.speaker_cluster for row in training.rows if row.speaker_cluster}
    hashes = {sha256_cached(clip): row.exported_path for row, clip in zip(training.rows, training.clips, strict=True)}
    problems: list[str] = []
    for ref in references:
        why: list[str] = []
        if ref.speaker_label and ref.speaker_label in labels:
            why.append(f"speaker {ref.speaker_label!r} is in training")
        if ref.speaker_cluster and ref.speaker_cluster in clusters:
            why.append(f"speaker cluster {ref.speaker_cluster!r} is in training")
        twin = hashes.get(sha256_cached(Path(ref.path)))
        if twin is not None:
            why.append(f"it is training clip {twin}")
        if why:
            problems.append(f"{ref.ref_id} ({ref.rel_path}): {'; '.join(why)}")
    if problems:
        hint = (
            " Cluster ids compare only between the two parts of one curator scan: export both parts from the same scan."
            if any("cluster" in p for p in problems)
            else ""
        )
        raise SpeakerOverlap(
            "held-out references overlap the training speakers, so they would measure memorisation: "
            + "; ".join(problems)
            + "."
            + hint
        )
    return SPEAKER_CHECK


__all__ = [
    "SPEAKER_CHECK",
    "SPEECH_MANIFEST_NAME",
    "SpeakerOverlap",
    "SpeechExport",
    "SpeechExportError",
    "SpeechExportRow",
    "check_speakers_disjoint",
    "held_out_references",
    "read_speech_export",
]
