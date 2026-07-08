from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from test_scoring import manifest  # reuse the RunManifest helper

from argus_proof.models import GeneratedImage
from argus_proof.scoring import score_run
from argus_proof.scoring.scorers import PhashDeduper, PhashDiversityScorer


def _save(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype("uint8")).save(path)


def vsplit() -> np.ndarray:
    a = np.zeros((128, 128, 3), np.uint8)
    a[:, 64:] = 255  # left black, right white
    return a


def checker() -> np.ndarray:
    a = np.zeros((128, 128, 3), np.uint8)
    for i in range(0, 128, 16):
        for j in range(0, 128, 16):
            if (i // 16 + j // 16) % 2 == 0:
                a[i : i + 16, j : j + 16] = 255
    return a


def noise(seed: int) -> np.ndarray:
    # Distinct high-frequency image; pHash-far from the structured patterns above.
    return np.random.default_rng(seed).integers(0, 256, (128, 128, 3), dtype=np.uint8)


def img(path: Path, seed: int) -> GeneratedImage:
    return GeneratedImage(image_id=path.stem, run_id="run-1", seed=seed, path=str(path), width=128, height=128)


def make_images(tmp_path: Path, patterns: list) -> list[GeneratedImage]:
    out = []
    for i, arr in enumerate(patterns):
        p = tmp_path / f"run-1-{i}.png"
        _save(p, arr)
        out.append(img(p, i))
    return out


# --------------------------------------------------------------------------
# deduper
# --------------------------------------------------------------------------


def test_deduper_available() -> None:
    assert PhashDeduper().is_available() is True


def test_deduper_groups_identical_separates_different(tmp_path: Path) -> None:
    # two identical frames + one clearly different (checker is ~18 bits from vsplit) -> [0, 0, 1]
    images = make_images(tmp_path, [vsplit(), vsplit(), checker()])
    labels = PhashDeduper().group(images)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]
    assert sorted(set(labels)) == [0, 1]


def test_deduper_transitive_clustering(tmp_path: Path) -> None:
    # three copies of the same image collapse into one group
    images = make_images(tmp_path, [checker(), checker(), checker()])
    assert PhashDeduper().group(images) == [0, 0, 0]


def test_deduper_empty() -> None:
    assert PhashDeduper().group([]) == []


def test_deduper_labels_are_first_appearance_order(tmp_path: Path) -> None:
    images = make_images(tmp_path, [checker(), vsplit(), checker()])
    labels = PhashDeduper().group(images)
    assert labels[0] == 0  # first image is always group 0
    assert labels[0] == labels[2] and labels[1] == 1


# --------------------------------------------------------------------------
# diversity
# --------------------------------------------------------------------------


def test_diversity_zero_for_identical(tmp_path: Path) -> None:
    images = make_images(tmp_path, [vsplit(), vsplit(), vsplit()])
    assert PhashDiversityScorer().score(images, ctx=None) == 0.0  # type: ignore[arg-type]


def test_diversity_positive_for_varied(tmp_path: Path) -> None:
    images = make_images(tmp_path, [vsplit(), checker(), noise(1)])
    score = PhashDiversityScorer().score(images, ctx=None)  # type: ignore[arg-type]
    assert 0.0 < score <= 1.0


def test_diversity_single_image_is_zero(tmp_path: Path) -> None:
    images = make_images(tmp_path, [vsplit()])
    assert PhashDiversityScorer().score(images, ctx=None) == 0.0  # type: ignore[arg-type]


def test_provenance() -> None:
    assert PhashDeduper().provenance().metric == "duplicate"
    assert PhashDiversityScorer().provenance().metric == "diversity"


# --------------------------------------------------------------------------
# end-to-end with the orchestrator (#6 acceptance: pass-rate over collapsed groups)
# --------------------------------------------------------------------------


def test_score_run_collapses_near_dups_into_one_group(tmp_path: Path) -> None:
    # 3 identical frames + 1 distinct -> 2 groups, not 4
    images = make_images(tmp_path, [vsplit(), vsplit(), vsplit(), checker()])
    report = score_run(manifest(), images, deduper=PhashDeduper(), diversity=PhashDiversityScorer())
    assert report.aggregate.n_images == 4
    assert report.aggregate.n_groups == 2
    assert report.aggregate.diversity is not None
    assert [p.name for p in report.scorers] == ["phash-dedup", "phash-diversity"]
