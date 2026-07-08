"""Perceptual-hash dedup + diversity — the lightweight Phase 2 scorers (#6).

Near-identical Monte-Carlo outputs shouldn't inflate a pass rate, and a LoRA
that renders the same frame every time should score as low-diversity. Both are
measured from the perceptual hash (pHash) of each image — CPU-only, no torch:

* :class:`PhashDeduper` groups images whose pHashes are within a Hamming
  distance, so a near-dup cluster collapses to one unit for the pass rate.
* :class:`PhashDiversityScorer` scores output variety as the mean pairwise
  Hamming distance (normalised to ``[0, 1]``) — near-zero under mode collapse.

Both need the ``[score]`` extra (Pillow + imagehash); :meth:`is_available`
reports whether it's installed so the orchestrator can skip them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from argus_proof.models import ScorerProvenance

if TYPE_CHECKING:
    from argus_proof.models import GeneratedImage
    from argus_proof.scoring.base import ScoreContext


def _imagehash_available() -> bool:
    try:
        import imagehash  # noqa: F401
        from PIL import Image  # noqa: F401

        return True
    except ImportError:
        return False


class _PhashMixin:
    """Shared pHash computation for the dedup + diversity scorers."""

    hash_size: int = 8  # 8 -> a 64-bit hash

    def is_available(self) -> bool:
        return _imagehash_available()

    def _phashes(self, images: list[GeneratedImage]):
        import imagehash
        from PIL import Image

        hashes = []
        for img in images:
            with Image.open(img.path) as im:
                hashes.append(imagehash.phash(im, hash_size=self.hash_size))
        return hashes

    @property
    def _bits(self) -> int:
        return self.hash_size * self.hash_size


class PhashDeduper(_PhashMixin):
    """Group images whose perceptual hashes are within ``threshold`` bits.

    Connected-components clustering: images A–B and B–C within threshold put all
    three in one group. Group labels are assigned in first-appearance order so
    the result is deterministic. ``threshold`` is a Hamming distance on the
    64-bit hash — ~5 treats visually near-identical frames as duplicates.
    """

    def __init__(self, threshold: int = 5, hash_size: int = 8) -> None:
        self.threshold = threshold
        self.hash_size = hash_size

    def provenance(self) -> ScorerProvenance:
        import imagehash

        return ScorerProvenance(
            name="phash-dedup",
            metric="duplicate",
            version=getattr(imagehash, "__version__", None),
            model=f"phash-{self._bits}bit@{self.threshold}",
        )

    def group(self, images: list[GeneratedImage]) -> list[int]:
        n = len(images)
        if n == 0:
            return []
        hashes = self._phashes(images)

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                if hashes[i] - hashes[j] <= self.threshold:  # imagehash '-' is Hamming distance
                    parent[find(i)] = find(j)

        # Relabel roots to 0,1,2… in first-appearance order for a stable result.
        labels: list[int] = []
        remap: dict[int, int] = {}
        for i in range(n):
            root = find(i)
            if root not in remap:
                remap[root] = len(remap)
            labels.append(remap[root])
        return labels


class PhashDiversityScorer(_PhashMixin):
    """Score output variety as the mean pairwise pHash Hamming distance in ``[0, 1]``.

    High when outputs differ, near zero under mode collapse (a LoRA that renders
    the same image regardless of prompt/seed). Returns ``0.0`` for fewer than two
    images (no variety to measure).
    """

    def __init__(self, hash_size: int = 8) -> None:
        self.hash_size = hash_size

    def provenance(self) -> ScorerProvenance:
        import imagehash

        return ScorerProvenance(
            name="phash-diversity",
            metric="diversity",
            version=getattr(imagehash, "__version__", None),
            model=f"phash-{self._bits}bit",
        )

    def score(self, images: list[GeneratedImage], ctx: ScoreContext) -> float:
        if len(images) < 2:
            return 0.0
        hashes = self._phashes(images)
        total = 0
        pairs = 0
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                total += hashes[i] - hashes[j]
                pairs += 1
        return (total / pairs) / self._bits


__all__ = ["PhashDeduper", "PhashDiversityScorer"]
