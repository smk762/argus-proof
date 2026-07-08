"""Identity scoring — "did the subject transfer?" (#4).

For an ``identity`` run, score each generated face by its ArcFace-embedding
cosine similarity to a **held-out reference set** of the subject (references that
were NOT in the training set — otherwise you measure memorisation, not likeness;
that contract lives on :class:`~argus_proof.scoring.base.ScoreContext`).

The embedding backend is injectable (:class:`Embedder`) so the scoring logic —
reference aggregation, normalisation, the no-face and no-reference cases — is
tested without a model; the real :class:`InsightFaceEmbedder` (buffalo_l, the
curator's stack) sits behind the ``[score]`` extra and is loaded lazily.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from argus_proof.models import ScorerProvenance

if TYPE_CHECKING:
    from argus_proof.scoring.base import ScoreContext


@runtime_checkable
class Embedder(Protocol):
    """Turns an image into a face/identity embedding, or ``None`` if none is found."""

    name: str

    def is_available(self) -> bool: ...
    def embed(self, image_path: Path) -> list[float] | None: ...


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors; 0.0 if either is a zero vector."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class IdentityScorer:
    """Score identity as embedding cosine vs a held-out reference set → ``[0, 1]``.

    Only applies to ``identity`` runs with a reference set; returns ``None``
    otherwise (the metric simply isn't produced). ``aggregate`` picks how the
    per-reference similarities combine: ``"max"`` (best-matching reference, the
    default) or ``"mean"`` (average — steadier against a noisy reference). A
    generated image with no detectable face scores ``0.0`` (identity absent).
    Cosine is clamped to ``[0, 1]``.
    """

    metric = "identity"

    def __init__(self, embedder: Embedder | None = None, aggregate: Literal["max", "mean"] = "max") -> None:
        self._embedder = embedder
        self.aggregate = aggregate
        self._ref_cache: dict[tuple[str, ...], list[list[float]]] = {}

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = InsightFaceEmbedder()
        return self._embedder

    def provenance(self) -> ScorerProvenance:
        return ScorerProvenance(name="identity", metric="identity", model=self.embedder.name)

    def is_available(self) -> bool:
        return self.embedder.is_available()

    def score(self, image_path: Path, ctx: ScoreContext) -> float | None:
        if ctx.profile.target_category != "identity":
            return None  # identity scoring only applies to identity runs
        refs = self._reference_vectors(ctx)
        if not refs:
            return None  # no held-out references -> identity can't be measured
        vec = self.embedder.embed(image_path)
        if vec is None:
            return 0.0  # no face detected -> the subject isn't present
        sims = [cosine(vec, ref) for ref in refs]
        agg = max(sims) if self.aggregate == "max" else sum(sims) / len(sims)
        return max(0.0, min(1.0, agg))

    def _reference_vectors(self, ctx: ScoreContext) -> list[list[float]]:
        """Embed the reference set once per run (cached by the reference paths)."""
        key = tuple(str(p) for p in ctx.reference_images)
        if key not in self._ref_cache:
            self._ref_cache[key] = [v for p in ctx.reference_images if (v := self.embedder.embed(p)) is not None]
        return self._ref_cache[key]


class InsightFaceEmbedder:
    """ArcFace embeddings via InsightFace ``buffalo_l`` (the curator's face stack).

    Detects faces, embeds the largest one, and returns its L2-normalised vector.
    Heavy (onnxruntime + models); lives behind the ``[score]`` extra and is
    loaded lazily on first use.
    """

    name = "insightface-buffalo_l"

    def __init__(self, model_name: str = "buffalo_l", det_size: tuple[int, int] = (640, 640)) -> None:
        self.model_name = model_name
        self.det_size = det_size
        self._app = None

    def is_available(self) -> bool:
        try:
            import insightface  # noqa: F401
            from PIL import Image  # noqa: F401

            return True
        except ImportError:
            return False

    def _load(self):  # noqa: ANN202 - insightface types aren't importable without the extra
        if self._app is None:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name=self.model_name)
            app.prepare(ctx_id=0, det_size=self.det_size)
            self._app = app
        return self._app

    def embed(self, image_path: Path) -> list[float] | None:
        import numpy as np
        from PIL import Image

        with Image.open(image_path) as im:
            rgb = np.array(im.convert("RGB"))
        bgr = rgb[:, :, ::-1]  # InsightFace expects BGR (OpenCV convention)
        faces = self._load().get(bgr)
        if not faces:
            return None
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return largest.normed_embedding.tolist()


__all__ = ["Embedder", "IdentityScorer", "InsightFaceEmbedder", "cosine"]
