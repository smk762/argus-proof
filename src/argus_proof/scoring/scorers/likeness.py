"""Likeness scoring — identity for subjects ArcFace can't see (#103).

:class:`~argus_proof.scoring.scorers.identity.IdentityScorer` embeds a human face,
so a helmeted, chrome-skinned or non-human character, or a scene, scores ``0.0``
there (no face detected). :class:`LikenessScorer` scores the same way, cosine
similarity to a **held-out reference set**, but over whole-image DINOv2
embeddings, so it measures "is this the same subject/set?" for anything.

DINOv2 rather than CLIP: CLIP's image embedding is trained to match captions, so
any two chrome robots land close together because both are "a chrome robot".
DINOv2 is self-supervised on images alone and keeps instance-level appearance
(shape, markings, material), which is what telling one character from a
look-alike needs. It loads through ``transformers``, which the ``[score]`` extra
already brings with torch for CLIPScore and pyiqa, so it adds no new stack.

It applies to every target category (identity, wardrobe, setting, …): the
reference set decides what it measures, and with no references it returns
``None``. The raw cosine is clamped to ``[0, 1]`` and is **not calibrated**:
unrelated photos still sit around 0.1-0.4 on DINOv2, so set ``--min-likeness``
from a real run, as with ``--min-identity``.

The embedder is injectable (:class:`~argus_proof.scoring.scorers.identity.Embedder`),
so the scoring logic is tested with a stub; :class:`Dinov2Embedder` is lazy and
behind the ``[score]`` extra. Once it has loaded, the weights file it loaded is
pinned by SHA256 in the scorer's provenance (``ScorerProvenance.model_file``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import structlog

from argus_proof.models import ModelRef, ScorerProvenance
from argus_proof.scoring.scorers._util import module_available
from argus_proof.scoring.scorers.identity import Embedder, IdentityScorer

if TYPE_CHECKING:
    from argus_proof.scoring.base import ScoreContext

logger = structlog.get_logger()

# ViT-B/14, 86M parameters, 768-d CLS embedding: the middle of the family (small is 22M, large 300M). Apache-2.0.
DEFAULT_LIKENESS_MODEL = "facebook/dinov2-base"
# The one weights file a DINOv2 checkpoint on the Hub (or saved with save_pretrained) carries.
WEIGHTS_FILE = "model.safetensors"


class Dinov2Embedder:
    """Whole-image DINOv2 embeddings via ``transformers``: the L2-normalised CLS token.

    *model* is a Hugging Face model id or a local directory saved with
    ``save_pretrained`` (an offline host points ``PROOF_LIKENESS_MODEL`` at one);
    *revision* pins a Hub commit. Always returns a vector: every image has an
    embedding, so "subject absent" shows as a low cosine rather than ``None``.
    """

    def __init__(self, model: str = DEFAULT_LIKENESS_MODEL, revision: str | None = None) -> None:
        self.model = model
        self.revision = revision
        self._processor = None
        self._model = None
        self._device = "cpu"

    @property
    def name(self) -> str:
        return f"dinov2:{self.model}" + (f"@{self.revision}" if self.revision else "")

    def is_available(self) -> bool:
        return module_available("torch", "transformers", "PIL", "numpy")

    def unload(self) -> None:
        """Drop the model and processor; scoring frees them before its GPU lease ends.

        A report's provenance is built inside the pass, while the model is still
        loaded, so :meth:`model_ref` has pinned the weights by then.
        """
        self._processor = None
        self._model = None

    def _load(self):  # noqa: ANN202 - transformers types aren't importable without the extra
        if self._model is None:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = AutoImageProcessor.from_pretrained(self.model, revision=self.revision)
            self._model = AutoModel.from_pretrained(self.model, revision=self.revision).eval().to(self._device)
        return self._processor, self._model

    def embed(self, image_path: Path) -> list[float] | None:
        import torch
        from PIL import Image, ImageOps

        processor, model = self._load()
        with Image.open(image_path) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
        inputs = processor(images=rgb, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            out = model(**inputs)
        # Dinov2Model's pooler_output is the layer-normed CLS token: the image-level embedding.
        vec = torch.nn.functional.normalize(out.pooler_output[0].float(), dim=0)
        return vec.cpu().tolist()

    def weights_file(self) -> Path | None:
        """The weights file this model id resolves to on this host, or ``None``.

        A local directory's ``model.safetensors``; for a Hub id, the cached copy
        (no network). ``None`` when it isn't there, e.g. before the first load.
        """
        local = Path(self.model).expanduser()
        if local.is_dir():
            path = local / WEIGHTS_FILE
            return path if path.is_file() else None
        if not module_available("huggingface_hub"):
            return None
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(self.model, WEIGHTS_FILE, revision=self.revision)
        return Path(cached) if isinstance(cached, str) else None

    def model_ref(self) -> ModelRef | None:
        """The loaded weights pinned by SHA256, or ``None`` if nothing has been loaded.

        Only after a load: the pin names the file that produced the scores, and
        a ``/scorers`` probe never hashes (or downloads) a few hundred MB.
        """
        if self._model is None:
            return None
        path = self.weights_file()
        if path is None:
            logger.warning("likeness.weights_unpinned", model=self.model, reason=f"no {WEIGHTS_FILE} found")
            return None
        from argus_proof.hashing import sha256_cached

        return ModelRef(name=self.name, sha256=sha256_cached(path))


class LikenessScorer(IdentityScorer):
    """Score likeness as whole-image embedding cosine vs a held-out reference set → ``[0, 1]``.

    The ``likeness`` metric. Same reference handling as :class:`IdentityScorer`
    (``aggregate`` ``"max"`` or ``"mean"``, a reference cache keyed by path, size
    and mtime, ``None`` without references, NaN passed to the orchestrator's
    guard), but over DINOv2 by default and for every target category.
    """

    metric = "likeness"

    def __init__(self, embedder: Embedder | None = None, aggregate: Literal["max", "mean"] = "max") -> None:
        super().__init__(embedder=embedder, aggregate=aggregate)

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Dinov2Embedder()
        return self._embedder

    def applies(self, ctx: ScoreContext) -> bool:
        """Every target category: the reference set (a character, a set) decides what is measured."""
        return True

    def provenance(self) -> ScorerProvenance:
        model_ref = getattr(self.embedder, "model_ref", None)
        return ScorerProvenance(
            name="likeness",
            metric="likeness",
            model=self.embedder.name,
            model_file=model_ref() if callable(model_ref) else None,
        )


__all__ = ["DEFAULT_LIKENESS_MODEL", "Dinov2Embedder", "LikenessScorer"]
