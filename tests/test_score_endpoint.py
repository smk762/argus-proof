"""POST /score (#103, argus-mint#51): caller-supplied images scored against an export's held-out references.

Stub embedders only: no model download, no GPU.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from argus_cortex.gpu import GpuLeaseRefused  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402
from test_evaluate import _photo  # noqa: E402
from test_gpu import Recording  # noqa: E402

import argus_proof.evaluate as evaluate  # noqa: E402
from argus_proof import gpu  # noqa: E402
from argus_proof.models import PROOF_VERSION, ModelRef, ScoreResult  # noqa: E402
from argus_proof.scoring.scorers import IdentityScorer, LikenessScorer  # noqa: E402
from argus_proof.server import create_app  # noqa: E402

RED, BLUE = (220, 20, 20), (20, 20, 220)


class ColourEmbedder:
    """Embeds an image as its mean colour, so a crop changes the score the way it would on DINOv2."""

    name = "stub-dinov2"

    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.embedded: list[str] = []
        self.loaded = False
        self.unloads = 0

    def is_available(self) -> bool:
        return self.available

    def embed(self, image_path: Path) -> list[float]:
        self.loaded = True
        self.embedded.append(Path(image_path).name)
        with Image.open(image_path) as im:
            r, g, b = im.convert("RGB").resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
        return [float(r), float(g), float(b)]

    def model_ref(self) -> ModelRef | None:
        return ModelRef(name=self.name, sha256="cd" * 32) if self.loaded else None

    def unload(self) -> None:
        self.loaded = False
        self.unloads += 1


def png(*colours: tuple[int, int, int], size: int = 64) -> bytes:
    """An image split into vertical bands, one per colour, left to right."""
    im = Image.new("RGB", (size * len(colours), size))
    for i, colour in enumerate(colours):
        im.paste(colour, (i * size, 0, (i + 1) * size, size))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def dirs(tmp_path: Path) -> dict[str, Path]:
    export = tmp_path / "exports" / "android"
    (export / "references").mkdir(parents=True)
    (export / "references" / "ref.png").write_bytes(png(RED))
    _photo(export / "train" / "t.png", seed=1)  # a training image, far from the reference under pHash
    return {"exports": tmp_path / "exports", "reports": tmp_path / "reports", "runs": tmp_path / "runs"}


@pytest.fixture
def embedder(monkeypatch: pytest.MonkeyPatch) -> ColourEmbedder:
    emb = ColourEmbedder()
    built: dict[str, object] = {
        "likeness": lambda: LikenessScorer(embedder=emb),
        "identity": lambda: IdentityScorer(embedder=ColourEmbedder()),
    }
    monkeypatch.setattr(evaluate, "reference_scorers", lambda names: [built[n]() for n in names])
    return emb


@pytest.fixture
def coordinator(monkeypatch: pytest.MonkeyPatch) -> Recording:
    rec = Recording()
    monkeypatch.setattr(gpu, "default_coordinator", lambda: rec)
    return rec


@pytest.fixture
def client(dirs: dict[str, Path]) -> TestClient:
    return TestClient(
        create_app(reports_dir=str(dirs["reports"]), runs_dir=str(dirs["runs"]), exports_dir=str(dirs["exports"]))
    )


def post(client: TestClient, request: dict, *images: tuple[str, bytes]):  # noqa: ANN201
    return client.post(
        "/score",
        data={"request": json.dumps(request)},
        files=[("images", (name, data, "image/png")) for name, data in images],
    )


# --------------------------------------------------------------------------
# the happy path, and the shape mint codes against
# --------------------------------------------------------------------------


def test_score_returns_per_image_metrics_means_and_pinned_provenance(
    client: TestClient, embedder: ColourEmbedder, coordinator: Recording, dirs: dict[str, Path]
) -> None:
    resp = post(
        client,
        {"export": "android", "scorers": ["likeness"], "target_category": "identity"},
        ("teaser-1.png", png(RED)),
        ("teaser-2.png", png(BLUE)),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    ScoreResult.model_validate(body)
    assert body["proof_version"] == PROOF_VERSION == "1.11"
    assert body["n_images"] == 2
    [same, other] = body["images"]
    assert (same["index"], same["filename"], same["crop"]) == (0, "teaser-1.png", None)
    assert same["metrics"]["likeness"] == pytest.approx(1.0)
    assert other["metrics"]["likeness"] < 0.5
    assert same["metrics"]["identity"] is None  # not requested
    assert body["means"]["likeness"] == pytest.approx((same["metrics"]["likeness"] + other["metrics"]["likeness"]) / 2)
    likeness, held_out = body["scorers"]
    assert (likeness["name"], likeness["model"]) == ("likeness", "stub-dinov2")
    assert likeness["model_file"] == {"name": "stub-dinov2", "sha256": "cd" * 32}  # pinned while loaded
    assert (held_out["name"], held_out["model"]) == ("held-out-references", "sha256+phash<=6")
    assert not dirs["reports"].exists() or not any(dirs["reports"].iterdir())  # nothing stored


def test_a_crop_scores_the_character_not_the_background(
    client: TestClient, embedder: ColourEmbedder, coordinator: Recording
) -> None:
    teaser = png(BLUE, RED, BLUE)  # a red character between blue background bands, 192x64
    crop = {"x": 64, "y": 0, "width": 64, "height": 64}
    resp = post(
        client,
        {"export": "android", "scorers": ["likeness"], "crops": [crop, None]},
        ("cropped.png", teaser),
        ("whole.png", teaser),
    )
    assert resp.status_code == 200, resp.text
    cropped, whole = resp.json()["images"]
    assert cropped["crop"] == crop and whole["crop"] is None
    assert cropped["metrics"]["likeness"] == pytest.approx(1.0)
    assert whole["metrics"]["likeness"] < cropped["metrics"]["likeness"]


def test_identity_and_likeness_together_and_identity_needs_an_identity_run(
    client: TestClient, embedder: ColourEmbedder, coordinator: Recording
) -> None:
    for category, identity_measured in (("identity", True), ("setting", False)):
        resp = post(
            client,
            {"export": "android", "scorers": ["identity", "likeness"], "target_category": category},
            ("t.png", png(RED)),
        )
        assert resp.status_code == 200, resp.text
        [image] = resp.json()["images"]
        assert (image["metrics"]["identity"] is not None) is identity_measured
        assert image["metrics"]["likeness"] == pytest.approx(1.0)
        assert [s["metric"] for s in resp.json()["scorers"]] == ["identity", "likeness", "identity"]


# --------------------------------------------------------------------------
# the GPU lease
# --------------------------------------------------------------------------


def test_score_holds_one_interactive_score_lease_and_frees_the_models(
    client: TestClient, embedder: ColourEmbedder, coordinator: Recording
) -> None:
    assert post(client, {"export": "android", "scorers": ["likeness"]}, ("t.png", png(RED))).status_code == 200
    [call] = coordinator.calls
    assert (call["kind"], call["target"], call["priority"], call["preemptible"]) == (
        "score",
        "in-process",
        "interactive",
        False,
    )
    assert call["caller"].startswith("score-")
    assert embedder.unloads == 1 and embedder.loaded is False
    assert coordinator.exits == [None]


def test_a_refused_lease_is_503_and_scores_nothing(
    client: TestClient, embedder: ColourEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gpu, "default_coordinator", lambda: Recording(raises=GpuLeaseRefused("no GPU free")))
    resp = post(client, {"export": "android", "scorers": ["likeness"]}, ("t.png", png(RED)))
    assert resp.status_code == 503
    assert "refused the lease" in resp.json()["detail"]
    assert embedder.embedded == []


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_references_that_are_training_images_are_refused_before_scoring(
    client: TestClient, embedder: ColourEmbedder, coordinator: Recording, dirs: dict[str, Path]
) -> None:
    export = dirs["exports"] / "android"
    (export / "train" / "copy.png").write_bytes((export / "references" / "ref.png").read_bytes())
    resp = post(client, {"export": "android", "scorers": ["likeness"]}, ("t.png", png(RED)))
    assert resp.status_code == 400
    assert "training images" in resp.json()["detail"]
    assert embedder.embedded == [] and coordinator.calls == []


def test_an_unavailable_scorer_is_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluate, "reference_scorers", lambda names: [LikenessScorer(embedder=ColourEmbedder(available=False))]
    )
    resp = post(client, {"export": "android", "scorers": ["likeness"]}, ("t.png", png(RED)))
    assert resp.status_code == 503
    assert "likeness" in resp.json()["detail"]


def test_a_scorer_breaking_its_range_is_a_500_with_a_reason(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, coordinator: Recording
) -> None:
    class NanScorer(LikenessScorer):
        def score(self, image_path, ctx):  # noqa: ANN001, ANN201
            return float("nan")

    monkeypatch.setattr(evaluate, "reference_scorers", lambda names: [NanScorer(embedder=ColourEmbedder())])
    resp = TestClient(client.app, raise_server_exceptions=False).post(
        "/score",
        data={"request": json.dumps({"export": "android", "scorers": ["likeness"]})},
        files=[("images", ("t.png", png(RED), "image/png"))],
    )
    assert resp.status_code == 500
    assert "scoring failed" in resp.json()["detail"]


def test_read_only_refuses_score(dirs: dict[str, Path], embedder: ColourEmbedder) -> None:
    ro = TestClient(create_app(reports_dir=str(dirs["reports"]), exports_dir=str(dirs["exports"]), read_only=True))
    resp = post(ro, {"export": "android", "scorers": ["likeness"]}, ("t.png", png(RED)))
    assert resp.status_code == 403
    assert embedder.embedded == []
    assert "score" not in ro.get("/health").json()["capabilities"]


def test_health_advertises_score(client: TestClient) -> None:
    assert "score" in client.get("/health").json()["capabilities"]


@pytest.mark.parametrize(
    ("request_body", "images", "status", "detail"),
    [
        pytest.param({"export": "ghost", "scorers": ["likeness"]}, 1, 404, "no export named", id="unknown-export"),
        pytest.param({"export": "../etc", "scorers": ["likeness"]}, 1, 400, "invalid export", id="traversal"),
        pytest.param({"export": "android", "scorers": []}, 1, 422, "invalid request", id="no-scorer"),
        pytest.param(
            {"export": "android", "scorers": ["clip_score"]}, 1, 422, "invalid request", id="not-a-ref-scorer"
        ),
        pytest.param({"export": "android", "scorers": ["likeness", "likeness"]}, 1, 422, "repeats", id="repeated"),
        pytest.param(
            {"export": "android", "scorers": ["likeness"], "crops": [None, None]}, 1, 400, "one per image", id="crops"
        ),
        pytest.param(
            {"export": "android", "scorers": ["likeness"], "crops": [{"x": 40, "y": 0, "width": 64, "height": 64}]},
            1,
            400,
            "outside",
            id="crop-outside",
        ),
        pytest.param({"export": "android", "scorers": ["likeness"]}, 17, 400, "1 to 16 images", id="too-many"),
    ],
)
def test_bad_requests(
    client: TestClient,
    embedder: ColourEmbedder,
    coordinator: Recording,
    request_body: dict,
    images: int,
    status: int,
    detail: str,
) -> None:
    resp = post(client, request_body, *[(f"{i}.png", png(RED)) for i in range(images)])
    assert resp.status_code == status, resp.text
    assert detail in resp.json()["detail"]
    assert embedder.embedded == []


def test_not_an_image_is_400(client: TestClient, embedder: ColourEmbedder, coordinator: Recording) -> None:
    resp = post(client, {"export": "android", "scorers": ["likeness"]}, ("t.png", b"not an image"))
    assert resp.status_code == 400
    assert "image 0: not a readable image" in resp.json()["detail"]


def test_an_export_without_references_is_400(
    client: TestClient, embedder: ColourEmbedder, coordinator: Recording, dirs: dict[str, Path]
) -> None:
    (dirs["exports"] / "android" / "references" / "ref.png").unlink()
    resp = post(client, {"export": "android", "scorers": ["likeness"]}, ("t.png", png(RED)))
    assert resp.status_code == 400
    assert "no references/ images" in resp.json()["detail"]


def test_invalid_json_request_is_422(client: TestClient) -> None:
    resp = client.post("/score", data={"request": "{"}, files=[("images", ("t.png", png(RED), "image/png"))])
    assert resp.status_code == 422


# --------------------------------------------------------------------------
# the crop helper
# --------------------------------------------------------------------------


def test_a_crop_is_in_upright_pixels(tmp_path: Path) -> None:
    from argus_proof.models import CropBox

    im = Image.new("RGB", (40, 20), BLUE)
    im.paste(RED, (0, 0, 20, 20))  # stored: red on the left
    exif = Image.Exif()
    exif[0x0112] = 6  # displayed rotated 90 CW: 20 wide, 40 tall, red on top
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif, quality=95)
    dest = tmp_path / "out.png"
    size = evaluate.prepare_scoring_image(buf.getvalue(), CropBox(x=0, y=0, width=20, height=20), dest, max_side=2048)
    assert size == (20, 20)
    with Image.open(dest) as out:
        r, g, b = out.convert("RGB").getpixel((10, 10))
    assert r > 150 and b < 100  # the top square upright is the red half


def test_an_oversized_image_is_refused(tmp_path: Path) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (300, 10)).save(buf, "PNG")
    with pytest.raises(evaluate.EvaluateError, match="at most 256 px"):
        evaluate.prepare_scoring_image(buf.getvalue(), None, tmp_path / "o.png", max_side=256)


def test_an_upload_filename_is_echoed_never_used_as_a_path(
    client: TestClient, embedder: ColourEmbedder, coordinator: Recording, tmp_path: Path
) -> None:
    name = f"../../{tmp_path.name}-evil.png"
    resp = post(client, {"export": "android", "scorers": ["likeness"]}, (name, png(RED)))
    assert resp.status_code == 200, resp.text
    assert resp.json()["images"][0]["filename"] == name
    assert embedder.embedded[-1] == "0.png"  # scored from the temp dir by index
    assert not (tmp_path.parent / f"{tmp_path.name}-evil.png").exists()
