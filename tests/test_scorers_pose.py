"""The run-level pose scorer: argus-puppet over HTTP, through a fake transport (#122).

Puppet's answers are small payloads hand-built from its wire schema (tests/fixtures/puppet/); nothing here imports
argus-puppet or reaches a network.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fakebackend import FakeBackend, save_png
from fakepuppet import HOST, PORT, URL, FakePuppet, payload, resolve

import argus_proof.evaluate as evaluate
from argus_proof.hashing import sha256_file
from argus_proof.models import GeneratedImage, LoRASpec, RunSpec, SamplingParams
from argus_proof.reports import HitlImageUpdate, HitlRequest, apply_hitl
from argus_proof.scoring.scorers import pose as pose_mod
from argus_proof.scoring.scorers.pose import PoseScorer, PuppetClient, PuppetError

SAMPLING = SamplingParams(sampler="euler", scheduler="normal", steps=8, cfg=5.0, width=16, height=16)


def _images(tmp_path: Path, n: int = 12) -> list[GeneratedImage]:
    out = []
    for i in range(1, n + 1):
        path = tmp_path / "run" / f"run-1-{i}.png"
        save_png(path, color=(i * 10, 30, 30))
        out.append(GeneratedImage(image_id=f"run-1-{i}", run_id="run-1", seed=i, path=str(path), width=16, height=16))
    return out


def _training(tmp_path: Path, n: int = 3) -> tuple[Path, list[Path]]:
    export = tmp_path / "export"
    frames = []
    for i in range(n):
        save_png(export / "train" / f"f{i}.png", color=(30, i * 20, 30))
        frames.append(export / "train" / f"f{i}.png")
    (export / "manifest.jsonl").write_text("", encoding="utf-8")
    return export, frames


def _run_dir(tmp_path: Path, *, seeds: int = 12, source_manifest: str | None = None) -> Path:
    spec = RunSpec(
        run_id="run-1",
        base_checkpoint="sdxl.safetensors",
        loras=[LoRASpec(name="subject.safetensors")],
        sampling=SAMPLING,
        prompt="a basketball player",
        seeds=list(range(1, seeds + 1)),
        source_manifest=source_manifest,
    )
    FakeBackend().generate(spec, tmp_path / "run-1")
    return tmp_path / "run-1"


def _use(monkeypatch: pytest.MonkeyPatch, fake: FakePuppet) -> None:
    client = fake.client()
    monkeypatch.setattr(evaluate, "default_pose_scorer", lambda: PoseScorer(client))


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------


def test_the_pose_block_maps_puppets_answers_back_to_proofs_image_ids(tmp_path: Path) -> None:
    fake = FakePuppet()
    images = _images(tmp_path)
    export, frames = _training(tmp_path)
    pose = PoseScorer(fake.client()).score(images, training=frames, export_dir=export)

    assert (pose.count, pose.posed, pose.usable, pose.partial) == (12, 12, 12, False)
    assert (pose.diversity, pose.comparable_share, pose.largest_cluster_share, pose.cluster_count) == (
        0.62,
        1.0,
        0.5,
        3,
    )
    assert pose.largest_cluster_medoids == ["run-1-3"]  # out-00003.png, mapped back
    assert pose.coverage is not None and pose.coverage.stance["jumping"] == 3
    prov = pose.provenance
    assert (prov.puppet_version, prov.package_version) == ("1.1", "0.2.0")
    assert (prov.pose_model, prov.detector_model) == ("dwpose-l-384x288", "yolox_m")
    assert prov.params["cluster_threshold"] == 0.25 and prov.training == "compared: 3 training frame(s)"
    assert prov.not_checked == []

    cmp = pose.comparison
    assert cmp is not None
    assert (cmp.close_share, cmp.baseline_close_share, cmp.close_excess, cmp.close_eligible) == (
        0.1667,
        0.0,
        0.1667,
        12,
    )
    assert cmp.output_nn_distance is not None and cmp.output_nn_distance.p50 == 0.12
    assert cmp.coverage_divergence == 0.18 and cmp.threshold == 0.04 and cmp.n_close == 2
    # closest first, named by proof's image_id and the frame's path in the export, pinned by SHA256
    assert [(p.image_id, p.training_frame) for p in cmp.close_pairs] == [
        ("run-1-2", "train/f1.png"),
        ("run-1-5", "train/f1.png"),
    ]
    assert cmp.close_pairs[0].training_sha256 == sha256_file(frames[1])

    assert fake.paths() == ["/health"] + ["/pose"] * 12 + ["/stats"] + ["/pose"] * 3 + ["/compare", "/stats"]
    names = [r.url.params["name"] for r in fake.requests if r.url.path == "/pose"]
    assert names[:2] == ["out-00001.png", "out-00002.png"] and names[-1] == "ref-00003.png"


def test_only_the_primary_person_is_forwarded(tmp_path: Path) -> None:
    fake = FakePuppet()  # the fixture's PoseResult has two people, the second primary
    PoseScorer(fake.client()).score(_images(tmp_path, 2))
    [stats_body] = fake.bodies("/stats")
    for result in stats_body["results"]:
        assert result["primary"] == 0 and len(result["people"]) == 1 and result["people"][0]["primary"] is True
        assert result["people"][0]["det_score"] == 0.93  # the primary one
    assert [r["image"] for r in stats_body["results"]] == ["out-00001.png", "out-00002.png"]


def test_the_pass_priority_is_forwarded_as_x_argus_priority(tmp_path: Path) -> None:
    fake = FakePuppet()
    PoseScorer(fake.client()).score(_images(tmp_path, 1), priority="batch")
    assert {r.headers["x-argus-priority"] for r in fake.requests} == {"batch"}


def test_score_images_fills_report_pose_and_names_puppet_in_the_scorers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export, _ = _training(tmp_path)
    fake = FakePuppet()
    _use(monkeypatch, fake)
    report = evaluate.score_run_dir(_run_dir(tmp_path, source_manifest=str(export / "manifest.jsonl")))
    assert report.pose is not None and report.pose.usable == 12 and report.pose.comparison is not None
    assert report.pose.largest_cluster_medoids == ["run-1-3"]
    [row] = [s for s in report.scorers if s.metric == "pose"]
    assert (row.name, row.version, row.model) == ("argus-puppet", "0.2.0", "dwpose-l-384x288+yolox_m")
    assert {r.headers["x-argus-priority"] for r in fake.requests} == {"interactive"}  # score's pass priority


# --------------------------------------------------------------------------
# degrading: unset, down, busy, partial, unreadable export
# --------------------------------------------------------------------------


def test_without_a_url_pose_is_unavailable_and_the_report_has_no_pose_block(tmp_path: Path) -> None:
    scorer = evaluate.default_pose_scorer()
    assert not scorer.is_available()
    assert "pose" in evaluate.unavailable_metrics()
    assert "pose" in {s.provenance().metric for s in evaluate.all_reporting_scorers()}
    report = evaluate.score_run_dir(_run_dir(tmp_path))
    assert report.pose is None and not any(s.metric == "pose" for s in report.scorers)


def test_a_url_turns_pose_on_and_a_bad_one_leaves_it_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(pose_mod.ENV_PUPPET_URL, "http://puppet:8220")
    assert evaluate.default_pose_scorer().is_available()
    monkeypatch.setenv(pose_mod.ENV_PUPPET_URL, "ftp://puppet")
    assert not evaluate.default_pose_scorer().is_available()


def test_puppet_down_never_fails_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakePuppet()
    fake.down = True
    _use(monkeypatch, fake)
    report = evaluate.score_run_dir(_run_dir(tmp_path))  # no exception
    pose = report.pose
    assert pose is not None and (pose.count, pose.posed) == (12, 0)
    assert pose.usable is None and pose.diversity is None and pose.largest_cluster_share is None
    assert "could not be reached" in pose.provenance.not_checked[0]


def test_puppet_going_down_mid_pass_stops_and_marks_the_rest_failed(tmp_path: Path) -> None:
    fake = FakePuppet()
    fake.script["/pose"] = [httpx.Response(200, json={**payload("pose"), "image": "out-00001.png"})]
    original = fake.handler

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/pose" and not fake.script["/pose"]:
            raise httpx.ConnectError("connection refused", request=request)
        return original(request)

    fake.handler = handler  # type: ignore[method-assign]
    pose = PoseScorer(fake.client()).score(_images(tmp_path, 4))
    assert (pose.posed, pose.partial, pose.usable) == (1, True, None)
    assert pose.failed_images == ["run-1-2", "run-1-3", "run-1-4"]
    assert any("stopped" in n for n in pose.provenance.not_checked)
    assert "/stats" not in fake.paths()


def test_a_busy_503_is_retried_honouring_retry_after(tmp_path: Path) -> None:
    fake = FakePuppet()
    busy = httpx.Response(503, headers={"Retry-After": "1"}, json={"detail": "busy: try again"})
    fake.script["/pose"] = [busy, httpx.Response(503, headers={"Retry-After": "5"}, json={"detail": "GPU busy"})]
    pose = PoseScorer(fake.client()).score(_images(tmp_path, 12))
    assert fake.sleeps == [1.0, 5.0]
    assert (pose.posed, pose.usable, pose.partial) == (12, 12, False)


def test_retry_after_is_capped_and_a_missing_one_backs_off() -> None:
    assert PuppetClient._retry_after({"Retry-After": "3600"}, 0) == pose_mod.MAX_RETRY_AFTER_S
    assert [PuppetClient._retry_after({}, a) for a in range(3)] == [1.0, 2.0, 4.0]


def test_still_busy_after_the_retries_stops_the_pass_without_failing_the_run(tmp_path: Path) -> None:
    fake = FakePuppet()
    fake.script["/pose"] = [httpx.Response(503, json={"detail": "busy: try again"}) for _ in range(10)]
    pose = PoseScorer(fake.client()).score(_images(tmp_path, 3))
    assert len(fake.sleeps) == pose_mod.MAX_RETRIES
    assert (pose.posed, pose.usable) == (0, None) and len(pose.failed_images) == 3
    assert any("HTTP 503" in n for n in pose.provenance.not_checked)


def test_a_partial_set_is_recorded_as_partial(tmp_path: Path) -> None:
    fake = FakePuppet()
    fake.fail_pose = {"out-00002.png", "out-00007.png"}
    pose = PoseScorer(fake.client()).score(_images(tmp_path, 12))
    assert (pose.count, pose.posed, pose.partial) == (12, 10, True)
    assert pose.failed_images == ["run-1-2", "run-1-7"]
    assert "2 image(s) not posed" in pose.provenance.not_checked[0]
    assert len(fake.bodies("/stats")[0]["results"]) == 10


def test_an_unreadable_export_is_not_compared_and_says_why(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakePuppet()
    _use(monkeypatch, fake)
    report = evaluate.score_run_dir(_run_dir(tmp_path, source_manifest=str(tmp_path / "elsewhere" / "manifest.jsonl")))
    assert report.pose is not None and report.pose.comparison is None
    assert report.pose.provenance.training == "not checked: training export not readable here"
    assert "/compare" not in fake.paths() and report.pose.diversity == 0.62


def test_no_source_manifest_is_not_compared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakePuppet()
    _use(monkeypatch, fake)
    report = evaluate.score_run_dir(_run_dir(tmp_path))
    assert report.pose is not None and report.pose.provenance.training == "not checked: no training export"


def test_a_large_training_set_is_sampled_deterministically(tmp_path: Path) -> None:
    fake = FakePuppet()
    export, frames = _training(tmp_path, 5)
    pose = PoseScorer(fake.client(), max_training_frames=2).score(
        _images(tmp_path, 2), training=frames, export_dir=export
    )
    sent = [r.url.params["name"] for r in fake.requests if r.url.params.get("name", "").startswith("ref-")]
    assert sent == ["ref-00001.png", "ref-00002.png"]
    assert pose.provenance.training == "compared: 2 training frame(s) sampled evenly from 5"
    assert pose_mod._evenly(list("abcde"), 2) == ["a", "c"]


def test_close_pairs_are_capped_but_counted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pose_mod, "POSE_MAX_CLOSE_PAIRS", 1)
    fake = FakePuppet()
    export, frames = _training(tmp_path)
    cmp = PoseScorer(fake.client()).score(_images(tmp_path), training=frames, export_dir=export).comparison
    assert cmp is not None and cmp.n_close == 2 and [p.image_id for p in cmp.close_pairs] == ["run-1-2"]


def test_a_training_frame_outside_the_export_is_named_by_file_name_only(tmp_path: Path) -> None:
    fake = FakePuppet()
    export, frames = _training(tmp_path)
    outside = tmp_path / "dataset" / "secret-dir" / "f1.png"  # sorts first: ref-00001
    save_png(outside)
    for i in (1, 4):
        fake.compare["matches"][i]["nearest"] = "ref-00001.png"
    cmp = PoseScorer(fake.client()).score(
        _images(tmp_path), training=[frames[0], outside, frames[2]], export_dir=export
    )
    assert cmp.comparison is not None
    assert {p.training_frame for p in cmp.comparison.close_pairs} == {"f1.png"}


# --------------------------------------------------------------------------
# the wire: unknown fields, majors, never a made-up 0.0
# --------------------------------------------------------------------------


def test_unknown_puppet_fields_are_ignored_and_forwarded_untouched(tmp_path: Path) -> None:
    fake = FakePuppet()
    fake.health["gpu_tenants"] = ["pose"]
    fake.pose["future"] = {"x": 1}
    fake.pose["people"][1]["mesh"] = [1, 2, 3]
    fake.stats["entropy"] = 0.7
    fake.stats["clusters"][0]["spread"] = 0.1
    fake.compare["matches"][1]["iou"] = 0.5
    export, frames = _training(tmp_path)
    pose = PoseScorer(fake.client()).score(_images(tmp_path), training=frames, export_dir=export)
    assert pose.diversity == 0.62 and pose.comparison is not None and pose.comparison.n_close == 2
    forwarded = fake.bodies("/stats")[0]["results"][0]
    assert forwarded["future"] == {"x": 1} and forwarded["people"][0]["mesh"] == [1, 2, 3]


def test_a_puppet_1_0_answer_is_read(tmp_path: Path) -> None:
    fake = FakePuppet()
    for doc in (fake.health, fake.pose, fake.stats):
        doc["puppet_version"] = "1.0"
    for person in fake.pose["people"]:
        del person["labels"]  # 1.0 had none
    assert PoseScorer(fake.client()).score(_images(tmp_path)).usable == 12


@pytest.mark.parametrize("doc", ["health", "pose", "stats"])
def test_a_wrong_puppet_major_is_refused_and_nothing_is_measured(tmp_path: Path, doc: str) -> None:
    fake = FakePuppet()
    getattr(fake, doc)["puppet_version"] = "2.0"
    pose = PoseScorer(fake.client()).score(_images(tmp_path))
    assert pose.usable is None and pose.diversity is None
    assert any("puppet_version '2.0'" in n for n in pose.provenance.not_checked)
    if doc == "health":
        assert fake.paths() == ["/health"]


def test_pose_never_fabricates_a_zero(tmp_path: Path) -> None:
    # puppet reads 0 for diversity under two usable poses, and 0 close share with none eligible: proof reads None.
    fake = FakePuppet()
    fake.stats.update(usable=1, comparable_pairs=0, comparable_share=0.0, diversity=0.0, mean_pairwise_distance=0.0)
    fake.compare.update(close_eligible=0, close_share=0.0, baseline_close_share=None, close_excess=None)
    fake.compare["output_nn_distance"] = None
    export, frames = _training(tmp_path)
    pose = PoseScorer(fake.client()).score(_images(tmp_path), training=frames, export_dir=export)
    assert (pose.usable, pose.diversity, pose.comparable_share, pose.mean_pairwise_distance) == (1, None, None, None)
    assert pose.comparison is not None and pose.comparison.close_share is None
    assert pose.comparison.close_excess is None and pose.comparison.output_nn_distance is None

    fake = FakePuppet()
    fake.stats.update(usable=0, comparable_pairs=0, comparable_share=0.0, diversity=0.0, largest_cluster_share=0.0)
    fake.stats.update(cluster_count=0, clusters=[])
    pose = PoseScorer(fake.client()).score(_images(tmp_path))
    assert (pose.diversity, pose.largest_cluster_share, pose.cluster_count, pose.coverage) == (None, None, None, None)


# --------------------------------------------------------------------------
# the URL is only the configured one
# --------------------------------------------------------------------------


def test_the_puppet_url_is_only_the_configured_one(tmp_path: Path) -> None:
    fake = FakePuppet()
    export, frames = _training(tmp_path)
    PoseScorer(fake.client()).score(_images(tmp_path), training=frames, export_dir=export)
    assert fake.requests and {r.headers["host"] for r in fake.requests} == {f"{HOST}:{PORT}"}


@pytest.mark.parametrize("path", ["/health", "/pose"])
def test_a_redirect_is_not_followed(tmp_path: Path, path: str) -> None:
    fake = FakePuppet()
    fake.script[path] = [httpx.Response(307, headers={"Location": "http://elsewhere.test/steal"})]
    pose = PoseScorer(fake.client()).score(_images(tmp_path, 2))
    assert {r.headers["host"] for r in fake.requests} == {f"{HOST}:{PORT}"}  # never elsewhere.test
    assert pose.usable is None and any("redirect" in n for n in pose.provenance.not_checked)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://puppet:8220",
        "http://user:pw@puppet:8220",
        "http://puppet:8220/?next=x",
        "http://puppet:8220/#f",
        "puppet",
    ],
)
def test_only_a_plain_http_origin_is_accepted(url: str) -> None:
    with pytest.raises(ValueError):
        pose_mod.puppet_origin(url)


def test_another_host_is_refused_by_the_fetch_policy() -> None:
    # The client allows exactly the configured host: a private address under any other name stays refused.
    client = PuppetClient(URL, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})), resolve=resolve)
    client.base = "http://other.test:8220"  # as if a path were steered elsewhere
    with pytest.raises(PuppetError, match="could not be reached"):
        client.health()


# --------------------------------------------------------------------------
# the report keeps it
# --------------------------------------------------------------------------


def test_a_hitl_re_roll_keeps_the_pose_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use(monkeypatch, FakePuppet())
    report = evaluate.score_run_dir(_run_dir(tmp_path))
    assert report.pose is not None
    reviewed = apply_hitl(report, HitlRequest(updates=[HitlImageUpdate(image_id="run-1-1", hitl_rating=5)]))
    assert reviewed.pose == report.pose
    assert type(report).model_validate_json(reviewed.model_dump_json()).pose == report.pose


def test_the_pose_scorer_takes_no_gpu_lease() -> None:
    assert PoseScorer.uses_gpu is False


def test_fixtures_are_tiny_and_hold_no_images() -> None:
    for name in ("health", "pose", "stats", "compare"):
        raw = (Path(__file__).parent / "fixtures" / "puppet" / f"{name}.json").read_text(encoding="utf-8")
        assert len(raw) < 20_000 and "base64" not in raw
        assert json.loads(raw)["puppet_version"].startswith("1.")
