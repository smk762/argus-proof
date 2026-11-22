"""Run-level pose diversity and pose-copying evidence from argus-puppet over HTTP (#122).

Proof's phash diversity only sees near-identical images; a concept LoRA can also collapse onto one
composition, so that every output is the same jump shot with a different person and background. argus-puppet
measures that from whole-body keypoints. Proof never imports it (its weights and ONNX Runtime stay in its own
service, and it takes its own GPU lease): it speaks puppet's wire.

* :class:`PuppetClient` posts to the one service at ``PROOF_PUPPET_URL`` through
  :func:`argus_cortex.net.fetch`: that host is the only one allowed, redirects are not followed, every call has a
  per-read timeout (``PROOF_PUPPET_TIMEOUT``) and an overall deadline, and a busy 503 is retried a bounded number of
  times, honouring ``Retry-After``. Proof's GPU priority goes along as ``X-Argus-Priority``. Every document is
  checked for ``puppet_version`` major 1; fields proof doesn't know are ignored.
* :class:`PoseScorer` poses each output image (``POST /pose``, named ``out-00001.png`` and so on and mapped back to
  proof's ``image_id`` here), runs ``POST /stats`` over the results, and, when the run's training export is readable
  on this host, poses the training frames and runs ``POST /compare`` (outputs against training) and ``POST /stats``
  on the training set. It returns a :class:`~argus_proof.models.PoseReport`.

Without a URL the scorer is unavailable (the capability probe says so, ``score`` warns, ``--require-scorers pose``
refuses). Puppet being down or erroring never fails the run: the pose block records what happened, and whatever
was not measured is ``None``, never ``0.0``.

Only each image's primary person is forwarded to ``/stats`` and ``/compare``: every puppet metric reads the primary
person alone, and trimming keeps a set of thousands well inside puppet's body caps.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from argus_proof.models import (
    PoseClosePair,
    PoseComparison,
    PoseCoverage,
    PoseDistanceQuantiles,
    PoseProvenance,
    PoseReport,
    ScorerProvenance,
)

if TYPE_CHECKING:
    from argus_proof.models import GeneratedImage

logger = structlog.get_logger()

# The puppet service (its origin) and per-read timeout, kept in lock-step with .env.example.
ENV_PUPPET_URL = "PROOF_PUPPET_URL"
ENV_PUPPET_TIMEOUT = "PROOF_PUPPET_TIMEOUT"
DEFAULT_PUPPET_TIMEOUT = 60.0
# A call's overall deadline is this many timeouts: a slow drip can't hold a scoring pass for ever.
DEADLINE_FACTOR = 3.0
# Busy 503s are retried at most this many times; Retry-After is honoured up to MAX_RETRY_AFTER_S.
MAX_RETRIES = 3
MAX_RETRY_AFTER_S = 30.0
# The largest answer proof reads (a 5000-image /stats or /compare is a few MB).
MAX_REPLY_BYTES = 64 * 1024 * 1024

# The puppet wire majors proof understands.
SUPPORTED_PUPPET_MAJORS: tuple[str, ...] = ("1",)

# Puppet's documented default caps (docs/server.md): a pose set holds at most 5000 results and a JSON body at
# most 128 MiB. A server configured lower answers 413, which is recorded, not raised.
PUPPET_MAX_SET_RESULTS = 5000
PUPPET_MAX_JSON_BYTES = 128 * 1024 * 1024
# Training frames posed per run: each is one POST /pose, and nothing is cached between runs, so a large export is
# sampled (evenly over its sorted paths, so it is deterministic). Puppet's leave-one-out baseline needs only 20.
MAX_TRAINING_FRAMES = 1000
# Close pairs kept in the report, closest first (PoseComparison.n_close counts them all).
POSE_MAX_CLOSE_PAIRS = 50
# Image ids of failed images kept in the report.
MAX_FAILED_LISTED = 100


class PuppetError(RuntimeError):
    """A puppet call failed. ``fatal`` when the service can't serve at all (unreachable, still busy after the
    retries, a model missing), so the rest of the pass is skipped; otherwise only this request failed."""

    def __init__(self, message: str, *, status: int | None = None, fatal: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.fatal = fatal


# ---------------------------------------------------------------------------
# puppet's wire, mirrored: the few fields proof reads. Unknown fields are ignored (a 1.x reader).
# ---------------------------------------------------------------------------


class _PuppetDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    puppet_version: str

    @field_validator("puppet_version")
    @classmethod
    def _major(cls, v: str) -> str:
        major = v.split(".", 1)[0]
        if major not in SUPPORTED_PUPPET_MAJORS:
            raise ValueError(f"puppet_version {v!r}: proof reads major {', '.join(SUPPORTED_PUPPET_MAJORS)} only")
        return v


class _Part(BaseModel):
    model_config = ConfigDict(extra="ignore")


class PuppetHealth(_PuppetDoc):
    version: str | None = None
    pose_model: str | None = None
    detector_model: str | None = None


class _Cluster(_Part):
    id: int
    size: int
    share: float
    medoid: str


class _Coverage(_Part):
    stance: dict[str, int] = Field(default_factory=dict)
    arms: dict[str, int] = Field(default_factory=dict)
    facing: dict[str, int] = Field(default_factory=dict)

    def to_proof(self) -> PoseCoverage:
        return PoseCoverage(stance=self.stance, arms=self.arms, facing=self.facing)


class PuppetStats(_PuppetDoc):
    count: int
    usable: int
    comparable_share: float
    diversity: float
    mean_pairwise_distance: float
    cluster_count: int
    largest_cluster_share: float
    clusters: list[_Cluster] = Field(default_factory=list)
    coverage: _Coverage = Field(default_factory=_Coverage)
    params: dict[str, Any] = Field(default_factory=dict)


class _Quantiles(_Part):
    n: int
    p01: float
    p10: float
    p50: float
    p90: float

    def to_proof(self) -> PoseDistanceQuantiles:
        return PoseDistanceQuantiles(n=self.n, p01=self.p01, p10=self.p10, p50=self.p50, p90=self.p90)


class _Match(_Part):
    image: str
    nearest: str | None = None
    distance: float | None = None
    close: bool = False


class PuppetComparison(_PuppetDoc):
    output_count: int
    matched_count: int
    reference_count: int
    matches: list[_Match] = Field(default_factory=list)
    close_eligible: int
    close_share: float
    baseline_eligible: int
    baseline_close_share: float | None = None
    close_excess: float | None = None
    output_nn_distance: _Quantiles | None = None
    reference_nn_distance: _Quantiles | None = None
    threshold: float
    coverage_divergence: float
    reference_coverage: _Coverage = Field(default_factory=_Coverage)
    params: dict[str, Any] = Field(default_factory=dict)


def check_pose_result(doc: object) -> dict:
    """A ``PoseResult`` from ``POST /pose``, checked just enough to forward: a JSON object with a supported
    ``puppet_version``, a ``people`` list and a ``primary`` index. Raises :class:`PuppetError`."""
    if not isinstance(doc, dict):
        raise PuppetError("puppet's /pose answer is not a JSON object")
    try:
        _PuppetDoc.model_validate(doc)
    except ValidationError as exc:
        raise PuppetError(f"puppet's /pose answer was refused: {_first_error(exc)}", fatal=True) from exc
    people, primary = doc.get("people"), doc.get("primary")
    if not isinstance(people, list) or not (primary is None or (type(primary) is int and 0 <= primary < len(people))):
        raise PuppetError("puppet's /pose answer has no valid people/primary")
    return doc


def primary_only(result: dict, name: str) -> dict:
    """*result* renamed *name*, keeping only its primary person (every puppet metric reads that one alone).

    Unknown fields travel unchanged: the result goes back to the service that produced it."""
    out = dict(result)
    out["image"] = name
    primary = result.get("primary")
    if primary is None:
        out["people"], out["primary"] = [], None
    else:
        person = dict(result["people"][primary])
        person["primary"] = True
        out["people"], out["primary"] = [person], 0
    return out


def _detail(body: bytes) -> str:
    """Puppet's ``{"detail": ...}`` error text, short (a 422's list collapses to its first message)."""
    try:
        detail = json.loads(body).get("detail")
    except (ValueError, AttributeError):
        return ""
    if isinstance(detail, list) and detail and isinstance(detail[0], dict):
        detail = detail[0].get("msg")
    if isinstance(detail, dict):
        detail = detail.get("message")
    return str(detail)[:200] if isinstance(detail, str) else ""


def _first_error(exc: ValidationError) -> str:
    err = exc.errors()[0]
    where = ".".join(str(p) for p in err.get("loc", ()))
    return f"{where}: {err.get('msg', '')}".strip(": ")


# ---------------------------------------------------------------------------
# the client
# ---------------------------------------------------------------------------


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def puppet_origin(url: str) -> tuple[str, str]:
    """``(base, host:port)`` of a configured puppet URL; :class:`ValueError` for anything but a plain http(s) origin
    (optionally with a path prefix): no credentials, query or fragment."""
    from argus_proof.service_origin import service_origin

    return service_origin(url, ENV_PUPPET_URL)


class PuppetClient:
    """argus-puppet's HTTP API at *base_url* (the configured one only).

    *transport* and *resolve* go to :func:`argus_cortex.net.fetch` (tests use an ``httpx.MockTransport``);
    *sleep* waits between retries.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_PUPPET_TIMEOUT,
        priority: str | None = None,
        retries: int = MAX_RETRIES,
        transport: Any = None,
        resolve: Callable[[str, int], list[str]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base, allowed = puppet_origin(base_url)
        self._allow = (allowed,)
        self.timeout = timeout
        self.deadline = timeout * DEADLINE_FACTOR
        self.priority = priority
        self.retries = retries
        self._transport = transport
        self._resolve = resolve
        self._sleep = sleep

    def _call(self, method: str, path: str, *, content: bytes | None = None, content_type: str | None = None) -> Any:
        """One API call → its decoded JSON. Retries a busy 503; raises :class:`PuppetError`."""
        from argus_cortex.net import UrlRefused, fetch

        headers = {"Accept": "application/json"}
        if content_type:
            headers["Content-Type"] = content_type
        if self.priority:
            headers["X-Argus-Priority"] = self.priority
        for attempt in range(self.retries + 1):
            try:
                reply = fetch(
                    self.base + path,
                    method=method,
                    content=content,
                    headers=headers,
                    allow_hosts=self._allow,
                    timeout=self.timeout,
                    deadline=self.deadline,
                    max_bytes=MAX_REPLY_BYTES,
                    max_redirects=0,
                    resolve=self._resolve,
                    transport=self._transport,
                )
            except (UrlRefused, OSError) as exc:
                raise PuppetError(f"puppet could not be reached: {exc}", fatal=True) from exc
            if reply.status == 503 and attempt < self.retries:
                self._sleep(self._retry_after(reply.headers, attempt))
                continue
            break
        if 300 <= reply.status < 400:
            raise PuppetError(f"puppet answered a redirect (HTTP {reply.status}); it is not followed", fatal=True)
        if reply.status >= 400:
            detail = _detail(reply.body)
            # 503 after the retries: still busy, or it can't run a model at all. Either way the pass stops.
            raise PuppetError(
                f"puppet answered HTTP {reply.status}{f': {detail}' if detail else ''}",
                status=reply.status,
                fatal=reply.status == 503,
            )
        try:
            return json.loads(reply.body)
        except ValueError as exc:
            raise PuppetError("puppet's answer is not JSON", status=reply.status) from exc

    @staticmethod
    def _retry_after(headers: Any, attempt: int) -> float:
        value = str({k.lower(): v for k, v in dict(headers).items()}.get("retry-after", "")).strip()
        try:
            wait = float(value)
        except ValueError:
            wait = float(2**attempt)  # 1, 2, 4 s without a header
        if not math.isfinite(wait) or wait < 0:
            wait = float(2**attempt)
        return min(wait, MAX_RETRY_AFTER_S)

    def _doc(self, model: type[_PuppetDoc], payload: Any, route: str) -> Any:
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            raise PuppetError(f"puppet's {route} answer was refused: {_first_error(exc)}", fatal=True) from exc

    def health(self) -> PuppetHealth:
        return self._doc(PuppetHealth, self._call("GET", "/health"), "/health")

    def pose(self, data: bytes, name: str) -> dict:
        """``POST /pose`` for one image named *name* (a plain file name) → its ``PoseResult`` as a dict."""
        doc = self._call(
            "POST", f"/pose?name={quote(name, safe='')}", content=data, content_type="application/octet-stream"
        )
        return check_pose_result(doc)

    def _json(self, route: str, body: dict) -> Any:
        content = json.dumps(body, separators=(",", ":")).encode()
        if len(content) > PUPPET_MAX_JSON_BYTES:
            raise PuppetError(f"the {route} body would be {len(content)} bytes, over puppet's {PUPPET_MAX_JSON_BYTES}")
        return self._call("POST", route, content=content, content_type="application/json")

    def stats(self, results: Sequence[dict]) -> PuppetStats:
        return self._doc(PuppetStats, self._json("/stats", {"results": list(results)}), "/stats")

    def compare(self, outputs: Sequence[dict], reference: Sequence[dict]) -> PuppetComparison:
        body = {"outputs": {"results": list(outputs)}, "reference": {"results": list(reference)}}
        return self._doc(PuppetComparison, self._json("/compare", body), "/compare")


# ---------------------------------------------------------------------------
# the scorer
# ---------------------------------------------------------------------------


def _evenly(items: Sequence, limit: int) -> list:
    """At most *limit* of *items*, evenly spaced (deterministic)."""
    if len(items) <= limit:
        return list(items)
    return [items[(i * len(items)) // limit] for i in range(limit)]


def _wire_name(prefix: str, index: int, path: Path) -> str:
    suffix = path.suffix.lower() if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") else ".png"
    return f"{prefix}-{index:05d}{suffix}"


def _frame_label(path: Path, export_dir: Path | None) -> str:
    """A training frame as the report names it: relative to the export, else its file name (never a host path)."""
    if export_dir is not None:
        try:
            return path.resolve().relative_to(export_dir.resolve()).as_posix()
        except ValueError:
            pass
    return path.name


class PoseScorer:
    """Run-level pose metrics from argus-puppet (``PROOF_PUPPET_URL``); see the module docstring.

    *client* injects a :class:`PuppetClient` (tests); otherwise one is built per pass from the environment, with the
    pass's priority. Takes no GPU lease itself (puppet leases its own).
    """

    metric = "pose"
    name = "pose"
    uses_gpu = False  # puppet runs as its own service and takes its own lease; proof holds no model

    def __init__(
        self,
        client: PuppetClient | None = None,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        max_training_frames: int = MAX_TRAINING_FRAMES,
    ) -> None:
        self._client = client
        self._base_url = base_url
        self._timeout = timeout
        self.max_training_frames = max_training_frames
        self._health: PuppetHealth | None = None

    def _url(self) -> str:
        return (self._base_url if self._base_url is not None else os.environ.get(ENV_PUPPET_URL, "")).strip()

    def is_available(self) -> bool:
        if self._client is not None:
            return True
        url = self._url()
        if not url:
            return False
        try:
            puppet_origin(url)
        except ValueError:
            logger.warning("pose.bad_url", env=ENV_PUPPET_URL)
            return False
        return True

    def client(self, priority: str | None = None) -> PuppetClient:
        if self._client is not None:
            if priority is not None:
                self._client.priority = priority
            return self._client
        timeout = self._timeout or _float_env(ENV_PUPPET_TIMEOUT, DEFAULT_PUPPET_TIMEOUT)
        return PuppetClient(self._url(), timeout=timeout, priority=priority)

    def provenance(self) -> ScorerProvenance:
        """Never calls the service: the models are named once a pass has read puppet's ``/health``."""
        h = self._health
        models = "+".join(m for m in (h.pose_model, h.detector_model) if m) if h is not None else ""
        return ScorerProvenance(
            name="argus-puppet",
            metric="pose",
            version=h.version if h is not None else None,
            model=models or None,
        )

    def score(
        self,
        images: Sequence[GeneratedImage],
        *,
        training: Sequence[Path] | None = None,
        export_dir: Path | None = None,
        training_note: str | None = None,
        priority: str | None = None,
    ) -> PoseReport:
        """Pose *images* (and *training*, when given) through puppet → a :class:`PoseReport`. Never raises for a
        puppet failure: what couldn't be measured is recorded and left ``None``.

        *training* is the run's training frames when the export is readable here (*export_dir*, used to name them);
        otherwise *training_note* says why they are not compared.
        """
        client = self.client(priority)
        prov = PoseProvenance(training=training_note)
        report = PoseReport(count=len(images), posed=0, provenance=prov)
        self._health = None  # a pass that can't reach puppet must not report the previous pass's models
        try:
            self._health = health = client.health()
        except PuppetError as exc:
            prov.not_checked.append(f"pose: {exc}")
            return self._finish(report)
        prov.puppet_version, prov.package_version = health.puppet_version, health.version
        prov.pose_model, prov.detector_model = health.pose_model, health.detector_model

        outputs, failed, fatal = self._pose_set(client, list(images), "out", lambda img: Path(img.path))
        names = {name: img.image_id for name, img, _ in outputs}
        report.posed = len(outputs)
        report.failed_images = [img.image_id for img, _ in failed][:MAX_FAILED_LISTED]
        report.partial = report.posed < report.count
        if failed:
            more = f" (listing {MAX_FAILED_LISTED})" if len(failed) > MAX_FAILED_LISTED else ""
            prov.not_checked.append(f"{len(failed)} image(s) not posed{more}: {failed[0][1]}")
        if fatal is not None:
            prov.not_checked.append(f"pose: stopped, {fatal}")
            return self._finish(report)
        if not outputs:
            prov.not_checked.append("pose: no image could be posed")
            return self._finish(report)

        results = [r for _, _, r in outputs]
        try:
            stats = client.stats(results)
        except PuppetError as exc:
            prov.not_checked.append(f"pose stats: {exc}")
            return self._finish(report)
        self._fill_stats(report, stats, names)

        if training is None:
            return self._finish(report)
        report.comparison = self._compare(client, results, list(training), export_dir, names, prov, stats.usable)
        return self._finish(report)

    def _finish(self, report: PoseReport) -> PoseReport:
        logger.info(
            "pose.scored",
            count=report.count,
            posed=report.posed,
            usable=report.usable,
            diversity=report.diversity,
            compared=report.comparison is not None,
            not_checked=report.provenance.not_checked,
        )
        return report

    def _pose_set(
        self, client: PuppetClient, items: list, prefix: str, path_of: Callable[[Any], Path]
    ) -> tuple[list[tuple[str, Any, dict]], list[tuple[Any, str]], str | None]:
        """POST each item's image → ``(posed [(wire name, item, primary-only result)], failed [(item, why)],
        fatal reason or None)``. A fatal error marks every remaining item failed and stops."""
        posed: list[tuple[str, Any, dict]] = []
        failed: list[tuple[Any, str]] = []
        for index, item in enumerate(items, start=1):
            path = path_of(item)
            name = _wire_name(prefix, index, path)
            try:
                data = path.read_bytes()
            except OSError as exc:
                failed.append((item, f"unreadable: {exc.strerror or exc}"))
                continue
            try:
                result = client.pose(data, name)
            except PuppetError as exc:
                if exc.fatal:
                    failed.extend((rest, str(exc)) for rest in items[index - 1 :])
                    return posed, failed, str(exc)
                failed.append((item, str(exc)))
                continue
            posed.append((name, item, primary_only(result, name)))
        return posed, failed, None

    @staticmethod
    def _fill_stats(report: PoseReport, stats: PuppetStats, names: dict[str, str]) -> None:
        """Copy puppet's set stats into *report*, leaving ``None`` what puppet's conventions fill with a 0."""
        report.usable = stats.usable
        report.provenance.params = {k: float(v) for k, v in stats.params.items() if isinstance(v, (int, float))}
        if stats.usable >= 2:  # puppet reads 0 under two usable poses: no pair, nothing measured
            report.comparable_share = stats.comparable_share
            report.diversity = stats.diversity if stats.comparable_share > 0 else None
            report.mean_pairwise_distance = stats.mean_pairwise_distance if stats.comparable_share > 0 else None
        else:
            report.provenance.not_checked.append(f"pose diversity: {stats.usable} usable pose(s), need at least 2")
        if stats.usable >= 1:
            report.cluster_count = stats.cluster_count
            report.largest_cluster_share = stats.largest_cluster_share
            report.coverage = stats.coverage.to_proof()
            top = stats.clusters[0].size if stats.clusters else 0
            report.largest_cluster_medoids = [names.get(c.medoid, c.medoid) for c in stats.clusters if c.size == top]

    def _compare(
        self,
        client: PuppetClient,
        outputs: list[dict],
        training: list[Path],
        export_dir: Path | None,
        names: dict[str, str],
        prov: PoseProvenance,
        output_usable: int,
    ) -> PoseComparison | None:
        if not training:
            prov.training = "not checked: the training export has no images"
            return None
        frames = sorted(training)
        if len(outputs) > PUPPET_MAX_SET_RESULTS:
            prov.training = f"not checked: {len(outputs)} outputs, over puppet's set cap of {PUPPET_MAX_SET_RESULTS}"
            return None
        cap = min(self.max_training_frames, PUPPET_MAX_SET_RESULTS)
        sampled = _evenly(frames, cap)
        posed, failed, fatal = self._pose_set(client, sampled, "ref", lambda p: p)
        if fatal is not None:
            prov.training = f"not checked: posing the training frames stopped, {fatal}"
            return None
        if not posed:
            prov.training = "not checked: no training frame could be posed"
            return None
        reference = [r for _, _, r in posed]
        try:
            comparison = client.compare(outputs, reference)
        except PuppetError as exc:
            prov.training = f"not checked: {exc}"
            return None
        note = f"compared: {len(posed)} training frame(s)"
        if len(sampled) < len(frames):
            note += f" sampled evenly from {len(frames)}"
        if failed:
            note += f", {len(failed)} not posed"
        prov.training = note
        frame_of = {name: path for name, path, _ in posed}
        try:
            training_stats = client.stats(reference)
        except PuppetError as exc:
            training_stats = None
            prov.not_checked.append(f"training pose stats: {exc}")
        return self._comparison(comparison, training_stats, frame_of, export_dir, names, len(posed), output_usable)

    @staticmethod
    def _comparison(
        cmp: PuppetComparison,
        training_stats: PuppetStats | None,
        frame_of: dict[str, Path],
        export_dir: Path | None,
        names: dict[str, str],
        n_frames: int,
        output_usable: int,
    ) -> PoseComparison:
        from argus_proof.hashing import sha256_cached

        close = sorted(
            (m for m in cmp.matches if m.close and m.nearest in frame_of and m.distance is not None),
            key=lambda m: (m.distance, m.image),
        )
        pairs = [
            PoseClosePair(
                image_id=names.get(m.image, m.image),
                training_frame=_frame_label(frame_of[m.nearest], export_dir),
                training_sha256=sha256_cached(frame_of[m.nearest]),
                distance=m.distance,
            )
            for m in close[:POSE_MAX_CLOSE_PAIRS]
        ]
        measured = training_stats is not None and training_stats.usable >= 2 and training_stats.comparable_share > 0
        return PoseComparison(
            training_frames=n_frames,
            output_count=cmp.output_count,
            matched_count=cmp.matched_count,
            reference_count=cmp.reference_count,
            close_eligible=cmp.close_eligible,
            close_share=cmp.close_share if cmp.close_eligible > 0 else None,  # puppet reads 0 with none eligible
            baseline_eligible=cmp.baseline_eligible,
            baseline_close_share=cmp.baseline_close_share,
            close_excess=cmp.close_excess,
            output_nn_distance=cmp.output_nn_distance.to_proof() if cmp.output_nn_distance else None,
            reference_nn_distance=cmp.reference_nn_distance.to_proof() if cmp.reference_nn_distance else None,
            # 0 or 1 by convention when a side has no usable pose: not a measured divergence
            coverage_divergence=cmp.coverage_divergence if output_usable and cmp.reference_count else None,
            reference_coverage=cmp.reference_coverage.to_proof() if cmp.reference_count else None,
            training_diversity=training_stats.diversity if measured else None,
            training_largest_cluster_share=(
                training_stats.largest_cluster_share if training_stats is not None and training_stats.usable else None
            ),
            threshold=cmp.threshold,
            n_close=len(close),
            close_pairs=pairs,
        )


__all__ = [
    "DEFAULT_PUPPET_TIMEOUT",
    "ENV_PUPPET_TIMEOUT",
    "ENV_PUPPET_URL",
    "MAX_TRAINING_FRAMES",
    "POSE_MAX_CLOSE_PAIRS",
    "PoseScorer",
    "PuppetClient",
    "PuppetError",
    "check_pose_result",
    "primary_only",
    "puppet_origin",
]
