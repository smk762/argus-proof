"""GPU leases from the suite's broker (argus-iris), through argus-cortex's coordinator.

A proof run is a grid of seeds, so its generation work is ``batch`` and preemptible at every seed boundary:

* the ``comfyui`` backend holds a lease on ``comfyui`` (kind ``generate``) and sends the work to the ComfyUI the grant
  names (``base_url``; ``None`` means proof's own ``COMFYUI_BASE_URL``);
* the ``diffusers`` backend renders in this process, so it leases ``in-process`` (kind ``generate-local``).

Scoring is the third kind: ``score`` (``in-process``) covers one scoring pass over a run's images with the learned
scorers (InsightFace, CLIP-IQA, CLIPScore), and is not preemptible (a pass is short and has no seed boundary). It
leases at ``interactive`` priority unless the caller is part of a batch run (``/run/stream`` sends ``batch``). The
scorers load their models inside the lease and free them before it ends, so proof keeps nothing resident.

One generation lease covers a run. Between seeds :class:`SeedLeases` checks ``should_yield()``; when the broker wants the GPU
back it releases the lease and queues for a new one, **pinned to the node the run started on**: the manifest hashed
the model files once, so the rest of the run must not move to a ComfyUI that may hold different ones.

The coordinator comes from the host-wide ``ARGUS_GPU_*`` environment (:func:`argus_cortex.gpu.coordinator_from_env`):
unset is ``none``, no gating, which is how proof ran before. ``broker`` falls back to ``ARGUS_GPU_BROKER_FALLBACK``
(``lease`` by default) when iris cannot be reached.

The figures below are the ones declared in ``gpu-tenant.yaml``; ``tests/test_gpu.py`` keeps the two in step. The
broker uses its learned p95 per kind when that is larger (a Flux checkpoint is caught after a few leases).

A refused or timed-out lease, or a broker error, is a :class:`~argus_proof.backends.base.BackendError` that says so.
Nothing retries it.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

from argus_cortex.gpu import GpuCoordinator, GpuLease, GpuLeaseRefused, GpuLeaseTimeout, coordinator_from_env
from argus_cortex.gpu_broker import BrokerError
from argus_cortex.gpu_wire import IN_PROCESS, GpuConstraints, PriorityClass

TENANT = "argus-proof"
TARGET_COMFYUI = "comfyui"
DEFAULT_PRIORITY: PriorityClass = "batch"

LEASE_ERRORS: tuple[type[Exception], ...] = (GpuLeaseRefused, GpuLeaseTimeout, BrokerError)


@dataclass(frozen=True)
class Footprint:
    kind: str
    target: str
    vram_mb: int  # the whole footprint: weights plus working set
    model_mb: int  # the weights that stay loaded after the lease
    preemptible: bool = True  # a grid yields between seeds


GENERATE = Footprint("generate", TARGET_COMFYUI, vram_mb=9000, model_mb=6500)  # an SDXL checkpoint in ComfyUI
GENERATE_LOCAL = Footprint("generate-local", IN_PROCESS, vram_mb=10000, model_mb=7000)  # diffusers SDXL, fp16
# One scoring pass: measured on an RTX 3090 (see gpu-tenant.yaml for the breakdown). model_mb 0: the scorers free
# their models before the lease ends, so nothing stays loaded.
SCORE = Footprint("score", IN_PROCESS, vram_mb=3500, model_mb=0, preemptible=False)
SCORE_PRIORITY: PriorityClass = "interactive"


def default_coordinator() -> GpuCoordinator:
    return coordinator_from_env()


def score_lease(
    coordinator: GpuCoordinator, *, caller: str, priority: PriorityClass = SCORE_PRIORITY
) -> contextlib.AbstractContextManager[GpuLease]:
    """The lease one scoring pass runs under (:data:`SCORE`, not preemptible)."""
    return coordinator.lease(
        caller=caller,
        min_vram_mb=SCORE.vram_mb,
        model_mb=SCORE.model_mb,
        kind=SCORE.kind,
        target=SCORE.target,
        priority=priority,
        preemptible=SCORE.preemptible,
    )


def lease_failure(exc: Exception) -> str:
    """The error message for a lease that was not granted."""
    if isinstance(exc, GpuLeaseRefused):
        return f"the GPU broker refused the lease: {exc}"
    if isinstance(exc, GpuLeaseTimeout):
        return f"no GPU lease in time: {exc}"
    return f"the GPU broker could not be used: {exc}"


class SeedLeases:
    """One lease across a run's seeds, given back and re-queued when the broker asks.

    Use as a context manager, and call :meth:`current` before each seed: it returns the held lease, first releasing it
    and taking a fresh one if ``should_yield()`` is set. A re-acquired lease is constrained to the node of the first
    grant (when the broker named one). Lease errors propagate as the coordinator raised them (:data:`LEASE_ERRORS`).
    """

    def __init__(
        self,
        coordinator: GpuCoordinator,
        footprint: Footprint,
        *,
        caller: str,
        priority: PriorityClass = DEFAULT_PRIORITY,
    ) -> None:
        self.coordinator = coordinator
        self.footprint = footprint
        self.caller = caller
        self.priority = priority
        self.yields = 0
        self._node: str | None = None
        self._stack: contextlib.ExitStack | None = None
        self._lease: GpuLease | None = None

    def __enter__(self) -> SeedLeases:
        self._acquire()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
        # Hand the exception to the lease, so the broker records a failed run as failed, not ok.
        stack, self._stack, self._lease = self._stack, None, None
        if stack is not None:
            stack.__exit__(exc_type, exc, tb)  # type: ignore[arg-type]

    def current(self) -> GpuLease:
        """The lease to run the next seed under."""
        if self._lease is None or self._lease.should_yield():
            if self._lease is not None:
                self.yields += 1
            self._release()
            self._acquire()
        assert self._lease is not None
        return self._lease

    def _acquire(self) -> None:
        stack = contextlib.ExitStack()
        constraints = GpuConstraints(node=self._node) if self._node else None
        lease = stack.enter_context(
            self.coordinator.lease(
                caller=self.caller,
                min_vram_mb=self.footprint.vram_mb,
                model_mb=self.footprint.model_mb,
                kind=self.footprint.kind,
                target=self.footprint.target,
                priority=self.priority,
                preemptible=self.footprint.preemptible,
                constraints=constraints,
            )
        )
        self._stack, self._lease = stack, lease
        if self._node is None and lease.node:
            self._node = lease.node

    def _release(self) -> None:
        stack, self._stack, self._lease = self._stack, None, None
        if stack is not None:
            stack.close()
