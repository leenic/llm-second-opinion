"""Background-job registry for the submit/poll tools.

A *job* is one upstream generation living independently of any tool call.
This module holds the job record, its state machine, the in-memory registry
(with the `request_key` idempotency index and lazy TTL eviction) and the
named constants of the timing model v2. It deliberately contains no asyncio
driving code: the pollers live in `server.py` next to `_run_bounded`, whose
cancellation discipline they reuse.

State machine (DESIGN-submit-poll.md §5):

    submit ──▶ submitting ──(upstream acknowledges)──▶ running ──┬──▶ succeeded
                   │                                             ├──▶ failed
                   └── submit fails: no job, submit returns an error  └──▶ cancelled

`submitting` is internal — a record is only registered once acknowledged, so
the client never observes it. Terminal states are immutable and keep their
full envelope cached until TTL eviction.

Everything here mutates on the event loop from a single process, so no
locking is needed (the v0.1 posture).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# --- Timing model v2 constants (design §7, §10) -----------------------------

# A background create is a fast call: if it cannot be acknowledged in 30s
# something is wrong and the structured error should say so. Also bounds the
# other control calls (poll, cancel) — all of them are "fast calls".
SUBMIT_BUDGET_SECONDS = 30.0

# Cap on `get_second_opinion(wait_seconds=…)`. Sits far under every client
# cap observed (240s here, ~60s reported on some hosts) so that even a hostile
# intermediary timer is cleared: 45 + CANCEL_GRACE_SECONDS ≈ 51s worst case.
MAX_POLL_WAIT_SECONDS = 45.0

# How often a provider-backed job is polled upstream while running.
POLL_INTERVAL_SECONDS = 4.0

# Concurrent non-terminal jobs. Generous for a single-user tool; bounds both
# memory and worst-case vendor spend.
MAX_ACTIVE_JOBS = 8

# Terminal records survive this long from their terminal transition so a
# result can be re-fetched after relay flakiness, then evict lazily.
TERMINAL_JOB_TTL_SECONDS = 1800.0

# Backing types (design §2).
BACKING_PROVIDER_BACKGROUND = "provider_background"
BACKING_LOCAL_TASK = "local_task"

STATUS_SUBMITTING = "submitting"  # internal only
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

TERMINAL_STATUSES = frozenset({STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELLED})

_TRANSITIONS: dict[str, frozenset[str]] = {
    STATUS_SUBMITTING: frozenset({STATUS_RUNNING}),
    STATUS_RUNNING: TERMINAL_STATUSES,
    STATUS_SUCCEEDED: frozenset(),
    STATUS_FAILED: frozenset(),
    STATUS_CANCELLED: frozenset(),
}


def new_job_id() -> str:
    """Same grammar as `request_id`, distinct value space; logged as `jid=`.

    Provider response/interaction ids are never used as the client-facing
    key (design §4, invariant 12).
    """
    return uuid.uuid4().hex[:12]


class JobStateError(Exception):
    """An illegal state transition was attempted (a bug, never a tool error)."""


@dataclass
class JobRecord:
    job_id: str
    provider: str
    target_model: str
    model: str
    backing: str
    # Request echo for logs and the terminal envelope — sizes and knobs only,
    # never content (invariant 7).
    reasoning_effort: str | None = None
    web_search: bool = False
    max_tokens: int | None = None
    focus: bool = False
    request_key: str | None = None
    # The provider adapter instance the job was submitted through; the
    # pollers and cancel need it. `provider` above is its name, for logs.
    adapter: Any = None
    # Provider response/interaction id for provider-backed jobs. Logged as
    # `upstream_id=`, never returned in a tool result.
    upstream_id: str | None = None
    # The in-process generate() task for local-task jobs.
    task: asyncio.Task | None = None
    # The poller/driver task, whichever backing.
    driver: asyncio.Task | None = None
    status: str = STATUS_SUBMITTING
    submitted_monotonic: float = field(default_factory=time.monotonic)
    submitted_wall: float = field(default_factory=time.time)
    terminal_monotonic: float | None = None
    # Cached terminal envelope (without a request_id — each get adds its own).
    envelope: dict[str, Any] | None = None
    # Set on terminal transition; long-polls wait on it.
    done: asyncio.Event = field(default_factory=asyncio.Event)
    # Set when the reservation resolves either way (registered as running, or
    # released after a failed submission); a concurrent same-key submit waits
    # on it instead of starting a second upstream job.
    acknowledged: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def age_seconds(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        return now - self.submitted_monotonic

    def job_elapsed_ms(self, now: float | None = None) -> int:
        """Wall-clock of the job: submit → terminal (or → now while running)."""
        end = self.terminal_monotonic if self.terminal_monotonic is not None else (
            time.monotonic() if now is None else now
        )
        return int((end - self.submitted_monotonic) * 1000)

    def attach_task(self, task: asyncio.Task) -> None:
        self.task = task

    def transition(self, new_status: str) -> None:
        allowed = _TRANSITIONS.get(self.status, frozenset())
        if new_status not in allowed:
            raise JobStateError(
                f"job {self.job_id}: illegal transition {self.status} -> {new_status}"
            )
        self.status = new_status


class JobRegistry:
    """In-memory `job_id -> JobRecord` plus the `request_key -> job_id` index.

    TTL eviction is a lazy sweep run on every access — no background timer.
    """

    def __init__(
        self,
        *,
        max_active: int = MAX_ACTIVE_JOBS,
        ttl_seconds: float = TERMINAL_JOB_TTL_SECONDS,
        logger: logging.Logger | None = None,
    ):
        self.max_active = max_active
        self.ttl_seconds = ttl_seconds
        self._jobs: dict[str, JobRecord] = {}
        self._by_key: dict[str, str] = {}
        self._log = logger or logging.getLogger("llm_second_opinion")

    # -- access ---------------------------------------------------------

    def sweep(self, now: float | None = None) -> list[str]:
        """Evict terminal records older than the TTL. Returns evicted ids."""
        now = time.monotonic() if now is None else now
        evicted: list[str] = []
        for job_id, rec in list(self._jobs.items()):
            if rec.terminal_monotonic is None:
                continue
            age = now - rec.terminal_monotonic
            if age >= self.ttl_seconds:
                del self._jobs[job_id]
                if rec.request_key is not None and self._by_key.get(rec.request_key) == job_id:
                    del self._by_key[rec.request_key]
                evicted.append(job_id)
                self._log.info(
                    "jid=%s job=evicted status=%s terminal_age_s=%.0f ttl_s=%.0f",
                    job_id, rec.status, age, self.ttl_seconds,
                )
        return evicted

    def get(self, job_id: str) -> JobRecord | None:
        """Look a job up by id. A reservation that is still `submitting` is
        not addressable — its id has not been disclosed to any client."""
        self.sweep()
        rec = self._jobs.get(job_id)
        if rec is not None and rec.status == STATUS_SUBMITTING:
            return None
        return rec

    def find_by_key(self, request_key: str) -> JobRecord | None:
        self.sweep()
        job_id = self._by_key.get(request_key)
        return self._jobs.get(job_id) if job_id is not None else None

    def active(self) -> list[JobRecord]:
        """Non-terminal jobs, oldest first."""
        self.sweep()
        return sorted(
            (r for r in self._jobs.values() if not r.is_terminal),
            key=lambda r: r.submitted_monotonic,
        )

    def at_capacity(self) -> bool:
        return len(self.active()) >= self.max_active

    def __len__(self) -> int:
        return len(self._jobs)

    def __contains__(self, job_id: object) -> bool:
        return job_id in self._jobs

    # -- mutation -------------------------------------------------------

    def reserve(self, record: JobRecord) -> JobRecord:
        """Hold a slot and the request_key for a submission that is awaiting
        upstream acknowledgement.

        Reserving *before* the upstream call is what makes the capacity cap and
        the idempotency key hold under concurrency: a second submit with the
        same key finds the reservation and waits on `acknowledged` instead of
        starting a second, separately billed job, and N parallel submits cannot
        all pass the cap check. The record stays `submitting`, which `get()`
        hides — its id is not disclosed until `register`.
        """
        if record.status != STATUS_SUBMITTING:
            raise JobStateError(f"job {record.job_id}: can only reserve a submitting job")
        self._jobs[record.job_id] = record
        if record.request_key is not None:
            self._by_key[record.request_key] = record.job_id
        return record

    def release(self, record: JobRecord) -> None:
        """Drop a reservation whose submission failed; wakes any waiter."""
        if record.status == STATUS_SUBMITTING and self._jobs.get(record.job_id) is record:
            del self._jobs[record.job_id]
            if record.request_key is not None and self._by_key.get(record.request_key) == record.job_id:
                del self._by_key[record.request_key]
        record.acknowledged.set()

    def register(self, record: JobRecord) -> JobRecord:
        """Admit an acknowledged job: `submitting -> running`, index its key.

        Called once the upstream has acknowledged (or, for local tasks, once
        the task is started), so a client never observes `submitting`. Works
        with or without a prior `reserve`.
        """
        record.transition(STATUS_RUNNING)
        self._jobs[record.job_id] = record
        if record.request_key is not None:
            self._by_key[record.request_key] = record.job_id
        record.acknowledged.set()
        return record

    def finish(self, record: JobRecord, status: str, envelope: dict[str, Any]) -> bool:
        """Move `record` to a terminal state with its cached envelope.

        Returns False (and changes nothing) if the record is already
        terminal — terminal states are immutable (invariant 13), so whoever
        lost the race simply observes the winner's state.
        """
        if record.is_terminal:
            return False
        record.transition(status)
        record.terminal_monotonic = time.monotonic()
        record.envelope = envelope
        record.done.set()
        err = envelope.get("error") if isinstance(envelope, dict) else None
        self._log.info(
            "jid=%s job=terminal status=%s provider=%s model=%s backing=%s "
            "job_elapsed_ms=%d%s",
            record.job_id, status, record.provider, record.model, record.backing,
            record.job_elapsed_ms(),
            f" error_type={err.get('type')}" if isinstance(err, dict) else "",
        )
        return True


def retry_after_ms(job_age_seconds: float) -> int:
    """Polling hint that escalates with job age: 5s -> 15s -> 30s."""
    if job_age_seconds < 60:
        return 5000
    if job_age_seconds < 300:
        return 15000
    return 30000


__all__ = [
    "SUBMIT_BUDGET_SECONDS",
    "MAX_POLL_WAIT_SECONDS",
    "POLL_INTERVAL_SECONDS",
    "MAX_ACTIVE_JOBS",
    "TERMINAL_JOB_TTL_SECONDS",
    "BACKING_PROVIDER_BACKGROUND",
    "BACKING_LOCAL_TASK",
    "STATUS_SUBMITTING",
    "STATUS_RUNNING",
    "STATUS_SUCCEEDED",
    "STATUS_FAILED",
    "STATUS_CANCELLED",
    "TERMINAL_STATUSES",
    "JobRecord",
    "JobRegistry",
    "JobStateError",
    "new_job_id",
    "retry_after_ms",
]
