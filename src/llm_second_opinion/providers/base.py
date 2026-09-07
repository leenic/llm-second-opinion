"""Common types and base class for provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


# Error type taxonomy surfaced back to Claude. Keep values stable — Claude may
# present them to the user verbatim.
ERROR_TYPES = {
    "missing_api_key",
    "auth_failed",
    "rate_limit",
    "timeout",
    "network_error",
    "upstream_error",
    "bad_request",
    "content_blocked",
    "invalid_input",
    "internal_error",
    # Background-job additions (DESIGN-submit-poll.md §8). Additive only.
    "unknown_job",  # job_id not in the registry: typo, TTL eviction, or restart orphan
    "job_limit",  # MAX_ACTIVE_JOBS reached; retriable once a job finishes
}


@dataclass
class SecondOpinionRequest:
    summary: str
    focus: str | None
    system_prompt: str
    temperature: float | None
    max_tokens: int | None


@dataclass
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    # Reasoning/thinking tokens — subset of output_tokens spent on internal
    # reasoning that isn't visible in the final message. Helps explain
    # "high output_tokens but empty/short reply" outcomes.
    reasoning_tokens: int | None = None

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass
class SecondOpinionResponse:
    provider: str
    model: str
    text: str
    usage: TokenUsage | None
    latency_ms: int


class ProviderError(Exception):
    """Structured error from a provider adapter. Carries enough metadata for
    the MCP layer to return a clean error payload to Claude."""

    def __init__(self, error_type: str, message: str, *, retriable: bool = False, status: int | None = None):
        super().__init__(message)
        if error_type not in ERROR_TYPES:
            error_type = "internal_error"
        self.error_type = error_type
        self.message = message
        self.retriable = retriable
        self.status = status

    def to_dict(self) -> dict:
        out: dict = {
            "type": self.error_type,
            "message": self.message,
            "retriable": self.retriable,
        }
        if self.status is not None:
            out["status"] = self.status
        return out


@dataclass
class BackgroundPoll:
    """One observation of a background job's upstream state.

    `done=False` means still running. `done=True` carries exactly one of
    `response` (the job succeeded) or `error` (the job reached a terminal
    failure upstream — mapped through the same code as the synchronous path,
    so the taxonomy is identical). A failure of the *poll call itself*
    (network, auth, rate limit) is raised as `ProviderError` instead, so the
    poller can tell "the job failed" from "I could not ask".
    """

    done: bool
    response: SecondOpinionResponse | None = None
    error: ProviderError | None = None
    upstream_status: str | None = None


class Provider(ABC):
    """Adapter for a single external LLM provider."""

    name: str = ""

    # True when the vendor offers real background execution (submit -> id ->
    # poll). Providers without it are run as in-process tasks by the server.
    supports_background: bool = False

    @abstractmethod
    async def generate(self, req: SecondOpinionRequest) -> SecondOpinionResponse:
        """Send the second-opinion request and return the model's reply."""

    async def submit_background(self, req: SecondOpinionRequest, timeout: float) -> str:
        """Start the generation upstream and return the provider's id for it.

        Must return as soon as the upstream acknowledges. `timeout` bounds
        this one control call. Raises `ProviderError` on failure.
        """
        raise NotImplementedError(f"{self.name} has no background mode")

    async def poll_background(self, upstream_id: str, timeout: float) -> BackgroundPoll:
        """Observe the upstream job once. See `BackgroundPoll`."""
        raise NotImplementedError(f"{self.name} has no background mode")

    async def cancel_background(
        self, upstream_id: str, timeout: float
    ) -> BackgroundPoll | None:
        """Cancel the upstream job. Idempotent: cancelling a finished job is
        not an error.

        Returns the upstream state the cancel call reported, when the vendor
        reports one: a job that had already finished comes back as a `done`
        poll carrying its response (or its mapped error) so the caller can
        preserve a completed, billed result instead of discarding it; a job
        that was actually stopped comes back `done` with neither. `None`
        means "cancelled, nothing more known".
        """
        raise NotImplementedError(f"{self.name} has no background mode")

    @abstractmethod
    async def check_reachable(self) -> tuple[bool, str | None]:
        """Lightweight reachability check. Returns (ok, error_message)."""

    @abstractmethod
    def model_id(self) -> str:
        """Configured default model identifier for this provider."""

    def build_user_content(self, req: SecondOpinionRequest) -> str:
        """Compose the user message: focus (if any) followed by the summary."""
        if req.focus:
            return f"Focus on: {req.focus}\n\n{req.summary}"
        return req.summary
