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


class Provider(ABC):
    """Adapter for a single external LLM provider."""

    name: str = ""

    @abstractmethod
    async def generate(self, req: SecondOpinionRequest) -> SecondOpinionResponse:
        """Send the second-opinion request and return the model's reply."""

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
