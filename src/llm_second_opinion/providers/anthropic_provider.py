"""Claude provider via Anthropic's Messages API (official `anthropic` SDK >= 1.11).

Request shape (`client.beta.messages.stream`):
- model, max_tokens (required on this API), system, messages=[{user}]
- output_config: {effort}  — the `reasoning_effort` knob
- tools: [{type: "web_search_20260209", name: "web_search"}] when enabled
- fallbacks: "default" + beta `server-side-fallback-2026-07-01` on models
  that support it, so a safety-classifier decline is retried server-side on
  Anthropic's recommended substitute instead of coming back as a refusal
  (the substitute's id is reported as `model`, so the caller sees it).

`thinking` is never sent: current models run adaptive thinking when it is
omitted, and Claude Fable 5.1 / Claude Opus 5.5 reject every explicit
non-adaptive setting. Depth is governed by `output_config.effort` alone.

The call is streamed and collected with `get_final_message()`: the SDK
refuses large non-streaming `max_tokens` values, and streaming keeps a long
reasoning run from tripping an idle-read timeout. The whole call is still
bounded by `asyncio.wait_for`.

There is no stored-response or background mode on the Messages API: nothing
is retained as a pollable object (DESIGN §18.1 has nothing to opt out of),
and jobs run as in-process tasks like Grok's.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    RequestTooLargeError,
    UnprocessableEntityError,
)

from .base import (
    Provider,
    ProviderError,
    SecondOpinionRequest,
    SecondOpinionResponse,
    TokenUsage,
)

log = logging.getLogger(__name__)

# `max_tokens` is mandatory on the Messages API, so "unbounded" (the caller
# passed none and `default_max_tokens` is null) needs a concrete value. 64000
# is the largest output every current model accepts (Claude Haiku 4.5's cap;
# the rest go to 128000).
UNBOUNDED_MAX_TOKENS = 64000

WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}

# Server-side refusal fallback (`fallbacks: "default"`). Sent only to models
# documented to accept it; elsewhere the field could turn every call into a
# 400 over a feature the caller never asked for.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODEL_PREFIXES = (
    "claude-fable-5-1",
    "claude-opus-5",  # and claude-opus-5-5
    "claude-sonnet-5-5",
)

# Sampling parameters were removed from Claude Opus 4.7 onwards (Fable,
# Mythos, Opus 5.x, Sonnet 5.x all return 400 on `temperature`). Only the
# older lines below still take it; for everything else it is dropped as
# advisory, the same call `openai_provider.DROPPABLE_PARAMS` makes.
SAMPLING_MODEL_PREFIXES = (
    "claude-3",
    "claude-haiku-4",
    "claude-sonnet-4",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-opus-4-1",
    "claude-opus-4-2025",
)

# Server-side tools (web search) run a sampling loop that can stop with
# `pause_turn`; the turn is resumed by re-sending it. Bounded so a runaway
# search cannot spin until the deadline on its own.
MAX_PAUSE_CONTINUATIONS = 5


def _has_prefix(model: str, prefixes: tuple[str, ...]) -> bool:
    return any(model.startswith(p) for p in prefixes)


class AnthropicProvider(Provider):
    name = "anthropic"

    # The Messages API has no submit/poll mode (Message Batches is a 24h
    # queue, not a fit), so jobs run as in-process tasks over generate().
    supports_background = False

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float,
        reasoning_effort: str | None = None,
        web_search: bool = False,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.web_search = web_search

    def model_id(self) -> str:
        return self.model

    def _client(self) -> AsyncAnthropic:
        # max_retries=0 for the same reason as the Responses adapter: SDK
        # retries restart the whole call and can blow past the MCP client's
        # per-call cap. Our errors carry `retriable` instead.
        return AsyncAnthropic(api_key=self.api_key, timeout=self.timeout, max_retries=0)

    def _build_kwargs(self, req: SecondOpinionRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": req.max_tokens if req.max_tokens is not None else UNBOUNDED_MAX_TOKENS,
            "messages": [{"role": "user", "content": self.build_user_content(req)}],
        }
        if req.system_prompt:
            kwargs["system"] = req.system_prompt
        if req.temperature is not None:
            if _has_prefix(self.model, SAMPLING_MODEL_PREFIXES):
                kwargs["temperature"] = req.temperature
            else:
                log.debug(
                    "%s model=%s takes no sampling parameters; dropping temperature",
                    self.name, self.model,
                )
        if self.reasoning_effort:
            kwargs["output_config"] = {"effort": self.reasoning_effort}
        if self.web_search:
            kwargs["tools"] = [WEB_SEARCH_TOOL]
        if _has_prefix(self.model, FALLBACK_MODEL_PREFIXES):
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    async def generate(self, req: SecondOpinionRequest) -> SecondOpinionResponse:
        kwargs = self._build_kwargs(req)
        client = self._client()

        start = time.monotonic()
        message, usage = await self._run_turn(client, kwargs, start)
        latency_ms = int((time.monotonic() - start) * 1000)
        return self._build_response(message, latency_ms, usage)

    async def _run_turn(
        self, client: AsyncAnthropic, kwargs: dict[str, Any], start: float
    ) -> tuple[Any, TokenUsage | None]:
        """Stream the turn to completion, resuming `pause_turn` stops.

        Every resume is charged against the one original deadline. Usage is
        summed across the continuations so the reported tokens are what the
        whole turn billed.
        """
        message = await self._stream_once(client, kwargs, self.timeout)
        totals = _UsageTotals()
        totals.add(message)
        for _ in range(MAX_PAUSE_CONTINUATIONS):
            if getattr(message, "stop_reason", None) != "pause_turn":
                break
            remaining = self.timeout - (time.monotonic() - start)
            if remaining <= 0:
                break
            # Re-send the paused assistant turn as-is; the API sees the
            # trailing server_tool_use block and resumes where it stopped.
            kwargs = {
                **kwargs,
                "messages": [
                    *kwargs["messages"],
                    {"role": "assistant", "content": message.content},
                ],
            }
            message = await self._stream_once(client, kwargs, remaining)
            totals.add(message)
        return message, totals.to_usage()

    async def _stream_once(
        self, client: AsyncAnthropic, kwargs: dict[str, Any], timeout: float
    ) -> Any:
        """One streamed Messages call under `timeout`, errors mapped to ProviderError."""

        async def run() -> Any:
            async with client.beta.messages.stream(**kwargs) as stream:
                return await stream.get_final_message()

        try:
            return await asyncio.wait_for(run(), timeout=timeout)
        except (asyncio.TimeoutError, APITimeoutError) as e:
            raise ProviderError(
                "timeout",
                f"{self.name} request timed out after {timeout:g}s",
                retriable=True,
            ) from e
        except AuthenticationError as e:
            raise ProviderError("auth_failed", f"{self.name} authentication failed: {e}",
                                retriable=False, status=e.status_code) from e
        except PermissionDeniedError as e:
            raise ProviderError("auth_failed", f"{self.name} permission denied: {e}",
                                retriable=False, status=e.status_code) from e
        except RateLimitError as e:
            raise ProviderError("rate_limit", f"{self.name} rate limit hit: {e}",
                                retriable=True, status=e.status_code) from e
        except APIConnectionError as e:
            raise ProviderError("network_error", f"{self.name} network error: {e}",
                                retriable=True) from e
        except NotFoundError as e:
            raise ProviderError("bad_request", f"{self.name} not found (check model name): {e}",
                                retriable=False, status=e.status_code) from e
        except (BadRequestError, RequestTooLargeError, UnprocessableEntityError) as e:
            raise ProviderError("bad_request", f"{self.name} rejected request: {e}",
                                retriable=False, status=e.status_code) from e
        except APIStatusError as e:
            # Includes 529 overloaded, which is transient.
            status = e.status_code or 0
            if status >= 500:
                raise ProviderError("upstream_error",
                                    f"{self.name} server error ({status}): {e}",
                                    retriable=True, status=status) from e
            raise ProviderError("upstream_error", f"{self.name} error ({status}): {e}",
                                retriable=False, status=status) from e
        except Exception as e:  # noqa: BLE001 - mid-stream SDK errors surface as other classes
            raise ProviderError("upstream_error", f"{self.name} error: {e}",
                                retriable=False) from e

    def _build_response(
        self, message: Any, latency_ms: int, usage: TokenUsage | None = None
    ) -> SecondOpinionResponse:
        """Validate a final Message and extract the answer.

        `stop_reason` is checked before `content` is read: a refusal is an
        HTTP 200 whose content may be empty or partial.
        """
        if usage is None:
            usage = _extract_usage(message)
        stop_reason = getattr(message, "stop_reason", None)

        if stop_reason == "refusal":
            # Reaching here means any configured fallback declined too.
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None)
            explanation = getattr(details, "explanation", None)
            detail = ", ".join(
                p for p in (f"category={category}" if category else "", explanation or "") if p
            )
            raise ProviderError(
                "content_blocked",
                f"{self.name} declined the request" + (f" ({detail})" if detail else ""),
                retriable=False,
            )

        if stop_reason == "max_tokens":
            reasoning_tokens = usage.reasoning_tokens if usage else None
            raise ProviderError(
                "upstream_error",
                f"{self.name} response was cut short (stop_reason='max_tokens', "
                f"reasoning_tokens={reasoning_tokens}). Retry with a higher "
                f"max_tokens, a lower reasoning_effort, or omit max_tokens entirely.",
                retriable=True,
            )

        if stop_reason == "pause_turn":
            raise ProviderError(
                "upstream_error",
                f"{self.name} kept pausing its server-side tool loop and did not "
                f"finish within the budget. Retry with web_search off or a "
                f"narrower prompt.",
                retriable=True,
            )

        if stop_reason == "model_context_window_exceeded":
            raise ProviderError(
                "bad_request",
                f"{self.name} ran out of context window. Send less material.",
                retriable=False,
            )

        text = _final_text(message).strip()
        if not text:
            reasoning_tokens = usage.reasoning_tokens if usage else None
            raise ProviderError(
                "upstream_error",
                f"{self.name} returned no visible text (stop_reason={stop_reason!r}, "
                f"reasoning_tokens={reasoning_tokens}). Retry with a lower "
                f"reasoning_effort or a higher max_tokens.",
                retriable=True,
            )

        return SecondOpinionResponse(
            provider=self.name,
            # The serving model: differs from the configured one when a
            # refusal fallback answered.
            model=getattr(message, "model", None) or self.model,
            text=text,
            usage=usage,
            latency_ms=latency_ms,
        )

    async def check_reachable(self) -> tuple[bool, str | None]:
        try:
            client = self._client()
            await client.with_options(timeout=5.0).models.retrieve(self.model)
        except AuthenticationError:
            return False, "authentication failed"
        except PermissionDeniedError:
            return False, "permission denied"
        except NotFoundError:
            return False, f"model {self.model!r} not found"
        except APIConnectionError as e:
            return False, f"network error: {e}"
        except APIStatusError as e:
            return False, f"http {e.status_code}"
        except Exception as e:  # noqa: BLE001
            return False, f"unexpected error: {e}"
        return True, None


def _final_text(message: Any) -> str:
    """The trailing run of text blocks in `message.content`.

    With web search the content interleaves text with `server_tool_use` /
    `web_search_tool_result` blocks, and the model may narrate before a
    search; a refusal fallback inserts a `fallback` block at the switch.
    Only the trailing run of text is the final answer — the same rule as
    `openai_provider._final_message_text` and `gemini._join_output_text`.
    One cited answer arrives as several adjacent text blocks, which the run
    keeps together. Trailing non-text blocks are skipped.
    """
    parts: list[str] = []
    collecting = False
    for block in reversed(getattr(message, "content", None) or []):
        if getattr(block, "type", None) != "text":
            if collecting:
                break
            continue
        collecting = True
        t = getattr(block, "text", None)
        if t:
            parts.append(t)
    parts.reverse()
    return "".join(parts)


def _extract_usage(message: Any) -> TokenUsage | None:
    usage = getattr(message, "usage", None)
    if usage is None:
        return None
    totals = _UsageTotals()
    totals.add(message)
    return totals.to_usage()


class _UsageTotals:
    """Sums usage across the messages of one (possibly resumed) turn."""

    def __init__(self) -> None:
        self.input: int | None = None
        self.output: int | None = None
        self.reasoning: int | None = None

    @staticmethod
    def _plus(acc: int | None, value: Any) -> int | None:
        if not isinstance(value, int):
            return acc
        return value if acc is None else acc + value

    def add(self, message: Any) -> None:
        usage = getattr(message, "usage", None)
        if usage is None:
            return
        # Prompt tokens are split three ways; the sum is what was read.
        for attr in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            self.input = self._plus(self.input, getattr(usage, attr, None))
        self.output = self._plus(self.output, getattr(usage, "output_tokens", None))
        details = getattr(usage, "output_tokens_details", None)
        self.reasoning = self._plus(self.reasoning, getattr(details, "thinking_tokens", None))

    def to_usage(self) -> TokenUsage | None:
        if self.input is None and self.output is None:
            return None
        total = (self.input or 0) + (self.output or 0)
        return TokenUsage(
            input_tokens=self.input,
            output_tokens=self.output,
            total_tokens=total,
            reasoning_tokens=self.reasoning,
        )
