"""Gemini provider via Google's Interactions API (google-genai SDK >= 1.55).

The Interactions API is Google's stateful counterpart to OpenAI's Responses
API. We use it in single-turn mode — no `previous_interaction_id`, no
server-side state — matching the v1 spec.

Request shape (from `client.aio.interactions.create`):
- input: str (the user's message)
- model: str
- system_instruction: str  (top-level, NOT nested under config)
- generation_config: {temperature, max_output_tokens, thinking_level, ...}
- tools: list[{type, ...}]

Response shape (`Interaction`), as of the May 2026 Interactions breaking
change (google-genai >= 2.0 — see ai.google.dev/gemini-api/docs/
interactions-breaking-changes-may-2026):
- status: "completed" | "failed" | "cancelled" | "incomplete" |
  "in_progress" | "requires_action" | "budget_exceeded"
- steps: list[Step] (replaces the old `outputs` list). The model's text
  lives in steps with type=="model_output", whose `.content` is a list of
  blocks where each text block has type=="text" and `.text`. Other step
  types (thoughts, tool calls/results) are interleaved and ignored here.
- usage: total_input_tokens / total_output_tokens / total_tokens /
  total_thought_tokens (unchanged across the 2.0 migration)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from google import genai
from google.genai import errors as genai_errors

from .base import (
    Provider,
    ProviderError,
    SecondOpinionRequest,
    SecondOpinionResponse,
    TokenUsage,
)


class GeminiProvider(Provider):
    name = "gemini"

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
        # `thinking_level` on the Interactions API takes exactly this enum,
        # so we pass it through unchanged.
        self.reasoning_effort = reasoning_effort
        self.web_search = web_search

    def model_id(self) -> str:
        return self.model

    def _client(self) -> genai.Client:
        return genai.Client(api_key=self.api_key)

    def _build_kwargs(self, req: SecondOpinionRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": self.build_user_content(req),
        }
        if req.system_prompt:
            kwargs["system_instruction"] = req.system_prompt

        gen_cfg: dict[str, Any] = {}
        if req.temperature is not None:
            gen_cfg["temperature"] = req.temperature
        if req.max_tokens is not None:
            gen_cfg["max_output_tokens"] = req.max_tokens
        if self.reasoning_effort:
            gen_cfg["thinking_level"] = self.reasoning_effort
        if gen_cfg:
            kwargs["generation_config"] = gen_cfg

        if self.web_search:
            kwargs["tools"] = [{"type": "google_search"}]

        return kwargs

    async def generate(self, req: SecondOpinionRequest) -> SecondOpinionResponse:
        kwargs = self._build_kwargs(req)
        client = self._client()

        start = time.monotonic()
        try:
            coro = client.aio.interactions.create(**kwargs)
            response = await asyncio.wait_for(coro, timeout=self.timeout)
        except asyncio.TimeoutError as e:
            raise ProviderError(
                "timeout",
                f"gemini request timed out after {self.timeout}s",
                retriable=True,
            ) from e
        except genai_errors.APIError as e:
            raise _translate_genai_error(e)
        except Exception as e:  # noqa: BLE001 - SDK may surface other classes
            raise ProviderError(
                "upstream_error",
                f"gemini error: {e}",
                retriable=False,
            ) from e

        latency_ms = int((time.monotonic() - start) * 1000)

        status = getattr(response, "status", None)
        if status == "failed":
            raise ProviderError(
                "upstream_error",
                f"gemini interaction failed (status={status})",
                retriable=False,
            )
        if status == "budget_exceeded":
            raise ProviderError(
                "upstream_error",
                f"gemini interaction exceeded its budget (status={status})",
                retriable=False,
            )
        if status in ("cancelled", "incomplete"):
            raise ProviderError(
                "content_blocked",
                f"gemini interaction did not complete (status={status})",
                retriable=False,
            )

        text = _join_output_text(response).strip()
        if not text:
            raise ProviderError(
                "upstream_error",
                f"gemini returned no text output (status={status!r})",
                retriable=False,
            )

        usage = _extract_usage(response)
        actual_model = _extract_model_id(response, fallback=self.model)

        return SecondOpinionResponse(
            provider=self.name,
            model=actual_model,
            text=text,
            usage=usage,
            latency_ms=latency_ms,
        )

    async def check_reachable(self) -> tuple[bool, str | None]:
        try:
            client = self._client()
            coro = client.aio.models.get(model=self.model)
            await asyncio.wait_for(coro, timeout=5.0)
        except asyncio.TimeoutError:
            return False, "reachability probe timed out"
        except genai_errors.APIError as e:
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            if status in (401, 403):
                return False, "authentication failed"
            if status == 404:
                return False, f"model {self.model!r} not found"
            return False, f"api error: {e}"
        except Exception as e:  # noqa: BLE001
            return False, f"unexpected error: {e}"
        return True, None


def _translate_genai_error(e: genai_errors.APIError) -> ProviderError:
    status = getattr(e, "code", None) or getattr(e, "status_code", None) or 0
    try:
        status = int(status) if status else 0
    except (TypeError, ValueError):
        status = 0
    msg = getattr(e, "message", None) or str(e)
    if status in (401, 403):
        return ProviderError("auth_failed", f"gemini authentication failed: {msg}",
                             retriable=False, status=status)
    if status == 429:
        return ProviderError("rate_limit", f"gemini rate limit hit: {msg}",
                             retriable=True, status=status)
    if status == 404:
        return ProviderError("bad_request", f"gemini not found (check model name): {msg}",
                             retriable=False, status=status)
    if status >= 500:
        return ProviderError("upstream_error", f"gemini server error ({status}): {msg}",
                             retriable=True, status=status)
    if status >= 400:
        return ProviderError("bad_request", f"gemini rejected request ({status}): {msg}",
                             retriable=False, status=status)
    return ProviderError("upstream_error", f"gemini error: {msg}", retriable=False)


def _join_output_text(response: Any) -> str:
    """Concatenate the model's text from `response.steps`.

    Since the May 2026 Interactions breaking change the response is a `steps`
    timeline rather than a flat `outputs` list. We pull text only from
    `model_output` steps (skipping thoughts and tool-call/result steps), and
    within each, only the `text`-type content blocks.

    The SDK exposes an `output_text` convenience property, but we walk the
    steps ourselves with `getattr` so parsing stays robust if the SDK swaps
    model classes and never hard-depends on that property existing.
    """
    parts: list[str] = []
    for step in getattr(response, "steps", None) or []:
        if getattr(step, "type", None) != "model_output":
            continue
        for block in getattr(step, "content", None) or []:
            if getattr(block, "type", None) == "text":
                t = getattr(block, "text", None)
                if t:
                    parts.append(t)
    return "".join(parts)


def _extract_usage(response: Any) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return TokenUsage(
        input_tokens=getattr(usage, "total_input_tokens", None),
        output_tokens=getattr(usage, "total_output_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        reasoning_tokens=getattr(usage, "total_thought_tokens", None),
    )


def _extract_model_id(response: Any, fallback: str) -> str:
    m = getattr(response, "model", None)
    if m is None:
        return fallback
    if isinstance(m, str):
        return m
    return getattr(m, "id", None) or getattr(m, "name", None) or str(m)
