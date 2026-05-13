"""Gemini provider via Google's Interactions API (google-genai SDK >= 1.55).

The Interactions API is the stateful counterpart to OpenAI's Responses API.
We use it in single-turn mode — no server-side state is carried between calls,
matching the v1 spec (no conversation history forwarded).
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

    # Map the common `reasoning_effort` string to a Gemini thinking_budget
    # (in tokens). "minimal" disables thinking; higher tiers let it think longer.
    EFFORT_TO_THINKING_BUDGET = {
        "minimal": 0,
        "low": 1024,
        "medium": 4096,
        "high": 16384,
    }

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

    def _client(self) -> genai.Client:
        return genai.Client(api_key=self.api_key)

    def _build_config(self, req: SecondOpinionRequest) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        if req.system_prompt:
            cfg["system_instruction"] = req.system_prompt
        if req.temperature is not None:
            cfg["temperature"] = req.temperature
        if req.max_tokens is not None:
            cfg["max_output_tokens"] = req.max_tokens
        if self.reasoning_effort:
            budget = self.EFFORT_TO_THINKING_BUDGET.get(self.reasoning_effort)
            if budget is not None:
                cfg["thinking_config"] = {"thinking_budget": budget}
        if self.web_search:
            cfg["tools"] = [{"google_search": {}}]
        return cfg

    async def generate(self, req: SecondOpinionRequest) -> SecondOpinionResponse:
        config = self._build_config(req)
        client = self._client()
        user_input = self.build_user_content(req)

        start = time.monotonic()
        try:
            coro = client.aio.interactions.create(
                model=self.model,
                input=user_input,
                config=config or None,
            )
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

        # Block / safety detection.
        blocked_reason = _detect_block(response)
        if blocked_reason:
            raise ProviderError(
                "content_blocked",
                f"gemini blocked the response: {blocked_reason}",
                retriable=False,
            )

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            text = _join_candidate_text(response)

        usage = _extract_usage(response)
        actual_model = (
            getattr(response, "model_version", None)
            or getattr(response, "model", None)
            or self.model
        )

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


def _detect_block(response: Any) -> str | None:
    # Prompt-level block.
    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback is not None:
        block_reason = getattr(prompt_feedback, "block_reason", None)
        if block_reason:
            return f"prompt blocked: {block_reason}"
    # Candidate-level block.
    candidates = getattr(response, "candidates", None) or []
    for c in candidates:
        fr = getattr(c, "finish_reason", None)
        if fr is None:
            continue
        # SDK exposes finish_reason as an enum; compare by name.
        fr_name = getattr(fr, "name", None) or str(fr)
        if fr_name in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"}:
            return f"finish_reason={fr_name}"
    return None


def _join_candidate_text(response: Any) -> str:
    parts: list[str] = []
    for cand in getattr(response, "candidates", None) or []:
        content = getattr(cand, "content", None)
        if content is None:
            continue
        for p in getattr(content, "parts", None) or []:
            t = getattr(p, "text", None)
            if t:
                parts.append(t)
    return "".join(parts)


def _extract_usage(response: Any) -> TokenUsage | None:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return None
    return TokenUsage(
        input_tokens=getattr(meta, "prompt_token_count", None),
        output_tokens=getattr(meta, "candidates_token_count", None),
        total_tokens=getattr(meta, "total_token_count", None),
    )
