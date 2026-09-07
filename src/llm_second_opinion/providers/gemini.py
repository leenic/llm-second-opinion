"""Gemini provider via Google's Interactions API (google-genai SDK >= 2.0).

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
    BackgroundPoll,
    Provider,
    ProviderError,
    SecondOpinionRequest,
    SecondOpinionResponse,
    TokenUsage,
)

# Interaction statuses that mean "still working" on a background interaction
# (ai.google.dev/gemini-api/docs/background-execution, verified 2026-09-07).
_NON_TERMINAL_STATUSES = frozenset({"queued", "in_progress"})


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
        response = await self._call(
            lambda: client.aio.interactions.create(**kwargs), self.timeout
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        return self._build_response(response, latency_ms)

    async def _call(self, make_coro: Any, timeout: float) -> Any:
        """Run one SDK call under `timeout` with its errors mapped to ProviderError.

        The genai client takes no per-request timeout on this surface, so the
        bound is `asyncio.wait_for`. `make_coro` is called inside the guard so
        a synchronous raise from the SDK is mapped too.
        """
        try:
            return await asyncio.wait_for(make_coro(), timeout=timeout)
        except asyncio.TimeoutError as e:
            raise ProviderError(
                "timeout",
                f"gemini request timed out after {timeout}s",
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

    # -- background mode (design §2, §3.3; verified against the live docs) ---
    #
    # `background=True` makes create() return once the interaction is queued;
    # it is then polled with `interactions.get(id=…)` and stopped with
    # `interactions.cancel(id=…)`. Interactions are stored by default (paid
    # tier: 55 days, free tier: 1 day) and `store=false` is incompatible with
    # background execution, so nothing about storage is sent here.

    supports_background = True

    async def submit_background(self, req: SecondOpinionRequest, timeout: float) -> str:
        kwargs = self._build_kwargs(req)
        kwargs["background"] = True
        client = self._client()
        response = await self._call(lambda: client.aio.interactions.create(**kwargs), timeout)
        upstream_id = getattr(response, "id", None)
        if not upstream_id:
            raise ProviderError(
                "upstream_error",
                f"gemini accepted the background interaction but returned no id "
                f"to poll by (status={getattr(response, 'status', None)!r})",
                retriable=True,
            )
        return str(upstream_id)

    async def poll_background(self, upstream_id: str, timeout: float) -> BackgroundPoll:
        client = self._client()
        response = await self._call(lambda: client.aio.interactions.get(id=upstream_id), timeout)
        return self._classify(response)

    async def cancel_background(
        self, upstream_id: str, timeout: float
    ) -> BackgroundPoll | None:
        client = self._client()
        response = await self._call(lambda: client.aio.interactions.cancel(id=upstream_id), timeout)
        status = getattr(response, "status", None)
        if status in _NON_TERMINAL_STATUSES or status in (None, "cancelled"):
            # Stopped, or the status has not caught up with the cancel yet
            # ("clean-up actions on the server can cause a slight delay").
            return BackgroundPoll(True, upstream_status=status)
        # Already finished before the cancel landed: keep that result.
        return self._classify(response)

    def _classify(self, response: Any) -> BackgroundPoll:
        """Map a retrieved Interaction to a BackgroundPoll: `queued` /
        `in_progress` ⇒ running; `requires_action` ⇒ terminal failure; any
        other status through `_build_response`, the synchronous path's code."""
        status = getattr(response, "status", None)
        if status in _NON_TERMINAL_STATUSES:
            return BackgroundPoll(False, upstream_status=status)
        if status == "requires_action":
            # Paused for client tool input — nothing here will ever provide it.
            return BackgroundPoll(
                True,
                error=ProviderError(
                    "upstream_error",
                    "gemini interaction paused waiting for client input "
                    "(status=requires_action), which this server does not support",
                    retriable=False,
                ),
                upstream_status=status,
            )
        try:
            return BackgroundPoll(
                True, response=self._build_response(response, 0), upstream_status=status
            )
        except ProviderError as e:
            return BackgroundPoll(True, error=e, upstream_status=status)

    def _build_response(self, response: Any, latency_ms: int) -> SecondOpinionResponse:
        """Validate a terminal Interaction and extract the final answer.

        Shared by the synchronous path and the background poller, so the
        status mappings are identical on both.
        """
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
    """Return Gemini's final answer text from the Interaction.

    Since the May 2026 Interactions breaking change the response is a `steps`
    timeline rather than a flat `outputs` list, and a single interaction may
    carry several `model_output` steps interleaved with thought and
    tool-call/result steps (e.g. when `web_search` grounding runs). We must
    return only the *trailing* run of model output — concatenating every
    `model_output` step would prepend intermediate model chatter to the final
    answer.

    We prefer the SDK's `output_text` property, which already returns exactly
    that trailing run. If it is unavailable we fall back to a manual walk that
    mirrors the same semantics, matching on the `type` discriminator with
    `getattr` so parsing stays robust if the SDK swaps model classes.
    """
    sdk_text = getattr(response, "output_text", None)
    if isinstance(sdk_text, str):
        return sdk_text

    # Fallback: walk the timeline backwards, collecting the trailing run of
    # model-output text and stopping at the first non-model-output boundary.
    parts: list[str] = []
    collecting = False
    for step in reversed(getattr(response, "steps", None) or []):
        step_type = getattr(step, "type", None)
        if step_type == "user_input":
            break
        content = getattr(step, "content", None)
        if step_type != "model_output" or not content:
            if collecting:
                break
            continue
        hit_barrier = False
        for block in reversed(content):
            if getattr(block, "type", None) == "text":
                collecting = True
                t = getattr(block, "text", None)
                if t:
                    parts.append(t)
            elif collecting:
                hit_barrier = True
                break
        if hit_barrier:
            break
    parts.reverse()
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
