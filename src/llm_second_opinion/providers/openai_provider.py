"""OpenAI (ChatGPT) provider via the Responses API.

Uses the official `openai` SDK (>=2.36). The same shape works for xAI (Grok)
via the OpenAI-compatible endpoint — see grok.py.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from .base import (
    Provider,
    ProviderError,
    SecondOpinionRequest,
    SecondOpinionResponse,
    TokenUsage,
)

log = logging.getLogger(__name__)

# Sampling knobs we can drop and still answer the user's actual question.
# Reasoning-only models reject these outright rather than ignoring them:
# gpt-5.6-sol returns `400 Unsupported parameter: 'temperature' is not
# supported with this model.` Deliberately excludes `max_output_tokens` — a
# length cap bounds cost and truncation, so silently dropping it could return
# a far longer and more expensive reply than the caller asked for. Better to
# surface that as an error.
DROPPABLE_PARAMS = frozenset({"temperature", "top_p"})

_UNSUPPORTED_PARAM_RE = re.compile(r"[Uu]nsupported parameter: '([^']+)'")


class ResponsesAPIProvider(Provider):
    """Adapter for any OpenAI-compatible Responses API endpoint.

    Subclasses set `name` and `base_url`. `base_url=None` means use the
    SDK's default (OpenAI).
    """

    name = ""
    base_url: str | None = None

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

    def _client(self) -> AsyncOpenAI:
        # max_retries=0: the SDK defaults to 2, but on a slow reasoning+web_search
        # call each retry restarts the full request. Stacked retries blow past the
        # MCP client's ~240s tool-call deadline, so the call is cancelled before we
        # can return our own `timeout` error. With retries off, `self.timeout`
        # actually bounds wall-clock time and a timeout surfaces cleanly. Claude can
        # re-issue the whole tool call (our errors carry `retriable`).
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": 0,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return AsyncOpenAI(**kwargs)

    def _build_kwargs(self, req: SecondOpinionRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": self.build_user_content(req),
        }
        if req.system_prompt:
            kwargs["instructions"] = req.system_prompt
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            kwargs["max_output_tokens"] = req.max_tokens
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.web_search:
            kwargs["tools"] = [{"type": "web_search"}]
        return kwargs

    async def generate(self, req: SecondOpinionRequest) -> SecondOpinionResponse:
        kwargs = self._build_kwargs(req)
        client = self._client()

        start = time.monotonic()
        try:
            response = await self._create(client, kwargs)
        except ProviderError as e:
            # A reasoning-only model rejects sampling knobs instead of ignoring
            # them, failing the whole call over a parameter that was only ever
            # advisory. Drop the offending one and ask again rather than
            # spending the user's round-trip on a 400. Retried at most once:
            # we only ever send one droppable param, and a second rejection
            # means something we can't paper over.
            dropped = _droppable_unsupported_param(e, kwargs)
            if dropped is None:
                raise
            log.warning(
                "%s model=%s rejected %r; retrying without it",
                self.name, self.model, dropped,
            )
            kwargs.pop(dropped)
            response = await self._create(client, kwargs)

        latency_ms = int((time.monotonic() - start) * 1000)
        return self._build_response(response, latency_ms)

    async def _create(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> Any:
        """One call to the Responses API, with SDK errors mapped to ProviderError."""
        try:
            return await client.responses.create(**kwargs)
        except APITimeoutError as e:
            raise ProviderError(
                "timeout",
                f"{self.name} request timed out after {self.timeout}s",
                retriable=True,
            ) from e
        except AuthenticationError as e:
            raise ProviderError("auth_failed", f"{self.name} authentication failed: {e}",
                                retriable=False, status=getattr(e, "status_code", 401)) from e
        except PermissionDeniedError as e:
            raise ProviderError("auth_failed", f"{self.name} permission denied: {e}",
                                retriable=False, status=getattr(e, "status_code", 403)) from e
        except RateLimitError as e:
            raise ProviderError("rate_limit", f"{self.name} rate limit hit: {e}",
                                retriable=True, status=getattr(e, "status_code", 429)) from e
        except APIConnectionError as e:
            raise ProviderError("network_error", f"{self.name} network error: {e}",
                                retriable=True) from e
        except NotFoundError as e:
            raise ProviderError("bad_request", f"{self.name} not found (check model name): {e}",
                                retriable=False, status=getattr(e, "status_code", 404)) from e
        except BadRequestError as e:
            raise ProviderError("bad_request", f"{self.name} rejected request: {e}",
                                retriable=False, status=getattr(e, "status_code", 400)) from e
        except APIStatusError as e:
            status = getattr(e, "status_code", None) or 0
            if status >= 500:
                raise ProviderError("upstream_error",
                                    f"{self.name} server error ({status}): {e}",
                                    retriable=True, status=status) from e
            raise ProviderError("upstream_error", f"{self.name} error ({status}): {e}",
                                retriable=False, status=status) from e

    def _build_response(self, response: Any, latency_ms: int) -> SecondOpinionResponse:
        """Validate a completed Responses payload and extract the final answer."""
        usage = _extract_usage(response)
        status = getattr(response, "status", None)
        incomplete = getattr(response, "incomplete_details", None)
        incomplete_reason = getattr(incomplete, "reason", None) if incomplete else None

        if status == "failed":
            err = getattr(response, "error", None)
            err_msg = getattr(err, "message", None) if err else None
            raise ProviderError(
                "upstream_error",
                f"{self.name} response failed: {err_msg or 'unknown error'}",
                retriable=False,
            )

        if status == "incomplete":
            reasoning_tokens = usage.reasoning_tokens if usage else None
            if incomplete_reason == "content_filter":
                raise ProviderError(
                    "content_blocked",
                    f"{self.name} response blocked by content filter",
                    retriable=False,
                )
            # The common "reasoning consumed the whole output budget" case.
            raise ProviderError(
                "upstream_error",
                f"{self.name} response was cut short (reason="
                f"{incomplete_reason!r}, reasoning_tokens={reasoning_tokens}). "
                f"Retry with a higher max_tokens, a lower reasoning_effort, "
                f"or omit max_tokens entirely.",
                retriable=True,
            )

        # Deliberately not `response.output_text` — see _final_message_text.
        text = _final_message_text(response).strip()
        if not text:
            refusal = _extract_refusal(response)
            if refusal:
                raise ProviderError(
                    "content_blocked",
                    f"{self.name} refused the request: {refusal}",
                    retriable=False,
                )

        if not text:
            # status was 'completed' but no visible text — usually means the
            # model emitted only reasoning items, or every output got consumed
            # by tool calls. Don't pretend this was a success.
            reasoning_tokens = usage.reasoning_tokens if usage else None
            output_count = len(getattr(response, "output", None) or [])
            raise ProviderError(
                "upstream_error",
                f"{self.name} returned no visible text (status={status!r}, "
                f"output_items={output_count}, reasoning_tokens={reasoning_tokens}). "
                f"Retry with a lower reasoning_effort, a higher max_tokens, "
                f"or rephrase so the model answers directly.",
                retriable=True,
            )

        actual_model = getattr(response, "model", None) or self.model

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
            await client.with_options(timeout=5.0).models.list()
        except AuthenticationError:
            return False, "authentication failed"
        except PermissionDeniedError:
            return False, "permission denied"
        except APIConnectionError as e:
            return False, f"network error: {e}"
        except APIStatusError as e:
            return False, f"http {getattr(e, 'status_code', 'error')}"
        except Exception as e:  # noqa: BLE001
            return False, f"unexpected error: {e}"
        return True, None


def _droppable_unsupported_param(
    error: ProviderError, kwargs: dict[str, Any]
) -> str | None:
    """Name of the sampling param this error blames, if we can safely drop it.

    Returns None — meaning "let the error propagate" — unless all of:
    the failure was a 400, the message names a parameter, that parameter is
    one we consider advisory (`DROPPABLE_PARAMS`), and we actually sent it.
    The last check matters: retrying without a param we never sent would just
    replay the identical request and burn a second round-trip.
    """
    if error.error_type != "bad_request":
        return None
    match = _UNSUPPORTED_PARAM_RE.search(error.message)
    if not match:
        return None
    name = match.group(1)
    if name not in DROPPABLE_PARAMS or name not in kwargs:
        return None
    return name


def _extract_usage(response: Any) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    reasoning_tokens: int | None = None
    details = getattr(usage, "output_tokens_details", None)
    if details is not None:
        reasoning_tokens = getattr(details, "reasoning_tokens", None)
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        reasoning_tokens=reasoning_tokens,
    )


def _extract_refusal(response: Any) -> str | None:
    """Walk `response.output` looking for a refusal item."""
    output = getattr(response, "output", None) or []
    for item in output:
        content = getattr(item, "content", None) or []
        for c in content:
            ctype = getattr(c, "type", None)
            if ctype == "refusal":
                return getattr(c, "refusal", None) or getattr(c, "text", None)
    return None


def _final_message_text(response: Any) -> str:
    """Return only the model's final answer from `response.output`.

    We can't use `response.output_text`: it concatenates the text of *every*
    message item in the timeline. A reasoning model that narrates before
    calling a tool emits those asides as ordinary message items, so
    `output_text` glues them onto the front of the real answer with no
    separator — e.g. "I need current best practices...**Do not store 30-day
    tokens...". Measured on grok-4.5 + web_search at roughly half of runs
    (message items at output indices [1, 8, 19]); gpt-5.6-sol did not do it,
    but the shape is model behaviour, not provider behaviour.

    So we walk the timeline backwards and keep only the trailing run of
    message items, stopping at the first non-message item — a reasoning step
    or a tool call marks the boundary of the final turn. Leading non-message
    items are skipped so a trailing reasoning item can't hide the answer.
    Mirrors `gemini._join_output_text`, which solves the same problem on the
    Interactions API `steps` timeline.
    """
    parts: list[str] = []
    collecting = False
    for item in reversed(getattr(response, "output", None) or []):
        if getattr(item, "type", None) != "message":
            if collecting:
                break
            continue
        chunk: list[str] = []
        for c in getattr(item, "content", None) or []:
            if getattr(c, "type", None) == "output_text":
                t = getattr(c, "text", None)
                if t:
                    chunk.append(t)
        if chunk:
            collecting = True
            parts.append("".join(chunk))
    parts.reverse()
    return "".join(parts)


class OpenAIProvider(ResponsesAPIProvider):
    name = "openai"
    base_url = None  # SDK default — https://api.openai.com/v1
