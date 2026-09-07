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
    BackgroundPoll,
    Provider,
    ProviderError,
    SecondOpinionRequest,
    SecondOpinionResponse,
    TokenUsage,
)

log = logging.getLogger(__name__)

# Responses API statuses that mean "still working" on a background response
# (developers.openai.com/api/docs/guides/background, verified 2026-09-07).
# Everything else — completed, failed, incomplete, cancelled — is terminal.
_NON_TERMINAL_STATUSES = frozenset({"queued", "in_progress"})

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
        response = await self._create_with_fallback(client, kwargs, start, self.timeout)
        latency_ms = int((time.monotonic() - start) * 1000)
        return self._build_response(response, latency_ms)

    async def _create_with_fallback(
        self, client: AsyncOpenAI, kwargs: dict[str, Any], start: float, budget: float
    ) -> Any:
        """`_create`, plus the one sanctioned retry without a rejected sampling knob.

        Shared by the synchronous path and background submission: a 400 for
        an unsupported parameter arrives synchronously either way.
        """
        try:
            return await self._create(client, kwargs)
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
            # The retry must fit inside the *original* deadline. `self.timeout`
            # is bound to the client at construction, so a second `_create`
            # would silently get a fresh full budget and let one tool call run
            # up to 2x `timeout_seconds` — past the MCP client's ~240s cap,
            # which is the exact failure `max_retries=0` above exists to
            # prevent. Charge the retry only what's left.
            remaining = budget - (time.monotonic() - start)
            if remaining <= 0:
                raise
            log.warning(
                "%s model=%s rejected %r; retrying without it (%.1fs left)",
                self.name, self.model, dropped, remaining,
            )
            kwargs.pop(dropped)
            return await self._create(client.with_options(timeout=remaining), kwargs)

    async def _create(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> Any:
        """One call to the Responses API, with SDK errors mapped to ProviderError."""
        return await self._mapped(client.responses.create(**kwargs), self.timeout)

    # -- background mode (design §2, §3.3; verified against the live docs) ---
    #
    # `background: true` returns as soon as the response is queued; the
    # response is then polled by id. `store: true` is sent explicitly because
    # background responses are retained beyond a ~10-minute window only when
    # stored (Zero-Data-Retention projects run background with store=false).
    # Cancel is `POST /v1/responses/{id}/cancel`, idempotent: cancelling a
    # finished response just returns it.

    supports_background = True

    async def submit_background(self, req: SecondOpinionRequest, timeout: float) -> str:
        kwargs = self._build_kwargs(req)
        kwargs["background"] = True
        kwargs["store"] = True
        client = self._client().with_options(timeout=timeout)
        start = time.monotonic()
        response = await self._create_with_fallback(client, kwargs, start, timeout)
        upstream_id = getattr(response, "id", None)
        if not upstream_id:
            raise ProviderError(
                "upstream_error",
                f"{self.name} accepted the background request but returned no "
                f"response id to poll by (status={getattr(response, 'status', None)!r})",
                retriable=True,
            )
        return str(upstream_id)

    async def poll_background(self, upstream_id: str, timeout: float) -> BackgroundPoll:
        client = self._client().with_options(timeout=timeout)
        response = await self._mapped(client.responses.retrieve(upstream_id), timeout)
        poll = self._classify(response)
        if poll.done and poll.response is None and poll.error is None:
            # Cancelled from outside this server: nobody here asked for it.
            poll.error = ProviderError(
                "upstream_error",
                f"{self.name} response was cancelled upstream",
                retriable=False,
            )
        return poll

    async def cancel_background(
        self, upstream_id: str, timeout: float
    ) -> BackgroundPoll | None:
        client = self._client().with_options(timeout=timeout)
        # Idempotent upstream: a response that already finished is returned
        # as-is, so the caller can keep that result instead of losing it.
        response = await self._mapped(client.responses.cancel(upstream_id), timeout)
        poll = self._classify(response)
        if not poll.done:
            # The cancel was accepted but the status has not caught up yet;
            # from this server's point of view the job is cancelled.
            return BackgroundPoll(True, upstream_status=poll.upstream_status)
        return poll

    def _classify(self, response: Any) -> BackgroundPoll:
        """Map a retrieved/cancelled Responses payload to a BackgroundPoll.

        `queued`/`in_progress` ⇒ running; `cancelled` ⇒ done with neither
        response nor error; any other terminal status goes through
        `_build_response`, the exact code the synchronous path uses, so the
        failed/incomplete/refusal/empty mappings are shared.
        """
        status = getattr(response, "status", None)
        if status in _NON_TERMINAL_STATUSES:
            return BackgroundPoll(False, upstream_status=status)
        if status == "cancelled":
            return BackgroundPoll(True, upstream_status=status)
        try:
            return BackgroundPoll(
                True, response=self._build_response(response, 0), upstream_status=status
            )
        except ProviderError as e:
            return BackgroundPoll(True, error=e, upstream_status=status)

    async def _mapped(self, awaitable: Any, timeout: float) -> Any:
        """Await one SDK call with its exceptions mapped to ProviderError."""
        try:
            return await awaitable
        except APITimeoutError as e:
            raise ProviderError(
                "timeout",
                f"{self.name} request timed out after {timeout}s",
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
