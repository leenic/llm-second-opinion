"""OpenAI (ChatGPT) provider via the Responses API.

Uses the official `openai` SDK (>=2.36). The same shape works for xAI (Grok)
via the OpenAI-compatible endpoint — see grok.py.
"""

from __future__ import annotations

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
        kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return AsyncOpenAI(**kwargs)

    async def generate(self, req: SecondOpinionRequest) -> SecondOpinionResponse:
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

        start = time.monotonic()
        try:
            client = self._client()
            response = await client.responses.create(**kwargs)
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

        latency_ms = int((time.monotonic() - start) * 1000)

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

        text = (getattr(response, "output_text", None) or "").strip()
        if not text:
            refusal = _extract_refusal(response)
            if refusal:
                raise ProviderError(
                    "content_blocked",
                    f"{self.name} refused the request: {refusal}",
                    retriable=False,
                )
            # `output_text` is the concatenation of message-type items only;
            # if it's empty, try walking the full output list once more.
            text = _join_output_text(response).strip()

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


def _join_output_text(response: Any) -> str:
    parts: list[str] = []
    output = getattr(response, "output", None) or []
    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        content = getattr(item, "content", None) or []
        for c in content:
            if getattr(c, "type", None) == "output_text":
                t = getattr(c, "text", None)
                if t:
                    parts.append(t)
    return "".join(parts)


class OpenAIProvider(ResponsesAPIProvider):
    name = "openai"
    base_url = None  # SDK default — https://api.openai.com/v1
