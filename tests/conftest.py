"""Shared fakes for provider tests.

The provider adapters read SDK responses entirely through `getattr`, so plain
namespaces stand in for the real SDK models without pulling in the SDK.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="output_text", text=text)


def refusal_block(reason: str) -> SimpleNamespace:
    return SimpleNamespace(type="refusal", refusal=reason)


def message(*blocks: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(type="message", content=list(blocks))


def text_message(text: str) -> SimpleNamespace:
    return message(text_block(text))


def reasoning() -> SimpleNamespace:
    return SimpleNamespace(type="reasoning", content=[])


def web_search_call() -> SimpleNamespace:
    return SimpleNamespace(type="web_search_call", content=[])


def usage(
    input_tokens: int = 10,
    output_tokens: int = 20,
    total_tokens: int = 30,
    reasoning_tokens: int | None = 5,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )


def responses_result(
    *output: SimpleNamespace,
    status: str = "completed",
    model: str = "test-model",
    output_text: str | None = None,
    **extra: Any,
) -> SimpleNamespace:
    """Build a fake Responses-API result.

    `output_text` defaults to the SDK's real semantics — the concatenation of
    every message item — so tests can prove we do *not* rely on it.
    """
    items = list(output)
    if output_text is None:
        output_text = "".join(
            b.text
            for it in items
            if getattr(it, "type", None) == "message"
            for b in (getattr(it, "content", None) or [])
            if getattr(b, "type", None) == "output_text"
        )
    return SimpleNamespace(
        output=items,
        output_text=output_text,
        status=status,
        model=model,
        usage=extra.pop("usage", usage()),
        incomplete_details=extra.pop("incomplete_details", None),
        error=extra.pop("error", None),
        **extra,
    )


def bad_request(message: str) -> Exception:
    """A real `openai.BadRequestError`, so the provider's except-chain is
    exercised exactly as it would be against the live API."""
    import httpx
    from openai import BadRequestError

    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return BadRequestError(
        message, response=httpx.Response(400, request=request), body=None
    )


# Verbatim from a live gpt-5.6-sol call.
UNSUPPORTED_TEMPERATURE = (
    "Unsupported parameter: 'temperature' is not supported with this model."
)


class CallLog(list):
    """The kwargs of each `responses.create` call, plus `.timeouts` — the
    timeout override each retry was given."""

    timeouts: list


class FakeClient:
    """Stand-in for `AsyncOpenAI`.

    Models `with_options` because the retry path uses it to charge the second
    attempt only the time left on the original deadline; `timeouts` records
    what each attempt was given so tests can assert the budget shrinks.
    """

    def __init__(self, create, timeouts: list[float | None]):
        self.responses = SimpleNamespace(create=create)
        self.timeouts = timeouts

    def with_options(self, **options):
        self.timeouts.append(options.get("timeout"))
        return self


@pytest.fixture
def make_provider():
    """Build a provider whose HTTP client is replaced by a stub.

    Returns (provider, calls) — `calls` accumulates the kwargs each
    `responses.create` was invoked with, and `calls.timeouts` the timeout
    override passed to each retry.
    """

    def _make(result: Any, cls=None, **provider_kwargs):
        """`result` is a response, an Exception to raise, or a list of either
        to be served one per call (for retry paths)."""
        from llm_second_opinion.providers.openai_provider import OpenAIProvider

        cls = cls or OpenAIProvider
        calls = CallLog()
        calls.timeouts = []
        queue = list(result) if isinstance(result, list) else None

        async def _create(**kwargs):
            # Copy: the provider mutates its kwargs dict between attempts.
            calls.append(dict(kwargs))
            outcome = queue.pop(0) if queue is not None else result
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        kwargs = {"api_key": "test-key", "model": "test-model", "timeout": 30.0}
        kwargs.update(provider_kwargs)
        provider = cls(**kwargs)
        provider._client = lambda: FakeClient(  # type: ignore[method-assign]
            _create, calls.timeouts
        )
        return provider, calls

    return _make


@pytest.fixture
def request_factory():
    from llm_second_opinion.providers.base import SecondOpinionRequest

    def _make(**overrides):
        kwargs = {
            "summary": "a summary to review",
            "focus": None,
            "system_prompt": "be critical",
            "temperature": None,
            "max_tokens": None,
        }
        kwargs.update(overrides)
        return SecondOpinionRequest(**kwargs)

    return _make
