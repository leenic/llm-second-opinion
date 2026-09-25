"""Gemini terminal-status remap (DESIGN-submit-poll.md §18.2).

One fixture per Interactions API status, including unknown and missing, on
both the synchronous path (`generate`) and the background poll (`_classify`
via `poll_background`) — the two share `_build_response`, and these prove
it. `content_blocked` must never appear without an explicit block signal.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_second_opinion.providers.base import ProviderError, SecondOpinionRequest
from llm_second_opinion.providers.gemini import GeminiProvider, _block_signal

ANSWER = "the answer"
ALL_TERMINAL = ["completed", "failed", "incomplete", "cancelled", "budget_exceeded",
                "requires_action", "wibble", None]


class FakeClient:
    def __init__(self, result):
        self.result = result

        async def _create(**kwargs):
            return result

        async def _get(**kwargs):
            return result

        self.aio = SimpleNamespace(interactions=SimpleNamespace(create=_create, get=_get))


def provider_for(result) -> GeminiProvider:
    provider = GeminiProvider(api_key="k", model="gemini-test", timeout=30.0)
    provider._client = lambda: FakeClient(result)  # type: ignore[method-assign]
    return provider


def interaction(status, text=ANSWER, **extra) -> SimpleNamespace:
    return SimpleNamespace(
        id="int_1", status=status, output_text=text, model="gemini-test-001",
        usage=SimpleNamespace(total_input_tokens=1, total_output_tokens=2,
                              total_tokens=3, total_thought_tokens=7),
        **extra,
    )


def safety_error(code="SAFETY", message="Blocked for safety reasons") -> SimpleNamespace:
    return SimpleNamespace(code=code, message=message)


def request() -> SecondOpinionRequest:
    return SecondOpinionRequest("s", None, "sys", None, None)


async def sync_error(result) -> ProviderError:
    with pytest.raises(ProviderError) as exc:
        await provider_for(result).generate(request())
    return exc.value


async def poll_error(result) -> ProviderError:
    poll = await provider_for(result).poll_background("int_1", timeout=30.0)
    assert poll.done and poll.response is None and poll.error is not None
    return poll.error


PATHS = [sync_error, poll_error]


class TestCompleted:
    @pytest.mark.asyncio
    async def test_sync(self):
        resp = await provider_for(interaction("completed")).generate(request())
        assert resp.text == ANSWER

    @pytest.mark.asyncio
    async def test_poll(self):
        poll = await provider_for(interaction("completed")).poll_background("int_1", timeout=30.0)
        assert poll.done and poll.response.text == ANSWER


class TestFailed:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", PATHS)
    async def test_is_upstream_error_not_retriable(self, path):
        err = await path(interaction("failed", None, errors=[SimpleNamespace(code="INTERNAL", message="boom")]))
        assert err.error_type == "upstream_error"
        assert err.retriable is False
        assert "boom" in err.message


class TestIncomplete:
    """A token-cap outcome, not a safety signal."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", PATHS)
    async def test_without_block_signal_is_retriable_naming_max_tokens(self, path):
        err = await path(interaction("incomplete", None))
        assert err.error_type == "upstream_error"
        assert err.retriable is True
        assert "max_tokens" in err.message
        assert "reasoning_tokens=7" in err.message

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", PATHS)
    async def test_with_block_signal_is_content_blocked(self, path):
        err = await path(interaction("incomplete", None, errors=[safety_error()]))
        assert err.error_type == "content_blocked"
        assert err.retriable is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", PATHS)
    async def test_block_signal_on_a_step_counts(self, path):
        step = SimpleNamespace(
            type="model_output", content=[],
            error=SimpleNamespace(code=3, message="PROHIBITED_CONTENT", details=None),
        )
        err = await path(interaction("incomplete", None, steps=[step]))
        assert err.error_type == "content_blocked"


class TestCancelled:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", PATHS)
    async def test_is_upstream_error_not_retriable(self, path):
        err = await path(interaction("cancelled", None))
        assert err.error_type == "upstream_error"
        assert err.retriable is False
        assert "cancelled upstream" in err.message


class TestBudgetExceeded:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", PATHS)
    async def test_is_retriable_with_remedy(self, path):
        err = await path(interaction("budget_exceeded", None))
        assert err.error_type == "upstream_error"
        assert err.retriable is True
        assert "reasoning_effort" in err.message


class TestRequiresAction:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", PATHS)
    async def test_is_diagnostic_not_retriable(self, path):
        err = await path(interaction("requires_action", None))
        assert err.error_type == "upstream_error"
        assert err.retriable is False
        assert "requires_action" in err.message


class TestUnknownAndMissing:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", PATHS)
    async def test_unknown_status_carries_the_raw_value(self, path):
        err = await path(interaction("wibble"))
        assert err.error_type == "upstream_error"
        assert err.retriable is False
        assert "'wibble'" in err.message

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", PATHS)
    async def test_missing_status_is_not_a_success(self, path):
        err = await path(interaction(None))
        assert err.error_type == "upstream_error"
        assert err.retriable is False
        assert "missing" in err.message


class TestNeverContentBlockedWithoutASignal:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ALL_TERMINAL)
    @pytest.mark.parametrize("text", [ANSWER, None])
    async def test_sync(self, status, text):
        result = interaction(status, text)
        assert _block_signal(result) is None
        try:
            await provider_for(result).generate(request())
        except ProviderError as e:
            assert e.error_type != "content_blocked", (status, text)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ALL_TERMINAL)
    @pytest.mark.parametrize("text", [ANSWER, None])
    async def test_poll(self, status, text):
        result = interaction(status, text)
        poll = await provider_for(result).poll_background("int_1", timeout=30.0)
        assert poll.done
        if poll.error is not None:
            assert poll.error.error_type != "content_blocked", (status, text)

    def test_signal_detection_is_explicit(self):
        assert _block_signal(interaction("incomplete", None)) is None
        assert _block_signal(interaction("incomplete", None, errors=[safety_error()]))
        assert _block_signal(interaction("completed", None, block_reason="RECITATION"))
        assert _block_signal(interaction("completed", None, errors=[
            SimpleNamespace(code="DEADLINE_EXCEEDED", message="took too long")
        ])) is None

    @pytest.mark.asyncio
    async def test_completed_but_empty_with_block_signal_is_content_blocked(self):
        err = await sync_error(interaction("completed", None, errors=[safety_error()]))
        assert err.error_type == "content_blocked"
