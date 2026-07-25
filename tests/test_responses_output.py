"""Regression tests for final-answer extraction on the Responses API.

Motivating bug: grok-4.5 with `web_search` narrates before calling tools
("I need current best practices...") and those asides arrive as ordinary
message items. `response.output_text` concatenates every message item, so the
narration was glued onto the front of the real answer with no separator. This
reproduced on roughly half of live runs.
"""

from __future__ import annotations

import pytest

from llm_second_opinion.providers.base import ProviderError
from llm_second_opinion.providers.grok import GrokProvider
from llm_second_opinion.providers.openai_provider import (
    OpenAIProvider,
    _final_message_text,
)

from conftest import (
    UNSUPPORTED_TEMPERATURE,
    bad_request,
    message,
    reasoning,
    refusal_block,
    responses_result,
    text_block,
    text_message,
    web_search_call,
)

ANSWER = "**Do not store 30-day full-access tokens in localStorage.**"
ASIDE_1 = "I need current best practices for token storage."
ASIDE_2 = "Fetching detailed guidance from high-quality sources."


class TestFinalMessageText:
    def test_single_message(self):
        r = responses_result(reasoning(), text_message(ANSWER))
        assert _final_message_text(r) == ANSWER

    def test_drops_narration_before_tool_calls(self):
        """The exact live shape: messages at indices [1, 8, 19]."""
        r = responses_result(
            reasoning(),
            text_message(ASIDE_1),
            web_search_call(),
            reasoning(),
            text_message(ASIDE_2),
            web_search_call(),
            reasoning(),
            text_message(ANSWER),
        )
        assert _final_message_text(r) == ANSWER
        # Guard against a regression to output_text, which glues all three.
        assert r.output_text == ASIDE_1 + ASIDE_2 + ANSWER

    def test_keeps_adjacent_trailing_messages(self):
        """A final answer split across adjacent messages is still one answer."""
        r = responses_result(
            text_message(ASIDE_1),
            web_search_call(),
            text_message("Part one. "),
            text_message("Part two."),
        )
        assert _final_message_text(r) == "Part one. Part two."

    def test_multiple_text_blocks_in_one_message(self):
        r = responses_result(
            web_search_call(),
            message(text_block("alpha "), text_block("beta")),
        )
        assert _final_message_text(r) == "alpha beta"

    def test_trailing_reasoning_does_not_hide_answer(self):
        """A reasoning item after the message must not truncate to empty."""
        r = responses_result(text_message(ANSWER), reasoning())
        assert _final_message_text(r) == ANSWER

    def test_non_text_trailing_message_is_skipped(self):
        """A trailing refusal-only message shouldn't mask the real answer."""
        r = responses_result(text_message(ANSWER), message(refusal_block("no")))
        assert _final_message_text(r) == ANSWER

    @pytest.mark.parametrize(
        "r",
        [
            responses_result(),
            responses_result(reasoning(), web_search_call()),
        ],
        ids=["empty_output", "no_message_items"],
    )
    def test_no_text_returns_empty(self, r):
        assert _final_message_text(r) == ""

    def test_missing_output_attribute(self):
        class Bare:
            pass

        assert _final_message_text(Bare()) == ""


class TestGenerateIntegration:
    """Same behaviour, but through the full `generate` path."""

    @pytest.mark.asyncio
    async def test_generate_returns_only_final_answer(
        self, make_provider, request_factory
    ):
        result = responses_result(
            reasoning(),
            text_message(ASIDE_1),
            web_search_call(),
            text_message(ANSWER),
            model="grok-4.5",
        )
        provider, _ = make_provider(result, cls=GrokProvider, web_search=True)
        resp = await provider.generate(request_factory())

        assert resp.text == ANSWER
        assert ASIDE_1 not in resp.text
        assert resp.model == "grok-4.5"
        assert resp.provider == "grok"
        assert resp.usage.reasoning_tokens == 5

    @pytest.mark.asyncio
    async def test_generate_strips_surrounding_whitespace(
        self, make_provider, request_factory
    ):
        result = responses_result(web_search_call(), text_message(f"\n\n{ANSWER}\n "))
        provider, _ = make_provider(result)
        assert (await provider.generate(request_factory())).text == ANSWER

    @pytest.mark.asyncio
    async def test_refusal_surfaces_as_content_blocked(
        self, make_provider, request_factory
    ):
        result = responses_result(message(refusal_block("I can't help with that.")))
        provider, _ = make_provider(result)
        with pytest.raises(ProviderError) as exc:
            await provider.generate(request_factory())
        assert exc.value.error_type == "content_blocked"
        assert exc.value.retriable is False

    @pytest.mark.asyncio
    async def test_no_visible_text_is_an_error_not_a_silent_empty(
        self, make_provider, request_factory
    ):
        """Reasoning-only output must not be reported as a successful reply."""
        result = responses_result(reasoning(), web_search_call())
        provider, _ = make_provider(result)
        with pytest.raises(ProviderError) as exc:
            await provider.generate(request_factory())
        assert exc.value.error_type == "upstream_error"
        assert exc.value.retriable is True

    @pytest.mark.asyncio
    async def test_output_text_is_not_trusted_as_a_fallback(
        self, make_provider, request_factory
    ):
        """If output_text disagrees with the timeline, the timeline wins."""
        result = responses_result(
            reasoning(), output_text="stale text the SDK computed"
        )
        provider, _ = make_provider(result)
        with pytest.raises(ProviderError):
            await provider.generate(request_factory())


class TestRequestShape:
    """Guards on what we send — these encode live-tested provider constraints."""

    @pytest.mark.asyncio
    async def test_web_search_and_reasoning_are_forwarded(
        self, make_provider, request_factory
    ):
        provider, calls = make_provider(
            responses_result(text_message(ANSWER)),
            reasoning_effort="high",
            web_search=True,
        )
        await provider.generate(request_factory(focus="security"))

        assert calls[0]["reasoning"] == {"effort": "high"}
        assert calls[0]["tools"] == [{"type": "web_search"}]
        assert calls[0]["instructions"] == "be critical"
        assert calls[0]["input"].startswith("Focus on: security")

    @pytest.mark.asyncio
    async def test_optional_params_omitted_when_unset(
        self, make_provider, request_factory
    ):
        """gpt-5.6-sol rejects `temperature`; never send it unasked."""
        provider, calls = make_provider(responses_result(text_message(ANSWER)))
        await provider.generate(request_factory())

        assert "temperature" not in calls[0]
        assert "max_output_tokens" not in calls[0]
        assert "reasoning" not in calls[0]
        assert "tools" not in calls[0]

    @pytest.mark.asyncio
    async def test_optional_params_forwarded_when_set(
        self, make_provider, request_factory
    ):
        provider, calls = make_provider(responses_result(text_message(ANSWER)))
        await provider.generate(request_factory(temperature=0.2, max_tokens=2000))

        assert calls[0]["temperature"] == 0.2
        assert calls[0]["max_output_tokens"] == 2000

    def test_grok_targets_the_xai_base_url(self):
        assert GrokProvider.base_url == "https://api.x.ai/v1"
        assert OpenAIProvider.base_url is None


class TestUnsupportedParamRetry:
    """gpt-5.6-sol rejects `temperature` outright instead of ignoring it.

    Failing the whole call over an advisory sampling knob costs the user a
    round-trip, so we drop the offending param and ask once more.
    """

    @pytest.mark.asyncio
    async def test_temperature_rejection_is_retried_without_it(
        self, make_provider, request_factory
    ):
        provider, calls = make_provider(
            [bad_request(UNSUPPORTED_TEMPERATURE), responses_result(text_message(ANSWER))]
        )
        resp = await provider.generate(request_factory(temperature=0.2, max_tokens=500))

        assert resp.text == ANSWER
        assert len(calls) == 2
        assert calls[0]["temperature"] == 0.2
        assert "temperature" not in calls[1]
        # Everything else must survive the retry unchanged.
        assert calls[1]["max_output_tokens"] == 500
        assert calls[1]["input"] == calls[0]["input"]
        assert calls[1]["instructions"] == calls[0]["instructions"]

    @pytest.mark.asyncio
    async def test_latency_covers_both_attempts(self, make_provider, request_factory):
        provider, _ = make_provider(
            [bad_request(UNSUPPORTED_TEMPERATURE), responses_result(text_message(ANSWER))]
        )
        resp = await provider.generate(request_factory(temperature=0.2))
        assert resp.latency_ms >= 0  # measured across the retry, not reset

    @pytest.mark.asyncio
    async def test_no_retry_when_temperature_was_not_sent(
        self, make_provider, request_factory
    ):
        """Retrying an identical request would just burn a second round-trip."""
        provider, calls = make_provider(bad_request(UNSUPPORTED_TEMPERATURE))
        with pytest.raises(ProviderError) as exc:
            await provider.generate(request_factory())

        assert exc.value.error_type == "bad_request"
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_unrelated_bad_request_is_not_retried(
        self, make_provider, request_factory
    ):
        provider, calls = make_provider(bad_request("Invalid value for 'input'."))
        with pytest.raises(ProviderError) as exc:
            await provider.generate(request_factory(temperature=0.2))

        assert exc.value.error_type == "bad_request"
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_max_output_tokens_is_never_dropped(
        self, make_provider, request_factory
    ):
        """Dropping a length cap would change cost and truncation behaviour,
        so it must surface as an error rather than being papered over."""
        provider, calls = make_provider(
            bad_request("Unsupported parameter: 'max_output_tokens' is not supported.")
        )
        with pytest.raises(ProviderError):
            await provider.generate(request_factory(max_tokens=500))
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_second_rejection_propagates(self, make_provider, request_factory):
        """At most one retry — no unbounded strip-and-retry loop."""
        provider, calls = make_provider(
            [
                bad_request(UNSUPPORTED_TEMPERATURE),
                bad_request("Unsupported parameter: 'top_p' is not supported."),
            ]
        )
        with pytest.raises(ProviderError) as exc:
            await provider.generate(request_factory(temperature=0.2))

        assert "top_p" in exc.value.message
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_non_bad_request_errors_are_untouched(
        self, make_provider, request_factory
    ):
        """A rate limit must stay retriable, not get swallowed by the retry."""
        from openai import RateLimitError
        import httpx

        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        err = RateLimitError(
            "slow down", response=httpx.Response(429, request=request), body=None
        )
        provider, calls = make_provider(err)
        with pytest.raises(ProviderError) as exc:
            await provider.generate(request_factory(temperature=0.2))

        assert exc.value.error_type == "rate_limit"
        assert exc.value.retriable is True
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_grok_keeps_temperature_when_accepted(
        self, make_provider, request_factory
    ):
        """grok-4.5 and gemini-3.6-flash accept temperature — no retry, and
        the value must actually reach the API."""
        provider, calls = make_provider(
            responses_result(text_message(ANSWER)), cls=GrokProvider
        )
        await provider.generate(request_factory(temperature=0.2))

        assert len(calls) == 1
        assert calls[0]["temperature"] == 0.2


class TestDroppableParamHelper:
    @pytest.mark.parametrize(
        "message,sent,expected",
        [
            (UNSUPPORTED_TEMPERATURE, {"temperature": 0.2}, "temperature"),
            ("unsupported parameter: 'temperature' bla", {"temperature": 0}, "temperature"),
            (UNSUPPORTED_TEMPERATURE, {}, None),
            ("Unsupported parameter: 'max_output_tokens'.", {"max_output_tokens": 5}, None),
            ("Unsupported parameter: 'reasoning'.", {"reasoning": {}}, None),
            ("something else entirely", {"temperature": 0.2}, None),
        ],
    )
    def test_matching(self, message, sent, expected):
        from llm_second_opinion.providers.openai_provider import (
            _droppable_unsupported_param,
        )

        err = ProviderError("bad_request", message)
        assert _droppable_unsupported_param(err, sent) == expected

    def test_only_bad_request_qualifies(self):
        from llm_second_opinion.providers.openai_provider import (
            _droppable_unsupported_param,
        )

        err = ProviderError("upstream_error", UNSUPPORTED_TEMPERATURE)
        assert _droppable_unsupported_param(err, {"temperature": 0.2}) is None


class TestTerminalStatuses:
    @pytest.mark.asyncio
    async def test_incomplete_from_token_budget_is_retriable(
        self, make_provider, request_factory
    ):
        from types import SimpleNamespace

        result = responses_result(
            reasoning(),
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        )
        provider, _ = make_provider(result)
        with pytest.raises(ProviderError) as exc:
            await provider.generate(request_factory())
        assert exc.value.error_type == "upstream_error"
        assert exc.value.retriable is True

    @pytest.mark.asyncio
    async def test_incomplete_from_content_filter_is_blocked(
        self, make_provider, request_factory
    ):
        from types import SimpleNamespace

        result = responses_result(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="content_filter"),
        )
        provider, _ = make_provider(result)
        with pytest.raises(ProviderError) as exc:
            await provider.generate(request_factory())
        assert exc.value.error_type == "content_blocked"
        assert exc.value.retriable is False

    @pytest.mark.asyncio
    async def test_failed_status_reports_upstream_message(
        self, make_provider, request_factory
    ):
        from types import SimpleNamespace

        result = responses_result(
            status="failed", error=SimpleNamespace(message="upstream exploded")
        )
        provider, _ = make_provider(result)
        with pytest.raises(ProviderError) as exc:
            await provider.generate(request_factory())
        assert "upstream exploded" in exc.value.message
