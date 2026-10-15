"""Claude adapter (providers/anthropic_provider.py).

Request shaping and stop-reason mapping are tested on plain fakes; the wire
contract (beta header, `fallbacks`, no `thinking`, no `temperature` on
models that reject it) is tested through the real SDK client with an httpx2
mock transport answering a streamed Messages response, so what is asserted
is what would leave the machine.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx2
import pytest
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient

from llm_second_opinion.config import AppConfig, ProviderConfig
from llm_second_opinion.providers import build_provider
from llm_second_opinion.providers import anthropic_provider as mod
from llm_second_opinion.providers.anthropic_provider import AnthropicProvider, _final_text
from llm_second_opinion.providers.base import ProviderError, SecondOpinionRequest

ANSWER = "The migration plan misses the rollback path."


def request(**overrides) -> SecondOpinionRequest:
    base = dict(summary="a summary", focus=None, system_prompt="be critical",
                temperature=None, max_tokens=500)
    base.update(overrides)
    return SecondOpinionRequest(**base)


def provider(model: str = "claude-fable-5-1", **kw) -> AnthropicProvider:
    return AnthropicProvider(api_key="k", model=model, timeout=30.0, **kw)


def text(t: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=t)


def block(kind: str) -> SimpleNamespace:
    return SimpleNamespace(type=kind)


def message(*content, stop_reason="end_turn", model="claude-fable-5-1",
            input_tokens=100, output_tokens=40, thinking_tokens=None, stop_details=None):
    usage = SimpleNamespace(
        input_tokens=input_tokens, output_tokens=output_tokens,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
        output_tokens_details=SimpleNamespace(thinking_tokens=thinking_tokens),
    )
    return SimpleNamespace(content=list(content), stop_reason=stop_reason, model=model,
                           usage=usage, stop_details=stop_details)


class TestBuildKwargs:
    def test_fable_request_shape(self):
        kw = provider(reasoning_effort="high", web_search=True)._build_kwargs(
            request(temperature=0.3)
        )
        assert kw["model"] == "claude-fable-5-1"
        assert kw["max_tokens"] == 500
        assert kw["system"] == "be critical"
        assert kw["messages"] == [{"role": "user", "content": "a summary"}]
        assert kw["output_config"] == {"effort": "high"}
        assert kw["tools"] == [{"type": "web_search_20260209", "name": "web_search"}]
        assert kw["fallbacks"] == "default"
        assert kw["betas"] == ["server-side-fallback-2026-07-01"]
        # Fable rejects sampling parameters and any explicit thinking config.
        assert "temperature" not in kw
        assert "thinking" not in kw

    def test_defaults_send_no_optional_knobs(self):
        kw = provider()._build_kwargs(request())
        assert "output_config" not in kw
        assert "tools" not in kw

    def test_unbounded_max_tokens_gets_a_concrete_value(self):
        kw = provider()._build_kwargs(request(max_tokens=None))
        assert kw["max_tokens"] == mod.UNBOUNDED_MAX_TOKENS

    @pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"])
    def test_fallbacks_on_supported_models(self, model):
        assert provider(model)._build_kwargs(request())["fallbacks"] == "default"

    @pytest.mark.parametrize("model", ["claude-sonnet-4-6", "claude-haiku-4-5", "claude-opus-4-8"])
    def test_no_fallbacks_elsewhere(self, model):
        kw = provider(model)._build_kwargs(request())
        assert "fallbacks" not in kw and "betas" not in kw

    @pytest.mark.parametrize("model", ["claude-sonnet-4-6", "claude-haiku-4-5", "claude-opus-4-6"])
    def test_temperature_kept_where_sampling_exists(self, model):
        assert provider(model)._build_kwargs(request(temperature=0.2))["temperature"] == 0.2

    @pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-opus-4-7", "claude-sonnet-5-5"])
    def test_temperature_dropped_where_sampling_was_removed(self, model):
        assert "temperature" not in provider(model)._build_kwargs(request(temperature=0.2))


class TestFinalText:
    def test_drops_narration_before_search(self):
        m = message(
            block("thinking"), text("Let me check current guidance."),
            block("server_tool_use"), block("web_search_tool_result"),
            text("The plan "), text("misses the rollback path."),
        )
        assert _final_text(m) == "The plan misses the rollback path."

    def test_answer_after_fallback_switch(self):
        m = message(text("partial from first model"), block("fallback"), text(ANSWER))
        assert _final_text(m) == ANSWER

    def test_trailing_non_text_does_not_hide_answer(self):
        assert _final_text(message(text(ANSWER), block("thinking"))) == ANSWER

    def test_no_text(self):
        assert _final_text(message(block("thinking"))) == ""


class TestBuildResponse:
    def test_success_reports_serving_model_and_usage(self):
        r = provider()._build_response(
            message(text(ANSWER), model="claude-opus-4-8", thinking_tokens=12), 50
        )
        assert r.text == ANSWER
        assert r.provider == "anthropic"
        assert r.model == "claude-opus-4-8"  # a refusal fallback answered
        assert r.usage.to_dict() == {
            "input_tokens": 100, "output_tokens": 40, "total_tokens": 140,
            "reasoning_tokens": 12,
        }

    def test_refusal_is_content_blocked(self):
        details = SimpleNamespace(category="cyber", explanation="declined")
        with pytest.raises(ProviderError) as ei:
            provider()._build_response(
                message(stop_reason="refusal", stop_details=details), 0
            )
        assert ei.value.error_type == "content_blocked"
        assert "category=cyber" in ei.value.message
        assert not ei.value.retriable

    def test_refusal_wins_over_partial_text(self):
        with pytest.raises(ProviderError) as ei:
            provider()._build_response(message(text("partial"), stop_reason="refusal"), 0)
        assert ei.value.error_type == "content_blocked"

    def test_max_tokens_is_retriable_truncation(self):
        with pytest.raises(ProviderError) as ei:
            provider()._build_response(message(text("half an"), stop_reason="max_tokens"), 0)
        assert ei.value.error_type == "upstream_error"
        assert ei.value.retriable
        assert "max_tokens" in ei.value.message

    def test_empty_text_is_an_error(self):
        with pytest.raises(ProviderError) as ei:
            provider()._build_response(message(block("thinking")), 0)
        assert ei.value.error_type == "upstream_error"


class TestPauseTurn:
    @pytest.mark.asyncio
    async def test_resumes_and_sums_usage(self, monkeypatch):
        paused = message(text("Searching."), block("server_tool_use"),
                         stop_reason="pause_turn", input_tokens=100, output_tokens=10)
        final = message(text(ANSWER), input_tokens=300, output_tokens=50)
        calls: list[dict] = []

        async def fake_stream_once(self, client, kwargs, timeout):
            calls.append(kwargs)
            return [paused, final][len(calls) - 1]

        monkeypatch.setattr(AnthropicProvider, "_stream_once", fake_stream_once)
        r = await provider(web_search=True).generate(request())

        assert r.text == ANSWER
        assert r.usage.input_tokens == 400 and r.usage.output_tokens == 60
        assert len(calls) == 2
        resumed = calls[1]["messages"]
        assert resumed[0]["role"] == "user"
        assert resumed[1] == {"role": "assistant", "content": paused.content}

    @pytest.mark.asyncio
    async def test_gives_up_after_the_continuation_cap(self, monkeypatch):
        async def always_paused(self, client, kwargs, timeout):
            return message(block("server_tool_use"), stop_reason="pause_turn")

        monkeypatch.setattr(AnthropicProvider, "_stream_once", always_paused)
        with pytest.raises(ProviderError) as ei:
            await provider(web_search=True).generate(request())
        assert ei.value.retriable


# -- wire level -------------------------------------------------------------


def sse_stream(content_text: str = ANSWER, stop_reason: str = "end_turn") -> bytes:
    events = [
        {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": "claude-fable-5-1", "content": [], "stop_reason": None,
            "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": content_text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
         "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


class Wire:
    def __init__(self, status: int = 200, body: bytes | None = None):
        self.status = status
        self.body = body
        self.requests: list[httpx2.Request] = []

    def install(self, monkeypatch):
        def handler(req: httpx2.Request) -> httpx2.Response:
            self.requests.append(req)
            if self.status != 200:
                return httpx2.Response(self.status, json={
                    "type": "error", "error": {"type": "x", "message": "boom"}})
            return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                                   content=self.body or sse_stream())

        def client(self_):
            return AsyncAnthropic(
                api_key="k", max_retries=0,
                http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
            )

        monkeypatch.setattr(AnthropicProvider, "_client", client)
        return self


class TestWire:
    @pytest.mark.asyncio
    async def test_serialised_request(self, monkeypatch):
        wire = Wire().install(monkeypatch)
        r = await provider(reasoning_effort="xhigh").generate(request(temperature=0.5))
        assert r.text == ANSWER
        assert r.usage.output_tokens == 7

        (req,) = wire.requests
        assert req.url.path == "/v1/messages"
        assert req.headers["anthropic-beta"] == "server-side-fallback-2026-07-01"
        body = json.loads(req.content)
        assert body["fallbacks"] == "default"
        assert body["output_config"] == {"effort": "xhigh"}
        assert body["stream"] is True
        assert "temperature" not in body and "thinking" not in body

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status,error_type,retriable", [
        (400, "bad_request", False),
        (401, "auth_failed", False),
        (404, "bad_request", False),
        (429, "rate_limit", True),
        (529, "upstream_error", True),
    ])
    async def test_http_errors_are_mapped(self, monkeypatch, status, error_type, retriable):
        Wire(status=status).install(monkeypatch)
        with pytest.raises(ProviderError) as ei:
            await provider().generate(request())
        assert (ei.value.error_type, ei.value.retriable) == (error_type, retriable)


class TestRegistry:
    def test_claude_target_builds_the_adapter(self):
        cfg = AppConfig(providers={"anthropic": ProviderConfig(api_key="k", model="claude-fable-5-1")})
        p = build_provider("claude", cfg)
        assert isinstance(p, AnthropicProvider)
        assert p.supports_background is False

    def test_missing_key_names_the_env_var(self):
        cfg = AppConfig(providers={"anthropic": ProviderConfig(api_key=None, model="m")})
        with pytest.raises(ProviderError) as ei:
            build_provider("claude", cfg)
        assert ei.value.error_type == "missing_api_key"
        assert "LLM_SECOND_OPINION_ANTHROPIC_API_KEY" in ei.value.message
