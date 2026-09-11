"""`store: false` on the synchronous path (DESIGN-submit-poll.md §18.1).

All three vendors store requests server-side by default. These tests drive
each adapter through its *real* SDK client with an httpx mock transport and
inspect the serialised request body — not the kwargs the adapter passed to
the SDK — so what is asserted is what would leave the machine. Background
submission keeps `store: true` (OpenAI) or sends nothing about storage
(Gemini), per §11.
"""

from __future__ import annotations

import json

import httpx
import pytest

from llm_second_opinion.providers import gemini as gemini_mod
from llm_second_opinion.providers import openai_provider as openai_mod
from llm_second_opinion.providers.base import SecondOpinionRequest
from llm_second_opinion.providers.gemini import GeminiProvider
from llm_second_opinion.providers.grok import GrokProvider
from llm_second_opinion.providers.openai_provider import OpenAIProvider


def request() -> SecondOpinionRequest:
    return SecondOpinionRequest(
        summary="a summary", focus=None, system_prompt="be critical",
        temperature=None, max_tokens=50,
    )


class Capture:
    """Records every request the mock transport sees and answers with `payload`."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.requests: list[httpx.Request] = []

    def transport(self) -> httpx.MockTransport:
        def handler(req: httpx.Request) -> httpx.Response:
            self.requests.append(req)
            return httpx.Response(200, json=self.payload)

        return httpx.MockTransport(handler)

    @property
    def body(self) -> dict:
        assert len(self.requests) == 1, [str(r.url) for r in self.requests]
        return json.loads(self.requests[0].content.decode("utf-8"))

    @property
    def url(self) -> str:
        return str(self.requests[0].url)


RESPONSES_COMPLETED = {
    "id": "resp_1", "object": "response", "status": "completed", "model": "m",
    "created_at": 0, "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
    "output": [{
        "type": "message", "id": "m1", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": "hi", "annotations": []}],
    }],
}
RESPONSES_QUEUED = {**RESPONSES_COMPLETED, "id": "resp_bg", "status": "queued", "output": []}

INTERACTION_COMPLETED = {
    "id": "int_1", "status": "completed", "model": "gemini-x",
    "steps": [{"type": "model_output", "content": [{"type": "text", "text": "hi"}]}],
}
INTERACTION_QUEUED = {"id": "int_bg", "status": "queued", "model": "gemini-x", "steps": []}


@pytest.fixture
def responses_capture(monkeypatch):
    """Route the adapter's own `AsyncOpenAI(...)` construction through a mock
    transport, keeping its base_url/timeout/max_retries choices intact."""

    def _install(payload: dict) -> Capture:
        cap = Capture(payload)
        real = openai_mod.AsyncOpenAI

        def factory(**kwargs):
            return real(http_client=httpx.AsyncClient(transport=cap.transport()), **kwargs)

        monkeypatch.setattr(openai_mod, "AsyncOpenAI", factory)
        return cap

    return _install


@pytest.fixture
def interactions_capture(monkeypatch):
    def _install(payload: dict) -> Capture:
        from types import SimpleNamespace

        from google import genai
        from google.genai import types

        cap = Capture(payload)

        def factory(**kwargs):
            return genai.Client(
                http_options=types.HttpOptions(
                    httpx_async_client=httpx.AsyncClient(transport=cap.transport())
                ),
                **kwargs,
            )

        monkeypatch.setattr(gemini_mod, "genai", SimpleNamespace(Client=factory))
        return cap

    return _install


class TestSynchronousPathSendsStoreFalse:
    @pytest.mark.asyncio
    async def test_openai(self, responses_capture):
        cap = responses_capture(RESPONSES_COMPLETED)
        provider = OpenAIProvider(api_key="k", model="m", timeout=30.0)
        resp = await provider.generate(request())
        assert resp.text == "hi"
        assert cap.url == "https://api.openai.com/v1/responses"
        assert cap.body["store"] is False
        assert "background" not in cap.body
        assert cap.body["max_output_tokens"] == 50, "max_output_tokens never dropped"

    @pytest.mark.asyncio
    async def test_xai(self, responses_capture):
        cap = responses_capture(RESPONSES_COMPLETED)
        provider = GrokProvider(api_key="k", model="grok-x", timeout=30.0)
        resp = await provider.generate(request())
        assert resp.text == "hi"
        assert cap.url == "https://api.x.ai/v1/responses"
        assert cap.body["store"] is False
        assert "background" not in cap.body

    @pytest.mark.asyncio
    async def test_gemini(self, interactions_capture):
        cap = interactions_capture(INTERACTION_COMPLETED)
        provider = GeminiProvider(api_key="k", model="gemini-x", timeout=30.0)
        resp = await provider.generate(request())
        assert resp.text == "hi"
        assert cap.url.endswith("/interactions")
        assert cap.body["store"] is False
        assert "background" not in cap.body
        assert cap.body["generation_config"]["max_output_tokens"] == 50


class TestBackgroundJobsKeepVendorStorage:
    @pytest.mark.asyncio
    async def test_openai_background_stores(self, responses_capture):
        cap = responses_capture(RESPONSES_QUEUED)
        provider = OpenAIProvider(api_key="k", model="m", timeout=30.0)
        assert await provider.submit_background(request(), timeout=30.0) == "resp_bg"
        assert cap.body["store"] is True
        assert cap.body["background"] is True

    @pytest.mark.asyncio
    async def test_gemini_background_sends_nothing_about_storage(self, interactions_capture):
        cap = interactions_capture(INTERACTION_QUEUED)
        provider = GeminiProvider(api_key="k", model="gemini-x", timeout=30.0)
        assert await provider.submit_background(request(), timeout=30.0) == "int_bg"
        assert "store" not in cap.body
        assert cap.body["background"] is True
