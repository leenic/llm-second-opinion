"""Grok provider via xAI's OpenAI-compatible Responses API.

xAI exposes the same `/v1/responses` shape as OpenAI, so we reuse the
ResponsesAPIProvider and only change the base URL.
"""

from __future__ import annotations

from .openai_provider import ResponsesAPIProvider


class GrokProvider(ResponsesAPIProvider):
    name = "grok"
    base_url = "https://api.x.ai/v1"
