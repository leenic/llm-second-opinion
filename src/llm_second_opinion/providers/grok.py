"""Grok provider via xAI's OpenAI-compatible Responses API.

xAI exposes the same `/v1/responses` shape as OpenAI, so we reuse the
ResponsesAPIProvider and only change the base URL.
"""

from __future__ import annotations

from .openai_provider import ResponsesAPIProvider


class GrokProvider(ResponsesAPIProvider):
    name = "grok"
    base_url = "https://api.x.ai/v1"

    # xAI has no background mode. Its docs describe `background` as "Not used
    # at the moment. Just for OpenResponses compatibility.", and live it is
    # worse than ignored: a create with background=true is rejected with
    # `400 Argument not supported: background`. Nor does generation survive a
    # dropped connection — a stored streaming response whose connection is
    # closed after `response.created` is never retrievable (404 for 90s+),
    # so there is no store-and-retrieve upgrade path either. Both measured
    # 2026-09-07 (DESIGN-submit-poll.md §12.1). The server therefore runs
    # grok jobs as in-process tasks over the ordinary generate() path, and
    # this flag must stay False: never send `background` to xAI.
    supports_background = False
