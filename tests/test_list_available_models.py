"""`list_available_models` reports the per-provider effort sets (DESIGN §18.3)
and the attachment configuration (§17). Driven with no API keys so the
reachability probes short-circuit and nothing touches the network."""

from __future__ import annotations

import pytest

from llm_second_opinion.config import REASONING_EFFORTS_BY_PROVIDER, AppConfig, ProviderConfig
from llm_second_opinion.server import build_server


def keyless_config() -> AppConfig:
    return AppConfig(providers={
        "openai": ProviderConfig(api_key=None, model="m"),
        "gemini": ProviderConfig(api_key=None, model="m"),
        "grok": ProviderConfig(api_key=None, model="m"),
    })


async def call_list(config: AppConfig) -> dict:
    result = await build_server(config).call_tool("list_available_models", {})
    return result[1] if isinstance(result, tuple) else result


class TestReasoningEffortsReported:
    @pytest.mark.asyncio
    async def test_each_provider_reports_its_allowed_set(self):
        result = await call_list(keyless_config())
        by_provider = {p["provider"]: p for p in result["providers"]}
        assert set(by_provider) == set(REASONING_EFFORTS_BY_PROVIDER)
        for name, entry in by_provider.items():
            assert entry["allowed_reasoning_efforts"] == sorted(REASONING_EFFORTS_BY_PROVIDER[name])
            assert entry["reason"] == "missing_api_key"

    @pytest.mark.asyncio
    async def test_attachments_default_disabled(self):
        result = await call_list(keyless_config())
        assert result["attachments"] == {
            "enabled": False, "roots": [], "max_attachment_bytes": 1_000_000,
        }

    @pytest.mark.asyncio
    async def test_writes_nothing_to_stdout(self, capsys):
        await call_list(keyless_config())
        assert capsys.readouterr().out == ""
