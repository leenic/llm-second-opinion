"""Provider adapters for external LLMs."""

from __future__ import annotations

from ..config import AppConfig
from .base import Provider, ProviderError, SecondOpinionRequest, SecondOpinionResponse, TokenUsage
from .gemini import GeminiProvider
from .grok import GrokProvider
from .openai_provider import OpenAIProvider

# target_model values the tool accepts, mapped to provider keys in config.
TARGET_TO_PROVIDER: dict[str, str] = {
    "chatgpt": "openai",
    "gemini": "gemini",
    "grok": "grok",
}

PROVIDER_TO_TARGET: dict[str, str] = {v: k for k, v in TARGET_TO_PROVIDER.items()}


def build_provider(target_model: str, config: AppConfig) -> Provider:
    """Build a provider for a `target_model` value from the tool input."""
    if target_model not in TARGET_TO_PROVIDER:
        raise ProviderError(
            "invalid_input",
            f"Unknown target_model {target_model!r}. Use one of: {sorted(TARGET_TO_PROVIDER)}",
            retriable=False,
        )

    provider_key = TARGET_TO_PROVIDER[target_model]
    pcfg = config.providers.get(provider_key)
    if pcfg is None or not pcfg.api_key:
        raise ProviderError(
            "missing_api_key",
            f"No API key configured for provider {provider_key!r}. "
            f"Add it under providers.{provider_key}.api_key in config.json, "
            f"or set LLM_SECOND_OPINION_{provider_key.upper()}_API_KEY.",
            retriable=False,
        )

    timeout = config.timeout_seconds
    kwargs = {
        "api_key": pcfg.api_key,
        "model": pcfg.model,
        "timeout": timeout,
        "reasoning_effort": pcfg.reasoning_effort,
        "web_search": pcfg.web_search,
    }
    if provider_key == "openai":
        return OpenAIProvider(**kwargs)
    if provider_key == "grok":
        return GrokProvider(**kwargs)
    if provider_key == "gemini":
        return GeminiProvider(**kwargs)
    raise ProviderError("internal_error", f"No adapter for {provider_key}", retriable=False)


__all__ = [
    "Provider",
    "ProviderError",
    "SecondOpinionRequest",
    "SecondOpinionResponse",
    "TokenUsage",
    "OpenAIProvider",
    "GrokProvider",
    "GeminiProvider",
    "TARGET_TO_PROVIDER",
    "PROVIDER_TO_TARGET",
    "build_provider",
]
