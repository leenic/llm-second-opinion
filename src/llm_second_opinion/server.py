"""MCP server exposing `second_opinion` and `list_available_models` tools."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from .config import AppConfig, ConfigError, load_config
from .providers import (
    PROVIDER_TO_TARGET,
    TARGET_TO_PROVIDER,
    ProviderError,
    SecondOpinionRequest,
    build_provider,
)
from .providers.gemini import GeminiProvider
from .providers.grok import GrokProvider
from .providers.openai_provider import OpenAIProvider

DEFAULT_SYSTEM_PROMPT = (
    "You are acting as an external reviewer for a conversation the user is "
    "having with another AI assistant. The user wants your independent view "
    "on the summary below. Be direct, concrete, and critical. If you disagree "
    "with the framing or see a stronger alternative, say so explicitly. Do "
    "not pad with praise. If a focus is provided, prioritise commenting on "
    "that aspect. State your confidence level when making factual claims."
)


def build_server(config: AppConfig, logger: logging.Logger | None = None) -> FastMCP:
    """Construct the FastMCP server. Separated from `main()` so tests can
    inspect or invoke tools without spawning a subprocess."""
    log = logger or logging.getLogger("llm_second_opinion")
    mcp = FastMCP("llm-second-opinion")

    @mcp.tool(
        description=(
            "Send a summary to an external LLM (Gemini, Grok, or ChatGPT) and "
            "return its independent, critical second opinion."
        )
    )
    async def second_opinion(
        summary: str,
        target_model: Literal["gemini", "grok", "chatgpt"],
        focus: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Route `summary` to the requested external LLM and return its reply.

        Args:
            summary: The content to review. Required.
            target_model: One of "gemini", "grok", or "chatgpt".
            focus: Optional aspect to prioritise in the review.
            system_prompt: Optional override for the default reviewer prompt.
            temperature: Optional sampling temperature.
            max_tokens: Optional maximum response length (tokens).
        """
        request_id = uuid.uuid4().hex[:12]

        if not isinstance(summary, str) or not summary.strip():
            log.warning("rid=%s tool=second_opinion error=invalid_input reason=empty_summary", request_id)
            return _error_response(
                request_id,
                target_model,
                "invalid_input",
                "`summary` must be a non-empty string.",
            )

        prompt = system_prompt if system_prompt and system_prompt.strip() else DEFAULT_SYSTEM_PROMPT

        try:
            provider = build_provider(target_model, config)
        except ProviderError as e:
            log.warning(
                "rid=%s tool=second_opinion target=%s error=%s",
                request_id, target_model, e.error_type,
            )
            return _error_response(request_id, target_model, e.error_type, e.message, retriable=e.retriable)

        req = SecondOpinionRequest(
            summary=summary,
            focus=focus,
            system_prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        log.info(
            "rid=%s tool=second_opinion provider=%s model=%s focus=%s temp=%s max_tokens=%s",
            request_id,
            provider.name,
            provider.model_id(),
            "yes" if focus else "no",
            temperature,
            max_tokens,
        )
        if config.log_prompts:
            log.debug("rid=%s prompt_summary=%r focus=%r", request_id, summary, focus)

        try:
            response = await provider.generate(req)
        except ProviderError as e:
            log.warning(
                "rid=%s tool=second_opinion provider=%s outcome=error type=%s status=%s",
                request_id, provider.name, e.error_type, e.status,
            )
            return _error_response(request_id, target_model, e.error_type, e.message,
                                   retriable=e.retriable, model=provider.model_id())
        except Exception as e:  # noqa: BLE001 - last-resort safety net
            log.exception("rid=%s tool=second_opinion provider=%s outcome=internal_error",
                          request_id, provider.name)
            return _error_response(request_id, target_model, "internal_error", str(e),
                                   retriable=False, model=provider.model_id())

        usage = response.usage.to_dict() if response.usage else None
        log.info(
            "rid=%s tool=second_opinion provider=%s model=%s outcome=ok latency_ms=%d "
            "input_tokens=%s output_tokens=%s",
            request_id, response.provider, response.model, response.latency_ms,
            usage.get("input_tokens") if usage else None,
            usage.get("output_tokens") if usage else None,
        )
        if config.log_prompts:
            log.debug("rid=%s response_text=%r", request_id, response.text)

        return {
            "success": True,
            "request_id": request_id,
            "target_model": target_model,
            "provider": response.provider,
            "model": response.model,
            "response": response.text,
            "usage": usage,
            "latency_ms": response.latency_ms,
        }

    @mcp.tool(
        description=(
            "Return the providers (gemini, grok, chatgpt) that are configured "
            "with an API key and pass a basic reachability check."
        )
    )
    async def list_available_models() -> dict[str, Any]:
        """List configured providers and whether each is currently reachable."""
        request_id = uuid.uuid4().hex[:12]
        log.info("rid=%s tool=list_available_models", request_id)

        checks = await _check_all_providers(config)

        available = [c["target_model"] for c in checks if c["available"]]
        log.info("rid=%s tool=list_available_models available=%s", request_id, available)
        return {
            "request_id": request_id,
            "providers": checks,
            "available_target_models": available,
            "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
        }

    return mcp


def _error_response(
    request_id: str,
    target_model: str,
    error_type: str,
    message: str,
    retriable: bool = False,
    model: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "success": False,
        "request_id": request_id,
        "target_model": target_model,
        "error": {
            "type": error_type,
            "message": message,
            "retriable": retriable,
        },
    }
    if model is not None:
        out["model"] = model
    return out


async def _check_all_providers(config: AppConfig) -> list[dict[str, Any]]:
    """Run a basic reachability probe for each configured provider in parallel."""
    items: list[tuple[str, str, Any]] = []  # (target_model, provider_key, provider_or_reason)
    for target_model, provider_key in TARGET_TO_PROVIDER.items():
        pcfg = config.providers.get(provider_key)
        if pcfg is None or not pcfg.api_key:
            items.append((target_model, provider_key, "missing_api_key"))
            continue
        kwargs = dict(
            api_key=pcfg.api_key,
            model=pcfg.model,
            timeout=config.timeout_seconds,
            reasoning_effort=pcfg.reasoning_effort,
            web_search=pcfg.web_search,
        )
        if provider_key == "openai":
            provider: Any = OpenAIProvider(**kwargs)
        elif provider_key == "grok":
            provider = GrokProvider(**kwargs)
        elif provider_key == "gemini":
            provider = GeminiProvider(**kwargs)
        else:
            items.append((target_model, provider_key, "unsupported_provider"))
            continue
        items.append((target_model, provider_key, provider))

    async def _probe(provider: Any) -> tuple[bool, str | None]:
        try:
            return await provider.check_reachable()
        except Exception as e:  # noqa: BLE001
            return False, f"unexpected error: {e}"

    coros = [
        _probe(item[2]) if not isinstance(item[2], str) else _noop_result(item[2])
        for item in items
    ]
    results = await asyncio.gather(*coros)

    out: list[dict[str, Any]] = []
    for (target_model, provider_key, p), (ok, reason) in zip(items, results):
        pcfg = config.providers[provider_key]
        model_id = p.model_id() if not isinstance(p, str) else pcfg.model
        out.append({
            "target_model": target_model,
            "provider": provider_key,
            "configured_model": model_id,
            "api_key_configured": isinstance(p, str) is False,
            "available": bool(ok),
            "reason": reason,
            "reasoning_effort": pcfg.reasoning_effort,
            "web_search": pcfg.web_search,
        })
    return out


async def _noop_result(reason: str) -> tuple[bool, str | None]:
    return False, reason


def main() -> None:
    """Console-script entry point: load config, build server, run on stdio."""
    from .logging_setup import setup_logging

    try:
        config = load_config()
    except ConfigError as e:
        # Fail fast with a clear stderr message; without config we can't do anything useful.
        import sys
        print(f"llm-second-opinion: configuration error: {e}", file=sys.stderr)
        sys.exit(2)

    logger = setup_logging(config.log_level)
    if config.config_path:
        logger.info("loaded config from %s", config.config_path)
    else:
        logger.warning(
            "no config file found; checked $LLM_SECOND_OPINION_CONFIG, ./config.json, "
            "%%APPDATA%%/llm-second-opinion/config.json, ~/.config/llm-second-opinion/config.json. "
            "Providers will report missing_api_key until one is configured."
        )

    configured = [
        PROVIDER_TO_TARGET[k] for k, v in config.providers.items()
        if v.api_key and k in PROVIDER_TO_TARGET
    ]
    logger.info(
        "starting llm-second-opinion MCP server (stdio). configured providers: %s",
        configured or "none",
    )

    server = build_server(config, logger=logger)
    server.run()  # FastMCP defaults to stdio transport.
