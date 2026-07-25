"""MCP server exposing `second_opinion` and `list_available_models` tools."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from .config import (
    CLIENT_HARD_CAP_SECONDS,
    AppConfig,
    ConfigError,
    load_config,
)
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

# How long a cancelled provider task gets to tear down before we return
# anyway. Kept small so `request_budget_seconds` + this still clears the MCP
# client's hard cap (200 + 5 = 205, under 240).
CANCEL_GRACE_SECONDS = 5.0

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
        started = time.monotonic()

        def elapsed_ms() -> int:
            return int((time.monotonic() - started) * 1000)

        if not isinstance(summary, str) or not summary.strip():
            log.warning(
                "rid=%s tool=second_opinion outcome=error type=invalid_input "
                "reason=empty_summary elapsed_ms=%d",
                request_id, elapsed_ms(),
            )
            return _error_response(
                request_id,
                target_model,
                "invalid_input",
                "`summary` must be a non-empty string.",
                elapsed_ms=elapsed_ms(),
            )

        prompt = system_prompt if system_prompt and system_prompt.strip() else DEFAULT_SYSTEM_PROMPT

        try:
            provider = build_provider(target_model, config)
        except ProviderError as e:
            log.warning(
                "rid=%s tool=second_opinion target=%s outcome=error type=%s elapsed_ms=%d",
                request_id, target_model, e.error_type, elapsed_ms(),
            )
            return _error_response(request_id, target_model, e.error_type, e.message,
                                   retriable=e.retriable, elapsed_ms=elapsed_ms())

        # An unbounded reply is the main driver of tail latency, so cap it when
        # the caller expressed no preference.
        effective_max_tokens = max_tokens if max_tokens is not None else config.default_max_tokens

        req = SecondOpinionRequest(
            summary=summary,
            focus=focus,
            system_prompt=prompt,
            temperature=temperature,
            max_tokens=effective_max_tokens,
        )

        budget = config.request_budget_seconds
        log.info(
            "rid=%s tool=second_opinion provider=%s model=%s focus=%s temp=%s "
            "max_tokens=%s budget_s=%.1f",
            request_id,
            provider.name,
            provider.model_id(),
            "yes" if focus else "no",
            temperature,
            effective_max_tokens,
            budget,
        )
        if config.log_prompts:
            log.debug("rid=%s prompt_summary=%r focus=%r", request_id, summary, focus)

        try:
            # The provider's own HTTP client is built with this same budget, so
            # the socket closes on its own in the normal case. This outer bound
            # is what guarantees the handler returns regardless — a provider
            # that stalls outside its HTTP timeout (DNS, a retry loop, a future
            # adapter that ignores the arg) would otherwise run past Desktop's
            # 240s cap, which cancels the call and discards the result.
            #
            response = await _run_bounded(provider.generate(req), budget, log)
        except (asyncio.TimeoutError, TimeoutError):
            took = elapsed_ms()
            log.warning(
                "rid=%s tool=second_opinion provider=%s model=%s outcome=timeout "
                "elapsed_ms=%d budget_s=%.1f",
                request_id, provider.name, provider.model_id(), took, budget,
            )
            return _error_response(
                request_id,
                target_model,
                "timeout",
                f"{provider.name} exceeded the {budget:g}s request budget and was "
                f"cancelled after {took / 1000:.1f}s. The upstream call may have "
                f"been close to finishing — retry, lower reasoning_effort, set a "
                f"smaller max_tokens, or raise request_budget_seconds (staying "
                f"below the {CLIENT_HARD_CAP_SECONDS:g}s cap the MCP client enforces).",
                retriable=True,
                model=provider.model_id(),
                elapsed_ms=took,
            )
        except ProviderError as e:
            log.warning(
                "rid=%s tool=second_opinion provider=%s outcome=error type=%s "
                "status=%s elapsed_ms=%d",
                request_id, provider.name, e.error_type, e.status, elapsed_ms(),
            )
            return _error_response(request_id, target_model, e.error_type, e.message,
                                   retriable=e.retriable, model=provider.model_id(),
                                   elapsed_ms=elapsed_ms())
        except Exception as e:  # noqa: BLE001 - last-resort safety net
            log.exception("rid=%s tool=second_opinion provider=%s outcome=internal_error "
                          "elapsed_ms=%d", request_id, provider.name, elapsed_ms())
            return _error_response(request_id, target_model, "internal_error", str(e),
                                   retriable=False, model=provider.model_id(),
                                   elapsed_ms=elapsed_ms())

        usage = response.usage.to_dict() if response.usage else None
        log.info(
            "rid=%s tool=second_opinion provider=%s model=%s outcome=ok latency_ms=%d "
            "elapsed_ms=%d input_tokens=%s output_tokens=%s",
            request_id, response.provider, response.model, response.latency_ms,
            elapsed_ms(),
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
            "elapsed_ms": elapsed_ms(),
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


async def _run_bounded(
    coro: Any, budget: float, log: logging.Logger | None = None
) -> Any:
    """Run `coro` under a hard wall-clock bound, raising TimeoutError if it
    overruns.

    Deliberately not `asyncio.wait_for`. If the awaited coroutine catches
    `CancelledError` and returns a value anyway, `wait_for` hands that value
    straight back and the deadline is silently ignored — a late answer would
    surface as a success well past the point the MCP client had given up. Here
    the result is discarded unconditionally once the deadline passes.
    """
    task = asyncio.ensure_future(coro)
    try:
        done, _pending = await asyncio.wait({task}, timeout=budget)
    except BaseException:
        # We are being cancelled — client disconnected, or the server is
        # shutting down. `task` has no other owner, so without this the
        # upstream request keeps running unmanaged: a billable API call
        # holding a socket that nothing will ever read.
        #
        # Cancel but deliberately do not await. Awaiting inside our own
        # cancellation is how shutdowns wedge: if the loop is already tearing
        # down, the continuation may never be scheduled. Cancelling is what
        # stops the leak; teardown can finish on the loop's own time.
        task.cancel()
        raise

    if task in done:
        return task.result()  # re-raises ProviderError etc. to the caller

    await _abandon(task, log)
    raise TimeoutError


async def _abandon(task: asyncio.Task, log: logging.Logger | None = None) -> None:
    """Cancel `task` and give it a bounded window to tear down.

    Bounded, not open-ended: a coroutine that catches `CancelledError` and
    keeps going would otherwise hold us here indefinitely and blow the very
    budget this function exists to enforce. That is the same misbehaving
    coroutine `_run_bounded` refuses to trust for its result, so we do not
    trust it to exit either. The grace period is small enough that
    budget + grace still clears the MCP client's hard cap.

    Whatever the task produces is dropped either way — the caller has already
    decided the deadline passed.
    """
    task.cancel()
    _done, still_running = await asyncio.wait({task}, timeout=CANCEL_GRACE_SECONDS)

    if still_running:
        # Rare and worth seeing: a provider that ignores cancellation leaves a
        # connection open behind us. We return regardless.
        if log is not None:
            log.warning(
                "provider task ignored cancellation after %.1fs; abandoning it "
                "(its result is discarded)",
                CANCEL_GRACE_SECONDS,
            )
        return

    # Retrieve the outcome so asyncio doesn't log "exception was never
    # retrieved" when the task is garbage collected.
    try:
        task.result()
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass


def _error_response(
    request_id: str,
    target_model: str,
    error_type: str,
    message: str,
    retriable: bool = False,
    model: str | None = None,
    elapsed_ms: int | None = None,
) -> dict[str, Any]:
    """Errors are returned as normal tool results, never raised.

    A raised exception reaches the calling model as a transport-level failure
    it can't reason about; this shape tells it what went wrong and whether
    retrying is worth it.
    """
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
    if elapsed_ms is not None:
        out["elapsed_ms"] = elapsed_ms
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
            timeout=config.request_budget_seconds,
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
    # Deferred from load_config(), which runs before the logger exists.
    for warning in config.warnings:
        logger.warning("config: %s", warning)
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
        "starting llm-second-opinion MCP server (stdio). configured providers: %s. "
        "request_budget_seconds=%.1f default_max_tokens=%s",
        configured or "none",
        config.request_budget_seconds,
        config.default_max_tokens,
    )

    server = build_server(config, logger=logger)
    server.run()  # FastMCP defaults to stdio transport.
