"""Configuration loading for the llm-second-opinion MCP server.

API keys are read from a JSON config file. Model names, timeout, and log
behavior can be overridden via environment variables. Environment variables
always take precedence over file values.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ENV_PREFIX = "LLM_SECOND_OPINION_"

DEFAULT_MODELS = {
    "openai": "gpt-5.6-sol",
    "gemini": "gemini-3.6-flash",
    "grok": "grok-4.5",
}

# One budget bounds the whole tool call: the outer `asyncio.wait_for` in the
# handler AND the provider's own HTTP client timeout are both set from it, so
# the socket actually closes instead of leaving a cancelled coroutine with an
# open connection.
#
# Claude Desktop enforces a hard, non-configurable 240s cap on tool calls and
# cancels with `MCP error -32001: Request timed out`, discarding the result
# even when the upstream call succeeded. Staying under that cap means we
# return our own structured error, which the calling model can act on.
DEFAULT_REQUEST_BUDGET_SECONDS = 200.0

# Above this, Desktop's cap would fire before ours and the result is lost.
# Not an error — another MCP client may allow longer — but always warned about.
CLIENT_HARD_CAP_SECONDS = 240.0
BUDGET_WARN_THRESHOLD_SECONDS = 235.0

# Applied when the caller omits `max_tokens`. Long calls correlate with large
# reasoning+output token counts, so an unbounded reply is the main driver of
# tail latency. 8000 is roughly 4x the largest reply observed in testing
# (grok-4.5 at 2021 output tokens), so it caps runaway generation without
# truncating realistic answers. Set to null to leave replies unbounded.
DEFAULT_MAX_TOKENS = 8000

REASONING_EFFORTS = {"minimal", "low", "medium", "high"}


@dataclass
class ProviderConfig:
    api_key: str | None = None
    model: str = ""
    # One of: "minimal", "low", "medium", "high", or None to leave at SDK default.
    # For Gemini this maps to a thinking_budget; for OpenAI/Grok it's the
    # reasoning effort field on the Responses API.
    reasoning_effort: str | None = None
    # If true, attach the provider's built-in web search tool to every call.
    web_search: bool = False


@dataclass
class AppConfig:
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    # Bounds one whole tool call — see DEFAULT_REQUEST_BUDGET_SECONDS.
    request_budget_seconds: float = DEFAULT_REQUEST_BUDGET_SECONDS
    # Cap applied when the caller passes no `max_tokens`. None = unbounded.
    default_max_tokens: int | None = DEFAULT_MAX_TOKENS
    log_prompts: bool = False
    log_level: str = "INFO"
    config_path: Path | None = None
    # Non-fatal config problems, logged to stderr by main() once the logger
    # exists. load_config() runs before logging is configured, so it cannot
    # emit these itself.
    warnings: list[str] = field(default_factory=list)

    @property
    def timeout_seconds(self) -> float:
        """Deprecated alias. The HTTP client timeout is the request budget."""
        return self.request_budget_seconds


def _candidate_paths() -> list[Path]:
    env_path = os.environ.get(f"{ENV_PREFIX}CONFIG")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path.cwd() / "config.json")
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "llm-second-opinion" / "config.json")
    candidates.append(Path.home() / ".config" / "llm-second-opinion" / "config.json")
    return candidates


def _load_file() -> tuple[dict, Path | None]:
    for path in _candidate_paths():
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f), path
            except (OSError, json.JSONDecodeError) as e:
                raise ConfigError(f"Failed to read config file {path}: {e}") from e
    return {}, None


class ConfigError(Exception):
    """Raised when the configuration is invalid or unreadable."""


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> AppConfig:
    raw, path = _load_file()

    raw_providers = raw.get("providers", {}) if isinstance(raw, dict) else {}
    if not isinstance(raw_providers, dict):
        raise ConfigError("`providers` in the config file must be an object")

    providers: dict[str, ProviderConfig] = {}
    for name, default_model in DEFAULT_MODELS.items():
        entry = raw_providers.get(name, {}) or {}
        if not isinstance(entry, dict):
            raise ConfigError(f"`providers.{name}` must be an object")
        file_key = entry.get("api_key")
        file_model = entry.get("model")

        env_key = os.environ.get(f"{ENV_PREFIX}{name.upper()}_API_KEY")
        env_model = os.environ.get(f"{ENV_PREFIX}{name.upper()}_MODEL")

        api_key = env_key or file_key or None
        # Empty placeholders count as unset.
        if isinstance(api_key, str) and (not api_key.strip() or "REPLACE-ME" in api_key):
            api_key = None

        model = env_model or file_model or default_model

        file_effort = entry.get("reasoning_effort")
        env_effort = os.environ.get(f"{ENV_PREFIX}{name.upper()}_REASONING_EFFORT")
        raw_effort = env_effort if env_effort is not None else file_effort
        reasoning_effort: str | None = None
        if raw_effort is not None and str(raw_effort).strip():
            normalised = str(raw_effort).strip().lower()
            if normalised not in REASONING_EFFORTS:
                raise ConfigError(
                    f"providers.{name}.reasoning_effort must be one of "
                    f"{sorted(REASONING_EFFORTS)} (got {raw_effort!r})"
                )
            reasoning_effort = normalised

        file_ws = entry.get("web_search")
        env_ws = os.environ.get(f"{ENV_PREFIX}{name.upper()}_WEB_SEARCH")
        if env_ws is not None:
            web_search = _truthy(env_ws)
        elif isinstance(file_ws, bool):
            web_search = file_ws
        else:
            web_search = False

        providers[name] = ProviderConfig(
            api_key=api_key,
            model=model,
            reasoning_effort=reasoning_effort,
            web_search=web_search,
        )

    warnings: list[str] = []
    request_budget_seconds = _load_request_budget(raw, warnings)
    default_max_tokens = _load_default_max_tokens(raw)

    log_prompts = bool(raw.get("log_prompts", False))
    if f"{ENV_PREFIX}LOG_PROMPTS" in os.environ:
        log_prompts = _truthy(os.environ[f"{ENV_PREFIX}LOG_PROMPTS"])

    log_level = os.environ.get(f"{ENV_PREFIX}LOG_LEVEL", "INFO").upper()

    return AppConfig(
        providers=providers,
        request_budget_seconds=request_budget_seconds,
        default_max_tokens=default_max_tokens,
        log_prompts=log_prompts,
        log_level=log_level,
        config_path=path,
        warnings=warnings,
    )


def _load_request_budget(raw: dict, warnings: list[str]) -> float:
    """Resolve the per-call budget.

    `timeout_seconds` is the pre-existing name for this value and still works,
    so upgrading doesn't silently reset an operator's tuned value. If both are
    present `request_budget_seconds` wins and the conflict is reported.
    """
    legacy = raw.get("timeout_seconds")
    current = raw.get("request_budget_seconds")
    if current is not None and legacy is not None:
        warnings.append(
            "config sets both `request_budget_seconds` and the deprecated "
            f"`timeout_seconds`; using request_budget_seconds={current!r} and "
            f"ignoring timeout_seconds={legacy!r}"
        )
    elif current is None and legacy is not None:
        warnings.append(
            "`timeout_seconds` is deprecated; rename it to "
            "`request_budget_seconds` (same meaning, bounds the whole call)"
        )

    value = current if current is not None else legacy
    # Env override. LLM_SECOND_OPINION_TIMEOUT kept working for the same reason.
    env_value = (
        os.environ.get(f"{ENV_PREFIX}REQUEST_BUDGET")
        or os.environ.get(f"{ENV_PREFIX}TIMEOUT")
    )
    if env_value:
        value = env_value

    if value is None:
        return DEFAULT_REQUEST_BUDGET_SECONDS
    try:
        budget = float(value)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"Invalid request_budget_seconds: {value!r}") from e
    if budget <= 0:
        raise ConfigError("request_budget_seconds must be > 0")
    if budget >= BUDGET_WARN_THRESHOLD_SECONDS:
        warnings.append(
            f"request_budget_seconds={budget} leaves no room under the "
            f"{CLIENT_HARD_CAP_SECONDS}s cap Claude Desktop enforces on tool "
            f"calls; Desktop will cancel the call and discard the result "
            f"before this budget fires. Use ~200 or lower."
        )
    return budget


def _load_default_max_tokens(raw: dict) -> int | None:
    """Reply cap used when the caller passes no `max_tokens`.

    Explicit null in the file means 'leave replies unbounded' and is honoured;
    only an absent key falls back to the default.
    """
    env_value = os.environ.get(f"{ENV_PREFIX}DEFAULT_MAX_TOKENS")
    if env_value is not None:
        if not env_value.strip() or env_value.strip().lower() in {"none", "null"}:
            return None
        value: object = env_value
    elif "default_max_tokens" in raw:
        value = raw["default_max_tokens"]
    else:
        return DEFAULT_MAX_TOKENS

    if value is None:
        return None
    try:
        tokens = int(value)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"Invalid default_max_tokens: {value!r}") from e
    if tokens <= 0:
        raise ConfigError("default_max_tokens must be > 0, or null to disable")
    return tokens
