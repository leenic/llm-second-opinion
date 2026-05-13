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
    "openai": "gpt-5.5",
    "gemini": "gemini-3.1-pro-preview",
    "grok": "grok-4.3",
}

DEFAULT_TIMEOUT_SECONDS = 180.0

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
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    log_prompts: bool = False
    log_level: str = "INFO"
    config_path: Path | None = None


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

    timeout_seconds = float(raw.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    env_timeout = os.environ.get(f"{ENV_PREFIX}TIMEOUT")
    if env_timeout:
        try:
            timeout_seconds = float(env_timeout)
        except ValueError as e:
            raise ConfigError(f"Invalid {ENV_PREFIX}TIMEOUT: {env_timeout}") from e
    if timeout_seconds <= 0:
        raise ConfigError("timeout_seconds must be > 0")

    log_prompts = bool(raw.get("log_prompts", False))
    if f"{ENV_PREFIX}LOG_PROMPTS" in os.environ:
        log_prompts = _truthy(os.environ[f"{ENV_PREFIX}LOG_PROMPTS"])

    log_level = os.environ.get(f"{ENV_PREFIX}LOG_LEVEL", "INFO").upper()

    return AppConfig(
        providers=providers,
        timeout_seconds=timeout_seconds,
        log_prompts=log_prompts,
        log_level=log_level,
        config_path=path,
    )
