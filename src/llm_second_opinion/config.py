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

# Applied when the caller omits `max_tokens`. Set to null to leave replies
# unbounded.
#
# Truncation is not a soft failure here: a reply that hits the cap comes back
# `incomplete` and surfaces as an error, so the user gets nothing at all. That
# makes a tight cap actively worse than a loose one, and an earlier value of
# 8000 was cutting real reviews short.
#
# The real ceiling is the request budget, not the API. Measured against the
# live providers: gemini-3.6-flash caps output at 65536 and sustains ~178
# tok/s (~35k inside a 200s budget); OpenAI and xAI publish no hard cap at all
# (both accepted max_output_tokens=10_000_000) but generate at ~62 and ~49
# tok/s, so the budget binds first at roughly 12k and 10k. At 60000 both of
# them ran out the full 200s and returned nothing.
#
# 32000 sits above every real answer observed (largest: 14129 billed tokens),
# under Gemini's hard limit, and left the slowest provider at 53% of budget on
# a full review. Beyond it the extra headroom is inert for OpenAI and xAI —
# they hit the deadline before the cap.
DEFAULT_MAX_TOKENS = 32000

# Wall-clock bound on one background *job* (submit_second_opinion), which
# lives independently of any tool call and is therefore not subject to the
# MCP client's per-call cap. 900s is ~4.5x the worst genuine workload
# observed (a full-document review with web search that died at 200s under
# the synchronous tool) while still bounding runaway vendor spend.
DEFAULT_JOB_BUDGET_SECONDS = 900.0

# Above this, warn: vendor-side background retention windows make very long
# jobs fragile (OpenAI keeps unstored background responses only ~10 minutes;
# stored ones follow the account's retention policy). Honoured, never clamped.
JOB_BUDGET_WARN_THRESHOLD_SECONDS = 3600.0

# Attachments (DESIGN-submit-poll.md §17.4). Disabled until at least one root
# is configured: the server never reads the filesystem at a model's direction
# unless the operator has named where. The cap is the total across all
# attachments on one call (~250K tokens: under every configured model's
# context with room for search ingestion and output).
DEFAULT_MAX_ATTACHMENT_BYTES = 1_000_000

# Per-provider allowed `reasoning_effort` values (DESIGN-submit-poll.md §18.3).
# A global enum both rejected valid values and passed invalid ones, because
# the vocabularies differ by vendor and shift by model generation. Each set
# below is the vendor's documented value set for its current model line,
# checked against primary documentation on 2026-09-11; narrowing per model is
# left to the vendor (a value a specific model rejects surfaces as
# `bad_request` with the vendor's message).
REASONING_EFFORTS_BY_PROVIDER: dict[str, frozenset[str]] = {
    # https://developers.openai.com/api/docs/guides/reasoning (checked
    # 2026-09-11): "Supported values are model-dependent and can include
    # none, minimal, low, medium, high, xhigh, and max."
    "openai": frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"}),
    # https://ai.google.dev/gemini-api/docs/thinking (checked 2026-09-11):
    # `thinking_level` per model — the 3.x Flash line (incl. the default
    # gemini-3.6-flash) takes minimal/low/medium/high; Pro lines take
    # low/medium/high or low/high. The provider set is the union.
    "gemini": frozenset({"minimal", "low", "medium", "high"}),
    # https://docs.x.ai/docs/guides/reasoning (checked 2026-09-11):
    # `reasoning_effort` accepts low, medium, high, xhigh; xhigh is available
    # on grok-4.6 and later and is treated as high on grok-4.5.
    "grok": frozenset({"low", "medium", "high", "xhigh"}),
}


@dataclass
class ProviderConfig:
    api_key: str | None = None
    model: str = ""
    # One of the provider's REASONING_EFFORTS_BY_PROVIDER values, or None to
    # leave the SDK default. For Gemini this is `thinking_level` on the
    # Interactions API; for OpenAI/Grok it is `reasoning.effort` on the
    # Responses API.
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
    # Bounds one background job — see DEFAULT_JOB_BUDGET_SECONDS. Governs
    # only the submit/poll tools; `request_budget_seconds` still governs the
    # synchronous tool alone.
    job_budget_seconds: float = DEFAULT_JOB_BUDGET_SECONDS
    # Directories whose files may be passed as `attachment_paths`. Empty (the
    # default) disables attachments entirely — see attachments.py.
    attachment_roots: list[str] = field(default_factory=list)
    # Total byte cap across all attachments on one call.
    max_attachment_bytes: int = DEFAULT_MAX_ATTACHMENT_BYTES
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
            allowed = REASONING_EFFORTS_BY_PROVIDER[name]
            if normalised not in allowed:
                raise ConfigError(
                    f"providers.{name}.reasoning_effort must be one of "
                    f"{sorted(allowed)} for provider {name!r} (got {raw_effort!r})"
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
    job_budget_seconds = _load_job_budget(raw, warnings)
    default_max_tokens = _load_default_max_tokens(raw)
    attachment_roots = _load_attachment_roots(raw, warnings)
    max_attachment_bytes = _load_max_attachment_bytes(raw)

    log_prompts = bool(raw.get("log_prompts", False))
    if f"{ENV_PREFIX}LOG_PROMPTS" in os.environ:
        log_prompts = _truthy(os.environ[f"{ENV_PREFIX}LOG_PROMPTS"])

    log_level = os.environ.get(f"{ENV_PREFIX}LOG_LEVEL", "INFO").upper()

    return AppConfig(
        providers=providers,
        request_budget_seconds=request_budget_seconds,
        job_budget_seconds=job_budget_seconds,
        default_max_tokens=default_max_tokens,
        attachment_roots=attachment_roots,
        max_attachment_bytes=max_attachment_bytes,
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


def _load_job_budget(raw: dict, warnings: list[str]) -> float:
    """Resolve the per-job budget: env `LLM_SECOND_OPINION_JOB_BUDGET` beats
    the file's `job_budget_seconds`, which beats the default."""
    value = raw.get("job_budget_seconds")
    env_value = os.environ.get(f"{ENV_PREFIX}JOB_BUDGET")
    if env_value:
        value = env_value

    if value is None:
        return DEFAULT_JOB_BUDGET_SECONDS
    try:
        budget = float(value)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"Invalid job_budget_seconds: {value!r}") from e
    if budget <= 0:
        raise ConfigError("job_budget_seconds must be > 0")
    if budget > JOB_BUDGET_WARN_THRESHOLD_SECONDS:
        warnings.append(
            f"job_budget_seconds={budget} is above {JOB_BUDGET_WARN_THRESHOLD_SECONDS:g}s; "
            f"vendor-side background retention windows make very long jobs "
            f"fragile (OpenAI keeps background responses for a limited window). "
            f"Honoured as set, but prefer a smaller value."
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


def _load_attachment_roots(raw: dict, warnings: list[str]) -> list[str]:
    """Directories whose files may be attached (DESIGN §17.4).

    Env `LLM_SECOND_OPINION_ATTACHMENT_ROOTS` (an `os.pathsep`-separated list)
    beats the file's `attachment_roots` list. Empty means disabled. Roots are
    kept as given — resolution (symlinks followed) happens per call — but a
    root that is not an existing directory is warned about at startup, since
    it can never admit a file.
    """
    env_value = os.environ.get(f"{ENV_PREFIX}ATTACHMENT_ROOTS")
    if env_value is not None:
        roots = [p.strip() for p in env_value.split(os.pathsep) if p.strip()]
    else:
        value = raw.get("attachment_roots")
        if value is None:
            roots = []
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            roots = [v.strip() for v in value if v.strip()]
        else:
            raise ConfigError("`attachment_roots` must be a list of directory paths")
    for root in roots:
        if not Path(root).expanduser().is_dir():
            warnings.append(
                f"attachment_roots entry {root!r} is not an existing directory; "
                f"no attachment can resolve inside it"
            )
    return roots


def _load_max_attachment_bytes(raw: dict) -> int:
    """Total byte cap across all attachments on one call (DESIGN §17.4)."""
    env_value = os.environ.get(f"{ENV_PREFIX}MAX_ATTACHMENT_BYTES")
    if env_value is not None and env_value.strip():
        value: object = env_value
    elif "max_attachment_bytes" in raw:
        value = raw["max_attachment_bytes"]
    else:
        return DEFAULT_MAX_ATTACHMENT_BYTES
    try:
        cap = int(value)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"Invalid max_attachment_bytes: {value!r}") from e
    if cap <= 0:
        raise ConfigError("max_attachment_bytes must be > 0")
    return cap
