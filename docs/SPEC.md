# llm-second-opinion — System Specification

**Version:** 0.2.0 · **Status:** current as of 2026-09-07 · **Audience:** contributors extending the server

> **0.2.0 amendment.** The submit/poll background-job extension specified in
> [DESIGN-submit-poll.md](DESIGN-submit-poll.md) has landed. This document has been amended where the
> extension touches it (§3, §5, §6.4, §8.2, §9, §10, §12–§16); the design document remains the authoritative
> rationale for the job subsystem and is not repeated here.

This document specifies what the application *does today*, precisely enough to serve as the foundation for
extensions. It is derived from an exhaustive review of the codebase (all source, tests, configuration, packaging,
and documentation). Where behaviour is intentional and load-bearing, it is called out as an **invariant**
(§15) that extensions must preserve.

---

## 1. Purpose

`llm-second-opinion` is a **local MCP (Model Context Protocol) server** that gives a Claude session (Claude Code
or Claude Desktop) the ability to obtain an **independent, critical second opinion** from a competing LLM vendor
— OpenAI (ChatGPT), Google (Gemini), or xAI (Grok).

The design goal is *not* to outvote Claude but to close information gaps: different training corpora, different
web-search behaviour, different perspectives. Claude remains the driver; the second opinion is one more
well-informed voice the user can weigh.

Deliberate scope boundaries of v1 (each is a candidate extension, see §16):

- **Single-turn only.** No conversation history is forwarded; each call is stateless and self-contained.
- **No streaming.** The full reply arrives in one tool result.
- **No caching, cost tracking, or budget enforcement.**
- **No server-side rate limiting** (upstream limits are relied on).
- **Stdio transport only.** No HTTP or remote-access mode.
- **Single-developer use.** No multi-tenancy, no auth of its own.

## 2. System context and data flow

```
┌────────────┐  MCP over stdio   ┌──────────────────────┐   HTTPS   ┌──────────────────┐
│   Claude    │ ────────────────▶ │  llm-second-opinion  │ ────────▶ │ OpenAI Responses │
│ (Code /     │  JSON-RPC on      │  (FastMCP server,    │           │ xAI Responses    │
│  Desktop)   │  stdout/stdin     │   local process)     │           │ Google           │
└────────────┘  logs on stderr    └──────────────────────┘           │   Interactions   │
                                                                     └──────────────────┘
```

One tool call maps to (at most) one upstream API request — plus at most one automatic retry in a narrowly
defined case (§9.3). The upstream reply is normalised into a structured JSON tool result.

**Transport discipline:** stdout carries the JSON-RPC stream and *nothing else*; every log line goes to stderr.
This is enforced by tests (§14).

## 3. Repository layout

| Path | Role |
|---|---|
| `src/llm_second_opinion/__init__.py` | Package marker; `__version__ = "0.2.0"` |
| `src/llm_second_opinion/__main__.py` | `python -m llm_second_opinion` entry point |
| `src/llm_second_opinion/server.py` | FastMCP server, all five tools, timing/cancellation logic, the background-job drivers, error envelopes |
| `src/llm_second_opinion/jobs.py` | Background-job record, state machine, in-memory registry with `request_key` index and lazy TTL eviction, timing-model-v2 constants (no asyncio driving code) |
| `src/llm_second_opinion/config.py` | Config file discovery, env overrides, validation, defaults, constants |
| `src/llm_second_opinion/logging_setup.py` | Stderr-only logger configuration |
| `src/llm_second_opinion/providers/base.py` | Provider ABC, request/response/usage dataclasses, `ProviderError`, error taxonomy |
| `src/llm_second_opinion/providers/__init__.py` | Target→provider registry, `build_provider` factory |
| `src/llm_second_opinion/providers/openai_provider.py` | `ResponsesAPIProvider` base + `OpenAIProvider`; final-answer extraction; droppable-param retry |
| `src/llm_second_opinion/providers/grok.py` | `GrokProvider` — `ResponsesAPIProvider` with xAI base URL |
| `src/llm_second_opinion/providers/gemini.py` | `GeminiProvider` — Google Interactions API adapter |
| `tests/` | Pytest suite (see §14) |
| `config.example.json` | Placeholder config shipped with the repo |
| `README.md` / `USAGE.md` | Operator setup reference / end-user walkthrough |
| `docs/index.html` | Self-contained GitHub Pages intro page with an animated demo re-enactment |
| `pyproject.toml` | Hatchling build; console script; pytest config |

**Dependencies:** `mcp>=1.2.0,<2` (2.x renamed `FastMCP` and changed the tool API), `openai>=2.36,<3`,
`google-genai>=2.0,<3`. Dev: `pytest>=8.0`, `pytest-asyncio>=0.23` (strict asyncio mode). Requires Python
≥ 3.10. License MIT.

## 4. Runtime and process model

### 4.1 Entry points

- Console script `llm-second-opinion` → `llm_second_opinion.__main__:main` → `server.main()`.
- `python -m llm_second_opinion` is equivalent.

### 4.2 Boot sequence (`server.main`)

1. `load_config()` — a `ConfigError` (unreadable/invalid config, bad values) prints
   `llm-second-opinion: configuration error: …` to stderr and exits with code **2**. A *missing* config file is
   not an error: the server starts with all providers unavailable.
2. `setup_logging(config.log_level)` — logger `llm_second_opinion`, single `StreamHandler` on **stderr**,
   format `%(asctime)s %(levelname)s %(name)s %(message)s` (ISO-ish datefmt), `propagate = False` (so a root
   handler elsewhere can never reach stdout). Idempotent on re-call (hot-reload safe).
3. Deferred config warnings (collected during `load_config`, which runs before logging exists) are emitted now.
4. Startup summary logged: config path (or the four searched locations), configured providers (as *target*
   names), `request_budget_seconds`, `job_budget_seconds`, `default_max_tokens`.
5. `build_server(config, logger, registry=None)` constructs `FastMCP("llm-second-opinion")` with all five
   tools registered and one `JobRegistry` (a fresh one unless a test injects its own); `server.run()` serves
   MCP over stdio.

`build_server` is deliberately separated from `main()` so tests can invoke tools in-process via
`server.call_tool(...)` without a subprocess.

## 5. Configuration subsystem (`config.py`)

### 5.1 File discovery (first match wins)

1. `$LLM_SECOND_OPINION_CONFIG` (path, `~` expanded)
2. `./config.json` (current working directory)
3. `%APPDATA%\llm-second-opinion\config.json` (only if `APPDATA` is set — Windows)
4. `~/.config/llm-second-opinion/config.json` (Linux/macOS)

A file that exists but cannot be read or parsed raises `ConfigError` (fatal). No file at all yields an empty
config. `.gitignore` excludes every `config*.json` variant except `config.example.json`, so real keys never
enter version control.

### 5.2 Resolved model (`AppConfig` / `ProviderConfig` dataclasses)

Per provider (`openai`, `gemini`, `grok` — the keys of `DEFAULT_MODELS`; every provider always gets a
`ProviderConfig` entry even when absent from the file):

| Field | Resolution (highest precedence first) | Validation / semantics |
|---|---|---|
| `api_key` | env `LLM_SECOND_OPINION_<NAME>_API_KEY` → file `providers.<name>.api_key` | Blank or containing `REPLACE-ME` ⇒ `None` ⇒ provider **unavailable** (server still runs) |
| `model` | env `…_<NAME>_MODEL` → file `model` → `DEFAULT_MODELS[name]` | Defaults: `openai: gpt-5.6-sol`, `gemini: gemini-3.6-flash`, `grok: grok-4.5` (each vendor's current flagship — the env override exists so a new flagship needs no code change) |
| `reasoning_effort` | env `…_<NAME>_REASONING_EFFORT` → file value | Normalised (strip + lowercase); must be one of `minimal, low, medium, high` else `ConfigError`; unset/empty ⇒ `None` ⇒ SDK default thinking depth (reasoning **on** for all three current flagships) |
| `web_search` | env `…_<NAME>_WEB_SEARCH` (truthy: `1/true/yes/on`) → file boolean → `False` | Non-boolean file value is ignored (treated as `False`) |

Top-level:

| Field | Resolution | Validation / semantics |
|---|---|---|
| `request_budget_seconds` | env `…_REQUEST_BUDGET` or legacy `…_TIMEOUT` → file `request_budget_seconds` (wins over legacy `timeout_seconds`; both present ⇒ warning) → **200.0** | Float > 0 else `ConfigError`. Value ≥ **235** produces a warning (Desktop's 240 s cap would fire first) but is **honoured, never clamped** — another MCP client may allow longer. §10 explains the model. |
| `job_budget_seconds` | env `…_JOB_BUDGET` → file `job_budget_seconds` → **900.0** | Float > 0 else `ConfigError`. Value > **3600** produces a warning (vendor-side background retention windows make very long jobs fragile) but is honoured. Governs only the background-job tools (§6.4); `request_budget_seconds` continues to govern only the synchronous tool. |
| `default_max_tokens` | env `…_DEFAULT_MAX_TOKENS` (`""`/`none`/`null` ⇒ unbounded) → file key **if present** (explicit JSON `null` ⇒ unbounded — distinct from an absent key) → **32000** | Int > 0 or null else `ConfigError`. §11 explains the choice of 32000. |
| `log_prompts` | env `…_LOG_PROMPTS` → file boolean → `False` | Gates DEBUG lines carrying prompt/response content |
| `log_level` | env `…_LOG_LEVEL` only → `INFO` | Uppercased. **No config-file field exists for this** — env only. |

`AppConfig.timeout_seconds` is a deprecated read-only alias for `request_budget_seconds`, kept so existing code
and operator muscle memory don't break.

`AppConfig.warnings` accumulates non-fatal problems during load (deprecated key used, both keys set, budget at
or above the warn threshold); `main()` logs them once the logger exists.

### 5.3 Named constants (in `config.py`)

| Constant | Value | Meaning |
|---|---|---|
| `DEFAULT_REQUEST_BUDGET_SECONDS` | 200.0 | Wall-clock bound on one whole tool call |
| `CLIENT_HARD_CAP_SECONDS` | 240.0 | Claude Desktop's non-configurable tool-call cap (fires as `MCP error -32001`, result discarded) |
| `BUDGET_WARN_THRESHOLD_SECONDS` | 235.0 | Budget at/above this is warned about |
| `DEFAULT_MAX_TOKENS` | 32000 | Reply cap when the caller passes no `max_tokens` |
| `REASONING_EFFORTS` | `{minimal, low, medium, high}` | Accepted effort values (shared enum across all three providers) |
| `DEFAULT_JOB_BUDGET_SECONDS` | 900.0 | Wall-clock bound on one background job (not subject to the client cap) |
| `JOB_BUDGET_WARN_THRESHOLD_SECONDS` | 3600.0 | Job budget above this is warned about |
| `CANCEL_GRACE_SECONDS` (in `server.py`) | 5.0 | Teardown window for a cancelled provider task; budget + grace must clear the client cap (200 + 5 < 240) |

Job constants (in `jobs.py`, design §10): `SUBMIT_BUDGET_SECONDS = 30.0` (bounds submit, poll and cancel
control calls), `MAX_POLL_WAIT_SECONDS = 45.0`, `POLL_INTERVAL_SECONDS = 4.0`, `MAX_ACTIVE_JOBS = 8`,
`TERMINAL_JOB_TTL_SECONDS = 1800.0`.

## 6. MCP interface

Server name: `llm-second-opinion`. Five tools: the synchronous `second_opinion` (§6.1), `list_available_models`
(§6.2), and the background-job trio `submit_second_opinion` / `get_second_opinion` / `cancel_second_opinion`
(§6.4). Tool schemas are derived by FastMCP from the Python signatures — the `Literal["gemini", "grok",
"chatgpt"]` annotation is what advertises the allowed `target_model` values to the client, and the docstring
`Args:` entries feed the parameter descriptions. The tool *descriptions* are behavioural steering for the
calling model (design §3.4): the sync tool's points heavyweight reviews at the job tools, and the job tools'
encode the polling protocol (wait 45 s, then report progress and poll after `retry_after_ms`, never in a
tight loop).

### 6.1 `second_opinion`

Send a summary to one external LLM and return its independent, critical reply.

| Arg | Type | Required | Semantics |
|---|---|---|---|
| `summary` | `str` | yes | The content to review. Must be non-empty after strip, else `invalid_input` (checked before any provider work). |
| `target_model` | `Literal["gemini","grok","chatgpt"]` | yes | Which external LLM. Mapped to a provider via the registry (§8.3). |
| `focus` | `str \| None` | no | Aspect to emphasise; prepended to the user message (§7). |
| `system_prompt` | `str \| None` | no | Replaces the default reviewer prompt; blank/whitespace falls back to the default. |
| `temperature` | `float \| None` | no | Passed through when set. If the model rejects it, it is dropped and retried once (§9.3). |
| `max_tokens` | `int \| None` | no | Output cap. `None` ⇒ `default_max_tokens` applies. The check is `is not None`, so an explicit `0` is honoured, not replaced by the default. |

**Handler flow** (`server.build_server → second_opinion`):

1. Generate `request_id` = `uuid4().hex[:12]`; start a monotonic clock (`elapsed_ms` covers the whole handler).
2. Validate `summary`; resolve the effective system prompt.
3. `build_provider(target_model, config)` — raises `ProviderError` for unknown target (`invalid_input`) or a
   provider without a key (`missing_api_key`, message tells the user exactly which config field / env var to set).
4. Compute `effective_max_tokens`; build a `SecondOpinionRequest`; log the request-scoped INFO line (provider,
   model, focus present y/n, temperature, max_tokens, budget). With `log_prompts`, DEBUG-log the summary/focus.
5. `await _run_bounded(provider.generate(req), budget)` — the timing/cancellation core, §10.
6. Map the outcome to exactly one of the result shapes below. **Errors are returned as ordinary tool results,
   never raised** — a raised exception reaches the calling model as an opaque transport failure it cannot
   reason about, whereas this envelope tells it what went wrong and whether retrying is worth it.

**Success result:**

```json
{
  "success": true,
  "request_id": "ab12cd34ef56",
  "target_model": "gemini",
  "provider": "gemini",
  "model": "gemini-3.6-flash",
  "response": "…the external model's final answer…",
  "usage": {"input_tokens": 123, "output_tokens": 456, "total_tokens": 579, "reasoning_tokens": 40},
  "latency_ms": 1840,
  "elapsed_ms": 1873
}
```

- `model` is the **actual** model identifier reported by the upstream response (falls back to the configured
  name), so aliasing/rollouts upstream stay visible.
- `usage` may be `null`; individual fields may be `null` when unreported. `reasoning_tokens` is the subset of
  output tokens spent on hidden reasoning — it exists to explain "high output_tokens but short reply" outcomes.
- `latency_ms` is measured inside the provider adapter (upstream call, including the one permitted retry);
  `elapsed_ms` is the whole handler.

**Error result:**

```json
{
  "success": false,
  "request_id": "ab12cd34ef56",
  "target_model": "grok",
  "model": "grok-4.5",
  "error": {"type": "timeout", "message": "…", "retriable": true},
  "elapsed_ms": 200009
}
```

- `model` and `elapsed_ms` are included when known (e.g. absent on `missing_api_key`, present on timeouts).
- `error.type` ∈ the fixed taxonomy (§6.3); `retriable` tells the calling model whether re-issuing the call may
  help.
- The timeout message is prescriptive: it names the budget, how long the call ran, and the remedies (retry,
  lower `reasoning_effort`, smaller `max_tokens`, raise `request_budget_seconds` below the client's cap).
- Note: `ProviderError` carries an HTTP `status` and has a `to_dict()` including it, but the server's envelope
  builder (`_error_response`) does not surface `status` to the client — it appears only in logs. `to_dict()` is
  currently dormant. An extension surfacing `status` would be additive and safe.

**Catch-all:** any non-`ProviderError` exception from a provider becomes `internal_error`
(`retriable: false`) with a full traceback logged — the handler never crashes the server.

### 6.2 `list_available_models`

No arguments. Reports, for every registered target, its configuration plus a live reachability probe, so Claude
can answer "what's set up?" without burning a real call.

```json
{
  "request_id": "…",
  "providers": [
    {
      "target_model": "chatgpt",
      "provider": "openai",
      "configured_model": "gpt-5.6-sol",
      "api_key_configured": true,
      "available": true,
      "reason": null,
      "reasoning_effort": "medium",
      "web_search": false
    }
  ],
  "available_target_models": ["chatgpt", "gemini"],
  "default_system_prompt": "You are acting as an external reviewer…"
}
```

Behaviour (`_check_all_providers`):

- Iterates the registry; a provider without a key is reported immediately (`available: false`,
  `reason: "missing_api_key"`) with no network traffic.
- All probes run **concurrently** (`asyncio.gather`); each probe is bounded at **5 s** and exception-proofed
  (any surprise becomes `available: false` with a reason string).
- Probe mechanics differ by family: OpenAI/Grok issue `GET /models` (catches bad keys and outages but does
  **not** validate the configured model name); Gemini issues `models.get(model=…)` (which *does* 404 on an
  unknown model name). The README states the weaker guarantee; treat model-name validation as best-effort.
- Note: this function constructs providers directly (with the same kwargs as `build_provider`) rather than via
  the factory — **adding a provider requires touching both** (§16.1).

### 6.3 Error taxonomy

Defined in `providers/base.py::ERROR_TYPES`; values are **stable API** (Claude may show them verbatim):

`missing_api_key` · `auth_failed` · `rate_limit` · `timeout` · `network_error` · `upstream_error` ·
`bad_request` · `content_blocked` · `invalid_input` · `internal_error` · `unknown_job` (0.2.0: job_id not
in the registry — typo, TTL eviction, or post-restart orphan; not retriable) · `job_limit` (0.2.0:
`MAX_ACTIVE_JOBS` reached; retriable, message lists the active jobs)

`ProviderError(error_type, message, retriable=False, status=None)` coerces any unknown type to
`internal_error` at construction, so a typo in an adapter can never leak a novel type to the client.

### 6.4 Background-job tools (0.2.0)

Contracts, result shapes and rationale are specified in [DESIGN-submit-poll.md](DESIGN-submit-poll.md) §3–§8;
this section records what was built.

- **`submit_second_opinion`** — `second_opinion`'s arguments plus `request_key: str | None`. Validates
  `summary` and resolves the provider exactly as §6.1 steps 1–4 (same `invalid_input` / `missing_api_key`
  envelopes, same `default_max_tokens` rule). Then: a `request_key` that matches a known job (running *or*
  terminal, until eviction) returns that job's submit shape with `"reused_existing_job": true` and no upstream
  call; the active-job cap returns `job_limit`; otherwise a `JobRecord` is created and started per the
  provider's backing. Provider-backed (`supports_background`): `provider.submit_background(req, timeout=30)`
  under `_run_bounded(…, SUBMIT_BUDGET_SECONDS)`; a timeout returns a `timeout` envelope whose message says the
  work may be orphaned and to retry with a `request_key`; the job is registered only once the upstream id is
  in hand, so `submitting` is never observed. Local-task (grok): registered immediately and the driver starts
  `generate(req)`. Success shape: `success, request_id, job_id, target_model, provider, model, backing,
  status, job_budget_seconds, poll_after_ms`. Two identical submits that overlap in time (the first not yet
  acknowledged) both create jobs — the key is indexed at registration; the race the key guards is the
  *lost-reply retry*, which arrives after registration.
- **`get_second_opinion(job_id, wait_seconds=0)`** — unknown id ⇒ `unknown_job` (message names the three
  causes and the restart consequence per backing). `wait_seconds` is clamped to `[0, 45]` with a warning above
  the cap. A positive wait on a running job awaits the record's `done` event under `_run_bounded`; the wait,
  and only the wait, is discarded at the deadline. Running result: the submit fields plus `job_elapsed_ms`
  and `retry_after_ms` (5 s under 60 s of age, 15 s under 300 s, then 30 s). Terminal result: the cached
  envelope (`status` `succeeded` with the §6.1 success fields — `latency_ms` measured across the job — or
  `failed` with the §6.1 `error` object, or `cancelled` with `cancelled_by`/`cancelled_at`) plus this call's
  `request_id` and `elapsed_ms`, identical on every re-read until eviction.
- **`cancel_second_opinion(job_id)`** — unknown ⇒ `unknown_job`; terminal ⇒ the terminal state (not an error).
  Provider-backed: `provider.cancel_background(upstream_id, timeout=30)` under `_run_bounded`; a failure of
  that call is returned as its error and the job stays running. Then the record moves to `cancelled` and the
  driver task is cancelled. Local-task: the record moves to `cancelled` and the generate task is torn down with
  `_abandon` (cancel, wait at most `CANCEL_GRACE_SECONDS`, log and leave a task that ignores cancellation).

**Drivers.** Every job has one driver task, started at registration, and it is the *only* thing that talks to
the upstream job after submission. Provider-backed: loop — if `age > job_budget_seconds`, best-effort upstream
cancel and `failed(timeout)`; else `poll_background` bounded at `SUBMIT_BUDGET_SECONDS` (a poll-call timeout
or a *retriable* `ProviderError` is logged and retried next interval; a non-retriable one fails the job);
a `done` poll finishes the job with the response or the mapped error; otherwise sleep
`min(POLL_INTERVAL_SECONDS, remaining budget)`. Local-task: `_run_bounded(generate(req), job_budget_seconds,
on_start=record.attach_task)` — the same machinery as the sync path with the larger budget, so a task that
ignores cancellation is abandoned with the existing log line. A driver never surfaces an exception: any
unexpected one fails the job with `internal_error`. Consequence for the timing model: the job budget is
enforced whether or not anyone polls, and a `get` never touches upstream — it waits on an event.

**Registry (`jobs.py`).** `dict[job_id → JobRecord]` plus `request_key → job_id`; terminal transitions are
immutable (`finish` returns `False` for a second attempt, so whoever loses a cancel/complete race observes the
winner's state); terminal records are swept lazily on every access once older than
`TERMINAL_JOB_TTL_SECONDS`; running records are never evicted. `job_id = uuid4().hex[:12]` — the `request_id`
grammar in its own value space; provider ids are held as `upstream_id`, logged, never returned.

## 7. Prompt construction

**System prompt** (used verbatim unless the caller overrides; also returned by `list_available_models`):

> You are acting as an external reviewer for a conversation the user is having with another AI assistant. The
> user wants your independent view on the summary below. Be direct, concrete, and critical. If you disagree with
> the framing or see a stronger alternative, say so explicitly. Do not pad with praise. If a focus is provided,
> prioritise commenting on that aspect. State your confidence level when making factual claims.

**User message** (`Provider.build_user_content`): the summary, optionally prefixed —

```
Focus on: {focus}

{summary}
```

Nothing else is sent. No conversation history, no repo contents, no metadata.

## 8. Provider abstraction layer

### 8.1 Data types (`providers/base.py`)

- `SecondOpinionRequest(summary, focus, system_prompt, temperature, max_tokens)` — the normalised inbound
  request handed to every adapter.
- `SecondOpinionResponse(provider, model, text, usage, latency_ms)` — the normalised outbound reply.
- `TokenUsage(input_tokens, output_tokens, total_tokens, reasoning_tokens)` — all optional ints;
  `to_dict()` produces the `usage` object in results.
- `ProviderError` — see §6.3.

### 8.2 `Provider` ABC

Every adapter implements:

- `async generate(req: SecondOpinionRequest) -> SecondOpinionResponse` — one upstream call (plus the one
  sanctioned retry, §9.3), raising `ProviderError` on any failure.
- `async check_reachable() -> tuple[bool, str | None]` — cheap probe; never raises in practice (callers still
  guard).
- `model_id() -> str` — the configured model name.
- Class attribute `name` — the provider key used in config and logs.

Background mode (0.2.0) is optional per adapter, declared by the class attribute `supports_background`
(default `False`, in which case the server runs jobs as local tasks over `generate`). Adapters that set it
implement three *control calls*, each bounded by the `timeout` argument they receive:

- `async submit_background(req, timeout) -> str` — start the generation upstream, return the provider's id as
  soon as it is acknowledged; raises `ProviderError` on failure.
- `async poll_background(upstream_id, timeout) -> BackgroundPoll` — one observation. `BackgroundPoll(done,
  response, error, upstream_status)`: `done=False` while running; `done=True` with exactly one of `response`
  (succeeded) or `error` (a terminal upstream failure, mapped through **the same code as the synchronous
  path**). A failure of the poll call *itself* (network, auth, rate limit) is *raised* as `ProviderError`
  instead, so the driver can tell "the job failed" from "I could not ask".
- `async cancel_background(upstream_id, timeout) -> None` — idempotent.

All adapters share the constructor signature
`(api_key, model, timeout, reasoning_effort=None, web_search=False)` — the factory and the reachability checker
both rely on this.

### 8.3 Registry and factory (`providers/__init__.py`)

```python
TARGET_TO_PROVIDER = {"chatgpt": "openai", "gemini": "gemini", "grok": "grok"}
PROVIDER_TO_TARGET = inverse
```

`build_provider(target_model, config)`: validates the target (`invalid_input`), requires an API key
(`missing_api_key` with actionable message), then instantiates the matching adapter with
`timeout=config.request_budget_seconds` — **the same value as the handler's outer bound**, so the HTTP socket
closes on its own in the normal case rather than lingering behind a cancelled coroutine.

## 9. Provider adapters

### 9.1 `ResponsesAPIProvider` (OpenAI + any OpenAI-compatible endpoint)

Base class for `OpenAIProvider` (`base_url=None` ⇒ SDK default `https://api.openai.com/v1`) and
`GrokProvider` (`base_url="https://api.x.ai/v1"` — xAI exposes the same `/v1/responses` shape). A new
OpenAI-compatible provider is *just* a subclass setting `name` and `base_url`.

**Client construction:** `AsyncOpenAI(api_key, timeout=<budget>, max_retries=0, base_url?)`.
`max_retries=0` is deliberate: the SDK default of 2 restarts the *full* request per retry, and stacked retries
on a slow reasoning+web_search call blow past the MCP client's 240 s cap before the server can return its own
structured `timeout`. With retries off, `timeout` genuinely bounds wall-clock; Claude can re-issue the whole
tool call instead (errors carry `retriable`).

**Request construction** (`_build_kwargs`) — parameters are *omitted*, never sent as null:

| Condition | Parameter sent |
|---|---|
| always | `model`, `input` (user content, §7) |
| system prompt present | `instructions` |
| `temperature is not None` | `temperature` |
| `max_tokens is not None` | `max_output_tokens` |
| `reasoning_effort` configured | `reasoning={"effort": "<value>"}` |
| `web_search` configured | `tools=[{"type": "web_search"}]` |

**SDK error mapping** (`_create`):

| SDK exception | `error.type` | retriable | status |
|---|---|---|---|
| `APITimeoutError` | `timeout` | ✔ | — |
| `AuthenticationError` | `auth_failed` | ✘ | 401 |
| `PermissionDeniedError` | `auth_failed` | ✘ | 403 |
| `RateLimitError` | `rate_limit` | ✔ | 429 |
| `APIConnectionError` | `network_error` | ✔ | — |
| `NotFoundError` | `bad_request` ("check model name") | ✘ | 404 |
| `BadRequestError` | `bad_request` | ✘ | 400 |
| `APIStatusError` ≥ 500 | `upstream_error` | ✔ | n |
| `APIStatusError` < 500 | `upstream_error` | ✘ | n |

**Response validation** (`_build_response`) — a completed HTTP call is *not* assumed to be a good answer:

- `status == "failed"` → `upstream_error` carrying the upstream error message.
- `status == "incomplete"` + `incomplete_details.reason == "content_filter"` → `content_blocked` (not retriable).
- `status == "incomplete"` otherwise (typically the reply hit `max_output_tokens`, often because reasoning
  consumed the budget) → `upstream_error`, **retriable**, message includes `reasoning_tokens` and the remedies.
  This is why truncation surfaces as an error, not a shorter answer (§11).
- Empty extracted text + a `refusal` output item → `content_blocked` with the refusal text.
- Empty extracted text otherwise (reasoning-only output, or everything consumed by tool calls) →
  `upstream_error`, retriable, with diagnostics (`status`, output item count, `reasoning_tokens`). A silent
  empty "success" is never returned.
- Otherwise: success. `model` = `response.model` or the configured name; usage extracted including
  `output_tokens_details.reasoning_tokens`.

**Structure (0.2.0):** `generate` = `_build_kwargs` → `_create_with_fallback` (one `_create`, plus the §9.3
retry) → `_build_response`. `_create` delegates to `_mapped(awaitable, timeout)`, which holds the SDK error
mapping above, so the background control calls share it verbatim.

**Background mode (`OpenAIProvider`, `supports_background = True`):** `submit_background` sends
`_build_kwargs(req)` plus `background: true` and `store: true` (explicit — unstored background responses are
retained only ~10 minutes; stored ones follow the account's retention policy) through `_create_with_fallback`
on a client bound to the control timeout, and returns `response.id`. `poll_background` calls
`responses.retrieve(id)`: status `queued`/`in_progress` ⇒ running; `cancelled` ⇒ terminal `upstream_error`;
anything else is passed to `_build_response` so `failed`/`incomplete`/refusal/empty map exactly as on the
sync path. `cancel_background` calls `responses.cancel(id)` (idempotent upstream). `GrokProvider` sets
`supports_background = False` and must keep it: live on 2026-09-07 xAI rejected `background: true` with
`400 Argument not supported`, and a stored streaming response whose connection was closed after
`response.created` was never retrievable (404 through 90 s), so there is no store-and-retrieve path either
(design §12.1). Grok jobs run as local tasks.

### 9.2 Final-answer extraction (`_final_message_text`) — **invariant**

The SDK's `response.output_text` is **deliberately not used**: it concatenates the text of *every* message item
in the output timeline. A reasoning model that narrates before calling a tool ("I need current best
practices…") emits those asides as ordinary message items, and `output_text` glues them onto the front of the
real answer with no separator. Observed live on `grok-4.5` + `web_search` in roughly half of runs (message items
at output indices `[1, 8, 19]`); model behaviour, not provider behaviour, so it applies to the whole family.

Algorithm: walk `response.output` **backwards**; skip leading (i.e. trailing-in-time) non-message items so a
trailing reasoning item cannot hide the answer; collect the trailing *run* of message items' `output_text`
blocks; stop at the first non-message boundary once collecting (a reasoning step or tool call marks the edge of
the final turn); reverse and join. Adjacent trailing messages are kept together (an answer split across two
message items is one answer). All access is via `getattr` — see §14 for why that matters.

### 9.3 Droppable-parameter retry — **invariant boundaries**

Reasoning-only models (live example: `gpt-5.6-sol`) reject advisory sampling knobs outright —
`400 Unsupported parameter: 'temperature' is not supported with this model.` — instead of ignoring them.
Failing the whole call over an advisory knob wastes the user's round-trip, so:

- On a `bad_request` whose message matches `[Uu]nsupported parameter: '([^']+)'`, **and** the named parameter is
  in `DROPPABLE_PARAMS = {temperature, top_p}`, **and** it was actually sent: drop it, log a warning, retry
  **once**.
- The retry is charged **only the remaining time** on the original deadline
  (`client.with_options(timeout=remaining)`); if nothing remains, the original error propagates. Without this,
  a retry on a fresh client timeout would allow one tool call ~2× the budget — reintroducing exactly the
  overrun `max_retries=0` exists to prevent.
- `max_output_tokens` is **never** droppable: silently removing a length cap changes cost and truncation
  behaviour, so it must surface as an error.
- At most one retry ever happens (only one droppable param is sent per call; a second rejection propagates).
- `latency_ms` spans both attempts.

### 9.4 `GeminiProvider` (Google Interactions API)

Uses `google-genai >= 2.0`, `client.aio.interactions.create` — Google's stateful counterpart to the Responses
API, called in **single-turn mode** (no `previous_interaction_id`, no session state), matching the v1 spec.
The SDK currently emits `UserWarning: Interactions usage is experimental…` at first client construction; this
is expected (the API was promoted to recommended in May 2026, post the google-genai 2.0 breaking change).

**Request shape:** `model`, `input`; `system_instruction` (top-level, *not* nested under config);
`generation_config = {temperature?, max_output_tokens?, thinking_level?}` (the `reasoning_effort` enum passes
through unchanged — Gemini accepts the same four values); `tools=[{"type": "google_search"}]` when
`web_search` is on. All optional keys omitted when unset.

**Timeout:** the SDK call is wrapped in `asyncio.wait_for(coro, self.timeout)` (the genai client takes no
per-request timeout here); overrun → `ProviderError("timeout", retriable=True)`.

**Error mapping:** `genai_errors.APIError` translated by status code — 401/403 → `auth_failed`;
429 → `rate_limit` (retriable); 404 → `bad_request` ("check model name"); ≥500 → `upstream_error` (retriable);
other 4xx → `bad_request`; unknown → `upstream_error`. Any other exception class → `upstream_error`
(non-retriable) — the SDK may surface non-`APIError` classes.

**Interaction status handling:** `failed` → `upstream_error`; `budget_exceeded` → `upstream_error`;
`cancelled` / `incomplete` → `content_blocked` (note this differs from the Responses family, where `incomplete`
is usually a retriable token-budget outcome); empty text → `upstream_error`.

**Text extraction (`_join_output_text`):** the post-May-2026 response is a `steps` timeline (replacing the old
`outputs` list) where the answer lives in `type=="model_output"` steps whose `.content` holds
`type=="text"` blocks, interleaved with thought and tool-call/result steps. The SDK's `output_text` property
here *does* return exactly the trailing run, so it is preferred when present (as a `str`); the fallback is a
manual reverse walk with the same semantics as §9.2 (stop at `user_input`, collect the trailing run of
model-output text, a non-text block after collecting is a barrier).

**Usage mapping:** `total_input_tokens → input_tokens`, `total_output_tokens → output_tokens`,
`total_tokens → total_tokens`, `total_thought_tokens → reasoning_tokens`.

**Model id:** `response.model` may be a string or an object; `id`/`name` attributes are tried before `str()`.

No droppable-param retry exists on this path (`gemini-3.6-flash` accepts `temperature`).

**Structure (0.2.0):** `generate` = `_build_kwargs` → `_call(make_coro, timeout)` (the `wait_for` bound and
the error mapping above; the coroutine is created inside the guard) → `_build_response` (the status mapping,
text extraction, usage and model id above). **Background mode (`supports_background = True`):**
`submit_background` adds `background: true` (interactions are stored by default and `store=false` is
documented as incompatible with background execution, so nothing about storage is sent) and returns
`response.id`; `poll_background` calls `interactions.get(id=…)`: `queued`/`in_progress` ⇒ running;
`requires_action` ⇒ terminal `upstream_error` (nothing here can supply client input); anything else through
`_build_response`. `cancel_background` calls `interactions.cancel(id=…)`.

## 10. Timing and cancellation model

This is the most safety-critical subsystem. The forcing constraint: **Claude Desktop enforces a hard,
non-configurable 240 s cap on tool calls**; when it fires, Desktop cancels with `MCP error -32001` and the
result is *discarded* — even when the upstream call had effectively finished. The server's entire timing design
exists to get in front of that cap and return a structured, actionable result instead.

**One budget, three enforcement layers**, all derived from `request_budget_seconds` (default 200):

1. **Provider HTTP timeout** — the same value is the provider client's timeout, so in the normal case the
   socket is torn down by the HTTP layer itself, not left open behind a cancelled coroutine.
2. **Outer bound** (`_run_bounded`) — guarantees the handler returns regardless: a provider stalling *outside*
   its HTTP timeout (DNS, a retry loop, a future adapter that ignores the arg) is still cut off.
3. **Bounded teardown** (`_abandon`, `CANCEL_GRACE_SECONDS = 5`) — even the cleanup is capped, so
   budget + grace (205 s) clears the client cap with headroom.

**`_run_bounded(coro, budget)` semantics** — deliberately *not* `asyncio.wait_for`:

- `wait_for` has a trap: if the awaited coroutine catches `CancelledError` and returns a value anyway,
  `wait_for` hands that value back and the deadline is silently ignored — a late answer would surface as a
  success after the MCP client already gave up. Here, `asyncio.wait({task}, timeout=budget)` is used and once
  the deadline passes the task's result is **discarded unconditionally** (pinned by test: a provider returning
  `"LATE ANSWER"` from inside its `except CancelledError` still yields a `timeout` result).
- On timeout: `_abandon(task)` cancels and waits at most `CANCEL_GRACE_SECONDS`. A task that ignores
  cancellation is logged (`provider task ignored cancellation after …s; abandoning it`) and left behind —
  returning on time beats waiting on a misbehaving coroutine (the same coroutine `_run_bounded` refuses to
  trust for its result is not trusted to exit either). A task that does exit has its result/exception retrieved
  so asyncio never logs "exception was never retrieved".
- On **outer cancellation** (client disconnected, server shutting down — surfaces as `BaseException` from the
  `await`): the provider task is cancelled — otherwise the upstream request keeps running unmanaged, a billable
  API call holding a socket nothing will read — but deliberately **not awaited**: awaiting inside one's own
  cancellation is how shutdowns wedge (the continuation may never be scheduled on a tearing-down loop). Cancel
  stops the leak; teardown finishes on the loop's own time.

**Post-conditions** (all test-pinned): a timeout never wedges the server for the next request; no pending tasks
survive a handler return; subsequent calls succeed normally.

### 10.1 Observed failure modes (Aug 21 – Sep 5, 2026)

Live diagnosis of the model above, recorded in full in DESIGN-submit-poll.md §1, established:

1. **The structured server timeout is real but workload-specific.** A full-document review (large prompt
   ingestion × web-search loop × long multi-part output) breaches 200 s deterministically on `gpt-5.6-sol` at
   `medium` effort — `rid=0533d5147482`, `outcome=timeout`, `elapsed_ms=200015`, the first timeout outcome in
   the server's log history. The same configuration on a 600-word prompt took 161 s.
2. **The client-side four-minute hang can be pure dispatch loss.** A call can die with zero traces in either
   the bridge log or the server log — never dispatched, no upstream cost — after an idle gap on the
   claude.ai → Desktop bridge path. The identical call re-fired seconds after a successful liveness probe
   (`list_available_models`) succeeded in 56.7 s. Hence the *probe-then-fire* habit documented in USAGE.
3. **The three-layer budget machinery works end-to-end in production:** finding 1's envelope crossed the
   relay with ~40 s of headroom.
4. **A synchronous timeout burns unrecoverable money.** On a synchronous create the upstream id arrives only
   with the response, so a 200 s cancelled call is billed work with no handle to retrieve it by.
5. **Cost concentrates in search ingestion, not generation** (grok: 390,151 input tokens in 56.7 s; gpt:
   118,501 in 161 s), and latency does not scale with it.

Consequences: heavy workloads must escape the per-call envelope entirely; every individual tool call must be
short because the relay can eat one; upstream work must survive the client and the relay. That is the
background-job extension.

### 10.2 Timing model v2 (0.2.0)

The three-layer model is retained *per tool call* with per-tool budgets, and a **job** gets its own clock:

| Tool call | Budget | Grace | Worst case | Headroom vs 240 s cap |
|---|---|---|---|---|
| `submit_second_opinion` | `SUBMIT_BUDGET_SECONDS = 30` | 5 | 35 s | 205 s |
| `get_second_opinion` | `wait_seconds ≤ 45` (+ ~1 s) | 5 | ~51 s | ~189 s |
| `cancel_second_opinion` | 30 | 5 | 35 s | 205 s |
| `second_opinion` (sync) | `request_budget_seconds` (200) | 5 | 205 s | 35 s (grandfathered) |

`job_budget_seconds` (default 900) is enforced by the job's driver task (§6.4) for both backings: the
provider-backed poller cancels upstream and moves the job to `failed(timeout)`; the local-task driver reuses
`_run_bounded` with the job budget, including `_abandon`'s bounded teardown. A `get` call's own deadline
discards only its wait. **Cancellation is inverted for jobs:** the v0.1 rule "cancel on outer cancellation so a
billable call is not left running unmanaged" applies to the *wait* inside `get` (which owns nothing upstream)
but never to the job — background work decoupled from any listener is the point (invariant 11).

## 11. Reply-length model

`default_max_tokens` (default **32000**) is applied when the caller omits `max_tokens`. It is **not a
"shorten the answer" knob**: a reply that hits the cap comes back `incomplete` upstream and surfaces as an
error (§9.1) — the user gets *nothing*, not a shorter answer — so a tight cap is strictly worse than a loose
one (an earlier default of 8000 was cutting real reviews short; commit history records the raise).

The real ceiling is the time budget, not the provider API. Measured live (long-form prompt):

| Model | Hard output cap | Sustained rate | Reachable in 200 s |
|---|---|---|---|
| `gemini-3.6-flash` | 65,536 | ~178 tok/s | ~35,000 |
| `gpt-5.6-sol` | none published (accepted 10,000,000) | ~62 tok/s | ~12,000 |
| `grok-4.5` | none published (accepted 10,000,000) | ~49 tok/s | ~10,000 |

At 60,000 both OpenAI and xAI burned the full 200 s and returned nothing. 32,000 sits above every real answer
observed (largest: 14,129 billed tokens), under Gemini's hard limit, and left the slowest provider at 53% of
budget on a full review. Guidance encoded in the error messages: if replies are cut short, raise the cap; if
`timeout` appears instead, the answer doesn't fit the budget — narrow the prompt or lower `reasoning_effort`.

Latency reference points (with `web_search` on, ~60-word review prompt): gemini ~17–19 s (`high`),
gpt ~21 s (`medium`), grok ~44–50 s (`high`); a long open-ended prompt pushes gpt past 80 s.

## 12. Observability

- **All logging to stderr**; stdout is the JSON-RPC transport, exclusively (test-enforced, including a source
  scan for stray `print(`/`sys.stdout`).
- Every tool call carries a 12-hex-char `request_id`, echoed in the result and prefixed (`rid=…`) on every
  related log line, in stable `key=value` grammar:

  ```
  rid=f55cb0b43b57 tool=second_opinion provider=grok model=grok-4.5 outcome=timeout elapsed_ms=200009 budget_s=200.0
  rid=9e2ad9387f41 tool=second_opinion provider=gemini model=gemini-3.6-flash outcome=ok latency_ms=3931 elapsed_ms=3964 input_tokens=412 output_tokens=1031
  ```

  `outcome` ∈ `ok | timeout | error(type=…)`. This keeps the latency distribution measurable straight from the
  Desktop log.
- **Jobs (0.2.0):** the `rid=` grammar is extended, never changed. Every job-related line carries `jid=`;
  submit logs `outcome=submitted jid=… upstream_id=… backing=…`; get/cancel log
  `outcome=running|ok|cancelled|error(type=…) jid=… job_elapsed_ms=…`; the registry logs lifecycle lines
  `jid=… job=terminal status=…` and `jid=… job=evicted`; the poller logs `jid=… upstream_id=… poll=…` on
  retries. `upstream_id` appears in logs only, never in tool results. One `request_id` per tool call, one
  `job_id` per job; a job accumulates many `rid`s over its life.
- Prompt and response **content** is logged only at DEBUG and only when `log_prompts` is enabled (off by
  default — the content may be sensitive).

## 13. Security and privacy posture

- API keys live only in a git-ignored `config.json` or env vars; they are never logged and never appear in tool
  results.
- Only the caller-supplied `summary` (plus focus/system prompt) leaves the machine, to the one chosen vendor.
  No history, no repo access, nothing persisted server-side ("0 conversation data stored" is a public claim on
  the docs page — *by the server*).
- **Background jobs move data custody (0.2.0).** Background execution requires the vendor to store the request
  and response: OpenAI stored responses (sent with `store: true`) persist under the account's retention
  policy; Google interactions are stored by default (55 days paid tier, 1 day free tier) and `store=false` is
  incompatible with background execution. The docs state this plainly and point at the synchronous tool when
  it is unacceptable. Grok-backed jobs are an ordinary API call and store nothing beyond it. Job records hold
  prompt-adjacent metadata in memory (model, effort, `web_search`, `max_tokens`, whether a focus was given) —
  never content — and `log_prompts` continues to gate every content log line.
- No inbound network surface: stdio only, spawned by the MCP host.
- The server trusts its single local user; there is no auth/tenancy of its own (v1 scope).

## 14. Testing architecture

`pytest` + `pytest-asyncio` (strict mode). No network, no real SDK objects in the hot paths.

**Load-bearing convention:** the provider adapters read SDK responses *entirely through `getattr`*, so plain
`SimpleNamespace` fakes (`tests/conftest.py`) stand in for SDK models without importing SDK internals. **New
adapter code should preserve this reflection-friendly style** — it is what keeps the suite hermetic and robust
to SDK class churn. Exception paths, in contrast, use *real* SDK exception classes (`bad_request()` builds a
genuine `openai.BadRequestError`) so the except-chains are exercised exactly as against the live API.

What each module pins:

- `test_config.py` — default-model fallback (the path nobody exercises locally, which is how defaults once
  drifted a model generation behind), placeholder-key handling, env-beats-file precedence, budget resolution
  including the legacy `timeout_seconds` alias and its warnings, headroom under the 240 s cap,
  `default_max_tokens` incl. explicit-null and env-disable, `reasoning_effort` validation/normalisation,
  `web_search` defaults and env parsing.
- `test_request_budget.py` — the structured timeout shape (never an exception), budget-bounded elapsed time,
  cancellation observed by the provider, late results discarded, no leaked tasks, bounded teardown for
  cancellation-ignoring providers (with log line), outer-cancellation propagation to the provider task,
  `default_max_tokens` application incl. explicit `0`, stdout hygiene (three distinct guards), and
  server-survives-timeout. Uses a `StubProvider` driven through the **real FastMCP tool path**
  (`server.call_tool`), with `build_provider` monkeypatched.
- `test_responses_output.py` — the final-answer extraction matrix (narration dropped, adjacent trailing
  messages kept, trailing reasoning ignored, refusal-only trailing message skipped, `output_text` explicitly
  distrusted), refusal → `content_blocked`, reasoning-only output → error not silent empty, request-shape
  guards (params omitted unless set; forwarded when set; Grok base URL), the entire droppable-param retry
  contract (§9.3) including remaining-budget charging and single-retry limit, and terminal statuses
  (`incomplete`/`failed` mappings).
- `test_jobs.py` (0.2.0) — the registry and state machine (transitions, terminal immutability, key index,
  capacity, lazy eviction with `jid=` log line, the design constants and the per-tool headroom under the
  cap); the tool path through the real FastMCP tools with a `BackgroundStub` whose lifecycle the test drives
  and the v0.1 `StubProvider` for local tasks: submit shape and bound, request built like the sync path,
  idempotent `request_key` (one upstream create), submit-side errors and the 30 s submit timeout, `get` with
  `wait_seconds=0` / long-poll early return / clamp warning, the get-deadline-discards-only-the-wait test
  (the inverse of v0.1's), late results surfaced by a later get, **poll abandonment never cancels** (outer
  cancellation of a long-poll, and repeated timed-out polls), explicit cancel, cancel racing terminal,
  cancel-call failure leaves the job running, job-budget exhaustion with upstream cancel and the prescriptive
  message, late upstream completion not resurrected, terminal failures through the taxonomy, retriable vs
  non-retriable poll errors, poll-call timeouts, re-readable results, TTL eviction ⇒ `unknown_job`,
  `job_limit` listing jobs, the local-task path (success, error, budget under grace discipline, the
  cancellation-swallowing task's log line, cancel, abandoned get), the sync tool untouched alongside jobs,
  stdout hygiene over every job path, and the log grammar (`jid=` on every job line, `upstream_id` never in a
  result). Adapter tests use SimpleNamespace fakes with create/retrieve/cancel (OpenAI) and
  create/get/cancel (Gemini): `background`/`store` sent as specified, the droppable-param retry reused on
  submit, terminal payloads mapped through the sync code, transport errors raised with the sync mapping (real
  SDK exception classes), and the refactored sync `generate` pinned unchanged.

Run: `pip install -e .[dev]` then `pytest`.

## 15. Design invariants — extensions MUST preserve these

1. **stdout is the transport.** Nothing but JSON-RPC on stdout, ever. All diagnostics to stderr via the
   configured logger (`propagate=False`).
2. **Errors are results, not exceptions.** Tool handlers always return the `success:false` envelope; the
   taxonomy values (§6.3) are stable API and may only be *added to*, not renamed.
3. **Every tool call clears the client cap** *(rewritten in 0.2.0)*. Each tool call fits its own budget +
   `CANCEL_GRACE_SECONDS` under `CLIENT_HARD_CAP_SECONDS`, per the §10.2 table. No tool call's worst case may
   sit within 30 s of the cap except the legacy sync path, which is grandfathered and documented as such. New
   provider adapters must accept and honour the `timeout` constructor arg (and the `timeout` argument of the
   background control calls); the outer `_run_bounded` is the backstop, not the primary mechanism.
4. **Call deadlines are dead; job lifetimes are alive** *(rewritten in 0.2.0)*. Work completing after a tool
   call's own deadline is never surfaced **by that call**. Work completing within `job_budget_seconds` is
   surfaced by any later `get`, by design — that is the point of a job. Work completing after the job budget
   is dead at the job level (the job is already `failed(timeout)`; a late upstream completion is not
   resurrected). Never trust a coroutine that swallows cancellation (for its result *or* its exit).
5. **`max_output_tokens` is never silently dropped**; only advisory sampling knobs (`temperature`, `top_p`)
   are, at most once, on an explicit upstream rejection, charged against the remaining deadline.
6. **Final answer only.** Extraction must return the trailing run of visible output — never a concatenation
   that glues pre-tool narration onto the answer — and an empty answer is an error, never a silent success.
7. **Single-turn, minimal disclosure.** Only summary/focus/system prompt go upstream (until a history feature
   deliberately changes this); keys and prompt content stay out of logs by default.
8. **Config compatibility.** Deprecated names (`timeout_seconds`, `LLM_SECOND_OPINION_TIMEOUT`) keep working
   with a warning; renames must follow the same pattern (new key wins, clash reported). Placeholder/blank keys
   mean "unavailable", never an error at startup.
9. **Reflection-friendly adapters.** Read SDK response objects via `getattr` so the SimpleNamespace test fakes
   keep working.
10. **Explicit caller values win** over defaults even when falsy (`max_tokens=0` is honoured — `is not None`,
    not truthiness).
11. **Poll abandonment never cancels** *(0.2.0)*. Only `cancel_second_opinion`, job-budget exhaustion, or
    process shutdown (local tasks only) terminate upstream work. No `get` call, timed-out or otherwise, may
    propagate cancellation to a job.
12. **Job ids are server-scoped** *(0.2.0)*. Provider ids never appear as the client-facing job key, and never
    appear in a tool result at all.
13. **Terminal results are immutable and idempotently re-readable** *(0.2.0)* until TTL eviction.

Invariants 1, 2, 5–10 apply to the job code paths unchanged (notably: stdout hygiene, errors-as-results,
getattr-reflection adapters, explicit-caller-values-win).

## 16. Extension points

### 16.1 Adding a provider (the supported recipe)

1. Create `src/llm_second_opinion/providers/<name>.py` subclassing `Provider` (implement `generate`,
   `check_reachable`, `model_id`). If the vendor exposes an OpenAI-compatible Responses API, subclass
   `ResponsesAPIProvider` instead and set only `name` and `base_url` (see `grok.py` — 3 lines of code).
2. Register a default model in `DEFAULT_MODELS` (`config.py`). Config loading, env overrides
   (`LLM_SECOND_OPINION_<NAME>_*`), placeholder handling, `reasoning_effort` and `web_search` plumbing are
   then automatic.
3. Add the target mapping in `TARGET_TO_PROVIDER` and a branch in `build_provider`
   (`providers/__init__.py`) **and** the parallel branch in `_check_all_providers` (`server.py`) — the
   reachability path constructs providers itself. (Unifying these two construction sites would be a welcome
   refactor.)
4. Widen the `target_model: Literal[…]` annotation on `second_opinion` (`server.py`) so MCP advertises the new
   value.
5. Add adapter tests following §14's fake conventions; update README/USAGE tables and, if user-facing,
   `docs/index.html`.

### 16.2 Known-limitation-shaped extensions (declared candidates)

Each v1 boundary in §1 is an acknowledged, intentional gap — USAGE.md explicitly invites extending the server
when one matters. Constraints to respect per candidate:

- **Background jobs (submit/poll)** — **landed in 0.2.0** (§6.4, §10.2, DESIGN-submit-poll.md). Remaining
  follow-ups from that design: durable job persistence (a JSON sidecar, not SQLite — v0.3 candidate; today a
  restart orphans jobs, §6.4 of the design), MCP Tasks extension adoption once a host negotiates it, and the
  grok `provider_stored` upgrade gated on the design's §12.1 experiment.
- **Streaming** — would change the "full reply at once" contract and interact with §10 (partial results at
  deadline are currently defined as dead).
- **Multi-model fan-out in one call** — today "compare two models" is two tool calls composed by Claude;
  a fan-out tool must still fit one 240 s envelope.
- **Conversation history / multi-turn** — both upstream interfaces are stateful-first
  (`previous_response_id` / Interactions sessions) and are deliberately used statelessly; history would relax
  invariant 7 and needs an explicit privacy story.
- **HTTP / remote transport** — invalidates the "no inbound surface, single local user" posture (§13); needs
  auth.
- **Caching, cost tracking, budget enforcement, server-side rate limiting** — additive; usage data needed is
  already normalised in `TokenUsage`.
- **Surfacing `ProviderError.status`** in the error envelope — `to_dict()` already exists, unused (§6.1).
- **Config-file `log_level`** — currently env-only (§5.2); an obvious small addition.

### 16.3 Documentation surfaces to keep in sync

A behavioural change typically lands in five places: code, tests, `README.md` (operator reference),
`USAGE.md` (end-user walkthrough), and `docs/index.html` (public intro page — self-contained HTML with an
animated demo whose `STEPS` script re-enacts the one-call-per-request workflow; its tables mirror the README's
provider/default tables). `config.example.json` mirrors the default values and must track them. Version lives
in **both** `pyproject.toml` and `src/llm_second_opinion/__init__.py`.
