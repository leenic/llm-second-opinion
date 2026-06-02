# llm-second-opinion

A local MCP server that exposes a `second_opinion` tool. When Claude calls it, the server forwards a user-supplied summary to an external LLM (Gemini, Grok, or ChatGPT) and returns that model's independent, critical reply. Designed for single-developer use over stdio.

## What it does

- `second_opinion` — sends a `summary` to one of `gemini`, `grok`, or `chatgpt`, returns the external model's reply along with the actual model identifier, token usage (when reported), and upstream latency.
- `list_available_models` — returns which providers have an API key configured and pass a basic reachability check, so Claude can tell you up front which targets are usable.

Each call is single-turn. No conversation history is forwarded to the external model.

### Provider interfaces (as of 13 May 2026)

| Provider | Interface | Default model | SDK |
|---|---|---|---|
| ChatGPT (OpenAI) | Responses API (`client.responses.create`) | `gpt-5.5` | `openai>=2.36` |
| Gemini (Google) | Interactions API (`client.aio.interactions.create`) | `gemini-3.1-pro-preview` | `google-genai>=1.55` |
| Grok (xAI) | Responses API via OpenAI-compatible base URL (`https://api.x.ai/v1`) | `grok-4.3` | `openai>=2.36` |

These are the stateful/agentic-first interfaces each provider now recommends for new integrations. We call them in single-turn mode (no `previous_response_id`, no Interactions session state) because v1 of this server forwards no conversation history.

**Note:** xAI is retiring Grok-3 and several early Grok-4 variants on **15 May 2026**. The default `grok-4.3` is the current flagship — override with `LLM_SECOND_OPINION_GROK_MODEL` if you need a different variant.

## Install

Requires Python 3.10+.

```powershell
# from the project root
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Configure

Copy the example and fill in the API keys you actually have:

```powershell
copy config.example.json config.json
notepad config.json
```

The server looks for `config.json` in this order:

1. `$LLM_SECOND_OPINION_CONFIG` (if set)
2. `./config.json` (current working directory)
3. `%APPDATA%\llm-second-opinion\config.json` (Windows)
4. `~/.config/llm-second-opinion/config.json` (Linux/macOS)

A provider with no key (or with the `REPLACE-ME` placeholder) is treated as unavailable — the server still starts.

### Config file fields

| Field | Purpose |
|---|---|
| `providers.openai.api_key` | OpenAI API key (for `target_model: chatgpt`) |
| `providers.gemini.api_key` | Google AI Studio key (for `target_model: gemini`) |
| `providers.grok.api_key` | xAI API key (for `target_model: grok`) |
| `providers.<name>.model` | Optional model name override for that provider |
| `providers.<name>.reasoning_effort` | Optional. One of `minimal`, `low`, `medium`, `high`. Omit to use the provider's default thinking depth |
| `providers.<name>.web_search` | Optional `true`/`false`. Attaches the provider's built-in web search tool to every call. Default `false` |
| `timeout_seconds` | Per-request timeout to the upstream LLM (default 180). Reasoning-heavy flagships (gpt-5.5, gemini-3.1-pro, grok-4.3) take 30s–170s+ end-to-end via the Responses/Interactions APIs because there is no streaming — `high` reasoning + `web_search` on gpt-5.5 has been measured at ~170s. **MCP clients (e.g. Claude Desktop) cancel a tool call at ~240s regardless**, so keep this comfortably below that (≈200–210). Lower it if you want failures to surface faster; raise it (up to ~210) if you enable `web_search` on a flagship and see timeouts. The OpenAI/Grok client uses `max_retries=0` so this value bounds total wall-clock time — without that, SDK retries stack past the client's 240s cap and the call hangs with no result |
| `log_prompts` | If `true`, prompts and responses are written to the log. Off by default |

#### How `reasoning_effort` is applied per provider

| Provider | Where it goes |
|---|---|
| OpenAI / Grok | `reasoning={"effort": "<value>"}` on the Responses API call |
| Gemini | `generation_config.thinking_level = "<value>"` on the Interactions API call (Gemini accepts the same `minimal`/`low`/`medium`/`high` enum) |

If omitted, the SDK's own default applies — for all three current flagships that means reasoning is **on** at a provider-chosen depth.

#### How `web_search` is applied per provider

| Provider | Tool sent |
|---|---|
| OpenAI | `tools=[{"type": "web_search"}]` |
| Gemini | `tools=[{"type": "google_search"}]` on the Interactions API call |
| Grok | `tools=[{"type": "web_search"}]` via the OpenAI-compatible Responses layer. If xAI rejects this shape on your account, set `web_search: false` for `grok` and use Gemini or OpenAI for queries that need fresh facts |

Enabling web search adds latency and may add cost depending on the provider's billing.

### Environment variable overrides

All env vars are prefixed `LLM_SECOND_OPINION_`. They take precedence over the config file.

| Env var | Effect |
|---|---|
| `LLM_SECOND_OPINION_CONFIG` | Path to a config file |
| `LLM_SECOND_OPINION_OPENAI_API_KEY` | Override OpenAI key from the config file |
| `LLM_SECOND_OPINION_GEMINI_API_KEY` | Override Gemini key from the config file |
| `LLM_SECOND_OPINION_GROK_API_KEY` | Override Grok key from the config file |
| `LLM_SECOND_OPINION_OPENAI_MODEL` | Override the default OpenAI model |
| `LLM_SECOND_OPINION_GEMINI_MODEL` | Override the default Gemini model |
| `LLM_SECOND_OPINION_GROK_MODEL` | Override the default Grok model |
| `LLM_SECOND_OPINION_<PROVIDER>_REASONING_EFFORT` | Override `reasoning_effort` for that provider |
| `LLM_SECOND_OPINION_<PROVIDER>_WEB_SEARCH` | `true`/`false` — override `web_search` for that provider |
| `LLM_SECOND_OPINION_TIMEOUT` | Per-request timeout, seconds |
| `LLM_SECOND_OPINION_LOG_PROMPTS` | `true`/`false` — log prompt and response content |
| `LLM_SECOND_OPINION_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Default model names live in [src/llm_second_opinion/config.py](src/llm_second_opinion/config.py#L17-L22). The defaults target each provider's current flagship — if a new flagship ships, set the matching `*_MODEL` env var to use it without touching code.

## Run

```powershell
llm-second-opinion
# equivalent to
python -m llm_second_opinion
```

The server speaks MCP over stdio. All logs go to stderr; stdout carries the JSON-RPC protocol.

## Hook it up to Claude

### Claude Code

Add an entry to your MCP config:

```json
{
  "mcpServers": {
    "llm-second-opinion": {
      "command": "llm-second-opinion",
      "args": [],
      "env": {
        "LLM_SECOND_OPINION_CONFIG": "C:\\src\\llm-second-opinion\\config.json"
      }
    }
  }
}
```

Or use `claude mcp add` from a shell.

### Claude Desktop

Edit `%APPDATA%\Claude\claude_desktop_config.json` and add the same `mcpServers` entry. Restart Claude Desktop.

Then ask Claude something like: *"Use second_opinion with target_model=gemini to review the plan we just sketched."*

For a user-facing walkthrough of how to drive the tool from a Claude conversation — example prompts, picking which model, reading the results — see [USAGE.md](USAGE.md).

## Tool reference

### `second_opinion`

| Arg | Type | Required | Notes |
|---|---|---|---|
| `summary` | string | yes | The content to review |
| `target_model` | enum `gemini` \| `grok` \| `chatgpt` | yes | Which external LLM to call |
| `focus` | string | no | Aspect to emphasise in the review |
| `system_prompt` | string | no | Overrides the default reviewer prompt |
| `temperature` | number | no | Sampling temperature, passed through |
| `max_tokens` | int | no | Output cap, passed through |

Successful response:

```json
{
  "success": true,
  "request_id": "ab12cd34ef56",
  "target_model": "gemini",
  "provider": "gemini",
  "model": "gemini-3.1-pro-preview-2026-05",
  "response": "...",
  "usage": { "input_tokens": 123, "output_tokens": 456, "total_tokens": 579 },
  "latency_ms": 1840
}
```

Error response (non-crashing — the tool returns `success: false`):

```json
{
  "success": false,
  "request_id": "ab12cd34ef56",
  "target_model": "grok",
  "error": {
    "type": "missing_api_key",
    "message": "No API key configured for provider 'grok'. ...",
    "retriable": false
  }
}
```

Error `type` is one of: `missing_api_key`, `auth_failed`, `rate_limit`, `timeout`, `network_error`, `upstream_error`, `bad_request`, `content_blocked`, `invalid_input`, `internal_error`.

### `list_available_models`

No arguments. Returns the per-provider configuration + reachability state and the list of usable `target_model` values. Use this when the user asks "what's set up?" or when a `second_opinion` call fails with `missing_api_key`.

### Default reviewer prompt

When `system_prompt` is not provided, the external LLM receives:

> You are acting as an external reviewer for a conversation the user is having with another AI assistant. The user wants your independent view on the summary below. Be direct, concrete, and critical. If you disagree with the framing or see a stronger alternative, say so explicitly. Do not pad with praise. If a focus is provided, prioritise commenting on that aspect. State your confidence level when making factual claims.

## Adding a new provider

1. Create `src/llm_second_opinion/providers/<name>.py` with a class that subclasses `Provider` from [providers/base.py](src/llm_second_opinion/providers/base.py). Implement `generate`, `check_reachable`, and `model_id`. If the new provider exposes an OpenAI-compatible Responses API, subclass `ResponsesAPIProvider` from [providers/openai_provider.py](src/llm_second_opinion/providers/openai_provider.py) instead — just set `name` and `base_url`.
2. Register a default model in `DEFAULT_MODELS` in [config.py](src/llm_second_opinion/config.py).
3. Add an entry to `TARGET_TO_PROVIDER` and a branch in `build_provider` in [providers/__init__.py](src/llm_second_opinion/providers/__init__.py), and in `_check_all_providers` in [server.py](src/llm_second_opinion/server.py).
4. Widen the `target_model: Literal[...]` annotation on `second_opinion` in [server.py](src/llm_second_opinion/server.py) so MCP advertises the new value.

That's it — config loading, env-var overrides, and reachability picks the new provider up automatically.

## Known limitations

- Single-turn only. No conversation history is forwarded to the external model.
- No response streaming. The full reply arrives at once.
- No caching, cost tracking, or budget enforcement.
- No rate limiting on the server side — we rely on the upstream provider's limits.
- Stdio transport only. There is no HTTP or remote-access mode in v1.
- The "basic reachability check" used by `list_available_models` is a low-cost GET against each provider's models endpoint. It catches missing keys and outages but does not guarantee the configured model name is valid for that account.
- Prompts and responses are **not** logged by default. Set `log_prompts: true` (or `LLM_SECOND_OPINION_LOG_PROMPTS=true`) to enable for debugging. Be careful — they may contain sensitive content.
- `google-genai` currently emits `UserWarning: Interactions usage is experimental and may change in future versions.` at first client construction. This is expected — the Interactions API was promoted to the recommended interface in May 2026 and the SDK still flags it. If the surface changes, update [providers/gemini.py](src/llm_second_opinion/providers/gemini.py).
