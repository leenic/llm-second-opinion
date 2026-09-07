# llm-second-opinion

**📖 [Introduction &amp; live demo →](https://leenic.github.io/llm-second-opinion/)** — what it is, why it exists, and an animated walkthrough.

A local MCP server that exposes a `second_opinion` tool. When Claude calls it, the server forwards a user-supplied summary to an external LLM (Gemini, Grok, or ChatGPT) and returns that model's independent, critical reply. Designed for single-developer use over stdio.

## What it does

- `second_opinion` — sends a `summary` to one of `gemini`, `grok`, or `chatgpt`, returns the external model's reply along with the actual model identifier, token usage (when reported), and upstream latency. Synchronous: the reply comes back in the same tool call.
- `submit_second_opinion` / `get_second_opinion` / `cancel_second_opinion` — the same review run as a **background job**. `submit` returns a `job_id` within seconds; `get` polls it (optionally waiting up to 45 s); `cancel` stops it. Use these for heavyweight reviews — large documents, web search, high reasoning effort — that cannot finish inside the per-call time cap. See [Background jobs](#background-jobs-submitpoll).
- `list_available_models` — returns which providers have an API key configured and pass a basic reachability check, so Claude can tell you up front which targets are usable.

Each call is single-turn. No conversation history is forwarded to the external model.

### Provider interfaces (as of 25 July 2026)

| Provider | Interface | Default model | SDK |
|---|---|---|---|
| ChatGPT (OpenAI) | Responses API (`client.responses.create`) | `gpt-5.6-sol` | `openai>=2.36` |
| Gemini (Google) | Interactions API (`client.aio.interactions.create`) | `gemini-3.6-flash` | `google-genai>=2.0` |
| Grok (xAI) | Responses API via OpenAI-compatible base URL (`https://api.x.ai/v1`) | `grok-4.5` | `openai>=2.36` |

These are the stateful/agentic-first interfaces each provider now recommends for new integrations. We call them in single-turn mode (no `previous_response_id`, no Interactions session state) because v1 of this server forwards no conversation history.

**Note:** `gpt-5.6-sol` is reasoning-only and rejects `temperature` outright (`400 Unsupported parameter`) rather than ignoring it; `grok-4.5` and `gemini-3.6-flash` both accept it. You don't need to track which is which — if a model rejects an advisory sampling parameter, the Responses provider drops it and retries once, logging a warning. Sampling knobs only (`temperature`, `top_p`); `max_output_tokens` is never dropped, because silently removing a length cap would change cost and truncation. Override any default with `LLM_SECOND_OPINION_<PROVIDER>_MODEL`.

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
| `request_budget_seconds` | Wall-clock bound on one whole synchronous `second_opinion` call (default 200). See [Bounding call duration](#bounding-call-duration) below. Formerly `timeout_seconds`, which still works but warns |
| `job_budget_seconds` | Wall-clock bound on one background job started with `submit_second_opinion` (default 900). A job that overruns is cancelled upstream and fails with `timeout`. Not subject to the per-call cap; values above 3600 warn. See [Background jobs](#background-jobs-submitpoll) |
| `default_max_tokens` | Reply cap applied when the caller passes no `max_tokens` (default 32000). Set to `null` to leave replies unbounded. See [Choosing the reply cap](#choosing-the-reply-cap) — a cap that is too tight fails the call outright rather than returning a shorter answer |
| `log_prompts` | If `true`, prompts and responses are written to the log. Off by default |

#### Bounding call duration

**Claude Desktop enforces a hard, non-configurable 240s cap on tool calls.** When it fires, Desktop cancels with `MCP error -32001: Request timed out` and the result is lost — even though the upstream call has often completed. `request_budget_seconds` exists to get in front of that: at the default 200 the server cancels the call itself and returns a normal tool result the calling model can act on:

```json
{
  "success": false,
  "request_id": "ab12cd34ef56",
  "target_model": "grok",
  "model": "grok-4.5",
  "error": { "type": "timeout", "retriable": true, "message": "..." },
  "elapsed_ms": 200009
}
```

The same value is used for the provider's HTTP client timeout, so the socket is actually torn down rather than left open behind a cancelled coroutine. Keep it comfortably under 240 — the server warns at load if you set it at or above 235, and honours the value anyway in case your MCP client allows longer.

#### Choosing the reply cap

`default_max_tokens` is not a "shorten the answer" knob. A reply that hits the cap comes back `incomplete` and surfaces as an error — **you get nothing, not a shorter answer** — so setting it too tight is worse than setting it too loose.

The real ceiling is `request_budget_seconds`, not the provider API. Measured against the live providers on a long-form prompt:

| Model | Hard output cap | Sustained rate | Reachable in 200s |
|---|---|---|---|
| `gemini-3.6-flash` | 65,536 | ~178 tok/s | ~35,000 |
| `gpt-5.6-sol` | none published | ~62 tok/s | ~12,000 |
| `grok-4.5` | none published | ~49 tok/s | ~10,000 |

OpenAI and xAI accepted `max_output_tokens=10_000_000` without complaint, so nothing stops you setting a huge value — but at 60,000 both of them burned the full 200s budget and returned nothing at all. Past roughly 12k the extra headroom is inert for them; the deadline arrives first.

The 32,000 default sits above every real answer observed (largest: 14,129 billed tokens), under Gemini's hard limit, and left the slowest provider at 53% of budget on a full review. If you still see replies cut short, raise it — but if you start seeing `timeout` instead, the answer genuinely doesn't fit in the budget and the fix is a narrower prompt or a lower `reasoning_effort`, not a bigger cap.

This is a tail-latency guard, not a throughput fix. Measured on a ~60-word review prompt with `web_search` on: `gemini-3.6-flash` ~17–19s (`high`), `gpt-5.6-sol` ~21s (`medium`), `grok-4.5` ~44–50s (`high`). A long, open-ended review prompt pushes `gpt-5.6-sol` past 80s. The OpenAI/Grok client also uses `max_retries=0` so SDK retries can't stack past the budget.

Every call logs `provider`, `outcome` and `elapsed_ms` to stderr, so the latency distribution stays measurable from the Desktop log:

```
rid=f55cb0b43b57 tool=second_opinion provider=grok model=grok-4.5 outcome=timeout elapsed_ms=200009 budget_s=200.0
rid=9e2ad9387f41 tool=second_opinion provider=gemini model=gemini-3.6-flash outcome=ok latency_ms=3931 elapsed_ms=3964 ...
```

All logging goes to stderr. stdout carries the JSON-RPC transport and nothing else.

#### Background jobs (submit/poll)

The per-call budget above cannot help a review that genuinely needs longer than the cap: a full-document review with web search on `gpt-5.6-sol` was measured breaching 200 s deterministically, and a synchronous call that is cancelled at the deadline is billed work with nothing to show for it. The background tools decouple the review from any single tool call:

```
submit_second_opinion  →  starts a job, returns job_id in seconds (bounded at 30 s)
get_second_opinion     →  status check, or a long-poll of up to 45 s; returns the finished envelope when done
cancel_second_opinion  →  cancels the upstream work
```

Every tool call stays short, so a call the client drops costs only the price of re-issuing it, and the **job** carries its own clock, `job_budget_seconds` (default 900). How a job is backed depends on the provider:

| Provider | Backing | Survives a server restart? |
|---|---|---|
| ChatGPT | OpenAI Responses background mode (`background: true`, `store: true`), polled by response id | Upstream yes — but the job is unreachable afterwards, see below |
| Gemini | Interactions API background execution (`background: true`), polled by interaction id | Same |
| Grok | In-process task running the ordinary synchronous call. xAI rejects `background` (`400 Argument not supported`), and a stored response whose connection is dropped mid-generation is never retrievable — measured, see the design doc §12.1 | No — the job dies with the process |

**Restart semantics.** The job registry is process memory only. If the server restarts, `get_second_opinion` returns `unknown_job` for every earlier job: grok jobs died with the process; ChatGPT and Gemini jobs finish upstream, are billed, and are in principle retrievable from the vendor — but the `job_id` → response-id mapping is gone, so this server cannot reach them. Durable mapping is a deliberate non-goal for 0.2.

**Vendor-side storage.** Background execution requires the vendor to store the request and response: OpenAI keeps stored responses under its data-retention policy (the request is sent with `store: true` explicitly, because unstored background responses are kept only about ten minutes); Google stores interactions by default (55 days on the paid tier, 1 day on the free tier) and does not allow `store: false` with background execution. The "nothing stored server-side" claim still holds for this server, but for a background job the chosen vendor stores the content under its retention policy. Use the synchronous tool if that is unacceptable. Grok-backed jobs are an ordinary API call and store nothing beyond that.

**Cancellation is explicit.** Abandoning, timing out, or never issuing a `get` call never cancels a job. Only `cancel_second_opinion`, job-budget exhaustion, or (for grok) process shutdown stops upstream work.

**Capacity and retention.** At most 8 jobs may be running at once (`job_limit` otherwise, listing the active ids). Finished jobs — succeeded, failed or cancelled — stay readable for 30 minutes, then are evicted; every `get` of a finished job returns the same envelope.

Jobs log with a `jid=` in addition to the per-call `rid=`, so a job's history greps out of the Desktop log:

```
rid=1a2b3c4d5e6f tool=submit_second_opinion outcome=submitted jid=9f8e7d6c5b4a upstream_id=resp_… backing=provider_background provider=openai model=gpt-5.6-sol elapsed_ms=812
rid=0f1e2d3c4b5a tool=get_second_opinion jid=9f8e7d6c5b4a outcome=running job_elapsed_ms=45210 elapsed_ms=45003 wait_s=45
jid=9f8e7d6c5b4a job=terminal status=succeeded provider=openai model=gpt-5.6-sol backing=provider_background job_elapsed_ms=387554
rid=aa11bb22cc33 tool=get_second_opinion jid=9f8e7d6c5b4a outcome=ok status=succeeded job_elapsed_ms=387554 elapsed_ms=1
```

Provider response ids appear in the log as `upstream_id=` and never in a tool result.

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
| `LLM_SECOND_OPINION_REQUEST_BUDGET` | Override `request_budget_seconds` (the legacy `LLM_SECOND_OPINION_TIMEOUT` still works) |
| `LLM_SECOND_OPINION_JOB_BUDGET` | Override `job_budget_seconds` |
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
  "model": "gemini-3.6-flash",
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

Error `type` is one of: `missing_api_key`, `auth_failed`, `rate_limit`, `timeout`, `network_error`, `upstream_error`, `bad_request`, `content_blocked`, `invalid_input`, `internal_error`, and for the job tools `unknown_job`, `job_limit`.

### `submit_second_opinion`

Same arguments as `second_opinion`, plus:

| Arg | Type | Required | Notes |
|---|---|---|---|
| `request_key` | string | no | Client-chosen idempotency key. A second submit with the same key returns the existing job (running or finished) instead of creating a new one — pass one so a retry after a lost reply cannot start a second, separately billed review |

Returns as soon as the upstream acknowledges the job (bounded at 30 s):

```json
{
  "success": true,
  "request_id": "ab12cd34ef56",
  "job_id": "9f8e7d6c5b4a",
  "target_model": "chatgpt",
  "provider": "openai",
  "model": "gpt-5.6-sol",
  "backing": "provider_background",
  "status": "running",
  "job_budget_seconds": 900,
  "poll_after_ms": 5000
}
```

`backing` is `provider_background` (ChatGPT, Gemini) or `local_task` (Grok). A deduplicated submit adds `"reused_existing_job": true`. Errors use the same envelope as `second_opinion`; `job_limit` (retriable) means 8 jobs are already running and lists them.

### `get_second_opinion`

| Arg | Type | Required | Notes |
|---|---|---|---|
| `job_id` | string | yes | From `submit_second_opinion`. Unknown or expired ⇒ `unknown_job` (not retriable) |
| `wait_seconds` | number | no | Default 0: return the current status at once. Up to 45: wait that long for the job to finish first. Larger values are clamped, not rejected |

While running:

```json
{
  "success": true,
  "request_id": "…",
  "job_id": "9f8e7d6c5b4a",
  "status": "running",
  "target_model": "chatgpt", "provider": "openai", "model": "gpt-5.6-sol", "backing": "provider_background",
  "job_elapsed_ms": 123456,
  "retry_after_ms": 15000,
  "elapsed_ms": 45003
}
```

`retry_after_ms` escalates with job age (5 s → 15 s → 30 s). When finished, the result carries `status` of `succeeded`, `failed` or `cancelled` and embeds the existing envelope unchanged: a succeeded job carries `response`, `usage`, `latency_ms` (measured across the whole job) and `model` as reported upstream; a failed job carries the usual `error` object (a job that overran `job_budget_seconds` fails with type `timeout`); a cancelled job carries `cancelled_by` and `cancelled_at`. Finished results can be re-read for 30 minutes.

### `cancel_second_opinion`

| Arg | Type | Required | Notes |
|---|---|---|---|
| `job_id` | string | yes | As above |

Cancels the upstream work and returns the job's cancelled state. Cancelling a job that already finished returns its finished result, not an error.

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
- Background jobs are not persisted: a server restart orphans them (see [Background jobs](#background-jobs-submitpoll)). ChatGPT/Gemini jobs still complete and are billed upstream but cannot be retrieved through this server afterwards.
- No caching, cost tracking, or budget enforcement beyond the per-call and per-job time budgets.
- No rate limiting on the server side — we rely on the upstream provider's limits.
- Stdio transport only. There is no HTTP or remote-access mode in v1.
- The "basic reachability check" used by `list_available_models` is a low-cost GET against each provider's models endpoint. It catches missing keys and outages but does not guarantee the configured model name is valid for that account.
- Prompts and responses are **not** logged by default. Set `log_prompts: true` (or `LLM_SECOND_OPINION_LOG_PROMPTS=true`) to enable for debugging. Be careful — they may contain sensitive content.
- `google-genai` currently emits `UserWarning: Interactions usage is experimental and may change in future versions.` at first client construction. This is expected — the Interactions API was promoted to the recommended interface in May 2026 and the SDK still flags it. If the surface changes, update [providers/gemini.py](src/llm_second_opinion/providers/gemini.py).
