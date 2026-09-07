# Using llm-second-opinion with Claude

This guide is for **the person sitting in front of Claude** (Claude Code or Claude Desktop) after the MCP server has been installed and registered. If you haven't set up the server yet, see the [README](README.md) first.

## What this gives you in your Claude session

A new tool that Claude can call: **`second_opinion`**. When you trigger it, Claude pauses, ships a summary you (or it) wrote to an external LLM — ChatGPT, Gemini, or Grok — and brings back that model's reply. Use it when you want a perspective that isn't Claude's, on something Claude has been helping with.

For reviews that take minutes rather than seconds there is a **background** variant: Claude submits the review, tells you it's in progress, and fetches the result when it's ready. See [Long reviews](#long-reviews-ask-for-a-background-review) below.

Also available: **`list_available_models`**, which tells you which of the three external models are currently usable (have keys configured, pass a reachability check).

## When to reach for it

Good fits:

- **Sanity check on a plan**: "Claude, get a second opinion from Gemini on the migration approach we just wrote up."
- **Disagree-with-me mode**: "Ask ChatGPT to argue against this design — focus on the failure modes."
- **Cross-check facts**: "Ask Grok with web search on whether xyz library deprecated this API in 2026."
- **Style/quality review**: "Get Gemini to critique the tone of this PR description."
- **Tiebreaker**: when you and Claude have gone back and forth and you're not sure who's right.

Bad fits:

- Just chatting — the tool is single-turn, no memory between calls.
- Anything that needs the external model to read your repo. It only sees what you put in `summary`.
- Streaming or real-time work. The whole reply comes back in one shot.

## Triggering it — example prompts you can give Claude

You don't call the tool yourself; you ask Claude in plain language and Claude decides to call it. Phrases that reliably trigger it:

- "Get a second opinion from Gemini on …"
- "Run that past ChatGPT and tell me what they say."
- "Use the second_opinion tool with target_model=grok to review …"
- "What does Gemini think of this plan?" (Claude will usually pick the tool here)

If Claude misses the cue and just answers itself, be explicit: *"Use the second_opinion tool."*

If you don't know which models are wired up, just ask: *"Which second-opinion models are available?"* Claude will call `list_available_models`.

## Long reviews: ask for a background review

A single tool call is capped at 240 seconds by Claude Desktop, and the server stops a synchronous call at 200 seconds so you get a proper error instead of a dropped one. A heavyweight review — a whole design document, web search turned on, `reasoning_effort` high — can genuinely need longer than that, and when a synchronous call is cut off the external model's work is billed and lost.

For those, ask for the review **in the background**:

- "Submit a background second opinion from ChatGPT on the full spec, then check on it."
- "Run this past Gemini as a background job — it's long — and let me know when it's done."

What happens: Claude calls `submit_second_opinion` and gets a `job_id` back within seconds. It then calls `get_second_opinion`, which waits up to 45 seconds for the job to finish. If the review is still running, Claude should tell you so and check again later — after the interval the server suggests, or whenever you ask *"is the review back yet?"*. When it finishes, the result is exactly what the synchronous tool would have returned. A job may run for up to 15 minutes (`job_budget_seconds`) before the server gives up on it. A finished result stays fetchable for 30 minutes, so a dropped reply just means asking again.

You can also ask Claude to **cancel** a running review (`cancel_second_opinion`). Nothing else cancels one: Claude losing track of it, or a poll being cut off, never stops the upstream work — that is the point of a job.

Things to know:

- **Probe, then fire.** After the Claude session has sat idle for a while, the connection to the server can go stale and the *first* tool call after the gap is sometimes lost — it never reaches the server and produces a four-minute client error. Cheap habit: ask *"which second-opinion models are available?"* first. That call is instant and re-warms the connection; then ask for the review. If a submit does seem to vanish, just ask Claude to submit again — it passes a `request_key`, so a retry that turns out to be a duplicate returns the same job instead of starting a second paid one.
- **The vendor stores the job.** Background execution requires OpenAI and Google to keep the request and the reply on their side under their retention policies (OpenAI's standard policy for stored responses; Google: 55 days on the paid tier, 1 day on the free tier). The server itself still stores nothing. If that is unacceptable for the content in question, ask for the ordinary synchronous review instead. Grok reviews run in-process — xAI has no background mode — so they are an ordinary API call and store nothing beyond it.
- **Restarting the server loses running jobs.** The job list lives in the server's memory. If Claude Desktop or Claude Code restarts the MCP server, a Grok job is gone outright; ChatGPT and Gemini jobs finish upstream (and are billed) but the server can no longer find them, and asking about them returns `unknown_job`. Submit again.
- **At most eight jobs run at once.** A ninth submit is refused (`job_limit`) until one finishes or is cancelled.

## Picking which external model

There's no universal best. Rough heuristics:

| Situation | Try first |
|---|---|
| Long codebase summaries, careful reasoning, multimodal | **Gemini** (`gemini-3.6-flash`) — strong long-context, and the fastest of the three (~17–19s on a typical review). |
| Deep tool-use thinking, careful refactoring critique | **ChatGPT** (`gpt-5.6-sol`) — current OpenAI flagship, deepest tool-aware reasoning. Reasoning-only: it rejects `temperature`. |
| Want a sharper, more contrarian take; also good price/performance | **Grok** (`grok-4.5`) — searches the web most aggressively of the three, and is correspondingly the slowest (~44–50s). |
| Need fresh facts (post-cutoff news, library docs, current pricing) | Whichever has `web_search: true` in your config — see below. |

When in doubt, ask the same question of two of them and compare. Claude is happy to do that in one turn.

## Optional arguments

When you ask Claude to call the tool, you can shape the call by mentioning these in your message — Claude will pass them through:

- **`focus`** — narrow the reviewer's attention. *"…focus on whether the rollback plan is realistic."*
- **`system_prompt`** — replace the default reviewer persona. The default tells the external model to be direct, critical, and skip the praise. Override it only when you want a different kind of feedback (e.g., *"…use system_prompt: 'you are a hostile pentest reviewer'"*).
- **`temperature`** — pass a number if you want it more deterministic (0–0.3) or more creative (0.8+). Most flagships ignore this for reasoning tracks anyway, and `gpt-5.6-sol` rejects it outright — but you can pass it to any model regardless: if the model refuses it, the server drops it and retries automatically, so you get an answer rather than an error. The trade-off is that the reply then uses the model's own default sampling, and the call takes one extra round-trip.
- **`max_tokens`** — cap the length of the reply. Useful when you only want a quick verdict, but note that hitting the cap fails the call rather than returning a shorter answer, so don't set it tight to "save tokens". If you don't pass one, `default_max_tokens` from the config applies (32000 by default), which is well clear of a full-length review.

You don't need to remember the arg names — say what you want and Claude will map it.

## Two settings that live in `config.json`, not in the call

These are per-provider, set once, and apply to every call until you change them:

- **`reasoning_effort`** (`minimal` | `low` | `medium` | `high`) — how hard the external model thinks before answering. Higher = slower and more expensive but usually better. If you didn't set it, the model uses its own default.
- **`web_search`** (`true` | `false`) — whether the external model is allowed to hit the live web. Off by default. Turn it on for the model you want to use for fact-checking; leave it off otherwise to keep responses fast and bounded to model knowledge.

To change either of these you edit `config.json` and restart the MCP server (Claude Code or Claude Desktop). You can also override at launch time with env vars like `LLM_SECOND_OPINION_OPENAI_REASONING_EFFORT=high` — see the README.

If you want to check the current settings without opening the file: *"Which models are available and what are they set to?"* — Claude will call `list_available_models` and the response includes each provider's `reasoning_effort` and `web_search`.

## What you get back

A successful call returns something like:

```json
{
  "success": true,
  "provider": "gemini",
  "model": "gemini-3.6-flash",
  "response": "The plan has two problems...",
  "usage": { "input_tokens": 412, "output_tokens": 1031, "total_tokens": 1443 },
  "latency_ms": 4820
}
```

Claude will surface the `response` text to you and usually mention which model said it. If you want the metadata too, ask: *"What model and how long?"*

On failure (most often a missing key, rate limit, or content filter):

```json
{
  "success": false,
  "error": {
    "type": "missing_api_key",
    "message": "No API key configured for provider 'grok'...",
    "retriable": false
  }
}
```

Common `error.type` values you might see:

| Type | What to do |
|---|---|
| `missing_api_key` | Add a key for that provider to `config.json` and restart the MCP host. |
| `auth_failed` | The key is set but rejected — rotate it. |
| `rate_limit` | Wait a bit, or switch to a different `target_model`. |
| `timeout` | A synchronous call hit `request_budget_seconds` (200s by default) and was cancelled. The reply carries `elapsed_ms` so you can see how close it got. Ask for the review **in the background** instead (see above), or retry with a lower `reasoning_effort` / smaller `max_tokens`. For a background job, `timeout` means it overran `job_budget_seconds` (15 minutes by default) and was cancelled upstream — narrow the prompt, or raise the job budget. |
| `unknown_job` | Claude asked about a job the server doesn't know: a mistyped id, a result older than 30 minutes, or the server restarted since the job was submitted. Submit again. |
| `job_limit` | Eight background jobs are already running. Wait for one to finish, or ask Claude to cancel one. |
| `content_blocked` | The provider's safety filter rejected the prompt or response. Try a different model or rephrase. |
| `bad_request` | Usually a model name typo, or a parameter the model didn't accept. `temperature` no longer lands here — it's dropped and retried automatically. |

## Tips that pay off

- **Give the external model context, not your whole chat history.** The tool forwards only what you put in `summary`. Ask Claude to *"write a self-contained one-page summary"* before sending — that's what makes the second opinion actually useful.
- **State what you want from the reviewer.** "Are there cases I'm missing?" gets sharper feedback than "What do you think?"
- **Don't ask all three at once unless you genuinely want three views.** Each call costs tokens and time. Pick one, and only fan out if the first answer is suspicious.
- **The reviewer doesn't see prior turns.** If your question relies on something Claude established earlier, paste it in or ask Claude to inline it.
- **For factual questions, turn on `web_search` for at least one provider** — otherwise everything you get back is bounded by that model's training cutoff.

## Things this server intentionally doesn't do

- No conversation — every call is one-shot.
- No streaming — the reply arrives in one block (a background job's reply too, once it's done).
- No cost/budget tracking — watch your provider dashboards. Background jobs are bounded in time (15 minutes each, eight at once) but not in spend.
- No memory of background jobs across a server restart.
- No comparing multiple models in one call — ask Claude to call the tool twice.
- No memory between calls — re-supply the context every time.

If any of these matter for your workflow, mention it and we can extend the server.
