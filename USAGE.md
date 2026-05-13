# Using llm-second-opinion with Claude

This guide is for **the person sitting in front of Claude** (Claude Code or Claude Desktop) after the MCP server has been installed and registered. If you haven't set up the server yet, see the [README](README.md) first.

## What this gives you in your Claude session

A new tool that Claude can call: **`second_opinion`**. When you trigger it, Claude pauses, ships a summary you (or it) wrote to an external LLM — ChatGPT, Gemini, or Grok — and brings back that model's reply. Use it when you want a perspective that isn't Claude's, on something Claude has been helping with.

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

## Picking which external model

There's no universal best. Rough heuristics:

| Situation | Try first |
|---|---|
| Long codebase summaries, careful reasoning, multimodal | **Gemini** (`gemini-3.1-pro-preview`) — strong long-context and reasoning. |
| Deep tool-use thinking, careful refactoring critique | **ChatGPT** (`gpt-5.5`) — current OpenAI flagship, deepest tool-aware reasoning. |
| Want a sharper, more contrarian take; also good price/performance | **Grok** (`grok-4.3`). |
| Need fresh facts (post-cutoff news, library docs, current pricing) | Whichever has `web_search: true` in your config — see below. |

When in doubt, ask the same question of two of them and compare. Claude is happy to do that in one turn.

## Optional arguments

When you ask Claude to call the tool, you can shape the call by mentioning these in your message — Claude will pass them through:

- **`focus`** — narrow the reviewer's attention. *"…focus on whether the rollback plan is realistic."*
- **`system_prompt`** — replace the default reviewer persona. The default tells the external model to be direct, critical, and skip the praise. Override it only when you want a different kind of feedback (e.g., *"…use system_prompt: 'you are a hostile pentest reviewer'"*).
- **`temperature`** — pass a number if you want it more deterministic (0–0.3) or more creative (0.8+). Most flagships ignore this for reasoning tracks anyway.
- **`max_tokens`** — cap the length of the reply. Useful when you only want a quick verdict.

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
  "model": "gemini-3.1-pro-preview-2026-05",
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
| `timeout` | Network or upstream slow — retry, or raise `timeout_seconds`. |
| `content_blocked` | The provider's safety filter rejected the prompt or response. Try a different model or rephrase. |
| `bad_request` | Usually a model name typo or a parameter the model didn't accept (e.g. `temperature` on a reasoning-only model). |

## Tips that pay off

- **Give the external model context, not your whole chat history.** The tool forwards only what you put in `summary`. Ask Claude to *"write a self-contained one-page summary"* before sending — that's what makes the second opinion actually useful.
- **State what you want from the reviewer.** "Are there cases I'm missing?" gets sharper feedback than "What do you think?"
- **Don't ask all three at once unless you genuinely want three views.** Each call costs tokens and time. Pick one, and only fan out if the first answer is suspicious.
- **The reviewer doesn't see prior turns.** If your question relies on something Claude established earlier, paste it in or ask Claude to inline it.
- **For factual questions, turn on `web_search` for at least one provider** — otherwise everything you get back is bounded by that model's training cutoff.

## Things this server intentionally doesn't do

- No conversation — every call is one-shot.
- No streaming — the reply arrives in one block.
- No cost/budget tracking — watch your provider dashboards.
- No comparing multiple models in one call — ask Claude to call the tool twice.
- No memory between calls — re-supply the context every time.

If any of these matter for your workflow, mention it and we can extend the server.
