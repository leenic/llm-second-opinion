# llm-second-opinion — Submit/Poll Extension Design

**Version target:** 0.2.0 · **Status:** implemented 2026-09-07 (§12.1 experiment and §12.2 acceptance pending — see those sections) · **Companion to:** SPEC.md v0.2.0
**Audience:** the implementer (human or Claude Code session) and future contributors

This document specifies the background-job ("submit/poll") extension. It is written to the same standard as
SPEC.md: precise enough to implement from, with invariants stated explicitly. §1 records the empirical evidence
that motivates the design — read it before questioning any decision below, because most decisions trace to a
specific observed failure.

---

## 1. Motivation — the empirical record (Aug 21 – Sep 5, 2026)

The v0.1 timing model (SPEC §10) assumes the dominant risk is a slow upstream call breaching the 200 s budget
inside a healthy transport. Live diagnosis falsified half of that and confirmed the other half:

| # | Finding | Evidence |
|---|---|---|
| 1 | **Signature A (structured server timeout) is real but workload-specific.** Document review — large prompt ingestion × web-search loop × long multi-part output — breaches 200 s deterministically on `gpt-5.6-sol` at `medium` effort. | `rid=0533d5147482`, `outcome=timeout`, `elapsed_ms=200015` — the **first** timeout outcome in the server's entire log history. Same model/config on a 600-word prompt: 161 s (`rid=45fd94285052`, 118,501 input tokens from search ingestion). |
| 2 | **Signature B (client-side four-minute hang) can be pure dispatch loss.** A call can die with *zero* traces in either the bridge log or the server log — never dispatched, no upstream cost. Trigger: a stale relay binding after an idle gap on the claude.ai → Desktop bridge path. | Aug 30 grok call: 4-minute client error, no `tools/call`, no `rid`, in either log. Identical call re-fired seconds after a successful liveness probe: `rid=4287d8eb2c41`, `outcome=ok`, 56.7 s. |
| 3 | **The three-layer budget machinery works end-to-end in production**, not only under test: finding 1's envelope crossed the relay with ~40 s of headroom. | Same `rid=0533d5147482` call. |
| 4 | **A synchronous timeout burns unrecoverable money.** On a synchronous create, the upstream response id arrives only *with* the response — a 200 s cancelled call is billed work with no handle to retrieve it by. | Finding 1 ingested the full spec, searched, and generated for 200 s; all lost. |
| 5 | **Cost concentrates in search ingestion, not generation.** Latency does not scale with it. | grok: 390,151 input tokens in 56.7 s; gpt: 118,501 in 161 s. |
| 6 | **Provider async support is asymmetric.** OpenAI Responses and Google Interactions both offer real background execution (submit → id → poll). xAI accepts `background` but documents it as "Not used at the moment. Just for OpenResponses compatibility." | Verified against primary docs (platform.openai.com, ai.google.dev, docs.x.ai) during the diagnostic sessions. |

Consequences the design must answer to: **(a)** heavy workloads need to escape the per-call envelope entirely,
not gain 30 s of margin; **(b)** every individual tool call must be short, because the relay can eat a call and
retrying must be cheap; **(c)** upstream work must survive the client, the relay, and ideally the server
process; **(d)** Grok requires a different mechanism from the other two.

## 2. Design overview

Three new tools, two existing tools unchanged:

```
submit_second_opinion  ──▶  starts a background job, returns job_id in ≲ seconds
get_second_opinion     ──▶  bounded poll (optional short long-poll), returns status or the finished envelope
cancel_second_opinion  ──▶  explicitly cancels the upstream job

second_opinion         ──▶  unchanged synchronous path (fast calls, existing contracts intact)
list_available_models  ──▶  unchanged; doubles as the liveness probe in the probe-then-fire protocol
```

One **job** = one upstream generation, identified by a server-generated `job_id`, living independently of any
tool call. Tool calls become uniformly short; the job carries its own (much larger) lifetime budget. Losing a
tool call to the relay loses nothing but the price of re-issuing it.

Per-provider job backing:

| Provider | Mechanism | Job survives server restart? |
|---|---|---|
| openai | Responses API background mode: `background: true`, `store: true`; poll `GET /v1/responses/{id}` | Upstream yes; the local job_id→response_id mapping does not (v0.2 accepts orphaning, §6.4) |
| gemini | Interactions API background execution: `background: true`; poll interaction by id | Same as openai |
| grok | **In-process** `asyncio.Task` running the existing synchronous `generate()` path, registered locally | No — job dies with the process (documented; upgrade gate in §12.1) |

## 3. Tool contracts

### 3.1 `submit_second_opinion`

Arguments: identical to `second_opinion` (`summary`, `target_model`, `focus`, `system_prompt`, `temperature`,
`max_tokens` — same validation, same defaults) plus:

| Arg | Type | Semantics |
|---|---|---|
| `request_key` | `str \| None` | Client-chosen idempotency key. If a live or terminal job with the same key exists, that job is returned instead of creating a new one. Guards the ambiguous-submission race: submit dispatched, upstream accepted, relay lost the reply, Claude retries. Motivated directly by finding 2 — retries are *expected* on this transport. |

Success result (returns as soon as the upstream submission is acknowledged):

```json
{
  "success": true,
  "request_id": "…",            // this tool call
  "job_id": "…",                // the job — 12 hex chars, own namespace, NEVER a provider id
  "status": "running",
  "target_model": "chatgpt",
  "provider": "openai",
  "model": "gpt-5.6-sol",
  "backing": "provider_background",   // or "local_task" (grok)
  "job_budget_seconds": 900,
  "poll_after_ms": 5000
}
```

Submission itself is bounded by the standard three-layer model with `SUBMIT_BUDGET_SECONDS = 30` (§7): a
background create is a fast call; if it cannot be acknowledged in 30 s something is wrong and the structured
error should say so. Errors reuse the existing envelope and taxonomy (`missing_api_key`, `auth_failed`,
`rate_limit`, `bad_request`, …) plus `job_limit` (§8) when the active-job cap is reached.

### 3.2 `get_second_opinion`

| Arg | Type | Semantics |
|---|---|---|
| `job_id` | `str` | Required. Unknown/expired ⇒ error `unknown_job` (retriable: false), message explains the likely causes (typo, TTL eviction, server restart for grok-backed jobs). |
| `wait_seconds` | `float` | Default **0** (pure status check). Capped at `MAX_POLL_WAIT_SECONDS = 45` — values above are clamped, not rejected (log a warning). A long-poll blocks until the job reaches a terminal state or the wait expires, polling upstream internally every `POLL_INTERVAL_SECONDS = 4`. |

Non-terminal result:

```json
{
  "success": true,
  "request_id": "…",
  "job_id": "…",
  "status": "running",
  "job_elapsed_ms": 123456,
  "retry_after_ms": 15000
}
```

`retry_after_ms` is a hint, escalating with job age (5 s → 15 s → 30 s). Terminal results embed the **existing
envelopes unchanged**: `status: "succeeded"` carries exactly the SPEC §6.1 success shape (response, usage,
latency_ms measured across the job, model as reported upstream); `status: "failed"` carries the §6.1 error
object (`error.type` from the existing taxonomy — a job that exhausts `job_budget_seconds` fails with type
`timeout`); `status: "cancelled"` carries who/when. Terminal results are cached and **re-readable** until TTL
eviction (§6.3) — a re-fetch is idempotent, never an error, because the relay may eat the first delivery of a
result (finding 2 again).

The `get` call's *own* deadline (wait + grace) still discards in-flight poll work at the call level — but the
job is untouched. Call-level deadlines remain dead; job-level results are alive. See invariant rewrite, §9.

### 3.3 `cancel_second_opinion`

| Arg | Type | Semantics |
|---|---|---|
| `job_id` | `str` | Required; `unknown_job` as above. |

Cancels the upstream job: openai `POST /v1/responses/{id}/cancel`; gemini via the Interactions cancellation
call (verify exact surface during implementation, §12.3); grok `task.cancel()` honouring the existing
`CANCEL_GRACE_SECONDS` teardown discipline. Result: `{status: "cancelled"}` or the already-terminal state if
the race was lost (cancelling a finished job returns the finished job — not an error).

**Cancellation asymmetry (invariant):** abandoning, timing out, or never issuing a `get` call NEVER cancels a
job. Only `cancel_second_opinion` (or job-budget exhaustion, or server shutdown for local tasks) stops upstream
work. The v0.1 rule "cancel on outer cancellation so a billable call isn't left running unmanaged" (SPEC §10)
is deliberately *inverted* for jobs: background work being decoupled from any listener is now the point.

### 3.4 Tool descriptions steer the calling model

FastMCP derives schemas from signatures and docstrings (SPEC §6); for these tools the docstrings are
behavioural steering and must encode the protocol, approximately: *"For heavyweight reviews (large documents,
web search, high reasoning effort). After submitting, call get_second_opinion with wait_seconds=45; if still
running, tell the user the review is in progress and poll again after retry_after_ms or when the user asks —
do not poll in a tight loop. The synchronous second_opinion tool remains better for quick questions."* The
sync tool's docstring gains the mirror-image sentence. This is part of the contract, not documentation polish:
it is what makes Desktop sessions behave sanely without server-side enforcement.

## 4. Job identity

`job_id = uuid4().hex[:12]` — same grammar as `request_id`, distinct value space, logged as `jid=`. Provider
response/interaction ids are stored in the registry and logged (`upstream_id=…`) but never used as the
client-facing key: they are provider-scoped, unwieldy, and leaking them invites clients to poll vendors
directly. One request_id per *tool call*; one job_id per *job*; a job accumulates many request_ids over its
life (submit + N gets + maybe a cancel), and every log line carries both.

## 5. Job state machine

```
            ┌────────────┐   upstream acknowledges    ┌─────────┐
 submit ───▶│ submitting │ ─────────────────────────▶ │ running │
            └────────────┘                            └────┬────┘
                  │ submit fails                            │
                  ▼                                         ├──▶ succeeded   (terminal, cached)
             (no job — submit                               ├──▶ failed      (terminal, cached; incl. job-budget timeout)
              returns an error)                             └──▶ cancelled   (terminal, cached)
```

`submitting` is internal only — the client never observes it (submit returns after acknowledgement or fails).
Terminal states are immutable and cached with their full envelope. There is no client-visible `pending` vs
`running` distinction: upstream queueing states (openai `queued`) map to `running`.

## 6. The job registry

### 6.1 Structure

In-memory `dict[job_id → JobRecord]`; `JobRecord` holds: provider kind, backing type, upstream id or
`asyncio.Task`, `request_key`, submitted-at monotonic + wall time, the request echo (model, target, effort,
web_search, max_tokens — for logs and the terminal envelope), current status, and the cached terminal envelope.
A secondary index `request_key → job_id` implements idempotent submit. All mutation on the event loop (single
process, no locking needed — matches the v0.1 posture).

### 6.2 Capacity

`MAX_ACTIVE_JOBS = 8` concurrent non-terminal jobs. Submit beyond the cap returns error `job_limit`
(retriable: true — "wait for a job to finish or cancel one"; message lists active job ids and ages). Eight is
generous for a single-user tool and bounds both memory and worst-case vendor spend.

### 6.3 Terminal retention

Terminal records survive `TERMINAL_JOB_TTL_SECONDS = 1800` (30 min) from terminal transition, then evict on a
lazy sweep (checked on any registry access — no background timer needed). TTL exists so a result can be
re-fetched after relay flakiness but memory cannot grow unboundedly.

### 6.4 Restart semantics (v0.2 accepted limitation)

The registry is process memory. A server restart orphans: grok jobs die outright (task gone); openai/gemini
jobs **continue and complete upstream, billed, retrievable in principle but unreachable in practice** because
the job_id→upstream_id mapping is lost. `get` after restart returns `unknown_job` with a message that says
exactly this. USAGE must state it plainly. Durable mapping (a JSON sidecar file — not SQLite; single user,
tiny data) is the natural v0.3 step and is deliberately out of scope now: it adds a persistence layer, file
locking questions, and a privacy-surface change (job metadata at rest) that should not gate shipping the core
resilience win. Logged loudly at startup if the process previously exited with non-terminal jobs — which
requires writing nothing; it cannot be detected. Accept silently instead: document, don't half-build.

## 7. Timing model v2

The v0.1 three-layer model (SPEC §10) is retained *per tool call* and applied with per-tool budgets:

| Tool call | Budget | Grace | Worst case | Headroom vs 240 s cap |
|---|---|---|---|---|
| `submit_second_opinion` | `SUBMIT_BUDGET_SECONDS = 30` | 5 | 35 s | 205 s |
| `get_second_opinion` | `wait_seconds ≤ 45` (+ ~1 s status overhead) | 5 | ~51 s | ~189 s |
| `cancel_second_opinion` | 30 (same as submit) | 5 | 35 s | 205 s |
| `second_opinion` (sync) | `request_budget_seconds` (200 default) | 5 | 205 s | 35 s |

The **job** gets a new wall-clock: `job_budget_seconds`, default **900** (15 min), config/env overridable
(§10). It is enforced by the poller for provider-backed jobs (a `get` or the internal long-poll loop observing
`now - submitted_at > job_budget` cancels upstream and transitions to `failed(timeout)`) and by
`asyncio.wait`-based bounding for local grok tasks, reusing `_run_bounded`'s machinery with the larger budget.
The prescriptive timeout message pattern (SPEC §6.1) carries over with job-appropriate remedies.

Why these numbers: 30 s submit is ~6× a healthy background-create round trip; 45 s poll cap sits far enough
under every client cap observed or reported (240 s measured here; ~60 s reported on some hosts by external
sources) that even a hostile intermediary timer is cleared; 900 s job budget is ~4.5× the worst observed
genuine workload (finding 1 would have finished; nothing observed suggests needing more) while still bounding
runaway vendor spend. All three are named constants; none is load-bearing at its exact value.

## 8. Error taxonomy additions (additive, per SPEC invariant 2)

| New type | retriable | Meaning |
|---|---|---|
| `unknown_job` | ✘ | job_id not in registry: typo, TTL eviction, or post-restart orphan. Message distinguishes the causes where possible. |
| `job_limit` | ✔ | `MAX_ACTIVE_JOBS` reached. Message lists active jobs. |

Everything else reuses the existing taxonomy: job-budget exhaustion is `timeout`; upstream background failures
map through the existing per-provider mappings verbatim (the poller receives the same terminal payloads the
sync path receives — mapping code is shared, not duplicated).

## 9. Invariant amendments (SPEC §15)

- **Invariant 3 (rewritten):** *Every tool call clears the client cap.* Each tool call fits its own budget +
  `CANCEL_GRACE_SECONDS` under `CLIENT_HARD_CAP_SECONDS`, per the §7 table. No tool call's worst case may sit
  within 30 s of the cap except the legacy sync path, which is grandfathered and documented as such.
- **Invariant 4 (rewritten):** *Call deadlines are dead; job lifetimes are alive.* Work completing after a
  tool call's own deadline is never surfaced **by that call**. Work completing within `job_budget_seconds` is
  surfaced by any later `get`, by design — that is the point of a job. Work completing after the job budget is
  dead at the job level (the job is already `failed(timeout)`; a late upstream completion is not resurrected).
  The "never trust a coroutine that swallows cancellation" clause stands unchanged.
- **Invariant 11 (new):** *Poll abandonment never cancels.* Only `cancel_second_opinion`, job-budget
  exhaustion, or process shutdown (local tasks only) terminate upstream work. No `get` call, timed-out or
  otherwise, may propagate cancellation to a job.
- **Invariant 12 (new):** *Job ids are server-scoped.* Provider ids never appear as the client-facing job key.
- **Invariant 13 (new):** *Terminal results are immutable and idempotently re-readable* until TTL eviction.
- Invariants 1, 2, 5–10 apply to the new code paths unchanged (notably: stdout hygiene, errors-as-results,
  getattr-reflection adapters, explicit-caller-values-win).

## 10. Configuration additions

| Field | Resolution | Default | Validation |
|---|---|---|---|
| `job_budget_seconds` | env `LLM_SECOND_OPINION_JOB_BUDGET` → file | 900.0 | float > 0; values > 3600 warn (vendor-side background retention windows make very long jobs fragile — verify windows in §12.3) |

Constants (config.py / server.py): `SUBMIT_BUDGET_SECONDS = 30.0`, `MAX_POLL_WAIT_SECONDS = 45.0`,
`POLL_INTERVAL_SECONDS = 4.0`, `MAX_ACTIVE_JOBS = 8`, `TERMINAL_JOB_TTL_SECONDS = 1800.0`,
`DEFAULT_JOB_BUDGET_SECONDS = 900.0`. The existing `request_budget_seconds` continues to govern only the sync
tool and is untouched.

## 11. Privacy amendments (SPEC §13)

Background mode moves data custody, and §13's public claims must be amended honestly:

- openai: background requires stored responses (`store: true` sent explicitly — verify the requirement's exact
  current form in §12.3); stored responses persist at OpenAI per their retention policy (30 days per current
  docs) beyond the life of the call.
- gemini: interactions are stored by default; same class of disclosure.
- The "0 conversation data stored" claim remains true **of the server** but USAGE/docs must now say: *for
  background jobs, the chosen vendor stores the request and response under its retention policy; use the
  synchronous tool if that is unacceptable.* Grok-backed jobs, ironically, remain the most private (no
  vendor-side storage beyond a normal API call) precisely because xAI lacks background mode.
- Job records hold prompt-adjacent metadata in memory; `log_prompts` continues to gate any *content* logging
  (invariant 7 applies to every new log line — job lines log sizes and ids, never content).

## 12. Verification work folded into implementation

### 12.1 Grok upgrade gate — the disconnect-then-retrieve experiment

If xAI generation survives client disconnection and the stored response is retrievable by id, grok can move to
provider-backed jobs with zero local state. The naïve test is impossible — on a synchronous create the id only
arrives with the finished response (finding 4). The correct protocol: **submit with `store: true` and
`stream: true`; read SSE events only until `response.created` delivers the response id (first seconds);
deliberately drop the connection; wait; `GET /v1/responses/{id}`.** Outcomes: completed ⇒ upgrade path exists
(implement `backing: "provider_stored"` for grok, keeping the local-task path as fallback); cancelled/failed ⇒
local tasks remain the grok story and the doc note stands. Run once during implementation; record the result
here.

**Outcome (run 2026-09-07 against `grok-4.5` at `api.x.ai/v1`, three short calls):**

| Probe | Result |
|---|---|
| Control: synchronous create with `store: true`, then `GET /v1/responses/{id}` at +0 s and +5 s | `completed`, text retrievable both times — the retrieval surface works with this key. |
| Protocol: `store: true` + `stream: true`, read until `response.created` (arrived at 0.5–1.5 s, `status=in_progress`), close the connection, poll by id | **404 `not-found` at every poll: +5, +10, +20, +30, +45, +60, +90 s** — long past the time the ~700-word prompt completes when the connection is held. Run twice, identical. |
| `background: true` + `store: true`, non-streaming | **400 `Argument not supported: background`** — xAI now rejects the parameter outright (the docs' "Not used at the moment" wording understates it). |

Conclusion: generation does **not** survive client disconnection on xAI — nothing is ever stored for a
dropped stream, so there is no `provider_stored` upgrade path. `GrokProvider.supports_background` stays
`False`; grok jobs remain in-process `asyncio.Task`s over the synchronous `generate()` path, dying with the
server process as documented. Note for the future: because `background` is a hard 400, a grok adapter must
never send it even speculatively.

### 12.2 The acceptance test

The exact workload of finding 1 — the full SPEC.md review (research pass + red team + three role critiques),
`chatgpt`, `medium` effort, `web_search` on — must complete via submit/poll: submit returns < 5 s; the job
finishes in whatever time it needs; the full envelope is retrieved by `get`. The diagnostic session's failed
`rid=0533d5147482` is the before; this is the after. Secondary acceptance: the same call issued twice with one
`request_key` creates one job.

### 12.3 Point verifications against live docs at implementation time

Exact openai background+store requirement and retention window; gemini background cancellation surface and
retention; openai cancel endpoint shape; whether either vendor's poll responses include progress metadata worth
surfacing in `get` results. None of these change the architecture; all change wording or a field name.

**Verified 2026-09-07 during implementation** (developers.openai.com/api/docs/guides/background,
ai.google.dev/gemini-api/docs/background-execution and …/docs/interactions, docs.x.ai API reference):

- openai: `background: true`; poll `GET /v1/responses/{id}` (SDK `responses.retrieve`), non-terminal statuses
  `queued` and `in_progress`, terminal `completed | failed | incomplete | cancelled`; cancel
  `POST /v1/responses/{id}/cancel` (SDK `responses.cancel`), documented idempotent — "subsequent calls simply
  return the final Response object". `store` is *not* strictly required (ZDR projects run background with
  `store=false`) but background responses are retained beyond "roughly 10 minutes" only when stored, so
  `store: true` is sent explicitly as designed; stored responses then follow the account's retention policy.
  The docs also note background time-to-first-token is higher than synchronous. No progress metadata beyond
  `status` — nothing to surface.
- gemini: `background=True` on `interactions.create`; poll `interactions.get(id=…)`; cancel
  `interactions.cancel(id=…)` → status `cancelled` ("clean-up actions on the server can cause a slight delay
  before the status updates"). SDK status literal: `queued | in_progress | requires_action | completed |
  failed | cancelled | incomplete | budget_exceeded`; the poller treats `queued`/`in_progress` as running,
  `requires_action` as a terminal failure (nothing here can supply client input), the rest through the sync
  path's mapping. Interactions are stored by default — paid tier 55 days, free tier 1 day — and `store=false`
  is documented as incompatible with `background=true`, so nothing about storage is sent. No maximum
  background duration is documented; no progress metadata.
- xAI: `background` is still "Not used at the moment. Just for OpenResponses compatibility."; `store` exists
  ("Whether to store the input message(s) and model response for later retrieval") with
  `GET /v1/responses/{id}` and `DELETE`. The §12.1 experiment is therefore meaningful.
- SDK surfaces at implementation: `openai` 2.54 (`responses.create/retrieve/cancel` accept `background`,
  `store`); `google-genai` 2.22 (`aio.interactions.create/get/cancel`; `background` is an accepted create-body
  key). Note `mcp>=1.2.0` now resolves to mcp 2.x, which renamed FastMCP — pinned `<2`.

## 13. Test plan (extends SPEC §14, same conventions)

`test_jobs.py`, driven through the real FastMCP tool path with stub providers whose background lifecycle is
controllable from the test:

- submit returns under its bound; result carries job_id/backing/poll_after_ms; registry populated.
- idempotent submit: same `request_key` twice ⇒ same job_id, one upstream submission (stub counts creates).
- `get wait_seconds=0` returns `running` immediately; long-poll returns early on terminal transition; wait
  clamped at 45 with a warning; the get call's own deadline discards in-flight poll work without touching the
  job (job still running after — the inverse-of-v0.1 test, pinning rewritten invariant 4).
- late job results ARE surfaced by a later get (explicitly pins the invariant rewrite; the v0.1 "LATE ANSWER
  discarded" test continues to pass unchanged for the sync path).
- poll abandonment never cancels (stub asserts no cancel seen); `cancel_second_opinion` does cancel; cancel
  racing terminal returns the terminal state.
- job-budget exhaustion ⇒ `failed(timeout)` with prescriptive message; upstream cancel issued (stub asserts).
- terminal results re-readable; TTL eviction ⇒ `unknown_job`; `MAX_ACTIVE_JOBS` ⇒ `job_limit` listing jobs.
- grok local-task path: task cancelled on job-budget exhaustion under `CANCEL_GRACE_SECONDS` discipline; a
  cancellation-swallowing task is abandoned with the existing log line.
- stdout hygiene guards extended over every new code path; log grammar asserted (`jid=` on every job line,
  `upstream_id=` never in tool results).
- sync-path regression: the entire existing `test_request_budget.py` passes unmodified.

## 14. Observability additions

Existing `rid=` grammar extended, never changed: every job line carries `jid=`; submit logs
`outcome=submitted jid=… upstream_id=… backing=…`; get logs `outcome=running|ok|error(...) jid=…
job_elapsed_ms=…`; lifecycle lines for terminal transitions and TTL evictions. The stated goal carries over:
the job latency distribution must be measurable straight from the Desktop log with grep.

## 15. Documentation surfaces (SPEC §16.3 applies)

Code, tests, README (operator: new config key, restart semantics), USAGE (end-user: when to ask for a
background review, the probe-then-fire habit for stale bindings, the vendor-storage disclosure), docs/index.html
(the animated demo gains a submit → poll → result act), config.example.json (`job_budget_seconds`), version to
0.2.0 in both pyproject.toml and `__init__.py`, and SPEC.md itself: §10 gains findings 1–5 as a short
"observed failure modes" subsection; §15 gains the amended invariants; §16.2 marks this candidate as landed.

## 16. Explicitly out of scope for 0.2.0

Durable job persistence (sidecar file — v0.3 candidate, see §6.4); MCP Tasks extension adoption (idiomatic
long-running-operation support exists in the protocol's extension track but no host in use here negotiates it
yet — revisit when Desktop or claude.ai advertises it, at which point these tools become its compatibility
layer); streaming; multi-model fan-out; per-call `web_search` override (cheap, valuable — but orthogonal; do it
as its own small change so it doesn't ride this one's risk).
