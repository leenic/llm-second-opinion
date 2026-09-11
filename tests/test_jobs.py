"""Tests for the submit/poll background-job extension (DESIGN-submit-poll.md §13).

The tool-path tests drive the real FastMCP tools with stub providers whose
background lifecycle is controlled from the test. The stubs follow the
existing conventions: plain objects read via `getattr`, no SDK internals.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import pytest
import pytest_asyncio

from llm_second_opinion import jobs as jobs_mod
from llm_second_opinion.config import AppConfig, ProviderConfig
from llm_second_opinion.jobs import (
    BACKING_LOCAL_TASK,
    BACKING_PROVIDER_BACKGROUND,
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUBMITTING,
    STATUS_SUCCEEDED,
    JobRecord,
    JobRegistry,
    JobStateError,
    new_job_id,
    retry_after_ms,
)
from llm_second_opinion.providers.base import (
    BackgroundPoll,
    Provider,
    ProviderError,
    SecondOpinionResponse,
    TokenUsage,
)
from llm_second_opinion.server import build_server
from test_request_budget import StubProvider  # the v0.1 local-task stub


def make_record(job_id: str = "job000000001", **overrides) -> JobRecord:
    kwargs = dict(
        job_id=job_id,
        provider="stub",
        target_model="chatgpt",
        model="stub-model",
        backing=BACKING_PROVIDER_BACKGROUND,
    )
    kwargs.update(overrides)
    return JobRecord(**kwargs)


# ---------------------------------------------------------------------------
# Registry and state machine (design §5–6)
# ---------------------------------------------------------------------------


class TestJobIdentity:
    def test_job_id_has_request_id_grammar(self):
        jid = new_job_id()
        assert len(jid) == 12
        int(jid, 16)  # hex

    def test_job_ids_are_unique(self):
        assert len({new_job_id() for _ in range(200)}) == 200


class TestStateMachine:
    @pytest.mark.asyncio
    async def test_register_moves_submitting_to_running(self):
        reg = JobRegistry()
        rec = make_record()
        assert rec.status == STATUS_SUBMITTING
        reg.register(rec)
        assert rec.status == STATUS_RUNNING
        assert reg.get(rec.job_id) is rec

    @pytest.mark.asyncio
    @pytest.mark.parametrize("terminal", [STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELLED])
    async def test_running_reaches_each_terminal_state(self, terminal):
        reg = JobRegistry()
        rec = reg.register(make_record())
        assert reg.finish(rec, terminal, {"status": terminal}) is True
        assert rec.status == terminal
        assert rec.is_terminal
        assert rec.done.is_set()
        assert rec.envelope == {"status": terminal}

    @pytest.mark.asyncio
    async def test_terminal_states_are_immutable(self):
        """Invariant 13: whoever loses the race observes the winner's state."""
        reg = JobRegistry()
        rec = reg.register(make_record())
        reg.finish(rec, STATUS_SUCCEEDED, {"status": "succeeded", "response": "answer"})
        assert reg.finish(rec, STATUS_CANCELLED, {"status": "cancelled"}) is False
        assert rec.status == STATUS_SUCCEEDED
        assert rec.envelope["response"] == "answer"

    @pytest.mark.asyncio
    async def test_submitting_cannot_jump_to_terminal(self):
        rec = make_record()
        with pytest.raises(JobStateError):
            rec.transition(STATUS_SUCCEEDED)

    @pytest.mark.asyncio
    async def test_running_cannot_go_back_to_submitting(self):
        rec = JobRegistry().register(make_record())
        with pytest.raises(JobStateError):
            rec.transition(STATUS_SUBMITTING)


class TestRequestKeyIndex:
    @pytest.mark.asyncio
    async def test_key_resolves_to_the_job(self):
        reg = JobRegistry()
        rec = reg.register(make_record(request_key="k1"))
        assert reg.find_by_key("k1") is rec
        assert reg.find_by_key("nope") is None

    @pytest.mark.asyncio
    async def test_key_survives_terminal_transition_until_eviction(self):
        reg = JobRegistry(ttl_seconds=0.05)
        rec = reg.register(make_record(request_key="k1"))
        reg.finish(rec, STATUS_SUCCEEDED, {})
        assert reg.find_by_key("k1") is rec
        await asyncio.sleep(0.08)
        assert reg.find_by_key("k1") is None
        assert reg.get(rec.job_id) is None


class TestCapacityAndEviction:
    @pytest.mark.asyncio
    async def test_active_counts_only_non_terminal(self):
        reg = JobRegistry(max_active=2)
        a = reg.register(make_record("a" * 12))
        reg.register(make_record("b" * 12))
        assert reg.at_capacity()
        reg.finish(a, STATUS_FAILED, {})
        assert not reg.at_capacity()
        assert [r.job_id for r in reg.active()] == ["b" * 12]

    @pytest.mark.asyncio
    async def test_active_is_oldest_first(self):
        reg = JobRegistry()
        first = reg.register(make_record("a" * 12))
        await asyncio.sleep(0.01)
        reg.register(make_record("b" * 12))
        assert reg.active()[0] is first

    @pytest.mark.asyncio
    async def test_running_jobs_are_never_evicted(self):
        reg = JobRegistry(ttl_seconds=0.0)
        rec = reg.register(make_record())
        await asyncio.sleep(0.01)
        assert reg.sweep() == []
        assert reg.get(rec.job_id) is rec

    @pytest.mark.asyncio
    async def test_eviction_is_lazy_and_logged_with_jid(self, caplog):
        log = logging.getLogger("test_jobs_evict")
        reg = JobRegistry(ttl_seconds=0.02, logger=log)
        rec = reg.register(make_record())
        reg.finish(rec, STATUS_SUCCEEDED, {})
        await asyncio.sleep(0.05)
        with caplog.at_level(logging.INFO, logger="test_jobs_evict"):
            assert reg.get(rec.job_id) is None
        evict_lines = [r.message for r in caplog.records if "job=evicted" in r.message]
        assert evict_lines and all(f"jid={rec.job_id}" in m for m in evict_lines)


class TestRetryAfterHint:
    @pytest.mark.parametrize(
        "age,expected",
        [(0, 5000), (59, 5000), (60, 15000), (299, 15000), (300, 30000), (5000, 30000)],
    )
    def test_escalates_with_job_age(self, age, expected):
        assert retry_after_ms(age) == expected


class TestConstants:
    def test_values_match_the_design(self):
        """§10 — none is load-bearing at its exact value, but drift should be deliberate."""
        assert jobs_mod.SUBMIT_BUDGET_SECONDS == 30.0
        assert jobs_mod.MAX_POLL_WAIT_SECONDS == 45.0
        assert jobs_mod.POLL_INTERVAL_SECONDS == 4.0
        assert jobs_mod.MAX_ACTIVE_JOBS == 8
        assert jobs_mod.TERMINAL_JOB_TTL_SECONDS == 1800.0

    def test_every_tool_call_clears_the_client_cap(self):
        """Rewritten invariant 3: worst case per tool call ≥ 30s under the cap."""
        from llm_second_opinion.config import CLIENT_HARD_CAP_SECONDS
        from llm_second_opinion.server import CANCEL_GRACE_SECONDS

        cap = CLIENT_HARD_CAP_SECONDS
        assert jobs_mod.SUBMIT_BUDGET_SECONDS + CANCEL_GRACE_SECONDS <= cap - 30
        assert jobs_mod.MAX_POLL_WAIT_SECONDS + 1 + CANCEL_GRACE_SECONDS <= cap - 30


# ---------------------------------------------------------------------------
# Tool path (design §3, §7, §9) — through the real FastMCP tools
# ---------------------------------------------------------------------------

UPSTREAM_ID = "resp_UPSTREAM_SECRET_0123456789"


def stub_response(text: str = "stub answer") -> SecondOpinionResponse:
    return SecondOpinionResponse(
        provider="stub", model="stub-model", text=text,
        usage=TokenUsage(1, 2, 3, 0), latency_ms=1,
    )


class BackgroundStub(Provider):
    """Provider with a background lifecycle the test drives by hand."""

    name = "stub"
    supports_background = True

    def __init__(self, submit_delay: float = 0.0, submit_error=None):
        self.submit_delay = submit_delay
        self.submit_error = submit_error
        self.poll_error: Exception | None = None
        self.state = "running"
        self.response: SecondOpinionResponse | None = None
        self.error: ProviderError | None = None
        self.requests: list = []
        self.creates = 0
        self.polls = 0
        self.cancels = 0
        self.reasoning_effort = "medium"
        self.web_search = True

    def model_id(self) -> str:
        return "stub-model"

    async def generate(self, req):
        raise AssertionError("a background-capable provider must not be run synchronously by submit")

    async def check_reachable(self):
        return True, None

    async def submit_background(self, req, timeout):
        self.requests.append(req)
        self.creates += 1
        error = self.submit_error  # captured at call time: the in-flight attempt's fate
        await asyncio.sleep(self.submit_delay)
        if error is not None:
            raise error
        return UPSTREAM_ID

    async def poll_background(self, upstream_id, timeout):
        assert upstream_id == UPSTREAM_ID
        self.polls += 1
        if self.poll_error is not None:
            raise self.poll_error
        if self.state == "running":
            return BackgroundPoll(False, upstream_status="in_progress")
        if self.state == "failed":
            return BackgroundPoll(True, error=self.error, upstream_status="failed")
        return BackgroundPoll(True, response=self.response or stub_response(),
                              upstream_status="completed")

    async def cancel_background(self, upstream_id, timeout):
        assert upstream_id == UPSTREAM_ID
        self.cancels += 1
        self.state = "cancelled"

    def complete(self, text: str = "stub answer") -> None:
        self.state = "completed"
        self.response = stub_response(text)

    def fail(self, error: ProviderError) -> None:
        self.state = "failed"
        self.error = error


def make_config(
    budget: float = 0.05,
    job_budget: float = 5.0,
    default_max_tokens: int | None = 32000,
) -> AppConfig:
    return AppConfig(
        providers={
            "openai": ProviderConfig(api_key="k", model="m"),
            "gemini": ProviderConfig(api_key="k", model="m"),
            "grok": ProviderConfig(api_key="k", model="m"),
        },
        request_budget_seconds=budget,
        job_budget_seconds=job_budget,
        default_max_tokens=default_max_tokens,
    )


class Tools:
    def __init__(self, server, registry, provider):
        self.server = server
        self.registry = registry
        self.provider = provider

    async def call(self, name: str, **kwargs):
        result = await self.server.call_tool(name, kwargs)
        return result[1] if isinstance(result, tuple) else result

    async def submit(self, **kwargs):
        args = {"summary": "review this", "target_model": "chatgpt"}
        args.update(kwargs)
        return await self.call("submit_second_opinion", **args)

    async def get(self, job_id: str, **kwargs):
        return await self.call("get_second_opinion", job_id=job_id, **kwargs)

    async def cancel(self, job_id: str):
        return await self.call("cancel_second_opinion", job_id=job_id)

    async def wait_for(self, job_id: str, status: str, timeout: float = 2.0):
        """Poll with wait_seconds=0 until the job reports `status`."""
        deadline = time.monotonic() + timeout
        while True:
            result = await self.get(job_id)
            if result.get("status") == status:
                return result
            assert time.monotonic() < deadline, f"job never reached {status}: {result}"
            await asyncio.sleep(0.01)


@pytest_asyncio.fixture
async def tools(monkeypatch):
    """Build a server around a stub provider; tears down leftover job tasks."""
    import llm_second_opinion.server as server_mod

    registries: list[JobRegistry] = []

    def _setup(provider, config: AppConfig | None = None, registry: JobRegistry | None = None,
               logger: logging.Logger | None = None):
        if isinstance(provider, Exception):
            def _raise(*a, **k):
                raise provider
            monkeypatch.setattr(server_mod, "build_provider", _raise)
        else:
            monkeypatch.setattr(server_mod, "build_provider", lambda *a, **k: provider)
        monkeypatch.setattr(jobs_mod, "POLL_INTERVAL_SECONDS", 0.01)
        # `is not None`, not truthiness: an empty registry has len() == 0.
        registry = registry if registry is not None else JobRegistry(logger=logger)
        registries.append(registry)
        server = build_server(config or make_config(), logger=logger, registry=registry)
        return Tools(server, registry, provider)

    yield _setup

    for reg in registries:
        for rec in list(reg._jobs.values()):
            for task in (rec.driver, rec.task):
                if task is not None and not task.done():
                    task.cancel()
    await asyncio.sleep(0.05)


class TestSubmit:
    @pytest.mark.asyncio
    async def test_submit_returns_under_its_bound_with_the_job_shape(self, tools):
        t = tools(BackgroundStub())
        started = time.monotonic()
        result = await t.submit()
        took = time.monotonic() - started

        assert took < jobs_mod.SUBMIT_BUDGET_SECONDS
        assert result["success"] is True
        assert result["status"] == STATUS_RUNNING
        assert len(result["job_id"]) == 12
        assert result["request_id"] and result["request_id"] != result["job_id"]
        assert result["target_model"] == "chatgpt"
        assert result["provider"] == "stub"
        assert result["model"] == "stub-model"
        assert result["backing"] == BACKING_PROVIDER_BACKGROUND
        assert result["job_budget_seconds"] == 5.0
        assert result["poll_after_ms"] == 5000
        assert t.registry.get(result["job_id"]) is not None
        assert t.provider.creates == 1

    @pytest.mark.asyncio
    async def test_request_is_built_like_the_sync_path(self, tools):
        t = tools(BackgroundStub(), make_config(default_max_tokens=32000))
        await t.submit(focus="security", temperature=0.2)
        req = t.provider.requests[0]
        assert req.max_tokens == 32000
        assert req.focus == "security"
        assert req.temperature == 0.2
        assert req.system_prompt  # default reviewer prompt applied

    @pytest.mark.asyncio
    async def test_explicit_zero_max_tokens_is_honoured(self, tools):
        t = tools(BackgroundStub(), make_config(default_max_tokens=32000))
        await t.submit(max_tokens=0)
        assert t.provider.requests[0].max_tokens == 0

    @pytest.mark.asyncio
    async def test_job_id_is_never_the_upstream_id(self, tools):
        """Invariant 12."""
        t = tools(BackgroundStub())
        result = await t.submit()
        assert result["job_id"] != UPSTREAM_ID
        assert UPSTREAM_ID not in json.dumps(result)

    @pytest.mark.asyncio
    async def test_empty_summary_rejected_before_any_upstream_work(self, tools):
        t = tools(BackgroundStub())
        result = await t.submit(summary="   ")
        assert result["success"] is False
        assert result["error"]["type"] == "invalid_input"
        assert t.provider.creates == 0
        assert len(t.registry) == 0

    @pytest.mark.asyncio
    async def test_missing_key_is_the_usual_envelope(self, tools):
        t = tools(ProviderError("missing_api_key", "no key", retriable=False))
        result = await t.submit()
        assert result["success"] is False
        assert result["error"]["type"] == "missing_api_key"
        assert "job_id" not in result

    @pytest.mark.asyncio
    async def test_upstream_rejection_at_submit_creates_no_job(self, tools):
        t = tools(BackgroundStub(submit_error=ProviderError("rate_limit", "slow down", retriable=True)))
        result = await t.submit()
        assert result["success"] is False
        assert result["error"]["type"] == "rate_limit"
        assert result["error"]["retriable"] is True
        assert result["model"] == "stub-model"
        assert len(t.registry) == 0

    @pytest.mark.asyncio
    async def test_unacknowledged_submit_times_out_at_the_submit_budget(self, tools, monkeypatch):
        monkeypatch.setattr(jobs_mod, "SUBMIT_BUDGET_SECONDS", 0.05)
        t = tools(BackgroundStub(submit_delay=10))
        result = await t.submit()
        assert result["success"] is False
        assert result["error"]["type"] == "timeout"
        assert result["error"]["retriable"] is True
        assert "request_key" in result["error"]["message"]
        assert result["elapsed_ms"] < 2000
        assert len(t.registry) == 0


class TestSubmitReservation:
    """The slot and the request_key are held while the upstream call is in
    flight, so concurrency cannot duplicate a job or overrun the cap."""

    @pytest.mark.asyncio
    async def test_concurrent_same_key_submits_create_one_job(self, tools):
        t = tools(BackgroundStub(submit_delay=0.05))
        first, second = await asyncio.gather(
            t.submit(request_key="k-1"), t.submit(request_key="k-1"),
        )
        assert first["success"] and second["success"]
        assert first["job_id"] == second["job_id"]
        assert first["status"] == second["status"] == STATUS_RUNNING
        assert t.provider.creates == 1, "one upstream submission"
        assert {first.get("reused_existing_job"), second.get("reused_existing_job")} == {None, True}

    @pytest.mark.asyncio
    async def test_in_flight_submit_counts_toward_the_cap(self, tools):
        t = tools(BackgroundStub(submit_delay=0.05), registry=JobRegistry(max_active=1))
        first = asyncio.ensure_future(t.submit())
        await asyncio.sleep(0.01)
        second = await t.submit()
        assert second["success"] is False
        assert second["error"]["type"] == "job_limit"
        assert (await first)["success"] is True
        assert t.provider.creates == 1

    @pytest.mark.asyncio
    async def test_failed_submit_releases_slot_and_key(self, tools):
        t = tools(BackgroundStub(submit_error=ProviderError("rate_limit", "slow", retriable=True)),
                  registry=JobRegistry(max_active=1))
        first = await t.submit(request_key="k-1")
        assert first["error"]["type"] == "rate_limit"
        assert len(t.registry) == 0
        t.provider.submit_error = None
        second = await t.submit(request_key="k-1")
        assert second["success"] is True
        assert "reused_existing_job" not in second
        assert t.provider.creates == 2

    @pytest.mark.asyncio
    async def test_waiter_proceeds_fresh_when_the_original_fails(self, tools):
        t = tools(BackgroundStub(submit_delay=0.05,
                                 submit_error=ProviderError("rate_limit", "slow", retriable=True)))
        first = asyncio.ensure_future(t.submit(request_key="k-1"))
        await asyncio.sleep(0.01)
        t.provider.submit_error = None  # only the in-flight attempt fails
        second = await t.submit(request_key="k-1")
        assert (await first)["error"]["type"] == "rate_limit"
        assert second["success"] is True
        assert second["status"] == STATUS_RUNNING
        assert t.provider.creates == 2

    @pytest.mark.asyncio
    async def test_waiter_times_out_if_the_original_is_still_unacknowledged(self, tools, monkeypatch):
        """The waiter's bound is its own submit budget; it does not inherit
        the original's. Here the original still has time left when the
        waiter gives up, so the waiter reports what it is waiting on."""
        monkeypatch.setattr(jobs_mod, "SUBMIT_BUDGET_SECONDS", 10.0)
        t = tools(BackgroundStub(submit_delay=10))
        first = asyncio.ensure_future(t.submit(request_key="k-1"))
        await asyncio.sleep(0.01)
        monkeypatch.setattr(jobs_mod, "SUBMIT_BUDGET_SECONDS", 0.05)
        second = await t.submit(request_key="k-1")
        assert second["error"]["type"] == "timeout"
        assert second["error"]["retriable"] is True
        assert "k-1" in second["error"]["message"]
        assert t.provider.creates == 1, "the waiter must not start a second upstream job"
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

    @pytest.mark.asyncio
    async def test_waiter_proceeds_fresh_when_the_original_times_out(self, tools, monkeypatch):
        """A released reservation wakes the waiter, which then submits itself."""
        monkeypatch.setattr(jobs_mod, "SUBMIT_BUDGET_SECONDS", 0.05)
        t = tools(BackgroundStub(submit_delay=10))
        first = asyncio.ensure_future(t.submit(request_key="k-1"))
        await asyncio.sleep(0.01)
        t.provider.submit_delay = 0  # the retry is acknowledged at once
        second = await t.submit(request_key="k-1")
        assert (await first)["error"]["type"] == "timeout"
        assert second["success"] is True
        assert second["status"] == STATUS_RUNNING
        assert t.provider.creates == 2

    @pytest.mark.asyncio
    async def test_submitting_reservation_is_not_addressable(self, tools):
        t = tools(BackgroundStub(submit_delay=0.05))
        pending = asyncio.ensure_future(t.submit(request_key="k-1"))
        await asyncio.sleep(0.01)
        reserved = t.registry.find_by_key("k-1")
        assert reserved is not None and reserved.status == STATUS_SUBMITTING
        assert t.registry.get(reserved.job_id) is None
        assert (await t.get(reserved.job_id))["error"]["type"] == "unknown_job"
        assert (await pending)["job_id"] == reserved.job_id
        assert t.registry.get(reserved.job_id) is reserved


class TestLocalTaskTimeout:
    @pytest.mark.asyncio
    async def test_local_task_adapter_is_built_with_the_job_budget(self, tools, monkeypatch):
        """The adapter's HTTP timeout must span the job, not one sync call."""
        import llm_second_opinion.server as server_mod

        provider = StubProvider(delay=0)
        built: list[dict] = []

        def fake_build(target_model, config, **kwargs):
            built.append(dict(kwargs))
            return provider

        t = tools(provider, make_config(budget=0.05, job_budget=123.0))
        monkeypatch.setattr(server_mod, "build_provider", fake_build)
        job_id = (await t.submit(target_model="grok"))["job_id"]
        await t.wait_for(job_id, STATUS_SUCCEEDED)
        assert built[-1] == {"timeout": 123.0}

    @pytest.mark.asyncio
    async def test_background_adapter_keeps_the_default_construction(self, tools, monkeypatch):
        import llm_second_opinion.server as server_mod

        provider = BackgroundStub()
        built: list[dict] = []

        def fake_build(target_model, config, **kwargs):
            built.append(dict(kwargs))
            return provider

        t = tools(provider, make_config(job_budget=123.0))
        monkeypatch.setattr(server_mod, "build_provider", fake_build)
        await t.submit()
        assert built == [{}]

    def test_factory_timeout_override(self):
        from llm_second_opinion.providers import build_provider

        cfg = make_config(budget=200.0, job_budget=900.0)
        assert build_provider("grok", cfg).timeout == 200.0
        assert build_provider("grok", cfg, timeout=900.0).timeout == 900.0
        assert build_provider("chatgpt", cfg).timeout == 200.0


class TestIdempotentSubmit:
    @pytest.mark.asyncio
    async def test_same_request_key_returns_the_same_job(self, tools):
        t = tools(BackgroundStub())
        first = await t.submit(request_key="k-1")
        second = await t.submit(request_key="k-1")
        assert second["job_id"] == first["job_id"]
        assert second["request_id"] != first["request_id"]
        assert second["reused_existing_job"] is True
        assert "reused_existing_job" not in first
        assert t.provider.creates == 1, "one upstream submission"
        assert len(t.registry) == 1

    @pytest.mark.asyncio
    async def test_no_key_means_a_new_job_every_time(self, tools):
        t = tools(BackgroundStub())
        a = await t.submit()
        b = await t.submit()
        assert a["job_id"] != b["job_id"]
        assert t.provider.creates == 2

    @pytest.mark.asyncio
    async def test_key_of_a_finished_job_returns_the_finished_job(self, tools):
        t = tools(BackgroundStub())
        first = await t.submit(request_key="k-1")
        t.provider.complete()
        await t.wait_for(first["job_id"], STATUS_SUCCEEDED)
        again = await t.submit(request_key="k-1")
        assert again["job_id"] == first["job_id"]
        assert again["status"] == STATUS_SUCCEEDED
        assert again["poll_after_ms"] == 0
        assert t.provider.creates == 1


class TestGet:
    @pytest.mark.asyncio
    async def test_wait_zero_returns_running_immediately(self, tools):
        t = tools(BackgroundStub())
        job_id = (await t.submit())["job_id"]
        started = time.monotonic()
        result = await t.get(job_id)
        assert time.monotonic() - started < 0.5
        assert result == {
            "success": True,
            "request_id": result["request_id"],
            "job_id": job_id,
            "status": STATUS_RUNNING,
            "target_model": "chatgpt",
            "provider": "stub",
            "model": "stub-model",
            "backing": BACKING_PROVIDER_BACKGROUND,
            "job_elapsed_ms": result["job_elapsed_ms"],
            "retry_after_ms": 5000,
            "elapsed_ms": result["elapsed_ms"],
        }
        assert isinstance(result["job_elapsed_ms"], int)

    @pytest.mark.asyncio
    async def test_long_poll_returns_early_on_terminal_transition(self, tools):
        t = tools(BackgroundStub())
        job_id = (await t.submit())["job_id"]

        async def finish_soon():
            await asyncio.sleep(0.05)
            t.provider.complete("the answer")

        asyncio.ensure_future(finish_soon())
        started = time.monotonic()
        result = await t.get(job_id, wait_seconds=10)
        assert time.monotonic() - started < 2.0, "should return at the transition, not the wait"
        assert result["status"] == STATUS_SUCCEEDED
        assert result["response"] == "the answer"

    @pytest.mark.asyncio
    async def test_succeeded_carries_the_sync_success_shape(self, tools):
        t = tools(BackgroundStub())
        job_id = (await t.submit())["job_id"]
        t.provider.complete()
        result = await t.wait_for(job_id, STATUS_SUCCEEDED)
        assert result["success"] is True
        assert result["job_id"] == job_id
        assert result["target_model"] == "chatgpt"
        assert result["provider"] == "stub"
        assert result["model"] == "stub-model"
        assert result["response"] == "stub answer"
        assert result["usage"] == {
            "input_tokens": 1, "output_tokens": 2, "total_tokens": 3, "reasoning_tokens": 0,
        }
        assert isinstance(result["latency_ms"], int)
        assert result["latency_ms"] == result["job_elapsed_ms"]
        assert isinstance(result["elapsed_ms"], int)
        assert result["backing"] == BACKING_PROVIDER_BACKGROUND

    @pytest.mark.asyncio
    async def test_wait_is_clamped_with_a_warning(self, tools, caplog):
        log = logging.getLogger("test_jobs_clamp")
        t = tools(BackgroundStub(), logger=log)
        job_id = (await t.submit())["job_id"]
        t.provider.complete()
        await t.wait_for(job_id, STATUS_SUCCEEDED)
        with caplog.at_level(logging.WARNING, logger="test_jobs_clamp"):
            result = await t.get(job_id, wait_seconds=1000)
        assert result["status"] == STATUS_SUCCEEDED, "clamped, not rejected"
        clamp = [r.message for r in caplog.records if "clamp" in r.message.lower()]
        assert clamp and f"jid={job_id}" in clamp[0]
        assert "45" in clamp[0]

    @pytest.mark.asyncio
    async def test_gets_own_deadline_discards_the_wait_but_not_the_job(self, tools, monkeypatch):
        """Rewritten invariant 4 — the inverse of the v0.1 test: the call's
        deadline is dead, the job is alive."""
        import llm_second_opinion.server as server_mod

        monkeypatch.setattr(server_mod, "CANCEL_GRACE_SECONDS", 0.05)
        t = tools(BackgroundStub())
        job_id = (await t.submit())["job_id"]
        result = await t.get(job_id, wait_seconds=0.05)
        assert result["status"] == STATUS_RUNNING
        record = t.registry.get(job_id)
        assert record.status == STATUS_RUNNING
        assert not record.driver.done(), "the poller must keep running"
        assert t.provider.cancels == 0

    @pytest.mark.asyncio
    async def test_late_job_results_are_surfaced_by_a_later_get(self, tools):
        """Pins the invariant-4 rewrite: work completing after a get's own
        deadline is surfaced by the *next* get."""
        t = tools(BackgroundStub())
        job_id = (await t.submit())["job_id"]
        first = await t.get(job_id, wait_seconds=0.05)
        assert first["status"] == STATUS_RUNNING
        t.provider.complete("LATE ANSWER")
        later = await t.wait_for(job_id, STATUS_SUCCEEDED)
        assert later["response"] == "LATE ANSWER"

    @pytest.mark.asyncio
    async def test_unknown_job_explains_the_causes(self, tools):
        t = tools(BackgroundStub())
        result = await t.get("000000000000")
        assert result["success"] is False
        assert result["job_id"] == "000000000000"
        assert result["error"]["type"] == "unknown_job"
        assert result["error"]["retriable"] is False
        message = result["error"]["message"]
        assert "restart" in message
        assert "grok" in message
        assert "evict" in message or "expired" in message


class TestCancellationAsymmetry:
    """Invariant 11: only cancel_second_opinion, job-budget exhaustion, or
    process shutdown (local tasks) stop upstream work."""

    @pytest.mark.asyncio
    async def test_abandoned_long_poll_never_cancels_the_job(self, tools):
        t = tools(BackgroundStub())
        job_id = (await t.submit())["job_id"]

        # The relay eats the call: the client-side cancellation reaches the
        # handler as an outer cancel of the get call.
        get_call = asyncio.ensure_future(t.get(job_id, wait_seconds=10))
        await asyncio.sleep(0.05)
        get_call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await get_call
        await asyncio.sleep(0.05)

        record = t.registry.get(job_id)
        assert record.status == STATUS_RUNNING
        assert t.provider.cancels == 0, "poll abandonment must not propagate to the job"
        assert not record.driver.done()

        # ...and the job still completes normally afterwards.
        t.provider.complete("still here")
        result = await t.wait_for(job_id, STATUS_SUCCEEDED)
        assert result["response"] == "still here"
        assert t.provider.cancels == 0

    @pytest.mark.asyncio
    async def test_timed_out_get_never_cancels_the_job(self, tools, monkeypatch):
        import llm_second_opinion.server as server_mod

        monkeypatch.setattr(server_mod, "CANCEL_GRACE_SECONDS", 0.05)
        t = tools(BackgroundStub())
        job_id = (await t.submit())["job_id"]
        for _ in range(3):
            assert (await t.get(job_id, wait_seconds=0.02))["status"] == STATUS_RUNNING
        assert t.provider.cancels == 0
        assert t.registry.get(job_id).status == STATUS_RUNNING

    @pytest.mark.asyncio
    async def test_cancel_second_opinion_does_cancel(self, tools):
        t = tools(BackgroundStub())
        job_id = (await t.submit())["job_id"]
        result = await t.cancel(job_id)
        assert result["success"] is True
        assert result["status"] == STATUS_CANCELLED
        assert result["job_id"] == job_id
        assert result["cancelled_by"] == "cancel_second_opinion"
        assert result["cancelled_at"]
        assert isinstance(result["job_elapsed_ms"], int)
        assert t.provider.cancels == 1
        await asyncio.sleep(0.05)
        assert t.registry.get(job_id).driver.done(), "the poller stops with the job"
        # Re-reading the cancelled job is idempotent.
        again = await t.get(job_id)
        assert again["status"] == STATUS_CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_racing_terminal_returns_the_terminal_state(self, tools):
        t = tools(BackgroundStub())
        job_id = (await t.submit())["job_id"]
        t.provider.complete("done first")
        await t.wait_for(job_id, STATUS_SUCCEEDED)
        result = await t.cancel(job_id)
        assert result["success"] is True
        assert result["status"] == STATUS_SUCCEEDED
        assert result["response"] == "done first"
        assert t.provider.cancels == 0, "nothing to cancel upstream"

    @pytest.mark.asyncio
    async def test_cancel_preserves_a_result_that_finished_upstream_between_polls(self, tools):
        """The upstream finished after our last poll; its cancel endpoint
        hands the finished response back. A billed result is never discarded."""
        t = tools(BackgroundStub())
        job_id = (await t.submit())["job_id"]

        async def cancel_returns_finished(upstream_id, timeout):
            t.provider.cancels += 1
            return BackgroundPoll(True, response=stub_response("finished first"),
                                  upstream_status="completed")

        t.provider.cancel_background = cancel_returns_finished
        result = await t.cancel(job_id)
        assert result["success"] is True
        assert result["status"] == STATUS_SUCCEEDED
        assert result["response"] == "finished first"
        assert t.provider.cancels == 1
        assert (await t.get(job_id))["status"] == STATUS_SUCCEEDED
        await asyncio.sleep(0.05)
        assert t.registry.get(job_id).driver.done()

    @pytest.mark.asyncio
    async def test_cancel_reports_a_failure_that_landed_upstream_between_polls(self, tools):
        t = tools(BackgroundStub())
        job_id = (await t.submit())["job_id"]

        async def cancel_returns_failed(upstream_id, timeout):
            return BackgroundPoll(True, error=ProviderError("content_blocked", "refused"),
                                  upstream_status="incomplete")

        t.provider.cancel_background = cancel_returns_failed
        result = await t.cancel(job_id)
        assert result["status"] == STATUS_FAILED
        assert result["error"]["type"] == "content_blocked"

    @pytest.mark.asyncio
    async def test_cancel_unknown_job(self, tools):
        t = tools(BackgroundStub())
        result = await t.cancel("000000000000")
        assert result["error"]["type"] == "unknown_job"

    @pytest.mark.asyncio
    async def test_cancel_failing_upstream_leaves_the_job_running(self, tools):
        t = tools(BackgroundStub())
        job_id = (await t.submit())["job_id"]

        async def broken_cancel(upstream_id, timeout):
            raise ProviderError("network_error", "no route", retriable=True)

        t.provider.cancel_background = broken_cancel
        result = await t.cancel(job_id)
        assert result["success"] is False
        assert result["error"]["type"] == "network_error"
        assert result["error"]["retriable"] is True
        assert t.registry.get(job_id).status == STATUS_RUNNING


class TestJobBudget:
    @pytest.mark.asyncio
    async def test_exhaustion_fails_with_timeout_and_cancels_upstream(self, tools):
        t = tools(BackgroundStub(), make_config(job_budget=0.05))
        job_id = (await t.submit())["job_id"]
        result = await t.wait_for(job_id, STATUS_FAILED)

        assert result["success"] is False
        assert result["error"]["type"] == "timeout"
        assert result["error"]["retriable"] is True
        message = result["error"]["message"]
        assert "job_budget_seconds" in message
        assert "LLM_SECOND_OPINION_JOB_BUDGET" in message
        assert "reasoning_effort" in message
        assert t.provider.cancels == 1, "upstream cancel must be issued"
        assert result["model"] == "stub-model"
        assert result["target_model"] == "chatgpt"

    @pytest.mark.asyncio
    async def test_late_upstream_completion_is_not_resurrected(self, tools):
        t = tools(BackgroundStub(), make_config(job_budget=0.05))
        job_id = (await t.submit())["job_id"]
        await t.wait_for(job_id, STATUS_FAILED)
        t.provider.complete("too late")
        await asyncio.sleep(0.05)
        result = await t.get(job_id)
        assert result["status"] == STATUS_FAILED
        assert "too late" not in json.dumps(result)

    @pytest.mark.asyncio
    async def test_completion_observed_after_the_deadline_is_dead(self, tools, monkeypatch):
        """A poll in flight across the deadline: its bound is the remaining
        budget, and a terminal payload seen past the deadline is discarded."""
        import llm_second_opinion.server as server_mod

        monkeypatch.setattr(server_mod, "CANCEL_GRACE_SECONDS", 0.01)
        t = tools(BackgroundStub(), make_config(job_budget=0.1))
        job_id = (await t.submit())["job_id"]
        original = t.provider.poll_background
        seen_timeouts: list[float] = []

        async def slow_poll(upstream_id, timeout):
            seen_timeouts.append(timeout)
            await asyncio.sleep(0.15)  # completes only after the job deadline
            t.provider.complete("arrived late")
            return await original(upstream_id, timeout)

        t.provider.poll_background = slow_poll
        result = await t.wait_for(job_id, STATUS_FAILED)
        assert result["error"]["type"] == "timeout"
        assert "arrived late" not in json.dumps(result)
        assert t.provider.cancels == 1
        assert all(tm <= 0.1 for tm in seen_timeouts), seen_timeouts
        assert result["job_elapsed_ms"] < 1000, "cancel not delayed by the hanging poll"


class TestUpstreamOutcomes:
    @pytest.mark.asyncio
    async def test_terminal_failure_maps_through_the_existing_taxonomy(self, tools):
        t = tools(BackgroundStub())
        job_id = (await t.submit())["job_id"]
        t.provider.fail(ProviderError("content_blocked", "refused", retriable=False))
        result = await t.wait_for(job_id, STATUS_FAILED)
        assert result["success"] is False
        assert result["error"] == {"type": "content_blocked", "message": "refused", "retriable": False}
        assert t.provider.cancels == 0

    @pytest.mark.asyncio
    async def test_retriable_poll_errors_keep_polling(self, tools):
        t = tools(BackgroundStub())
        job_id = (await t.submit())["job_id"]
        t.provider.poll_error = ProviderError("rate_limit", "slow down", retriable=True)
        await asyncio.sleep(0.05)
        assert t.registry.get(job_id).status == STATUS_RUNNING
        t.provider.poll_error = None
        t.provider.complete()
        result = await t.wait_for(job_id, STATUS_SUCCEEDED)
        assert result["response"] == "stub answer"

    @pytest.mark.asyncio
    async def test_non_retriable_poll_error_fails_the_job(self, tools):
        t = tools(BackgroundStub())
        job_id = (await t.submit())["job_id"]
        t.provider.poll_error = ProviderError("auth_failed", "key revoked", retriable=False)
        result = await t.wait_for(job_id, STATUS_FAILED)
        assert result["error"]["type"] == "auth_failed"

    @pytest.mark.asyncio
    async def test_poll_call_timeouts_keep_polling(self, tools, monkeypatch):
        import llm_second_opinion.server as server_mod

        monkeypatch.setattr(server_mod, "CANCEL_GRACE_SECONDS", 0.01)
        monkeypatch.setattr(jobs_mod, "SUBMIT_BUDGET_SECONDS", 0.02)
        t = tools(BackgroundStub())
        job_id = (await t.submit())["job_id"]

        slow = {"on": True}
        original = t.provider.poll_background

        async def slow_poll(upstream_id, timeout):
            if slow["on"]:
                await asyncio.sleep(10)
            return await original(upstream_id, timeout)

        t.provider.poll_background = slow_poll
        await asyncio.sleep(0.1)
        assert t.registry.get(job_id).status == STATUS_RUNNING
        slow["on"] = False
        t.provider.complete()
        assert (await t.wait_for(job_id, STATUS_SUCCEEDED))["response"] == "stub answer"


class TestRetention:
    @pytest.mark.asyncio
    async def test_terminal_results_are_re_readable(self, tools):
        """Invariant 13."""
        t = tools(BackgroundStub())
        job_id = (await t.submit())["job_id"]
        t.provider.complete()
        first = await t.wait_for(job_id, STATUS_SUCCEEDED)
        second = await t.get(job_id)
        third = await t.get(job_id, wait_seconds=5)
        for r in (second, third):
            assert r["request_id"] != first["request_id"]
            for key in ("job_id", "status", "response", "usage", "latency_ms", "job_elapsed_ms"):
                assert r[key] == first[key]

    @pytest.mark.asyncio
    async def test_ttl_eviction_yields_unknown_job(self, tools):
        t = tools(BackgroundStub(), registry=JobRegistry(ttl_seconds=0.05))
        job_id = (await t.submit())["job_id"]
        t.provider.complete()
        await t.wait_for(job_id, STATUS_SUCCEEDED)
        await asyncio.sleep(0.08)
        result = await t.get(job_id)
        assert result["error"]["type"] == "unknown_job"

    @pytest.mark.asyncio
    async def test_active_job_cap_returns_job_limit_listing_jobs(self, tools):
        t = tools(BackgroundStub(), registry=JobRegistry(max_active=1))
        first = await t.submit()
        result = await t.submit()
        assert result["success"] is False
        assert result["error"]["type"] == "job_limit"
        assert result["error"]["retriable"] is True
        assert first["job_id"] in result["error"]["message"]
        assert "cancel_second_opinion" in result["error"]["message"]
        assert t.provider.creates == 1
        # A finished job frees the slot.
        t.provider.complete()
        await t.wait_for(first["job_id"], STATUS_SUCCEEDED)
        assert (await t.submit())["success"] is True


class TestLocalTaskBacking:
    """The grok story: an in-process task running the sync generate() path."""

    @pytest.mark.asyncio
    async def test_submit_and_complete(self, tools):
        provider = StubProvider(delay=0.02)
        t = tools(provider)
        submitted = await t.submit(target_model="grok")
        assert submitted["backing"] == BACKING_LOCAL_TASK
        assert submitted["status"] == STATUS_RUNNING
        result = await t.get(submitted["job_id"], wait_seconds=5)
        assert result["status"] == STATUS_SUCCEEDED
        assert result["response"] == "stub answer"
        assert result["backing"] == BACKING_LOCAL_TASK
        assert result["target_model"] == "grok"
        assert provider.requests[0].max_tokens == 32000

    @pytest.mark.asyncio
    async def test_provider_errors_fail_the_job(self, tools):
        provider = StubProvider(error=ProviderError("upstream_error", "boom", retriable=True))
        t = tools(provider)
        job_id = (await t.submit(target_model="grok"))["job_id"]
        result = await t.wait_for(job_id, STATUS_FAILED)
        assert result["error"]["type"] == "upstream_error"
        assert result["error"]["message"] == "boom"

    @pytest.mark.asyncio
    async def test_job_budget_cancels_the_task_under_grace_discipline(self, tools, monkeypatch):
        import llm_second_opinion.server as server_mod

        monkeypatch.setattr(server_mod, "CANCEL_GRACE_SECONDS", 0.05)
        provider = StubProvider(delay=10)
        t = tools(provider, make_config(job_budget=0.05))
        job_id = (await t.submit(target_model="grok"))["job_id"]
        result = await t.wait_for(job_id, STATUS_FAILED)
        assert result["error"]["type"] == "timeout"
        assert "job_budget_seconds" in result["error"]["message"]
        assert provider.started == 1
        assert provider.cancelled == 1, "the task must observe cancellation"
        assert provider.finished == 0

    @pytest.mark.asyncio
    async def test_cancellation_swallowing_task_is_abandoned_with_the_log_line(
        self, tools, monkeypatch, caplog
    ):
        import llm_second_opinion.server as server_mod

        monkeypatch.setattr(server_mod, "CANCEL_GRACE_SECONDS", 0.05)

        class StubbornProvider(StubProvider):
            async def generate(self, req):
                self.started += 1
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    await asyncio.sleep(10)

        log = logging.getLogger("test_jobs_stubborn")
        t = tools(StubbornProvider(), make_config(job_budget=0.05), logger=log)
        with caplog.at_level(logging.WARNING, logger="test_jobs_stubborn"):
            job_id = (await t.submit(target_model="grok"))["job_id"]
            result = await t.wait_for(job_id, STATUS_FAILED)
        assert result["error"]["type"] == "timeout"
        assert any("ignored cancellation" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_cancel_second_opinion_cancels_the_task(self, tools):
        provider = StubProvider(delay=10)
        t = tools(provider)
        job_id = (await t.submit(target_model="grok"))["job_id"]
        await asyncio.sleep(0.01)
        result = await t.cancel(job_id)
        assert result["status"] == STATUS_CANCELLED
        await asyncio.sleep(0.05)
        assert provider.cancelled == 1
        assert provider.finished == 0
        assert t.registry.get(job_id).driver.done()

    @pytest.mark.asyncio
    async def test_abandoned_get_never_cancels_the_task(self, tools):
        provider = StubProvider(delay=10)
        t = tools(provider)
        job_id = (await t.submit(target_model="grok"))["job_id"]
        get_call = asyncio.ensure_future(t.get(job_id, wait_seconds=10))
        await asyncio.sleep(0.05)
        get_call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await get_call
        await asyncio.sleep(0.05)
        assert provider.cancelled == 0
        assert t.registry.get(job_id).status == STATUS_RUNNING


class TestSyncPathUntouched:
    @pytest.mark.asyncio
    async def test_second_opinion_still_works_alongside_jobs(self, tools):
        provider = StubProvider(delay=0)
        t = tools(provider)
        result = await t.call("second_opinion", summary="review this", target_model="grok")
        assert result["success"] is True
        assert result["response"] == "stub answer"
        assert "job_id" not in result
        assert len(t.registry) == 0


class TestHygieneAndObservability:
    @pytest.mark.asyncio
    async def test_every_job_path_writes_nothing_to_stdout(self, tools, capsys, monkeypatch):
        import llm_second_opinion.server as server_mod

        monkeypatch.setattr(server_mod, "CANCEL_GRACE_SECONDS", 0.05)
        t = tools(BackgroundStub(), registry=JobRegistry(max_active=1))
        job_id = (await t.submit(request_key="k"))["job_id"]
        await t.submit(request_key="k")
        await t.submit()  # job_limit
        await t.get(job_id)
        await t.get(job_id, wait_seconds=0.02)
        await t.get("000000000000")
        await t.cancel(job_id)
        await t.cancel(job_id)
        await t.cancel("000000000000")
        await t.submit(summary=" ")
        assert capsys.readouterr().out == ""

    @pytest.mark.asyncio
    async def test_local_paths_write_nothing_to_stdout(self, tools, capsys, monkeypatch):
        import llm_second_opinion.server as server_mod

        monkeypatch.setattr(server_mod, "CANCEL_GRACE_SECONDS", 0.05)
        t = tools(StubProvider(delay=10), make_config(job_budget=0.05))
        job_id = (await t.submit(target_model="grok"))["job_id"]
        await t.wait_for(job_id, STATUS_FAILED)
        assert capsys.readouterr().out == ""

    @pytest.mark.asyncio
    async def test_log_grammar_jid_on_every_job_line_and_no_upstream_id_in_results(
        self, tools, caplog
    ):
        log = logging.getLogger("test_jobs_grammar")
        t = tools(BackgroundStub(), logger=log)
        results = []
        with caplog.at_level(logging.INFO, logger="test_jobs_grammar"):
            submitted = await t.submit(request_key="k")
            results.append(submitted)
            job_id = submitted["job_id"]
            results.append(await t.submit(request_key="k"))
            results.append(await t.get(job_id))
            results.append(await t.get(job_id, wait_seconds=0.02))
            t.provider.complete()
            results.append(await t.wait_for(job_id, STATUS_SUCCEEDED))
            results.append(await t.cancel(job_id))

        job_lines = [
            r.message for r in caplog.records
            if any(k in r.message for k in (
                "tool=get_second_opinion", "tool=cancel_second_opinion",
                "outcome=submitted", "outcome=deduplicated", "job=",
            ))
        ]
        assert job_lines, "expected job log lines"
        assert all("jid=" in m for m in job_lines), job_lines
        assert any("outcome=submitted" in m and f"upstream_id={UPSTREAM_ID}" in m
                   and "backing=provider_background" in m for m in job_lines)
        assert any("job=terminal" in m and "status=succeeded" in m for m in job_lines)
        assert any("outcome=running" in m and "job_elapsed_ms=" in m for m in job_lines)
        assert any("outcome=ok" in m and "tool=get_second_opinion" in m for m in job_lines)

        blob = json.dumps(results)
        assert "upstream_id" not in blob
        assert UPSTREAM_ID not in blob

    @pytest.mark.asyncio
    async def test_no_tasks_linger_after_a_job_finishes(self, tools):
        t = tools(BackgroundStub())
        job_id = (await t.submit())["job_id"]
        t.provider.complete()
        await t.wait_for(job_id, STATUS_SUCCEEDED)
        await asyncio.sleep(0.02)
        record = t.registry.get(job_id)
        assert record.driver.done()
        assert record.task is None


# ---------------------------------------------------------------------------
# Provider backends (design §2, §8, §12.3) — SimpleNamespace fakes read via
# getattr; real SDK exception classes on the error paths.
# ---------------------------------------------------------------------------

from types import SimpleNamespace

from conftest import (
    UNSUPPORTED_TEMPERATURE,
    bad_request,
    reasoning,
    responses_result,
    text_message,
    web_search_call,
)
from llm_second_opinion.providers.gemini import GeminiProvider
from llm_second_opinion.providers.grok import GrokProvider
from llm_second_opinion.providers.openai_provider import OpenAIProvider

ANSWER = "**Final answer.**"
ASIDE = "I need current best practices."


class FakeResponsesClient:
    """Stand-in for `AsyncOpenAI` with create/retrieve/cancel."""

    def __init__(self, create=None, retrieve=None, cancel=None):
        self.calls: dict[str, list] = {"create": [], "retrieve": [], "cancel": []}
        self.options: list[dict] = []
        self._handlers = {"create": create, "retrieve": retrieve, "cancel": cancel}
        self.responses = SimpleNamespace(
            create=self._make("create"),
            retrieve=self._make("retrieve"),
            cancel=self._make("cancel"),
        )

    def _make(self, name):
        async def _call(*args, **kwargs):
            self.calls[name].append((args, dict(kwargs)))
            handler = self._handlers[name]
            outcome = handler(*args, **kwargs) if callable(handler) else handler
            if isinstance(outcome, list):
                outcome = outcome.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return _call

    def with_options(self, **options):
        self.options.append(options)
        return self


def responses_provider(cls=OpenAIProvider, **handlers):
    client = FakeResponsesClient(**handlers)
    provider = cls(api_key="k", model="test-model", timeout=200.0, reasoning_effort="medium")
    provider._client = lambda: client  # type: ignore[method-assign]
    return provider, client


def queued(upstream_id: str = "resp_abc") -> SimpleNamespace:
    return SimpleNamespace(id=upstream_id, status="queued", output=[], usage=None)


class TestBackingSupport:
    def test_which_providers_have_background_mode(self):
        assert OpenAIProvider.supports_background is True
        assert GeminiProvider.supports_background is True
        assert GrokProvider.supports_background is False, "xAI: 'Not used at the moment'"

    @pytest.mark.asyncio
    async def test_default_provider_has_no_background_methods(self):
        provider = StubProvider()
        assert provider.supports_background is False
        with pytest.raises(NotImplementedError):
            await provider.submit_background(None, timeout=1)


class TestOpenAIBackground:
    @pytest.mark.asyncio
    async def test_submit_sends_background_and_store_explicitly(self, request_factory):
        provider, client = responses_provider(create=queued("resp_123"))
        upstream_id = await provider.submit_background(request_factory(max_tokens=500), timeout=30.0)

        assert upstream_id == "resp_123"
        _args, kwargs = client.calls["create"][0]
        assert kwargs["background"] is True
        assert kwargs["store"] is True
        assert kwargs["model"] == "test-model"
        assert kwargs["max_output_tokens"] == 500
        assert kwargs["reasoning"] == {"effort": "medium"}
        assert kwargs["instructions"] == "be critical"
        assert client.options == [{"timeout": 30.0}], "control calls use the submit budget"

    @pytest.mark.asyncio
    async def test_submit_reuses_the_droppable_param_retry(self, request_factory):
        provider, client = responses_provider(
            create=[bad_request(UNSUPPORTED_TEMPERATURE), queued("resp_123")]
        )
        upstream_id = await provider.submit_background(request_factory(temperature=0.2), timeout=30.0)
        assert upstream_id == "resp_123"
        first, second = client.calls["create"]
        assert first[1]["temperature"] == 0.2
        assert "temperature" not in second[1]
        assert second[1]["background"] is True and second[1]["store"] is True
        # Retry charged against the submit budget, not the 200s sync timeout.
        assert len(client.options) == 2
        assert 0 < client.options[1]["timeout"] <= 30.0

    @pytest.mark.asyncio
    async def test_submit_without_an_id_is_an_error(self, request_factory):
        provider, _ = responses_provider(create=SimpleNamespace(status="queued"))
        with pytest.raises(ProviderError) as exc:
            await provider.submit_background(request_factory(), timeout=30.0)
        assert exc.value.error_type == "upstream_error"
        assert exc.value.retriable is True

    @pytest.mark.asyncio
    async def test_submit_maps_sdk_errors_like_the_sync_path(self, request_factory):
        import httpx
        from openai import RateLimitError

        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        err = RateLimitError("slow down", response=httpx.Response(429, request=request), body=None)
        provider, _ = responses_provider(create=err)
        with pytest.raises(ProviderError) as exc:
            await provider.submit_background(request_factory(), timeout=30.0)
        assert exc.value.error_type == "rate_limit"
        assert exc.value.retriable is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["queued", "in_progress"])
    async def test_poll_non_terminal(self, status):
        provider, client = responses_provider(retrieve=SimpleNamespace(id="resp_1", status=status))
        poll = await provider.poll_background("resp_1", timeout=30.0)
        assert poll.done is False
        assert poll.upstream_status == status
        assert client.calls["retrieve"][0][0] == ("resp_1",)

    @pytest.mark.asyncio
    async def test_poll_completed_uses_final_answer_extraction(self):
        """The terminal payload goes through `_build_response`: narration
        before the tool call is dropped exactly as on the sync path."""
        result = responses_result(
            reasoning(), text_message(ASIDE), web_search_call(), text_message(ANSWER),
            model="gpt-5.6-sol",
        )
        provider, _ = responses_provider(retrieve=result)
        poll = await provider.poll_background("resp_1", timeout=30.0)
        assert poll.done is True
        assert poll.error is None
        assert poll.response.text == ANSWER
        assert poll.response.model == "gpt-5.6-sol"
        assert poll.response.provider == "openai"
        assert poll.response.usage.reasoning_tokens == 5

    @pytest.mark.asyncio
    async def test_poll_failed_is_a_terminal_error_not_a_raise(self):
        result = responses_result(status="failed", error=SimpleNamespace(message="upstream exploded"))
        provider, _ = responses_provider(retrieve=result)
        poll = await provider.poll_background("resp_1", timeout=30.0)
        assert poll.done is True and poll.response is None
        assert poll.error.error_type == "upstream_error"
        assert "upstream exploded" in poll.error.message

    @pytest.mark.asyncio
    async def test_poll_incomplete_keeps_the_sync_mappings(self):
        provider, _ = responses_provider(retrieve=responses_result(
            status="incomplete", incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        ))
        poll = await provider.poll_background("resp_1", timeout=30.0)
        assert poll.done and poll.error.error_type == "upstream_error" and poll.error.retriable

        provider, _ = responses_provider(retrieve=responses_result(
            status="incomplete", incomplete_details=SimpleNamespace(reason="content_filter"),
        ))
        poll = await provider.poll_background("resp_1", timeout=30.0)
        assert poll.done and poll.error.error_type == "content_blocked"

    @pytest.mark.asyncio
    async def test_poll_cancelled_upstream(self):
        provider, _ = responses_provider(retrieve=SimpleNamespace(id="resp_1", status="cancelled"))
        poll = await provider.poll_background("resp_1", timeout=30.0)
        assert poll.done is True
        assert poll.error.error_type == "upstream_error"
        assert "cancelled" in poll.error.message

    @pytest.mark.asyncio
    async def test_poll_transport_errors_raise_with_the_sync_mapping(self):
        import httpx
        from openai import APITimeoutError, AuthenticationError

        request = httpx.Request("GET", "https://api.openai.com/v1/responses/resp_1")
        provider, _ = responses_provider(retrieve=APITimeoutError(request=request))
        with pytest.raises(ProviderError) as exc:
            await provider.poll_background("resp_1", timeout=30.0)
        assert exc.value.error_type == "timeout"
        assert exc.value.retriable is True
        assert "30.0s" in exc.value.message

        err = AuthenticationError("bad key", response=httpx.Response(401, request=request), body=None)
        provider, _ = responses_provider(retrieve=err)
        with pytest.raises(ProviderError) as exc:
            await provider.poll_background("resp_1", timeout=30.0)
        assert exc.value.error_type == "auth_failed"
        assert exc.value.retriable is False

    @pytest.mark.asyncio
    async def test_cancel_posts_to_the_cancel_endpoint(self):
        provider, client = responses_provider(cancel=SimpleNamespace(id="resp_1", status="cancelled"))
        outcome = await provider.cancel_background("resp_1", timeout=30.0)
        assert client.calls["cancel"][0][0] == ("resp_1",)
        assert client.options == [{"timeout": 30.0}]
        assert outcome.done and outcome.response is None and outcome.error is None
        assert outcome.upstream_status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_of_a_finished_response_returns_it(self):
        """Idempotent upstream: the finished Response comes back and is kept."""
        provider, _ = responses_provider(cancel=responses_result(text_message(ANSWER)))
        outcome = await provider.cancel_background("resp_1", timeout=30.0)
        assert outcome.done and outcome.response.text == ANSWER

    @pytest.mark.asyncio
    async def test_cancel_with_lagging_status_counts_as_cancelled(self):
        provider, _ = responses_provider(cancel=SimpleNamespace(id="resp_1", status="in_progress"))
        outcome = await provider.cancel_background("resp_1", timeout=30.0)
        assert outcome.done and outcome.response is None and outcome.error is None

    @pytest.mark.asyncio
    async def test_sync_generate_is_unchanged_by_the_refactor(self, make_provider, request_factory):
        provider, calls = make_provider(
            [bad_request(UNSUPPORTED_TEMPERATURE), responses_result(text_message(ANSWER))],
            timeout=30.0,
        )
        resp = await provider.generate(request_factory(temperature=0.2))
        assert resp.text == ANSWER
        assert len(calls) == 2
        # The sync path asks the vendor not to store (DESIGN §18.1); it never
        # sends `background`.
        assert "background" not in calls[0] and calls[0]["store"] is False
        assert 0 < calls.timeouts[0] <= 30.0


class FakeInteractionsClient:
    def __init__(self, create=None, get=None, cancel=None):
        self.calls: dict[str, list] = {"create": [], "get": [], "cancel": []}
        self._handlers = {"create": create, "get": get, "cancel": cancel}
        self.aio = SimpleNamespace(interactions=SimpleNamespace(
            create=self._make("create"), get=self._make("get"), cancel=self._make("cancel"),
        ))

    def _make(self, name):
        async def _call(**kwargs):
            self.calls[name].append(dict(kwargs))
            outcome = self._handlers[name]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return _call


def gemini_provider(**handlers):
    client = FakeInteractionsClient(**handlers)
    provider = GeminiProvider(api_key="k", model="gemini-test", timeout=200.0,
                              reasoning_effort="high", web_search=True)
    provider._client = lambda: client  # type: ignore[method-assign]
    return provider, client


def interaction(status: str, text: str | None = ANSWER, upstream_id: str = "int_1") -> SimpleNamespace:
    return SimpleNamespace(
        id=upstream_id, status=status, output_text=text, model="gemini-test-001",
        usage=SimpleNamespace(total_input_tokens=1, total_output_tokens=2,
                              total_tokens=3, total_thought_tokens=1),
    )


class TestGeminiBackground:
    @pytest.mark.asyncio
    async def test_submit_sends_background_true(self, request_factory):
        provider, client = gemini_provider(create=interaction("in_progress", None, "int_42"))
        upstream_id = await provider.submit_background(request_factory(max_tokens=500), timeout=30.0)
        assert upstream_id == "int_42"
        sent = client.calls["create"][0]
        assert sent["background"] is True
        assert "store" not in sent, "stored by default; store=false is incompatible with background"
        assert sent["model"] == "gemini-test"
        assert sent["system_instruction"] == "be critical"
        assert sent["generation_config"] == {"max_output_tokens": 500, "thinking_level": "high"}
        assert sent["tools"] == [{"type": "google_search"}]

    @pytest.mark.asyncio
    async def test_submit_without_an_id_is_an_error(self, request_factory):
        provider, _ = gemini_provider(create=SimpleNamespace(status="in_progress"))
        with pytest.raises(ProviderError) as exc:
            await provider.submit_background(request_factory(), timeout=30.0)
        assert exc.value.error_type == "upstream_error"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["queued", "in_progress"])
    async def test_poll_non_terminal(self, status):
        provider, client = gemini_provider(get=interaction(status, None))
        poll = await provider.poll_background("int_1", timeout=30.0)
        assert poll.done is False
        assert client.calls["get"][0] == {"id": "int_1"}

    @pytest.mark.asyncio
    async def test_poll_completed(self):
        provider, _ = gemini_provider(get=interaction("completed"))
        poll = await provider.poll_background("int_1", timeout=30.0)
        assert poll.done is True
        assert poll.response.text == ANSWER
        assert poll.response.model == "gemini-test-001"
        assert poll.response.usage.reasoning_tokens == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status,error_type",
        # DESIGN §18.2: `content_blocked` only on a block signal; the rest
        # are upstream_error with retriability per status.
        [("failed", "upstream_error"), ("budget_exceeded", "upstream_error"),
         ("cancelled", "upstream_error"), ("incomplete", "upstream_error")],
    )
    async def test_poll_terminal_failures_keep_the_sync_mappings(self, status, error_type):
        provider, _ = gemini_provider(get=interaction(status))
        poll = await provider.poll_background("int_1", timeout=30.0)
        assert poll.done is True and poll.response is None
        assert poll.error.error_type == error_type

    @pytest.mark.asyncio
    async def test_poll_requires_action_fails_the_job(self):
        provider, _ = gemini_provider(get=interaction("requires_action", None))
        poll = await provider.poll_background("int_1", timeout=30.0)
        assert poll.done is True
        assert poll.error.error_type == "upstream_error"
        assert "requires_action" in poll.error.message

    @pytest.mark.asyncio
    async def test_poll_transport_errors_use_the_genai_mapping(self):
        from google.genai import errors as genai_errors

        err = genai_errors.APIError(429, {"error": {"message": "slow down", "status": "RESOURCE_EXHAUSTED"}})
        provider, _ = gemini_provider(get=err)
        with pytest.raises(ProviderError) as exc:
            await provider.poll_background("int_1", timeout=30.0)
        assert exc.value.error_type == "rate_limit"
        assert exc.value.retriable is True

    @pytest.mark.asyncio
    async def test_poll_is_bounded_by_the_control_timeout(self):
        async def hang(**kwargs):
            await asyncio.sleep(10)

        provider, client = gemini_provider()
        client.aio.interactions.get = hang
        with pytest.raises(ProviderError) as exc:
            await provider.poll_background("int_1", timeout=0.05)
        assert exc.value.error_type == "timeout"

    @pytest.mark.asyncio
    async def test_cancel_calls_the_cancel_surface(self):
        provider, client = gemini_provider(cancel=interaction("cancelled", None))
        outcome = await provider.cancel_background("int_1", timeout=30.0)
        assert client.calls["cancel"][0] == {"id": "int_1"}
        assert outcome.done and outcome.response is None and outcome.error is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["in_progress", "queued"])
    async def test_cancel_with_lagging_status_counts_as_cancelled(self, status):
        provider, _ = gemini_provider(cancel=interaction(status, None))
        outcome = await provider.cancel_background("int_1", timeout=30.0)
        assert outcome.done and outcome.response is None and outcome.error is None

    @pytest.mark.asyncio
    async def test_cancel_of_a_finished_interaction_returns_it(self):
        provider, _ = gemini_provider(cancel=interaction("completed"))
        outcome = await provider.cancel_background("int_1", timeout=30.0)
        assert outcome.done and outcome.response.text == ANSWER

    @pytest.mark.asyncio
    async def test_poll_cancelled_from_outside_is_the_sync_mapping(self):
        provider, _ = gemini_provider(get=interaction("cancelled"))
        poll = await provider.poll_background("int_1", timeout=30.0)
        assert poll.done and poll.error.error_type == "upstream_error"
        assert poll.error.retriable is False and "cancelled" in poll.error.message

    @pytest.mark.asyncio
    async def test_sync_generate_is_unchanged_by_the_refactor(self, request_factory):
        provider, client = gemini_provider(create=interaction("completed"))
        resp = await provider.generate(request_factory(temperature=0.3))
        assert resp.text == ANSWER
        assert resp.provider == "gemini"
        assert "background" not in client.calls["create"][0]
        assert client.calls["create"][0]["generation_config"]["temperature"] == 0.3

        provider, _ = gemini_provider(create=interaction("failed"))
        with pytest.raises(ProviderError) as exc:
            await provider.generate(request_factory())
        assert exc.value.error_type == "upstream_error"


class TestTaxonomyAdditions:
    def test_new_types_are_registered_and_nothing_renamed(self):
        from llm_second_opinion.providers.base import ERROR_TYPES

        assert {"unknown_job", "job_limit"} <= ERROR_TYPES
        assert {
            "missing_api_key", "auth_failed", "rate_limit", "timeout", "network_error",
            "upstream_error", "bad_request", "content_blocked", "invalid_input", "internal_error",
        } <= ERROR_TYPES
        assert ProviderError("unknown_job", "x").error_type == "unknown_job"
        assert ProviderError("job_limit", "x").error_type == "job_limit"
