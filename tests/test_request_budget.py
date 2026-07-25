"""Tests for the per-call request budget.

Claude Desktop enforces a hard 240s cap on tool calls and cancels with
`MCP error -32001: Request timed out`, discarding the result even when the
upstream call succeeded. The handler bounds itself below that cap so it
returns a structured error the calling model can act on instead.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from llm_second_opinion.config import AppConfig, ProviderConfig
from llm_second_opinion.providers.base import (
    Provider,
    ProviderError,
    SecondOpinionResponse,
    TokenUsage,
)
from llm_second_opinion.server import build_server


class StubProvider(Provider):
    """Provider whose timing and outcome the test controls directly."""

    name = "stub"

    def __init__(self, delay: float = 0.0, result=None, error=None):
        self.delay = delay
        self.result = result
        self.error = error
        self.requests: list = []
        self.started = 0
        self.finished = 0
        self.cancelled = 0

    def model_id(self) -> str:
        return "stub-model"

    async def generate(self, req):
        self.requests.append(req)
        self.started += 1
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        self.finished += 1
        if self.error is not None:
            raise self.error
        return self.result or SecondOpinionResponse(
            provider=self.name,
            model="stub-model",
            text="stub answer",
            usage=TokenUsage(1, 2, 3, 0),
            latency_ms=1,
        )

    async def check_reachable(self):
        return True, None


def make_config(budget: float = 0.05, default_max_tokens: int | None = 32000) -> AppConfig:
    return AppConfig(
        providers={
            "openai": ProviderConfig(api_key="k", model="m"),
            "gemini": ProviderConfig(api_key="k", model="m"),
            "grok": ProviderConfig(api_key="k", model="m"),
        },
        request_budget_seconds=budget,
        default_max_tokens=default_max_tokens,
    )


@pytest.fixture
def call_tool(monkeypatch):
    """Invoke `second_opinion` through the real FastMCP tool path."""

    def _setup(provider: StubProvider, config: AppConfig | None = None):
        import llm_second_opinion.server as server_mod

        monkeypatch.setattr(server_mod, "build_provider", lambda *a, **k: provider)
        server = build_server(config or make_config())

        async def _call(**kwargs):
            args = {"summary": "review this", "target_model": "chatgpt"}
            args.update(kwargs)
            result = await server.call_tool("second_opinion", args)
            return result[1] if isinstance(result, tuple) else result

        return _call

    return _setup


class TestTimeoutShape:
    @pytest.mark.asyncio
    async def test_slow_call_returns_structured_timeout_not_an_exception(self, call_tool):
        """A raised exception reaches the model as a transport failure it
        can't reason about. This must be an ordinary tool result."""
        provider = StubProvider(delay=10)
        result = await call_tool(provider, make_config(budget=0.05))()

        assert result["success"] is False
        assert result["error"]["type"] == "timeout"
        assert result["error"]["retriable"] is True
        assert result["request_id"]
        assert result["target_model"] == "chatgpt"
        assert result["model"] == "stub-model"
        assert isinstance(result["elapsed_ms"], int)

    @pytest.mark.asyncio
    async def test_timeout_fires_at_the_budget(self, call_tool):
        provider = StubProvider(delay=10)
        result = await call_tool(provider, make_config(budget=0.2))()

        # Bounded by the budget, not by the provider's 10s sleep.
        assert 150 <= result["elapsed_ms"] < 3000

    @pytest.mark.asyncio
    async def test_timeout_message_names_the_budget_and_the_client_cap(self, call_tool):
        provider = StubProvider(delay=10)
        result = await call_tool(provider, make_config(budget=0.05))()

        message = result["error"]["message"]
        assert "request budget" in message
        assert "240" in message, "should point at the cap that motivates the budget"


class TestCancellation:
    @pytest.mark.asyncio
    async def test_provider_task_is_cancelled_and_awaited(self, call_tool):
        """`wait_for` cancels the inner coroutine and awaits its cancellation
        before raising, so nothing is left running after the handler returns."""
        provider = StubProvider(delay=10)
        await call_tool(provider, make_config(budget=0.05))()

        assert provider.started == 1
        assert provider.cancelled == 1, "the coroutine must observe cancellation"
        assert provider.finished == 0, "no partial completion after cancel"

    @pytest.mark.asyncio
    async def test_no_pending_tasks_survive_the_handler(self, call_tool):
        before = len(asyncio.all_tasks())
        provider = StubProvider(delay=10)
        await call_tool(provider, make_config(budget=0.05))()
        await asyncio.sleep(0)  # let any stragglers get scheduled

        leftover = [t for t in asyncio.all_tasks() if not t.done()]
        assert len(leftover) <= before, f"leaked tasks: {leftover}"

    @pytest.mark.asyncio
    async def test_late_result_is_discarded_not_returned(self, call_tool):
        """A provider that finishes just after the deadline must not have its
        answer surface — that write would land after the handler returned."""

        class LateProvider(StubProvider):
            async def generate(self, req):
                self.started += 1
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    self.cancelled += 1
                    # Swallow cancellation and try to return anyway.
                    return SecondOpinionResponse(
                        provider="stub", model="stub-model",
                        text="LATE ANSWER", usage=None, latency_ms=1,
                    )

        provider = LateProvider()
        result = await call_tool(provider, make_config(budget=0.05))()

        assert result["success"] is False
        assert result["error"]["type"] == "timeout"
        assert "LATE ANSWER" not in str(result)


class TestNormalCallsUnchanged:
    @pytest.mark.asyncio
    async def test_fast_call_succeeds_with_the_usual_shape(self, call_tool):
        provider = StubProvider(delay=0)
        result = await call_tool(provider)()

        assert result["success"] is True
        assert result["response"] == "stub answer"
        assert result["provider"] == "stub"
        assert result["model"] == "stub-model"
        assert result["usage"] == {
            "input_tokens": 1, "output_tokens": 2,
            "total_tokens": 3, "reasoning_tokens": 0,
        }
        assert result["latency_ms"] == 1
        assert isinstance(result["elapsed_ms"], int)

    @pytest.mark.asyncio
    async def test_provider_errors_still_pass_through_untouched(self, call_tool):
        provider = StubProvider(error=ProviderError("rate_limit", "slow down", retriable=True))
        result = await call_tool(provider)()

        assert result["success"] is False
        assert result["error"]["type"] == "rate_limit"
        assert result["error"]["retriable"] is True
        assert result["elapsed_ms"] >= 0

    @pytest.mark.asyncio
    async def test_empty_summary_still_rejected_before_any_provider_call(self, call_tool):
        provider = StubProvider()
        result = await call_tool(provider)(summary="   ")

        assert result["error"]["type"] == "invalid_input"
        assert provider.started == 0


class TestDefaultMaxTokens:
    @pytest.mark.asyncio
    async def test_applied_when_caller_omits_max_tokens(self, call_tool):
        provider = StubProvider()
        await call_tool(provider, make_config(default_max_tokens=32000))()
        assert provider.requests[0].max_tokens == 32000

    @pytest.mark.asyncio
    async def test_caller_value_wins(self, call_tool):
        provider = StubProvider()
        await call_tool(provider, make_config(default_max_tokens=32000))(max_tokens=123)
        assert provider.requests[0].max_tokens == 123

    @pytest.mark.asyncio
    async def test_null_default_leaves_replies_unbounded(self, call_tool):
        provider = StubProvider()
        await call_tool(provider, make_config(default_max_tokens=None))()
        assert provider.requests[0].max_tokens is None

    @pytest.mark.asyncio
    async def test_explicit_zero_is_not_overridden_by_the_default(self, call_tool):
        """0 is falsy — the check must be `is not None`, not truthiness."""
        provider = StubProvider()
        await call_tool(provider, make_config(default_max_tokens=32000))(max_tokens=0)
        assert provider.requests[0].max_tokens == 0


class TestUncooperativeProviders:
    """`_run_bounded` exists because a coroutine may misbehave under
    cancellation. It must not then trust that same coroutine to exit."""

    @pytest.mark.asyncio
    async def test_provider_that_ignores_cancellation_still_times_out(self, monkeypatch):
        """Swallow the cancel and keep running: an unbounded teardown wait
        would hold us here forever and blow the budget entirely."""
        import llm_second_opinion.server as server_mod

        monkeypatch.setattr(server_mod, "CANCEL_GRACE_SECONDS", 0.05)
        entered = asyncio.Event()

        async def stubborn():
            entered.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(10)  # ignores the cancel and carries on

        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await server_mod._run_bounded(stubborn(), budget=0.05)
        took = time.monotonic() - started

        assert entered.is_set()
        assert took < 2.0, f"teardown was not bounded: {took:.2f}s"

    @pytest.mark.asyncio
    async def test_abandoned_task_is_logged(self, monkeypatch, caplog):
        import llm_second_opinion.server as server_mod

        monkeypatch.setattr(server_mod, "CANCEL_GRACE_SECONDS", 0.05)

        async def stubborn():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(10)

        log = logging.getLogger("test_abandon")
        with caplog.at_level(logging.WARNING, logger="test_abandon"):
            with pytest.raises(TimeoutError):
                await server_mod._run_bounded(stubborn(), budget=0.05, log=log)

        assert any("ignored cancellation" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_handler_timeout_still_returned_for_stubborn_provider(
        self, call_tool, monkeypatch
    ):
        """End to end: the tool result is still a structured timeout."""
        import llm_second_opinion.server as server_mod

        monkeypatch.setattr(server_mod, "CANCEL_GRACE_SECONDS", 0.05)

        class StubbornProvider(StubProvider):
            async def generate(self, req):
                self.started += 1
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    await asyncio.sleep(10)

        result = await call_tool(StubbornProvider(), make_config(budget=0.05))()
        assert result["success"] is False
        assert result["error"]["type"] == "timeout"


class TestOuterCancellation:
    """If the MCP client disconnects or the server shuts down mid-call, the
    upstream request must not keep running — it is billable and holds a
    socket nothing will read."""

    @pytest.mark.asyncio
    async def test_provider_task_is_cancelled_when_the_handler_is_cancelled(self):
        import llm_second_opinion.server as server_mod

        entered = asyncio.Event()
        observed = {"cancelled": False}

        async def provider_call():
            entered.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                observed["cancelled"] = True
                raise

        # A generous budget, so only the outer cancel can end this.
        outer = asyncio.ensure_future(
            server_mod._run_bounded(provider_call(), budget=30)
        )
        await entered.wait()
        outer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await outer
        await asyncio.sleep(0.05)  # let the cancellation reach the inner task

        assert observed["cancelled"], "upstream call leaked past handler cancellation"

    @pytest.mark.asyncio
    async def test_no_task_survives_handler_cancellation(self):
        import llm_second_opinion.server as server_mod

        entered = asyncio.Event()

        async def provider_call():
            entered.set()
            await asyncio.sleep(30)

        outer = asyncio.ensure_future(
            server_mod._run_bounded(provider_call(), budget=30)
        )
        await entered.wait()
        outer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await outer
        await asyncio.sleep(0.05)

        leftover = [
            t for t in asyncio.all_tasks()
            if not t.done() and t is not asyncio.current_task()
        ]
        assert not leftover, f"leaked tasks: {leftover}"


class TestStdoutIsReservedForTheTransport:
    """stdout carries the JSON-RPC stream; a stray write corrupts it and the
    client drops the connection."""

    @pytest.mark.asyncio
    async def test_handler_writes_nothing_to_stdout(self, call_tool, capsys):
        provider = StubProvider(delay=0)
        await call_tool(provider)()
        assert capsys.readouterr().out == ""

    @pytest.mark.asyncio
    async def test_timeout_path_writes_nothing_to_stdout(self, call_tool, capsys):
        provider = StubProvider(delay=10)
        await call_tool(provider, make_config(budget=0.05))()
        assert capsys.readouterr().out == ""

    def test_no_source_file_prints_to_stdout(self):
        """`print(...)` without file=sys.stderr anywhere in the package."""
        import pathlib

        import llm_second_opinion

        root = pathlib.Path(llm_second_opinion.__file__).parent
        offenders = []
        for path in root.rglob("*.py"):
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "print(" in stripped and "sys.stderr" not in stripped:
                    offenders.append(f"{path.name}:{lineno}: {stripped}")
                if "sys.stdout" in stripped:
                    offenders.append(f"{path.name}:{lineno}: {stripped}")
        assert not offenders, "writes to stdout: " + "; ".join(offenders)

    def test_logging_handler_targets_stderr(self):
        import logging
        import sys

        from llm_second_opinion.logging_setup import setup_logging

        logger = setup_logging("INFO")
        streams = [
            h.stream for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert streams, "expected a StreamHandler"
        assert all(s is sys.stderr for s in streams)
        assert logger.propagate is False, (
            "propagating to root could reach a stdout handler installed elsewhere"
        )


class TestServerSurvives:
    @pytest.mark.asyncio
    async def test_subsequent_calls_succeed_after_a_timeout(self, call_tool):
        """A timeout must not wedge the server for the next request."""
        provider = StubProvider(delay=10)
        call = call_tool(provider, make_config(budget=0.05))

        first = await call()
        assert first["error"]["type"] == "timeout"

        provider.delay = 0
        second = await call()
        assert second["success"] is True
        assert second["response"] == "stub answer"
