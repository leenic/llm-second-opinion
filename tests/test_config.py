"""Config loading tests, focused on the default-model fallback.

The defaults are the path nobody exercises locally — every developer has a
`config.json` that overrides them — which is how they drifted a full model
generation behind. These pin the fallback behaviour itself.
"""

from __future__ import annotations

import json

import pytest

from llm_second_opinion import config as config_mod
from llm_second_opinion.config import (
    CLIENT_HARD_CAP_SECONDS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODELS,
    DEFAULT_REQUEST_BUDGET_SECONDS,
    ConfigError,
    load_config,
)


@pytest.fixture
def config_env(tmp_path, monkeypatch):
    """Point config loading at a temp file and clear all env overrides."""
    for key in list(config_mod.os.environ):
        if key.startswith(config_mod.ENV_PREFIX):
            monkeypatch.delenv(key, raising=False)

    def _write(payload):
        path = tmp_path / "config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setenv(f"{config_mod.ENV_PREFIX}CONFIG", str(path))
        return path

    return _write


class TestDefaultModels:
    def test_every_provider_has_a_default(self):
        assert set(DEFAULT_MODELS) == {"openai", "gemini", "grok"}
        assert all(v and isinstance(v, str) for v in DEFAULT_MODELS.values())

    def test_omitted_model_falls_back_to_default(self, config_env):
        config_env({"providers": {"openai": {"api_key": "sk-test"}}})
        cfg = load_config()
        assert cfg.providers["openai"].model == DEFAULT_MODELS["openai"]

    def test_provider_absent_entirely_still_gets_default(self, config_env):
        config_env({"providers": {}})
        cfg = load_config()
        for name, default in DEFAULT_MODELS.items():
            assert cfg.providers[name].model == default
            assert cfg.providers[name].api_key is None

    def test_file_model_beats_default(self, config_env):
        config_env({"providers": {"grok": {"api_key": "x", "model": "grok-custom"}}})
        assert load_config().providers["grok"].model == "grok-custom"

    def test_env_model_beats_file(self, config_env, monkeypatch):
        config_env({"providers": {"grok": {"api_key": "x", "model": "grok-from-file"}}})
        monkeypatch.setenv(f"{config_mod.ENV_PREFIX}GROK_MODEL", "grok-from-env")
        assert load_config().providers["grok"].model == "grok-from-env"


class TestApiKeys:
    @pytest.mark.parametrize("key", ["", "   ", "sk-REPLACE-ME"])
    def test_placeholder_keys_count_as_unset(self, config_env, key):
        """config.example.json ships REPLACE-ME placeholders."""
        config_env({"providers": {"openai": {"api_key": key}}})
        assert load_config().providers["openai"].api_key is None

    def test_env_key_beats_file(self, config_env, monkeypatch):
        config_env({"providers": {"openai": {"api_key": "sk-file"}}})
        monkeypatch.setenv(f"{config_mod.ENV_PREFIX}OPENAI_API_KEY", "sk-env")
        assert load_config().providers["openai"].api_key == "sk-env"


class TestRequestBudget:
    def test_default_leaves_headroom_under_the_client_cap(self):
        """Desktop cancels at 240s and discards the result; our own error is
        only useful if it fires first."""
        assert DEFAULT_REQUEST_BUDGET_SECONDS < CLIENT_HARD_CAP_SECONDS
        assert CLIENT_HARD_CAP_SECONDS - DEFAULT_REQUEST_BUDGET_SECONDS >= 30

    def test_absent_key_uses_the_default(self, config_env):
        config_env({"providers": {}})
        cfg = load_config()
        assert cfg.request_budget_seconds == DEFAULT_REQUEST_BUDGET_SECONDS
        assert cfg.warnings == []

    def test_explicit_value_is_used(self, config_env):
        config_env({"providers": {}, "request_budget_seconds": 90})
        assert load_config().request_budget_seconds == 90.0

    def test_legacy_timeout_seconds_still_works(self, config_env):
        """An existing deployment must not silently lose its tuned value."""
        config_env({"providers": {}, "timeout_seconds": 150})
        cfg = load_config()
        assert cfg.request_budget_seconds == 150.0
        assert any("deprecated" in w for w in cfg.warnings)

    def test_new_key_wins_over_legacy_and_the_clash_is_reported(self, config_env):
        config_env({
            "providers": {},
            "request_budget_seconds": 90,
            "timeout_seconds": 150,
        })
        cfg = load_config()
        assert cfg.request_budget_seconds == 90.0
        assert any("both" in w for w in cfg.warnings)

    def test_budget_at_or_above_the_cap_is_warned_about(self, config_env):
        config_env({"providers": {}, "request_budget_seconds": 240})
        cfg = load_config()
        assert cfg.request_budget_seconds == 240.0  # honoured, not clamped
        assert any("240" in w for w in cfg.warnings)

    def test_env_override(self, config_env, monkeypatch):
        config_env({"providers": {}, "request_budget_seconds": 90})
        monkeypatch.setenv(f"{config_mod.ENV_PREFIX}REQUEST_BUDGET", "45")
        assert load_config().request_budget_seconds == 45.0

    def test_legacy_env_override_still_works(self, config_env, monkeypatch):
        config_env({"providers": {}})
        monkeypatch.setenv(f"{config_mod.ENV_PREFIX}TIMEOUT", "45")
        assert load_config().request_budget_seconds == 45.0

    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_is_rejected(self, config_env, value):
        config_env({"providers": {}, "request_budget_seconds": value})
        with pytest.raises(ConfigError, match="request_budget_seconds"):
            load_config()

    def test_unparseable_is_rejected(self, config_env):
        config_env({"providers": {}, "request_budget_seconds": "soon"})
        with pytest.raises(ConfigError, match="request_budget_seconds"):
            load_config()

    def test_timeout_seconds_alias_reads_the_budget(self, config_env):
        config_env({"providers": {}, "request_budget_seconds": 90})
        cfg = load_config()
        assert cfg.timeout_seconds == cfg.request_budget_seconds


class TestJobBudget:
    """`job_budget_seconds` bounds a background job, not a tool call, so it
    lives outside the 240s client cap entirely."""

    def test_default_is_well_above_the_per_call_budget(self):
        from llm_second_opinion.config import DEFAULT_JOB_BUDGET_SECONDS

        assert DEFAULT_JOB_BUDGET_SECONDS == 900.0
        assert DEFAULT_JOB_BUDGET_SECONDS > CLIENT_HARD_CAP_SECONDS

    def test_absent_key_uses_the_default(self, config_env):
        from llm_second_opinion.config import DEFAULT_JOB_BUDGET_SECONDS

        config_env({"providers": {}})
        cfg = load_config()
        assert cfg.job_budget_seconds == DEFAULT_JOB_BUDGET_SECONDS
        assert cfg.warnings == []

    def test_explicit_value_is_used(self, config_env):
        config_env({"providers": {}, "job_budget_seconds": 600})
        assert load_config().job_budget_seconds == 600.0

    def test_env_override(self, config_env, monkeypatch):
        config_env({"providers": {}, "job_budget_seconds": 600})
        monkeypatch.setenv(f"{config_mod.ENV_PREFIX}JOB_BUDGET", "1200")
        assert load_config().job_budget_seconds == 1200.0

    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_is_rejected(self, config_env, value):
        config_env({"providers": {}, "job_budget_seconds": value})
        with pytest.raises(ConfigError, match="job_budget_seconds"):
            load_config()

    def test_unparseable_is_rejected(self, config_env):
        config_env({"providers": {}, "job_budget_seconds": "later"})
        with pytest.raises(ConfigError, match="job_budget_seconds"):
            load_config()

    def test_very_long_budget_is_warned_about_not_clamped(self, config_env):
        config_env({"providers": {}, "job_budget_seconds": 7200})
        cfg = load_config()
        assert cfg.job_budget_seconds == 7200.0
        assert any("job_budget_seconds" in w for w in cfg.warnings)

    def test_threshold_itself_is_not_warned_about(self, config_env):
        config_env({"providers": {}, "job_budget_seconds": 3600})
        assert load_config().warnings == []

    def test_request_budget_is_independent(self, config_env):
        """The per-call budget still governs only the synchronous tool."""
        config_env({"providers": {}, "request_budget_seconds": 90, "job_budget_seconds": 600})
        cfg = load_config()
        assert cfg.request_budget_seconds == 90.0
        assert cfg.job_budget_seconds == 600.0


class TestDefaultMaxTokens:
    def test_absent_key_uses_the_default(self, config_env):
        config_env({"providers": {}})
        assert load_config().default_max_tokens == DEFAULT_MAX_TOKENS

    def test_explicit_value_is_used(self, config_env):
        config_env({"providers": {}, "default_max_tokens": 1234})
        assert load_config().default_max_tokens == 1234

    def test_explicit_null_disables_the_cap(self, config_env):
        """Distinct from an absent key, which falls back to the default."""
        config_env({"providers": {}, "default_max_tokens": None})
        assert load_config().default_max_tokens is None

    @pytest.mark.parametrize("value", [0, -5])
    def test_non_positive_is_rejected(self, config_env, value):
        config_env({"providers": {}, "default_max_tokens": value})
        with pytest.raises(ConfigError, match="default_max_tokens"):
            load_config()

    def test_unparseable_is_rejected(self, config_env):
        config_env({"providers": {}, "default_max_tokens": "lots"})
        with pytest.raises(ConfigError, match="default_max_tokens"):
            load_config()

    def test_env_override(self, config_env, monkeypatch):
        config_env({"providers": {}, "default_max_tokens": 1234})
        monkeypatch.setenv(f"{config_mod.ENV_PREFIX}DEFAULT_MAX_TOKENS", "555")
        assert load_config().default_max_tokens == 555

    @pytest.mark.parametrize("value", ["none", "null", ""])
    def test_env_can_disable_the_cap(self, config_env, monkeypatch, value):
        config_env({"providers": {}, "default_max_tokens": 1234})
        monkeypatch.setenv(f"{config_mod.ENV_PREFIX}DEFAULT_MAX_TOKENS", value)
        assert load_config().default_max_tokens is None


class TestValidation:
    def test_bad_reasoning_effort_is_rejected(self, config_env):
        config_env({"providers": {"openai": {"api_key": "x", "reasoning_effort": "max"}}})
        with pytest.raises(ConfigError, match="reasoning_effort"):
            load_config()

    def test_reasoning_effort_is_normalised(self, config_env):
        config_env({"providers": {"openai": {"api_key": "x", "reasoning_effort": " HIGH "}}})
        assert load_config().providers["openai"].reasoning_effort == "high"

    def test_non_positive_budget_is_rejected(self, config_env):
        config_env({"providers": {}, "request_budget_seconds": 0})
        with pytest.raises(ConfigError, match="request_budget_seconds"):
            load_config()

    def test_providers_must_be_an_object(self, config_env):
        config_env({"providers": []})
        with pytest.raises(ConfigError, match="providers"):
            load_config()

    def test_web_search_defaults_off(self, config_env):
        config_env({"providers": {"gemini": {"api_key": "x"}}})
        assert load_config().providers["gemini"].web_search is False

    @pytest.mark.parametrize(
        "value,expected",
        [("1", True), ("true", True), ("YES", True), ("0", False), ("off", False)],
    )
    def test_web_search_env_override(self, config_env, monkeypatch, value, expected):
        config_env({"providers": {"gemini": {"api_key": "x", "web_search": not expected}}})
        monkeypatch.setenv(f"{config_mod.ENV_PREFIX}GEMINI_WEB_SEARCH", value)
        assert load_config().providers["gemini"].web_search is expected
