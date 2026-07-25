"""Config loading tests, focused on the default-model fallback.

The defaults are the path nobody exercises locally — every developer has a
`config.json` that overrides them — which is how they drifted a full model
generation behind. These pin the fallback behaviour itself.
"""

from __future__ import annotations

import json

import pytest

from llm_second_opinion import config as config_mod
from llm_second_opinion.config import DEFAULT_MODELS, ConfigError, load_config


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


class TestValidation:
    def test_bad_reasoning_effort_is_rejected(self, config_env):
        config_env({"providers": {"openai": {"api_key": "x", "reasoning_effort": "max"}}})
        with pytest.raises(ConfigError, match="reasoning_effort"):
            load_config()

    def test_reasoning_effort_is_normalised(self, config_env):
        config_env({"providers": {"openai": {"api_key": "x", "reasoning_effort": " HIGH "}}})
        assert load_config().providers["openai"].reasoning_effort == "high"

    def test_non_positive_timeout_is_rejected(self, config_env):
        config_env({"providers": {}, "timeout_seconds": 0})
        with pytest.raises(ConfigError, match="timeout_seconds"):
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
