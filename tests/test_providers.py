"""Tests for the multi-provider factory (parsing logic only; API clients are
mocked so no live network calls happen)."""
from unittest.mock import MagicMock

import pytest

from promptraid.harness import providers as providers_mod
from promptraid.harness.providers import (
    AnthropicProvider,
    GeminiProvider,
    GroqProvider,
    OpenRouterProvider,
    get_provider,
)


@pytest.fixture(autouse=True)
def _mock_provider_sdks(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    monkeypatch.setattr(providers_mod, "anthropic", MagicMock())
    monkeypatch.setattr(providers_mod, "OpenAI", MagicMock())
    monkeypatch.setattr(providers_mod, "genai", MagicMock())


def test_get_provider_parses_anthropic_target():
    provider = get_provider("anthropic:claude-sonnet-4-6")
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-sonnet-4-6"


def test_get_provider_parses_gemini_target():
    provider = get_provider("gemini:gemini-2.0-flash")
    assert isinstance(provider, GeminiProvider)
    assert provider.model == "gemini-2.0-flash"


def test_get_provider_parses_groq_target():
    provider = get_provider("groq:llama-3.3-70b-versatile")
    assert isinstance(provider, GroqProvider)
    assert provider.model == "llama-3.3-70b-versatile"


def test_get_provider_parses_openrouter_target_with_slash_in_model():
    provider = get_provider("openrouter:qwen/qwen-2.5-72b-instruct")
    assert isinstance(provider, OpenRouterProvider)
    assert provider.model == "qwen/qwen-2.5-72b-instruct"


def test_get_provider_unknown_provider_raises():
    with pytest.raises(ValueError):
        get_provider("notreal:some-model")


def test_get_provider_missing_colon_raises():
    with pytest.raises(ValueError):
        get_provider("claude-sonnet-4-6")


def test_get_provider_empty_model_raises():
    with pytest.raises(ValueError):
        get_provider("anthropic:")


def test_get_provider_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        get_provider("groq:llama-3.3-70b-versatile")
