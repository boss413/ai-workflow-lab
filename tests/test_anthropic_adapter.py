"""
tests/test_anthropic_adapter.py

Unit tests for models/anthropic_adapter.py.
All Anthropic API calls are mocked — no network access or real credentials required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base_model import ModelResponse
from models.anthropic_adapter import AnthropicAdapter, _calculate_cost


# ---------------------------------------------------------------------------
# Helpers — fake Anthropic response object
# ---------------------------------------------------------------------------

def _make_anthropic_response(
    text: str = "Mocked Claude response.",
    input_tokens: int = 20,
    output_tokens: int = 10,
) -> SimpleNamespace:
    """Build a minimal fake Anthropic Message response."""
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _make_adapter(model: str = "claude-sonnet-4-5", **kwargs) -> AnthropicAdapter:
    return AnthropicAdapter(model_name=model, api_key="sk-ant-test", **kwargs)


def _patch_client(response: SimpleNamespace | None = None):
    """Patch _build_client so no real Anthropic client is created."""
    fake = response or _make_anthropic_response()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = fake
    return patch.object(AnthropicAdapter, "_build_client", return_value=mock_client)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_adapter_constructs_with_api_key():
    assert _make_adapter() is not None


def test_adapter_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
        AnthropicAdapter(model_name="claude-sonnet-4-5")


def test_adapter_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    assert AnthropicAdapter(model_name="claude-sonnet-4-5") is not None


def test_adapter_explicit_key_overrides_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    assert AnthropicAdapter(model_name="claude-sonnet-4-5", api_key="sk-ant-explicit") is not None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def test_provider_is_anthropic():
    assert _make_adapter().provider == "anthropic"


def test_model_name_matches_constructor():
    assert _make_adapter("claude-opus-4").model_name == "claude-opus-4"


def test_default_model_name():
    assert AnthropicAdapter(api_key="sk-ant-test").model_name == "claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# run() — response schema
# ---------------------------------------------------------------------------

def test_run_returns_model_response():
    with _patch_client():
        result = _make_adapter().run("Test prompt")
    assert isinstance(result, ModelResponse)


def test_run_response_text_matches_mock():
    fake = _make_anthropic_response(text="Hello from Claude.")
    with _patch_client(fake):
        result = _make_adapter().run("Test prompt")
    assert result.text == "Hello from Claude."


def test_run_response_text_is_string():
    with _patch_client():
        result = _make_adapter().run("Test prompt")
    assert isinstance(result.text, str)


def test_run_response_input_tokens_correct():
    fake = _make_anthropic_response(input_tokens=55)
    with _patch_client(fake):
        result = _make_adapter().run("Test prompt")
    assert result.input_tokens == 55


def test_run_response_output_tokens_correct():
    fake = _make_anthropic_response(output_tokens=22)
    with _patch_client(fake):
        result = _make_adapter().run("Test prompt")
    assert result.output_tokens == 22


def test_run_response_cost_is_float():
    with _patch_client():
        result = _make_adapter().run("Test prompt")
    assert isinstance(result.cost, float)


def test_run_response_cost_non_negative():
    with _patch_client():
        result = _make_adapter().run("Test prompt")
    assert result.cost >= 0.0


def test_run_response_latency_non_negative():
    with _patch_client():
        result = _make_adapter().run("Test prompt")
    assert result.latency >= 0.0


def test_run_response_latency_is_float():
    with _patch_client():
        result = _make_adapter().run("Test prompt")
    assert isinstance(result.latency, float)


# ---------------------------------------------------------------------------
# run() — token accounting
# ---------------------------------------------------------------------------

def test_run_total_tokens_correct():
    fake = _make_anthropic_response(input_tokens=30, output_tokens=15)
    with _patch_client(fake):
        result = _make_adapter().run("Prompt")
    assert result.total_tokens == 45


def test_run_zero_output_tokens():
    fake = _make_anthropic_response(input_tokens=10, output_tokens=0)
    with _patch_client(fake):
        result = _make_adapter().run("Prompt")
    assert result.output_tokens == 0


# ---------------------------------------------------------------------------
# run() — cost calculation
# ---------------------------------------------------------------------------

def test_run_cost_reflects_token_counts():
    fake = _make_anthropic_response(input_tokens=1000, output_tokens=1000)
    with _patch_client(fake):
        result = _make_adapter("claude-sonnet-4-5").run("Prompt")
    expected = _calculate_cost("claude-sonnet-4-5", 1000, 1000)
    assert abs(result.cost - expected) < 1e-9


def test_run_cost_zero_tokens():
    fake = _make_anthropic_response(input_tokens=0, output_tokens=0)
    with _patch_client(fake):
        result = _make_adapter().run("Prompt")
    assert result.cost == 0.0


# ---------------------------------------------------------------------------
# run() — prompt validation
# ---------------------------------------------------------------------------

def test_run_raises_on_empty_prompt():
    with pytest.raises(ValueError, match="empty"):
        _make_adapter().run("")


def test_run_raises_on_whitespace_prompt():
    with pytest.raises(ValueError, match="empty"):
        _make_adapter().run("   ")


# ---------------------------------------------------------------------------
# run() — API error handling
# ---------------------------------------------------------------------------

def test_run_raises_runtime_error_on_api_failure():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("connection reset")
    with patch.object(AnthropicAdapter, "_build_client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="Anthropic API call failed"):
            _make_adapter().run("Valid prompt")


def test_run_raises_on_malformed_response():
    malformed = SimpleNamespace(
        content=[],
        usage=SimpleNamespace(input_tokens=None, output_tokens=None),
    )
    with _patch_client(malformed):
        with pytest.raises(RuntimeError):
            _make_adapter().run("Valid prompt")


# ---------------------------------------------------------------------------
# run() — empty content list fallback
# ---------------------------------------------------------------------------

def test_run_handles_empty_content_list():
    fake = SimpleNamespace(
        content=[],
        usage=SimpleNamespace(input_tokens=5, output_tokens=0),
    )
    with _patch_client(fake):
        result = _make_adapter().run("Prompt")
    assert result.text == ""


# ---------------------------------------------------------------------------
# _calculate_cost
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,input_t,output_t,expected", [
    ("claude-sonnet-4-5",  1000, 0,    0.003),
    ("claude-sonnet-4-5",  0,    1000, 0.015),
    ("claude-sonnet-4-5",  1000, 1000, 0.018),
    ("claude-opus-4-5",    1000, 1000, 0.09),
    ("claude-haiku-4-5",   1000, 1000, 0.0048),
    ("claude-3-haiku-20240307", 1000, 1000, 0.00150),
])
def test_calculate_cost_known_models(model, input_t, output_t, expected):
    result = _calculate_cost(model, input_t, output_t)
    assert abs(result - expected) < 1e-9


def test_calculate_cost_unknown_model_uses_default():
    cost = _calculate_cost("claude-future-model", 1000, 1000)
    assert cost > 0.0


def test_calculate_cost_zero_tokens():
    assert _calculate_cost("claude-sonnet-4-5", 0, 0) == 0.0


def test_calculate_cost_returns_float():
    assert isinstance(_calculate_cost("claude-sonnet-4-5", 100, 50), float)


# ---------------------------------------------------------------------------
# API call parameters
# ---------------------------------------------------------------------------

def test_run_passes_model_name_to_api():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_anthropic_response()
    with patch.object(AnthropicAdapter, "_build_client", return_value=mock_client):
        _make_adapter("claude-opus-4").run("Prompt")
    kw = mock_client.messages.create.call_args[1]
    assert kw["model"] == "claude-opus-4"


def test_run_passes_max_tokens_to_api():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_anthropic_response()
    with patch.object(AnthropicAdapter, "_build_client", return_value=mock_client):
        AnthropicAdapter(api_key="sk-ant-test", max_tokens=512).run("Prompt")
    kw = mock_client.messages.create.call_args[1]
    assert kw["max_tokens"] == 512


def test_run_passes_temperature_to_api():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_anthropic_response()
    with patch.object(AnthropicAdapter, "_build_client", return_value=mock_client):
        AnthropicAdapter(api_key="sk-ant-test", temperature=0.5).run("Prompt")
    kw = mock_client.messages.create.call_args[1]
    assert kw["temperature"] == 0.5


def test_run_passes_prompt_as_user_message():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_anthropic_response()
    with patch.object(AnthropicAdapter, "_build_client", return_value=mock_client):
        _make_adapter().run("My Anthropic prompt")
    kw = mock_client.messages.create.call_args[1]
    messages = kw["messages"]
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "My Anthropic prompt"