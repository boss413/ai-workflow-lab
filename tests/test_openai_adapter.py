"""
tests/test_openai_adapter.py

Unit tests for models/openai_adapter.py.
All OpenAI API calls are mocked — no network access or real credentials required.
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
from models.openai_adapter import OpenAIAdapter, _calculate_cost

# ---------------------------------------------------------------------------
# Helpers — fake OpenAI response object
# ---------------------------------------------------------------------------

def _make_openai_response(
    content: str = "Mocked response text.",
    prompt_tokens: int = 20,
    completion_tokens: int = 10,
) -> SimpleNamespace:
    """Build a minimal fake OpenAI ChatCompletion response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content))
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def _make_adapter(model_name: str = "gpt-4o-mini", **kwargs) -> OpenAIAdapter:
    """Construct an OpenAIAdapter with a dummy API key."""
    return OpenAIAdapter(model_name=model_name, api_key="sk-test-key", **kwargs)


def _patch_client(response: SimpleNamespace | None = None):
    """Patch _build_client to return a mock that yields the given response."""
    fake_response = response or _make_openai_response()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = fake_response
    return patch.object(OpenAIAdapter, "_build_client", return_value=mock_client)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_adapter_constructs_with_api_key():
    adapter = _make_adapter()
    assert adapter is not None


def test_adapter_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        OpenAIAdapter(model_name="gpt-4o-mini")


def test_adapter_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    adapter = OpenAIAdapter(model_name="gpt-4o-mini")
    assert adapter is not None


def test_adapter_explicit_key_overrides_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    adapter = OpenAIAdapter(model_name="gpt-4o-mini", api_key="sk-explicit")
    assert adapter is not None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def test_provider_is_openai():
    assert _make_adapter().provider == "openai"


def test_model_name_matches_constructor():
    assert _make_adapter("gpt-4o").model_name == "gpt-4o"


def test_default_model_name():
    adapter = OpenAIAdapter(api_key="sk-test")
    assert adapter.model_name == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# run() — response schema
# ---------------------------------------------------------------------------

def test_run_returns_model_response():
    with _patch_client():
        result = _make_adapter().run("Test prompt")
    assert isinstance(result, ModelResponse)


def test_run_response_text_matches_mock():
    fake = _make_openai_response(content="Hello from mock.")
    with _patch_client(fake):
        result = _make_adapter().run("Test prompt")
    assert result.text == "Hello from mock."


def test_run_response_text_is_string():
    with _patch_client():
        result = _make_adapter().run("Test prompt")
    assert isinstance(result.text, str)


def test_run_response_input_tokens_correct():
    fake = _make_openai_response(prompt_tokens=42)
    with _patch_client(fake):
        result = _make_adapter().run("Test prompt")
    assert result.input_tokens == 42


def test_run_response_output_tokens_correct():
    fake = _make_openai_response(completion_tokens=17)
    with _patch_client(fake):
        result = _make_adapter().run("Test prompt")
    assert result.output_tokens == 17


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
    fake = _make_openai_response(prompt_tokens=30, completion_tokens=15)
    with _patch_client(fake):
        result = _make_adapter().run("Prompt")
    assert result.total_tokens == 45


def test_run_zero_output_tokens():
    fake = _make_openai_response(prompt_tokens=10, completion_tokens=0)
    with _patch_client(fake):
        result = _make_adapter().run("Prompt")
    assert result.output_tokens == 0


# ---------------------------------------------------------------------------
# run() — cost calculation
# ---------------------------------------------------------------------------

def test_run_cost_reflects_token_counts():
    fake = _make_openai_response(prompt_tokens=1000, completion_tokens=1000)
    with _patch_client(fake):
        result = _make_adapter("gpt-4o-mini").run("Prompt")
    expected = _calculate_cost("gpt-4o-mini", 1000, 1000)
    assert abs(result.cost - expected) < 1e-9


def test_run_cost_zero_tokens():
    fake = _make_openai_response(prompt_tokens=0, completion_tokens=0)
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
    mock_client.chat.completions.create.side_effect = Exception("connection timeout")
    with patch.object(OpenAIAdapter, "_build_client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="OpenAI API call failed"):
            _make_adapter().run("Valid prompt")


def test_run_raises_on_malformed_response():
    malformed = SimpleNamespace(choices=[], usage=SimpleNamespace(
        prompt_tokens=5, completion_tokens=3
    ))
    with _patch_client(malformed):
        with pytest.raises(RuntimeError):
            _make_adapter().run("Valid prompt")


# ---------------------------------------------------------------------------
# run() — null content fallback
# ---------------------------------------------------------------------------

def test_run_handles_none_content():
    fake = _make_openai_response(content=None)  # type: ignore[arg-type]
    fake.choices[0].message.content = None
    with _patch_client(fake):
        result = _make_adapter().run("Prompt")
    assert result.text == ""


# ---------------------------------------------------------------------------
# _calculate_cost
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,input_t,output_t,expected", [
    ("gpt-4o-mini", 1000, 0,    0.00015),
    ("gpt-4o-mini", 0,    1000, 0.0006),
    ("gpt-4o-mini", 1000, 1000, 0.00075),
    ("gpt-4o",      1000, 1000, 0.02),
    ("gpt-4",       1000, 1000, 0.09),
])
def test_calculate_cost_known_models(model, input_t, output_t, expected):
    result = _calculate_cost(model, input_t, output_t)
    assert abs(result - expected) < 1e-9


def test_calculate_cost_unknown_model_uses_default():
    cost = _calculate_cost("some-future-model", 1000, 1000)
    assert cost > 0.0


def test_calculate_cost_zero_tokens():
    assert _calculate_cost("gpt-4o-mini", 0, 0) == 0.0


def test_calculate_cost_returns_float():
    assert isinstance(_calculate_cost("gpt-4o-mini", 100, 50), float)


# ---------------------------------------------------------------------------
# API call parameters
# ---------------------------------------------------------------------------

def test_run_passes_model_name_to_api():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response()
    with patch.object(OpenAIAdapter, "_build_client", return_value=mock_client):
        _make_adapter("gpt-4o").run("Prompt")
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"


def test_run_passes_temperature_to_api():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response()
    with patch.object(OpenAIAdapter, "_build_client", return_value=mock_client):
        OpenAIAdapter(api_key="sk-test", temperature=0.7).run("Prompt")
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["temperature"] == 0.7


def test_run_passes_prompt_as_user_message():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response()
    with patch.object(OpenAIAdapter, "_build_client", return_value=mock_client):
        _make_adapter().run("My test prompt")
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "My test prompt"