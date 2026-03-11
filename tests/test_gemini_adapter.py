"""
tests/test_gemini_adapter.py

Unit tests for models/gemini_adapter.py.
All Gemini API calls are mocked — no network access or real credentials required.
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
from models.gemini_adapter import GeminiAdapter, _calculate_cost


# ---------------------------------------------------------------------------
# Helpers — fake Gemini response object
# ---------------------------------------------------------------------------

def _make_gemini_response(
    text: str = "Mocked Gemini response.",
    prompt_token_count: int = 20,
    candidates_token_count: int = 10,
) -> SimpleNamespace:
    """Build a minimal fake Gemini GenerateContentResponse."""
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_token_count,
            candidates_token_count=candidates_token_count,
        ),
    )


def _make_adapter(model: str = "gemini-2.5-flash", **kwargs) -> GeminiAdapter:
    return GeminiAdapter(model_name=model, api_key="test-gemini-key", **kwargs)


def _patch_client(response: SimpleNamespace | None = None):
    """Patch _build_client and _build_generation_config for offline testing."""
    fake = response or _make_gemini_response()
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = fake
    return patch.object(GeminiAdapter, "_build_client", return_value=mock_client)


def _patch_generation_config():
    """Patch _build_generation_config to return a plain dict."""
    return patch.object(GeminiAdapter, "_build_generation_config", return_value={})


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_adapter_constructs_with_api_key():
    assert _make_adapter() is not None


def test_adapter_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
        GeminiAdapter(model_name="gemini-2.5-flash")


def test_adapter_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key-from-env")
    assert GeminiAdapter(model_name="gemini-2.5-flash") is not None


def test_adapter_explicit_key_overrides_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key-from-env")
    assert GeminiAdapter(model_name="gemini-2.5-flash", api_key="key-explicit") is not None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def test_provider_is_google():
    assert _make_adapter().provider == "google"


def test_model_name_matches_constructor():
    assert _make_adapter("gemini-2.5-pro").model_name == "gemini-2.5-pro"


def test_default_model_name():
    assert GeminiAdapter(api_key="test-key").model_name == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# run() — response schema
# ---------------------------------------------------------------------------

def test_run_returns_model_response():
    with _patch_client(), _patch_generation_config():
        result = _make_adapter().run("Test prompt")
    assert isinstance(result, ModelResponse)


def test_run_response_text_matches_mock():
    fake = _make_gemini_response(text="Hello from Gemini.")
    with _patch_client(fake), _patch_generation_config():
        result = _make_adapter().run("Test prompt")
    assert result.text == "Hello from Gemini."


def test_run_response_text_is_string():
    with _patch_client(), _patch_generation_config():
        result = _make_adapter().run("Test prompt")
    assert isinstance(result.text, str)


def test_run_response_input_tokens_correct():
    fake = _make_gemini_response(prompt_token_count=77)
    with _patch_client(fake), _patch_generation_config():
        result = _make_adapter().run("Test prompt")
    assert result.input_tokens == 77


def test_run_response_output_tokens_correct():
    fake = _make_gemini_response(candidates_token_count=33)
    with _patch_client(fake), _patch_generation_config():
        result = _make_adapter().run("Test prompt")
    assert result.output_tokens == 33


def test_run_response_cost_is_float():
    with _patch_client(), _patch_generation_config():
        result = _make_adapter().run("Test prompt")
    assert isinstance(result.cost, float)


def test_run_response_cost_non_negative():
    with _patch_client(), _patch_generation_config():
        result = _make_adapter().run("Test prompt")
    assert result.cost >= 0.0


def test_run_response_latency_non_negative():
    with _patch_client(), _patch_generation_config():
        result = _make_adapter().run("Test prompt")
    assert result.latency >= 0.0


def test_run_response_latency_is_float():
    with _patch_client(), _patch_generation_config():
        result = _make_adapter().run("Test prompt")
    assert isinstance(result.latency, float)


# ---------------------------------------------------------------------------
# run() — token accounting
# ---------------------------------------------------------------------------

def test_run_total_tokens_correct():
    fake = _make_gemini_response(prompt_token_count=40, candidates_token_count=20)
    with _patch_client(fake), _patch_generation_config():
        result = _make_adapter().run("Prompt")
    assert result.total_tokens == 60


def test_run_zero_output_tokens():
    fake = _make_gemini_response(prompt_token_count=10, candidates_token_count=0)
    with _patch_client(fake), _patch_generation_config():
        result = _make_adapter().run("Prompt")
    assert result.output_tokens == 0


def test_run_none_token_counts_default_to_zero():
    fake = SimpleNamespace(
        text="Some text",
        usage_metadata=SimpleNamespace(
            prompt_token_count=None,
            candidates_token_count=None,
        ),
    )
    with _patch_client(fake), _patch_generation_config():
        result = _make_adapter().run("Prompt")
    assert result.input_tokens == 0
    assert result.output_tokens == 0


# ---------------------------------------------------------------------------
# run() — cost calculation
# ---------------------------------------------------------------------------

def test_run_cost_reflects_token_counts():
    fake = _make_gemini_response(prompt_token_count=1000, candidates_token_count=1000)
    with _patch_client(fake), _patch_generation_config():
        result = _make_adapter("gemini-2.5-flash").run("Prompt")
    expected = _calculate_cost("gemini-2.5-flash", 1000, 1000)
    assert abs(result.cost - expected) < 1e-9


def test_run_cost_zero_tokens():
    fake = _make_gemini_response(prompt_token_count=0, candidates_token_count=0)
    with _patch_client(fake), _patch_generation_config():
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
    mock_client.models.generate_content.side_effect = Exception("quota exceeded")
    with patch.object(GeminiAdapter, "_build_client", return_value=mock_client), \
         _patch_generation_config():
        with pytest.raises(RuntimeError, match="Gemini API call failed"):
            _make_adapter().run("Valid prompt")


def test_run_raises_on_malformed_response():
    malformed = SimpleNamespace(text="ok", usage_metadata=None)
    with _patch_client(malformed), _patch_generation_config():
        with pytest.raises(RuntimeError, match="Unexpected Gemini response"):
            _make_adapter().run("Valid prompt")


# ---------------------------------------------------------------------------
# run() — None text fallback
# ---------------------------------------------------------------------------

def test_run_handles_none_text():
    fake = _make_gemini_response(text=None)  # type: ignore[arg-type]
    fake.text = None
    with _patch_client(fake), _patch_generation_config():
        result = _make_adapter().run("Prompt")
    assert result.text == ""


# ---------------------------------------------------------------------------
# _calculate_cost
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,input_t,output_t,expected", [
    ("gemini-2.5-flash",  1000, 0,    0.000075),
    ("gemini-2.5-flash",  0,    1000, 0.0003),
    ("gemini-2.5-flash",  1000, 1000, 0.000375),
    ("gemini-2.5-pro",    1000, 1000, 0.01125),
    ("gemini-1.5-pro",    1000, 1000, 0.00625),
    ("gemini-1.5-flash",  1000, 1000, 0.000375),
])
def test_calculate_cost_known_models(model, input_t, output_t, expected):
    result = _calculate_cost(model, input_t, output_t)
    assert abs(result - expected) < 1e-9


def test_calculate_cost_unknown_model_uses_default():
    cost = _calculate_cost("gemini-future-model", 1000, 1000)
    assert cost > 0.0


def test_calculate_cost_zero_tokens():
    assert _calculate_cost("gemini-2.5-flash", 0, 0) == 0.0


def test_calculate_cost_returns_float():
    assert isinstance(_calculate_cost("gemini-2.5-flash", 100, 50), float)


# ---------------------------------------------------------------------------
# API call parameters
# ---------------------------------------------------------------------------

def test_run_passes_model_name_to_api():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_gemini_response()
    with patch.object(GeminiAdapter, "_build_client", return_value=mock_client), \
         _patch_generation_config():
        _make_adapter("gemini-2.5-pro").run("Prompt")
    kw = mock_client.models.generate_content.call_args[1]
    assert kw["model"] == "gemini-2.5-pro"


def test_run_passes_prompt_as_contents():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_gemini_response()
    with patch.object(GeminiAdapter, "_build_client", return_value=mock_client), \
         _patch_generation_config():
        _make_adapter().run("My Gemini prompt")
    kw = mock_client.models.generate_content.call_args[1]
    assert kw["contents"] == "My Gemini prompt"


def test_run_passes_generation_config():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_gemini_response()
    sentinel_config = object()
    with patch.object(GeminiAdapter, "_build_client", return_value=mock_client), \
         patch.object(GeminiAdapter, "_build_generation_config", return_value=sentinel_config):
        _make_adapter().run("Prompt")
    kw = mock_client.models.generate_content.call_args[1]
    assert kw["config"] is sentinel_config