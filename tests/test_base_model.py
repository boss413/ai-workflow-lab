"""
tests/test_base_model.py

Unit tests for models/base_model.py.
Covers ModelResponse validation, BaseModel interface enforcement,
_validate_prompt, _measure_latency, and to_dict serialization.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base_model import BaseModel, ModelResponse, _LatencyTimer


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing
# ---------------------------------------------------------------------------

class _EchoModel(BaseModel):
    """Concrete subclass that echoes the prompt as the response text."""

    @property
    def model_name(self) -> str:
        return "echo-1"

    @property
    def provider(self) -> str:
        return "test"

    def run(self, prompt: str) -> ModelResponse:
        self._validate_prompt(prompt)
        with self._measure_latency() as timer:
            text = f"Echo: {prompt}"
        return ModelResponse(
            text=text,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            cost=0.0,
            latency=timer.elapsed,
        )


class _AlwaysFailModel(BaseModel):
    """Concrete subclass that always raises RuntimeError."""

    def run(self, prompt: str) -> ModelResponse:
        self._validate_prompt(prompt)
        raise RuntimeError("Simulated API failure.")


# ---------------------------------------------------------------------------
# BaseModel — abstract enforcement
# ---------------------------------------------------------------------------

def test_cannot_instantiate_base_model_directly():
    with pytest.raises(TypeError):
        BaseModel()  # type: ignore[abstract]


def test_concrete_subclass_without_run_raises():
    class _Incomplete(BaseModel):
        pass

    with pytest.raises(TypeError):
        _Incomplete()  # type: ignore[abstract]


def test_concrete_subclass_instantiates_successfully():
    model = _EchoModel()
    assert model is not None


# ---------------------------------------------------------------------------
# BaseModel.run — happy path
# ---------------------------------------------------------------------------

def test_run_returns_model_response():
    model = _EchoModel()
    result = model.run("Hello world")
    assert isinstance(result, ModelResponse)


def test_run_response_text_is_string():
    result = _EchoModel().run("test prompt")
    assert isinstance(result.text, str)


def test_run_response_text_non_empty():
    result = _EchoModel().run("test prompt")
    assert len(result.text) > 0


def test_run_response_input_tokens_non_negative():
    result = _EchoModel().run("four word test prompt")
    assert result.input_tokens >= 0


def test_run_response_output_tokens_non_negative():
    result = _EchoModel().run("test")
    assert result.output_tokens >= 0


def test_run_response_cost_non_negative():
    result = _EchoModel().run("test")
    assert result.cost >= 0.0


def test_run_response_latency_non_negative():
    result = _EchoModel().run("test")
    assert result.latency >= 0.0


# ---------------------------------------------------------------------------
# BaseModel._validate_prompt
# ---------------------------------------------------------------------------

def test_validate_prompt_raises_on_empty_string():
    model = _EchoModel()
    with pytest.raises(ValueError, match="empty"):
        model.run("")


def test_validate_prompt_raises_on_whitespace_only():
    model = _EchoModel()
    with pytest.raises(ValueError, match="empty"):
        model.run("   ")


def test_validate_prompt_raises_on_non_string():
    model = _EchoModel()
    with pytest.raises(ValueError, match="str"):
        model._validate_prompt(123)  # type: ignore[arg-type]


def test_validate_prompt_passes_with_valid_prompt():
    model = _EchoModel()
    model._validate_prompt("A valid prompt.")  # should not raise


# ---------------------------------------------------------------------------
# BaseModel.provider and model_name
# ---------------------------------------------------------------------------

def test_model_name_property():
    assert _EchoModel().model_name == "echo-1"


def test_provider_property():
    assert _EchoModel().provider == "test"


def test_default_provider_is_unknown():
    class _MinimalModel(BaseModel):
        def run(self, prompt: str) -> ModelResponse:
            self._validate_prompt(prompt)
            return ModelResponse("x", 1, 1, 0.0, 0.1)

    assert _MinimalModel().provider == "unknown"


def test_default_model_name_is_class_name():
    class _MyModel(BaseModel):
        def run(self, prompt: str) -> ModelResponse:
            self._validate_prompt(prompt)
            return ModelResponse("x", 1, 1, 0.0, 0.1)

    assert _MyModel().model_name == "_MyModel"


# ---------------------------------------------------------------------------
# BaseModel._measure_latency
# ---------------------------------------------------------------------------

def test_measure_latency_returns_positive_elapsed():
    model = _EchoModel()
    with model._measure_latency() as timer:
        time.sleep(0.01)
    assert timer.elapsed >= 0.005


def test_measure_latency_elapsed_is_float():
    model = _EchoModel()
    with model._measure_latency() as timer:
        pass
    assert isinstance(timer.elapsed, float)


def test_latency_timer_context_manager():
    t = _LatencyTimer()
    assert t.elapsed == 0.0
    with t:
        pass
    assert t.elapsed >= 0.0


# ---------------------------------------------------------------------------
# ModelResponse — construction and validation
# ---------------------------------------------------------------------------

def test_model_response_valid_construction():
    r = ModelResponse(text="output", input_tokens=10, output_tokens=5, cost=0.001, latency=1.2)
    assert r.text == "output"
    assert r.input_tokens == 10
    assert r.output_tokens == 5
    assert r.cost == 0.001
    assert r.latency == 1.2


def test_model_response_zero_values_allowed():
    r = ModelResponse(text="", input_tokens=0, output_tokens=0, cost=0.0, latency=0.0)
    assert r.input_tokens == 0
    assert r.output_tokens == 0


def test_model_response_invalid_text_type():
    with pytest.raises(TypeError, match="text must be str"):
        ModelResponse(text=123, input_tokens=1, output_tokens=1, cost=0.0, latency=0.1)  # type: ignore


def test_model_response_negative_input_tokens():
    with pytest.raises(ValueError, match="input_tokens"):
        ModelResponse(text="x", input_tokens=-1, output_tokens=1, cost=0.0, latency=0.1)


def test_model_response_negative_output_tokens():
    with pytest.raises(ValueError, match="output_tokens"):
        ModelResponse(text="x", input_tokens=1, output_tokens=-1, cost=0.0, latency=0.1)


def test_model_response_negative_cost():
    with pytest.raises(ValueError, match="cost"):
        ModelResponse(text="x", input_tokens=1, output_tokens=1, cost=-0.01, latency=0.1)


def test_model_response_negative_latency():
    with pytest.raises(ValueError, match="latency"):
        ModelResponse(text="x", input_tokens=1, output_tokens=1, cost=0.0, latency=-1.0)


def test_model_response_non_numeric_cost():
    with pytest.raises((ValueError, TypeError)):
        ModelResponse(text="x", input_tokens=1, output_tokens=1, cost="cheap", latency=0.1)  # type: ignore


# ---------------------------------------------------------------------------
# ModelResponse.to_dict
# ---------------------------------------------------------------------------

def test_to_dict_returns_dict():
    r = ModelResponse("output", 10, 5, 0.002, 1.5)
    assert isinstance(r.to_dict(), dict)


def test_to_dict_has_required_keys():
    r = ModelResponse("output", 10, 5, 0.002, 1.5)
    d = r.to_dict()
    for key in ("text", "input_tokens", "output_tokens", "cost", "latency"):
        assert key in d


def test_to_dict_values_match():
    r = ModelResponse("hello", 3, 7, 0.005, 2.1)
    d = r.to_dict()
    assert d["text"] == "hello"
    assert d["input_tokens"] == 3
    assert d["output_tokens"] == 7
    assert d["cost"] == 0.005
    assert d["latency"] == 2.1


def test_to_dict_cost_is_float():
    r = ModelResponse("x", 1, 1, 0, 0)
    assert isinstance(r.to_dict()["cost"], float)


def test_to_dict_latency_is_float():
    r = ModelResponse("x", 1, 1, 0, 0)
    assert isinstance(r.to_dict()["latency"], float)


# ---------------------------------------------------------------------------
# ModelResponse.total_tokens
# ---------------------------------------------------------------------------

def test_total_tokens_sums_input_and_output():
    r = ModelResponse("x", input_tokens=10, output_tokens=5, cost=0.0, latency=0.1)
    assert r.total_tokens == 15


def test_total_tokens_zero():
    r = ModelResponse("x", input_tokens=0, output_tokens=0, cost=0.0, latency=0.0)
    assert r.total_tokens == 0


# ---------------------------------------------------------------------------
# RuntimeError propagation from subclass
# ---------------------------------------------------------------------------

def test_run_propagates_runtime_error():
    model = _AlwaysFailModel()
    with pytest.raises(RuntimeError, match="Simulated API failure"):
        model.run("valid prompt")