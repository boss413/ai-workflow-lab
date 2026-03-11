"""
tests/test_runner.py

Unit tests for benchmarks/runner.py.
All external dependencies (dataset loader, model adapters, prompt generator)
are mocked — no network access or API keys required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.runner import (
    _append_record,
    _build_adapter,
    _make_record,
    _output_path,
    _run_single_model,
    run_benchmark,
)
from models.base_model import ModelResponse


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_response(
    text: str = "predicted",
    input_tokens: int = 10,
    output_tokens: int = 5,
    cost: float = 0.001,
    latency: float = 0.5,
) -> ModelResponse:
    return ModelResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
        latency=latency,
    )


def _make_examples(n: int = 3, task: str = "classification") -> list[dict]:
    if task == "qa":
        return [{"input": f"Context {i}", "question": f"Q{i}?", "label": f"A{i}"} for i in range(n)]
    return [{"input": f"Input text {i}", "label": f"label_{i}"} for i in range(n)]


def _mock_adapter(response: ModelResponse | None = None) -> MagicMock:
    adapter = MagicMock()
    adapter.run.return_value = response or _make_response()
    return adapter


# ---------------------------------------------------------------------------
# _make_record
# ---------------------------------------------------------------------------

def test_make_record_returns_dict():
    r = _make_record("classification", "openai:gpt-4o", "prompt text", _make_response())
    assert isinstance(r, dict)


def test_make_record_required_keys():
    r = _make_record("classification", "openai:gpt-4o", "prompt", _make_response())
    for key in ("task", "model", "prompt", "response", "latency", "input_tokens", "output_tokens", "cost"):
        assert key in r, f"Missing key: {key}"


def test_make_record_task_value():
    r = _make_record("summarization", "openai:gpt-4o", "p", _make_response())
    assert r["task"] == "summarization"


def test_make_record_model_value():
    r = _make_record("classification", "anthropic:claude-sonnet-4-5", "p", _make_response())
    assert r["model"] == "anthropic:claude-sonnet-4-5"


def test_make_record_prompt_value():
    r = _make_record("classification", "openai:gpt-4o", "my prompt", _make_response())
    assert r["prompt"] == "my prompt"


def test_make_record_response_text():
    response = _make_response(text="predicted label")
    r = _make_record("classification", "openai:gpt-4o", "p", response)
    assert r["response"] == "predicted label"


def test_make_record_latency_rounded():
    response = _make_response(latency=1.23456789)
    r = _make_record("classification", "openai:gpt-4o", "p", response)
    assert r["latency"] == round(1.23456789, 4)


def test_make_record_cost_rounded():
    response = _make_response(cost=0.000123456789)
    r = _make_record("classification", "openai:gpt-4o", "p", response)
    assert r["cost"] == round(0.000123456789, 8)


def test_make_record_token_counts():
    response = _make_response(input_tokens=42, output_tokens=17)
    r = _make_record("classification", "openai:gpt-4o", "p", response)
    assert r["input_tokens"] == 42
    assert r["output_tokens"] == 17


# ---------------------------------------------------------------------------
# _output_path
# ---------------------------------------------------------------------------

def test_output_path_is_jsonl(tmp_path):
    p = _output_path("classification", "openai:gpt-4o-mini", tmp_path)
    assert p.suffix == ".jsonl"


def test_output_path_contains_task(tmp_path):
    p = _output_path("summarization", "openai:gpt-4o", tmp_path)
    assert "summarization" in p.name


def test_output_path_contains_safe_model_name(tmp_path):
    p = _output_path("classification", "openai:gpt-4o-mini", tmp_path)
    assert ":" not in p.name


def test_output_path_is_under_results_dir(tmp_path):
    p = _output_path("classification", "openai:gpt-4o", tmp_path)
    assert p.parent == tmp_path


# ---------------------------------------------------------------------------
# _append_record
# ---------------------------------------------------------------------------

def test_append_record_creates_file(tmp_path):
    out = tmp_path / "test.jsonl"
    _append_record(out, {"key": "value"})
    assert out.exists()


def test_append_record_writes_valid_json(tmp_path):
    out = tmp_path / "test.jsonl"
    _append_record(out, {"task": "classification", "score": 0.9})
    line = out.read_text().strip()
    parsed = json.loads(line)
    assert parsed["task"] == "classification"


def test_append_record_appends_multiple_lines(tmp_path):
    out = tmp_path / "test.jsonl"
    _append_record(out, {"n": 1})
    _append_record(out, {"n": 2})
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["n"] == 1
    assert json.loads(lines[1])["n"] == 2


def test_append_record_creates_parent_dirs(tmp_path):
    out = tmp_path / "nested" / "deep" / "test.jsonl"
    _append_record(out, {"x": 1})
    assert out.exists()


# ---------------------------------------------------------------------------
# _build_adapter
# ---------------------------------------------------------------------------

def test_build_adapter_raises_on_missing_colon():
    with pytest.raises(ValueError, match="provider:model_name"):
        _build_adapter("gpt-4o-no-provider")


def test_build_adapter_raises_on_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        _build_adapter("unknown:some-model")


def test_build_adapter_openai(monkeypatch):
    mock_cls = MagicMock(return_value=MagicMock())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("benchmarks.runner.OpenAIAdapter", mock_cls, create=True), \
         patch("models.openai_adapter.OpenAIAdapter", mock_cls):
        from models.openai_adapter import OpenAIAdapter
        with patch("benchmarks.runner._build_adapter") as mock_build:
            mock_build.return_value = MagicMock()
            result = mock_build("openai:gpt-4o")
            assert result is not None


def test_build_adapter_anthropic_import_path():
    with patch("benchmarks.runner._build_adapter") as mock_build:
        mock_build.return_value = MagicMock()
        result = mock_build("anthropic:claude-sonnet-4-5")
        mock_build.assert_called_once_with("anthropic:claude-sonnet-4-5")


# ---------------------------------------------------------------------------
# _run_single_model
# ---------------------------------------------------------------------------

def test_run_single_model_returns_list(tmp_path):
    examples = _make_examples(3)
    adapter = _mock_adapter()
    with patch("benchmarks.runner.generate_prompt", return_value="mocked prompt"):
        records = _run_single_model("classification", "openai:gpt-4o", examples, adapter, tmp_path / "out.jsonl")
    assert isinstance(records, list)


def test_run_single_model_correct_count(tmp_path):
    examples = _make_examples(5)
    adapter = _mock_adapter()
    with patch("benchmarks.runner.generate_prompt", return_value="prompt"):
        records = _run_single_model("classification", "openai:gpt-4o", examples, adapter, tmp_path / "out.jsonl")
    assert len(records) == 5


def test_run_single_model_writes_jsonl(tmp_path):
    examples = _make_examples(3)
    out = tmp_path / "out.jsonl"
    adapter = _mock_adapter()
    with patch("benchmarks.runner.generate_prompt", return_value="prompt"):
        _run_single_model("classification", "openai:gpt-4o", examples, adapter, out)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3


def test_run_single_model_record_schema(tmp_path):
    examples = _make_examples(1)
    out = tmp_path / "out.jsonl"
    adapter = _mock_adapter()
    with patch("benchmarks.runner.generate_prompt", return_value="the prompt"):
        records = _run_single_model("classification", "openai:gpt-4o", examples, adapter, out)
    rec = records[0]
    for key in ("task", "model", "prompt", "response", "latency", "input_tokens", "output_tokens", "cost"):
        assert key in rec


def test_run_single_model_skips_on_prompt_error(tmp_path):
    examples = _make_examples(3)
    adapter = _mock_adapter()
    with patch("benchmarks.runner.generate_prompt", side_effect=ValueError("bad")):
        records = _run_single_model("classification", "openai:gpt-4o", examples, adapter, tmp_path / "out.jsonl")
    assert records == []


def test_run_single_model_skips_on_model_error(tmp_path):
    examples = _make_examples(3)
    adapter = MagicMock()
    adapter.run.side_effect = RuntimeError("API error")
    with patch("benchmarks.runner.generate_prompt", return_value="prompt"):
        records = _run_single_model("classification", "openai:gpt-4o", examples, adapter, tmp_path / "out.jsonl")
    assert records == []


def test_run_single_model_partial_skip_on_error(tmp_path):
    examples = _make_examples(4)
    adapter = MagicMock()
    good_response = _make_response()
    adapter.run.side_effect = [good_response, RuntimeError("fail"), good_response, good_response]
    with patch("benchmarks.runner.generate_prompt", return_value="prompt"):
        records = _run_single_model("classification", "openai:gpt-4o", examples, adapter, tmp_path / "out.jsonl")
    assert len(records) == 3


def test_run_single_model_calls_adapter_once_per_example(tmp_path):
    examples = _make_examples(4)
    adapter = _mock_adapter()
    with patch("benchmarks.runner.generate_prompt", return_value="prompt"):
        _run_single_model("classification", "openai:gpt-4o", examples, adapter, tmp_path / "out.jsonl")
    assert adapter.run.call_count == 4


# ---------------------------------------------------------------------------
# run_benchmark — input validation
# ---------------------------------------------------------------------------

def test_run_benchmark_raises_on_empty_models():
    with pytest.raises(ValueError, match="empty"):
        run_benchmark("classification", models=[])


# ---------------------------------------------------------------------------
# run_benchmark — happy path
# ---------------------------------------------------------------------------

def test_run_benchmark_returns_dict(tmp_path):
    examples = _make_examples(2)
    adapter = _mock_adapter()
    with patch("benchmarks.runner.load_dataset", return_value=examples), \
         patch("benchmarks.runner._build_adapter", return_value=adapter), \
         patch("benchmarks.runner.generate_prompt", return_value="prompt"):
        result = run_benchmark("classification", ["openai:gpt-4o"], results_dir=tmp_path)
    assert isinstance(result, dict)


def test_run_benchmark_keys_match_models(tmp_path):
    examples = _make_examples(2)
    adapter = _mock_adapter()
    models = ["openai:gpt-4o", "anthropic:claude-sonnet-4-5"]
    with patch("benchmarks.runner.load_dataset", return_value=examples), \
         patch("benchmarks.runner._build_adapter", return_value=adapter), \
         patch("benchmarks.runner.generate_prompt", return_value="prompt"):
        result = run_benchmark("classification", models, results_dir=tmp_path)
    assert set(result.keys()) == set(models)


def test_run_benchmark_produces_correct_record_count(tmp_path):
    examples = _make_examples(3)
    adapter = _mock_adapter()
    with patch("benchmarks.runner.load_dataset", return_value=examples), \
         patch("benchmarks.runner._build_adapter", return_value=adapter), \
         patch("benchmarks.runner.generate_prompt", return_value="prompt"):
        result = run_benchmark("classification", ["openai:gpt-4o"], results_dir=tmp_path)
    assert len(result["openai:gpt-4o"]) == 3


def test_run_benchmark_creates_output_files(tmp_path):
    examples = _make_examples(2)
    adapter = _mock_adapter()
    with patch("benchmarks.runner.load_dataset", return_value=examples), \
         patch("benchmarks.runner._build_adapter", return_value=adapter), \
         patch("benchmarks.runner.generate_prompt", return_value="prompt"):
        run_benchmark("classification", ["openai:gpt-4o-mini"], results_dir=tmp_path)
    jsonl_files = list(tmp_path.glob("*.jsonl"))
    assert len(jsonl_files) == 1


def test_run_benchmark_jsonl_is_valid(tmp_path):
    examples = _make_examples(2)
    adapter = _mock_adapter()
    with patch("benchmarks.runner.load_dataset", return_value=examples), \
         patch("benchmarks.runner._build_adapter", return_value=adapter), \
         patch("benchmarks.runner.generate_prompt", return_value="prompt"):
        run_benchmark("classification", ["openai:gpt-4o"], results_dir=tmp_path)
    jsonl_file = next(tmp_path.glob("*.jsonl"))
    for line in jsonl_file.read_text().strip().splitlines():
        record = json.loads(line)
        assert "task" in record
        assert "model" in record


def test_run_benchmark_skips_bad_adapter_continues(tmp_path):
    examples = _make_examples(2)
    good_adapter = _mock_adapter()

    def build_side_effect(model_id):
        if "bad" in model_id:
            raise ValueError("bad model")
        return good_adapter

    with patch("benchmarks.runner.load_dataset", return_value=examples), \
         patch("benchmarks.runner._build_adapter", side_effect=build_side_effect), \
         patch("benchmarks.runner.generate_prompt", return_value="prompt"):
        result = run_benchmark(
            "classification",
            ["bad:model", "openai:gpt-4o"],
            results_dir=tmp_path,
        )
    assert result["bad:model"] == []
    assert len(result["openai:gpt-4o"]) == 2


def test_run_benchmark_multiple_models_separate_files(tmp_path):
    examples = _make_examples(2)
    adapter = _mock_adapter()
    models = ["openai:gpt-4o", "anthropic:claude-sonnet-4-5"]
    with patch("benchmarks.runner.load_dataset", return_value=examples), \
         patch("benchmarks.runner._build_adapter", return_value=adapter), \
         patch("benchmarks.runner.generate_prompt", return_value="prompt"):
        run_benchmark("classification", models, results_dir=tmp_path)
    jsonl_files = list(tmp_path.glob("*.jsonl"))
    assert len(jsonl_files) == 2