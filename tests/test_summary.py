"""
tests/test_summary.py

Unit tests for evaluation/summarize.py.
Uses temporary files and in-memory JSONL — no real benchmark runs required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.summarize import (
    _aggregate,
    _build_dataframe,
    _load_jsonl,
    _load_records,
    _validate_records,
    summarize,
    summarize_to_markdown,
)


# ---------------------------------------------------------------------------
# Fixtures — JSONL record factories
# ---------------------------------------------------------------------------

def _record(
    task: str = "classification",
    model: str = "openai:gpt-4o",
    latency: float = 0.5,
    cost: float = 0.001,
    input_tokens: int = 20,
    output_tokens: int = 10,
    correct: bool | None = None,
    score: float | None = None,
) -> dict:
    rec = {
        "task": task,
        "model": model,
        "prompt": "test prompt",
        "response": "test response",
        "latency": latency,
        "cost": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if correct is not None:
        rec["correct"] = correct
    if score is not None:
        rec["score"] = score
    return rec


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _basic_records(n: int = 4) -> list[dict]:
    return [
        _record("classification", "openai:gpt-4o",   latency=1.0, cost=0.002, correct=True),
        _record("classification", "openai:gpt-4o",   latency=2.0, cost=0.004, correct=False),
        _record("classification", "anthropic:claude", latency=0.5, cost=0.001, correct=True),
        _record("summarization",  "openai:gpt-4o",   latency=3.0, cost=0.006, score=0.8),
    ][:n]


# ---------------------------------------------------------------------------
# _load_jsonl
# ---------------------------------------------------------------------------

def test_load_jsonl_returns_list(tmp_path):
    f = tmp_path / "data.jsonl"
    _write_jsonl(f, [_record()])
    result = _load_jsonl(f)
    assert isinstance(result, list)


def test_load_jsonl_correct_count(tmp_path):
    f = tmp_path / "data.jsonl"
    _write_jsonl(f, [_record(), _record(task="summarization")])
    assert len(_load_jsonl(f)) == 2


def test_load_jsonl_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_jsonl(tmp_path / "nonexistent.jsonl")


def test_load_jsonl_raises_on_empty_file(tmp_path):
    f = tmp_path / "empty.jsonl"
    f.write_text("")
    with pytest.raises(ValueError, match="empty"):
        _load_jsonl(f)


def test_load_jsonl_raises_on_invalid_json(tmp_path):
    f = tmp_path / "bad.jsonl"
    f.write_text("not json\n")
    with pytest.raises(ValueError, match="Invalid JSON"):
        _load_jsonl(f)


def test_load_jsonl_skips_blank_lines(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text(json.dumps(_record()) + "\n\n" + json.dumps(_record()) + "\n")
    result = _load_jsonl(f)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# _load_records — directory mode
# ---------------------------------------------------------------------------

def test_load_records_from_directory(tmp_path):
    _write_jsonl(tmp_path / "a.jsonl", [_record("classification", "openai:gpt-4o")])
    _write_jsonl(tmp_path / "b.jsonl", [_record("summarization", "anthropic:claude")])
    records = _load_records(tmp_path)
    assert len(records) == 2


def test_load_records_dir_raises_no_jsonl(tmp_path):
    with pytest.raises(FileNotFoundError, match="No JSONL files"):
        _load_records(tmp_path)


def test_load_records_from_single_file(tmp_path):
    f = tmp_path / "data.jsonl"
    _write_jsonl(f, [_record(), _record()])
    assert len(_load_records(f)) == 2


def test_load_records_raises_path_not_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_records(tmp_path / "ghost")


# ---------------------------------------------------------------------------
# _validate_records
# ---------------------------------------------------------------------------

def test_validate_records_passes_valid():
    _validate_records([_record()])  # should not raise


def test_validate_records_raises_missing_task():
    bad = _record()
    del bad["task"]
    with pytest.raises(ValueError, match="missing required fields"):
        _validate_records([bad])


def test_validate_records_raises_missing_latency():
    bad = _record()
    del bad["latency"]
    with pytest.raises(ValueError, match="missing required fields"):
        _validate_records([bad])


def test_validate_records_raises_missing_cost():
    bad = _record()
    del bad["cost"]
    with pytest.raises(ValueError, match="missing required fields"):
        _validate_records([bad])


def test_validate_records_raises_missing_model():
    bad = _record()
    del bad["model"]
    with pytest.raises(ValueError, match="missing required fields"):
        _validate_records([bad])


# ---------------------------------------------------------------------------
# _build_dataframe
# ---------------------------------------------------------------------------

def test_build_dataframe_returns_dataframe():
    df = _build_dataframe([_record()])
    assert isinstance(df, pd.DataFrame)


def test_build_dataframe_latency_is_numeric():
    df = _build_dataframe([_record(latency=1.5)])
    assert pd.api.types.is_numeric_dtype(df["latency"])


def test_build_dataframe_cost_is_numeric():
    df = _build_dataframe([_record(cost=0.002)])
    assert pd.api.types.is_numeric_dtype(df["cost"])


def test_build_dataframe_correct_is_numeric_when_present():
    df = _build_dataframe([_record(correct=True), _record(correct=False)])
    assert pd.api.types.is_numeric_dtype(df["correct"])


# ---------------------------------------------------------------------------
# _aggregate
# ---------------------------------------------------------------------------

def test_aggregate_returns_dataframe():
    df = _build_dataframe(_basic_records())
    result = _aggregate(df)
    assert isinstance(result, pd.DataFrame)


def test_aggregate_has_task_and_model_columns():
    df = _build_dataframe(_basic_records())
    result = _aggregate(df)
    assert "task" in result.columns
    assert "model" in result.columns


def test_aggregate_has_latency_column():
    df = _build_dataframe(_basic_records())
    result = _aggregate(df)
    assert "latency" in result.columns


def test_aggregate_has_cost_columns():
    df = _build_dataframe(_basic_records())
    result = _aggregate(df)
    assert "cost_mean" in result.columns
    assert "cost_total" in result.columns


def test_aggregate_has_accuracy_when_correct_present():
    df = _build_dataframe(_basic_records())
    result = _aggregate(df)
    assert "accuracy" in result.columns


def test_aggregate_has_score_when_score_present():
    records = [_record(score=0.9), _record(score=0.7)]
    df = _build_dataframe(records)
    result = _aggregate(df)
    assert "score_mean" in result.columns


def test_aggregate_accuracy_calculation():
    records = [
        _record("classification", "openai:gpt-4o", correct=True),
        _record("classification", "openai:gpt-4o", correct=True),
        _record("classification", "openai:gpt-4o", correct=False),
    ]
    df = _build_dataframe(records)
    result = _aggregate(df)
    row = result[(result["task"] == "classification") & (result["model"] == "openai:gpt-4o")]
    assert abs(float(row["accuracy"].iloc[0]) - round(2/3, 4)) < 0.001


def test_aggregate_latency_mean():
    records = [
        _record("classification", "openai:gpt-4o", latency=1.0),
        _record("classification", "openai:gpt-4o", latency=3.0),
    ]
    df = _build_dataframe(records)
    result = _aggregate(df)
    row = result[result["task"] == "classification"]
    assert abs(float(row["latency"].iloc[0]) - 2.0) < 1e-4


def test_aggregate_cost_total():
    records = [
        _record("classification", "openai:gpt-4o", cost=0.001),
        _record("classification", "openai:gpt-4o", cost=0.003),
    ]
    df = _build_dataframe(records)
    result = _aggregate(df)
    row = result[result["task"] == "classification"]
    assert abs(float(row["cost_total"].iloc[0]) - 0.004) < 1e-6


def test_aggregate_groups_by_task_and_model():
    records = _basic_records()
    df = _build_dataframe(records)
    result = _aggregate(df)
    # classification has 2 models, summarization has 1
    assert len(result) == 3


def test_aggregate_n_examples_count():
    records = [
        _record("classification", "openai:gpt-4o"),
        _record("classification", "openai:gpt-4o"),
        _record("classification", "openai:gpt-4o"),
    ]
    df = _build_dataframe(records)
    result = _aggregate(df)
    assert int(result["n_examples"].iloc[0]) == 3


def test_aggregate_sorted_by_task_then_model():
    records = [
        _record("summarization", "openai:gpt-4o"),
        _record("classification", "openai:gpt-4o"),
        _record("classification", "anthropic:claude"),
    ]
    df = _build_dataframe(records)
    result = _aggregate(df)
    tasks = result["task"].tolist()
    assert tasks == sorted(tasks)


# ---------------------------------------------------------------------------
# summarize — end to end
# ---------------------------------------------------------------------------

def test_summarize_returns_dataframe(tmp_path):
    f = tmp_path / "results.jsonl"
    _write_jsonl(f, _basic_records())
    result = summarize(f)
    assert isinstance(result, pd.DataFrame)


def test_summarize_from_directory(tmp_path):
    _write_jsonl(tmp_path / "clf.jsonl", [_record("classification", "openai:gpt-4o")] * 3)
    _write_jsonl(tmp_path / "sum.jsonl", [_record("summarization", "anthropic:claude")] * 2)
    result = summarize(tmp_path)
    assert len(result) == 2


def test_summarize_raises_path_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        summarize(tmp_path / "nonexistent")


def test_summarize_raises_missing_required_field(tmp_path):
    bad = _record()
    del bad["cost"]
    f = tmp_path / "bad.jsonl"
    _write_jsonl(f, [bad])
    with pytest.raises(ValueError, match="missing required fields"):
        summarize(f)


def test_summarize_accuracy_in_output_when_correct_present(tmp_path):
    records = [_record(correct=True), _record(correct=False)]
    f = tmp_path / "r.jsonl"
    _write_jsonl(f, records)
    result = summarize(f)
    assert "accuracy" in result.columns


def test_summarize_no_accuracy_without_correct_field(tmp_path):
    f = tmp_path / "r.jsonl"
    _write_jsonl(f, [_record()])  # no 'correct' field
    result = summarize(f)
    assert "accuracy" not in result.columns


def test_summarize_accepts_string_path(tmp_path):
    f = tmp_path / "r.jsonl"
    _write_jsonl(f, [_record()])
    result = summarize(str(f))
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# summarize_to_markdown
# ---------------------------------------------------------------------------

def test_summarize_to_markdown_returns_string(tmp_path):
    f = tmp_path / "r.jsonl"
    _write_jsonl(f, _basic_records())
    result = summarize_to_markdown(f)
    assert isinstance(result, str)


def test_summarize_to_markdown_contains_headers(tmp_path):
    f = tmp_path / "r.jsonl"
    _write_jsonl(f, _basic_records())
    result = summarize_to_markdown(f)
    assert "task" in result
    assert "model" in result
    assert "latency" in result


def test_summarize_to_markdown_contains_pipe_chars(tmp_path):
    f = tmp_path / "r.jsonl"
    _write_jsonl(f, _basic_records())
    result = summarize_to_markdown(f)
    assert "|" in result