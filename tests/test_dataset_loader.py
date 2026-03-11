"""
tests/test_dataset_loader.py

Unit tests for data_loader/loader.py.

All HuggingFace network calls are mocked at the `_fetch_hf_dataset` boundary
so tests run fully offline without requiring the data_loader library.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader.loader import (
    SUPPORTED_TASKS,
    _normalize_classification,
    _normalize_extraction,
    _normalize_generation,
    _normalize_qa,
    _normalize_reasoning,
    _normalize_summarization,
    load_dataset,
)


# ---------------------------------------------------------------------------
# Fake dataset
# ---------------------------------------------------------------------------

class _FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._rows[idx]


def _make_rows(task: str, n: int = 50) -> list[dict[str, Any]]:
    factories: dict[str, Any] = {
        "classification": lambda i: {"text": f"Article text {i}", "label": i % 4},
        "extraction": lambda i: {
            "tokens": ["Apple", "is", "based", "in", "Cupertino"],
            "ner_tags": [3, 0, 0, 0, 5],
        },
        "summarization": lambda i: {
            "document": f"Long document {i}.",
            "summary": f"Short summary {i}.",
        },
        "qa": lambda i: {
            "context": f"Context passage {i}.",
            "question": f"What is {i}?",
            "answers": {"text": [f"Answer {i}"], "answer_start": [0]},
        },
        "reasoning": lambda i: {
            "question": f"Math problem {i}?",
            "answer": f"The answer is {i}.",
        },
        "generation": lambda i: {
            "concepts": ["apple", "tree", "garden"],
            "target": f"An apple grows on a tree. Example {i}.",
        },
    }
    return [factories[task](i) for i in range(n)]


def _fake_dataset(task: str, n: int = 50) -> _FakeDataset:
    return _FakeDataset(_make_rows(task, n))


def _patch(task: str, n: int = 50):
    return patch(
        "data_loader.loader._fetch_hf_dataset",
        return_value=_fake_dataset(task, n),
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_unsupported_task_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported task"):
        load_dataset("unknown_task", sample_size=10)


def test_zero_sample_size_raises_value_error():
    with pytest.raises(ValueError, match="sample_size must be a positive integer"):
        load_dataset("classification", sample_size=0)


def test_negative_sample_size_raises_value_error():
    with pytest.raises(ValueError, match="sample_size must be a positive integer"):
        load_dataset("classification", sample_size=-5)


def test_string_sample_size_raises_value_error():
    with pytest.raises(ValueError, match="sample_size must be a positive integer"):
        load_dataset("classification", sample_size="100")  # type: ignore[arg-type]


def test_hf_fetch_failure_raises_runtime_error():
    with patch("data_loader.loader._fetch_hf_dataset", side_effect=RuntimeError("network error")):
        with pytest.raises(RuntimeError):
            load_dataset("classification", sample_size=10)


# ---------------------------------------------------------------------------
# Sample size
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", sorted(SUPPORTED_TASKS))
def test_sample_size_respected(task: str):
    with _patch(task):
        results = load_dataset(task, sample_size=10)
    assert len(results) == 10


@pytest.mark.parametrize("task", sorted(SUPPORTED_TASKS))
def test_sample_size_larger_than_dataset_returns_all(task: str):
    with _patch(task, n=20):
        results = load_dataset(task, sample_size=9999)
    assert len(results) == 20


@pytest.mark.parametrize("task", sorted(SUPPORTED_TASKS))
def test_returns_list(task: str):
    with _patch(task):
        results = load_dataset(task, sample_size=5)
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Schema correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", sorted(t for t in SUPPORTED_TASKS if t != "qa"))
def test_schema_has_input_and_label(task: str):
    with _patch(task):
        results = load_dataset(task, sample_size=5)
    for ex in results:
        assert "input" in ex
        assert "label" in ex


def test_qa_schema_has_question_field():
    with _patch("qa"):
        results = load_dataset("qa", sample_size=5)
    for ex in results:
        assert "input" in ex
        assert "question" in ex
        assert "label" in ex


@pytest.mark.parametrize("task", sorted(SUPPORTED_TASKS))
def test_schema_values_are_strings(task: str):
    with _patch(task):
        results = load_dataset(task, sample_size=5)
    for ex in results:
        for key, value in ex.items():
            assert isinstance(value, str), (
                f"task={task} key='{key}' expected str, got {type(value).__name__}"
            )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_produces_same_results():
    rows = _make_rows("classification")
    with patch("data_loader.loader._fetch_hf_dataset", return_value=_FakeDataset(rows)):
        first = load_dataset("classification", sample_size=15, seed=42)
    with patch("data_loader.loader._fetch_hf_dataset", return_value=_FakeDataset(rows)):
        second = load_dataset("classification", sample_size=15, seed=42)
    assert first == second


def test_different_seeds_produce_different_results():
    rows = _make_rows("classification")
    with patch("data_loader.loader._fetch_hf_dataset", return_value=_FakeDataset(rows)):
        first = load_dataset("classification", sample_size=20, seed=1)
    with patch("data_loader.loader._fetch_hf_dataset", return_value=_FakeDataset(rows)):
        second = load_dataset("classification", sample_size=20, seed=999)
    assert first != second


# ---------------------------------------------------------------------------
# Normalizer unit tests — classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("int_label,str_label", [
    (0, "world"), (1, "sports"), (2, "business"), (3, "science/technology"),
])
def test_normalize_classification_all_labels(int_label: int, str_label: str):
    result = _normalize_classification({"text": "sample text", "label": int_label})
    assert result["label"] == str_label


def test_normalize_classification_strips_whitespace():
    result = _normalize_classification({"text": "  padded text  ", "label": 0})
    assert result["input"] == "padded text"


# ---------------------------------------------------------------------------
# Normalizer unit tests — extraction
# ---------------------------------------------------------------------------

def test_normalize_extraction_tokens_joined_as_input():
    row = {"tokens": ["Apple", "is", "great"], "ner_tags": [3, 0, 0]}
    result = _normalize_extraction(row)
    assert result["input"] == "Apple is great"


def test_normalize_extraction_entities_in_label():
    row = {"tokens": ["Apple", "is", "great"], "ner_tags": [3, 0, 0]}
    result = _normalize_extraction(row)
    assert "Apple/B-ORG" in result["label"]


def test_normalize_extraction_no_entities_returns_none():
    row = {"tokens": ["hello", "world"], "ner_tags": [0, 0]}
    result = _normalize_extraction(row)
    assert result["label"] == "none"


def test_normalize_extraction_multiple_entity_types():
    row = {
        "tokens": ["London", "is", "in", "UK"],
        "ner_tags": [5, 0, 0, 7],
    }
    result = _normalize_extraction(row)
    assert "London/B-LOC" in result["label"]
    assert "UK/B-MISC" in result["label"]


# ---------------------------------------------------------------------------
# Normalizer unit tests — summarization
# ---------------------------------------------------------------------------

def test_normalize_summarization_fields():
    row = {"document": "Long text here.", "summary": "Short text."}
    result = _normalize_summarization(row)
    assert result["input"] == "Long text here."
    assert result["label"] == "Short text."


def test_normalize_summarization_strips_whitespace():
    row = {"document": "  doc  ", "summary": "  sum  "}
    result = _normalize_summarization(row)
    assert result["input"] == "doc"
    assert result["label"] == "sum"


# ---------------------------------------------------------------------------
# Normalizer unit tests — qa
# ---------------------------------------------------------------------------

def test_normalize_qa_with_answers():
    row = {
        "context": "Paris is in France.",
        "question": "Where is Paris?",
        "answers": {"text": ["France"], "answer_start": [10]},
    }
    result = _normalize_qa(row)
    assert result["input"] == "Paris is in France."
    assert result["question"] == "Where is Paris?"
    assert result["label"] == "France"


def test_normalize_qa_empty_answers_returns_empty_label():
    row = {
        "context": "Some context.",
        "question": "What?",
        "answers": {"text": [], "answer_start": []},
    }
    result = _normalize_qa(row)
    assert result["label"] == ""


def test_normalize_qa_uses_first_answer():
    row = {
        "context": "ctx",
        "question": "q?",
        "answers": {"text": ["first", "second"], "answer_start": [0, 5]},
    }
    result = _normalize_qa(row)
    assert result["label"] == "first"


# ---------------------------------------------------------------------------
# Normalizer unit tests — reasoning
# ---------------------------------------------------------------------------

def test_normalize_reasoning_fields():
    row = {"question": "What is 2+2?", "answer": "4"}
    result = _normalize_reasoning(row)
    assert result["input"] == "What is 2+2?"
    assert result["label"] == "4"


# ---------------------------------------------------------------------------
# Normalizer unit tests — generation
# ---------------------------------------------------------------------------

def test_normalize_generation_concepts_comma_joined():
    row = {"concepts": ["dog", "run", "park"], "target": "The dog runs in the park."}
    result = _normalize_generation(row)
    assert result["input"] == "dog, run, park"
    assert result["label"] == "The dog runs in the park."


def test_normalize_generation_empty_concepts():
    row = {"concepts": [], "target": "Something."}
    result = _normalize_generation(row)
    assert result["input"] == ""


def test_normalize_generation_missing_target_defaults_empty():
    row = {"concepts": ["a", "b"]}
    result = _normalize_generation(row)
    assert result["label"] == ""


# ---------------------------------------------------------------------------
# SUPPORTED_TASKS coverage
# ---------------------------------------------------------------------------

def test_supported_tasks_contains_all_six():
    expected = {
        "classification", "extraction", "summarization",
        "qa", "reasoning", "generation",
    }
    assert SUPPORTED_TASKS == expected