"""
tests/test_prompts.py

Unit tests for prompts/templates.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts.templates import SUPPORTED_TASKS, generate_prompt, get_template


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLASSIFICATION_EXAMPLE = {"input": "My invoice is wrong and I need a refund."}
EXTRACTION_EXAMPLE = {"input": "Apple Inc. was founded by Steve Jobs in Cupertino."}
SUMMARIZATION_EXAMPLE = {
    "input": (
        "The quarterly earnings call revealed that revenue grew 12% year-over-year, "
        "driven by strong cloud segment performance. The CFO announced a new cost-cutting "
        "initiative targeting $200M in savings. Guidance for the next quarter was raised."
    )
}
QA_EXAMPLE = {
    "input": "The Eiffel Tower is located in Paris, France. It was completed in 1889.",
    "question": "When was the Eiffel Tower completed?",
}
REASONING_EXAMPLE = {
    "input": "A factory produces 120 widgets per hour. How many widgets are produced in an 8-hour shift?"
}
GENERATION_EXAMPLE = {"input": "cloud, security, enterprise"}

ALL_TASK_EXAMPLES: dict[str, dict] = {
    "classification": CLASSIFICATION_EXAMPLE,
    "extraction": EXTRACTION_EXAMPLE,
    "summarization": SUMMARIZATION_EXAMPLE,
    "qa": QA_EXAMPLE,
    "reasoning": REASONING_EXAMPLE,
    "generation": GENERATION_EXAMPLE,
}


# ---------------------------------------------------------------------------
# generate_prompt — input validation
# ---------------------------------------------------------------------------

def test_unsupported_task_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported task"):
        generate_prompt("unknown_task", {"input": "text"})


def test_missing_input_field_raises_value_error():
    with pytest.raises(ValueError, match="missing required fields"):
        generate_prompt("classification", {})


def test_qa_missing_question_field_raises_value_error():
    with pytest.raises(ValueError, match="missing required fields"):
        generate_prompt("qa", {"input": "some context"})


def test_qa_missing_input_field_raises_value_error():
    with pytest.raises(ValueError, match="missing required fields"):
        generate_prompt("qa", {"question": "What?"})


def test_empty_input_field_raises_value_error():
    with pytest.raises(ValueError, match="empty required fields"):
        generate_prompt("classification", {"input": "   "})


def test_empty_question_field_raises_value_error():
    with pytest.raises(ValueError, match="empty required fields"):
        generate_prompt("qa", {"input": "some context", "question": ""})


# ---------------------------------------------------------------------------
# generate_prompt — return type and structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", sorted(SUPPORTED_TASKS))
def test_generate_prompt_returns_string(task: str):
    result = generate_prompt(task, ALL_TASK_EXAMPLES[task])
    assert isinstance(result, str)


@pytest.mark.parametrize("task", sorted(SUPPORTED_TASKS))
def test_generate_prompt_is_non_empty(task: str):
    result = generate_prompt(task, ALL_TASK_EXAMPLES[task])
    assert len(result.strip()) > 0


@pytest.mark.parametrize("task", sorted(SUPPORTED_TASKS))
def test_generate_prompt_is_deterministic(task: str):
    first = generate_prompt(task, ALL_TASK_EXAMPLES[task])
    second = generate_prompt(task, ALL_TASK_EXAMPLES[task])
    assert first == second


# ---------------------------------------------------------------------------
# generate_prompt — variable substitution
# ---------------------------------------------------------------------------

def test_classification_prompt_contains_input():
    prompt = generate_prompt("classification", CLASSIFICATION_EXAMPLE)
    assert CLASSIFICATION_EXAMPLE["input"] in prompt


def test_classification_prompt_contains_category_labels():
    prompt = generate_prompt("classification", CLASSIFICATION_EXAMPLE)
    for label in ("billing", "technical", "account", "general"):
        assert label in prompt


def test_extraction_prompt_contains_input():
    prompt = generate_prompt("extraction", EXTRACTION_EXAMPLE)
    assert EXTRACTION_EXAMPLE["input"] in prompt


def test_extraction_prompt_requests_json():
    prompt = generate_prompt("extraction", EXTRACTION_EXAMPLE)
    assert "JSON" in prompt or "json" in prompt.lower()


def test_summarization_prompt_contains_input():
    prompt = generate_prompt("summarization", SUMMARIZATION_EXAMPLE)
    assert SUMMARIZATION_EXAMPLE["input"] in prompt


def test_qa_prompt_contains_context():
    prompt = generate_prompt("qa", QA_EXAMPLE)
    assert QA_EXAMPLE["input"] in prompt


def test_qa_prompt_contains_question():
    prompt = generate_prompt("qa", QA_EXAMPLE)
    assert QA_EXAMPLE["question"] in prompt


def test_reasoning_prompt_contains_input():
    prompt = generate_prompt("reasoning", REASONING_EXAMPLE)
    assert REASONING_EXAMPLE["input"] in prompt


def test_reasoning_prompt_requests_answer_format():
    prompt = generate_prompt("reasoning", REASONING_EXAMPLE)
    assert "Answer:" in prompt


def test_generation_prompt_contains_concepts():
    prompt = generate_prompt("generation", GENERATION_EXAMPLE)
    assert GENERATION_EXAMPLE["input"] in prompt


# ---------------------------------------------------------------------------
# generate_prompt — no leftover placeholders
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", sorted(SUPPORTED_TASKS))
def test_no_unsubstituted_placeholders(task: str):
    """Ensure no $variable tokens remain in the rendered prompt."""
    prompt = generate_prompt(task, ALL_TASK_EXAMPLES[task])
    import re
    unresolved = re.findall(r"\$[a-zA-Z_][a-zA-Z0-9_]*", prompt)
    assert unresolved == [], f"Unresolved placeholders in task='{task}': {unresolved}"


# ---------------------------------------------------------------------------
# generate_prompt — different inputs produce different prompts
# ---------------------------------------------------------------------------

def test_different_inputs_produce_different_prompts():
    example_a = {"input": "My invoice shows an incorrect amount."}
    example_b = {"input": "I cannot log into my account."}
    prompt_a = generate_prompt("classification", example_a)
    prompt_b = generate_prompt("classification", example_b)
    assert prompt_a != prompt_b


def test_different_qa_questions_produce_different_prompts():
    example_a = {**QA_EXAMPLE, "question": "Where is the Eiffel Tower?"}
    example_b = {**QA_EXAMPLE, "question": "When was the Eiffel Tower built?"}
    assert generate_prompt("qa", example_a) != generate_prompt("qa", example_b)


# ---------------------------------------------------------------------------
# get_template
# ---------------------------------------------------------------------------

def test_get_template_returns_string():
    tmpl = get_template("classification")
    assert isinstance(tmpl, str)


def test_get_template_unsupported_task_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported task"):
        get_template("invalid")


@pytest.mark.parametrize("task", sorted(SUPPORTED_TASKS))
def test_get_template_contains_input_placeholder(task: str):
    tmpl = get_template(task)
    assert "$input" in tmpl


def test_get_template_qa_contains_question_placeholder():
    tmpl = get_template("qa")
    assert "$question" in tmpl


# ---------------------------------------------------------------------------
# SUPPORTED_TASKS
# ---------------------------------------------------------------------------

def test_supported_tasks_contains_all_six():
    expected = {
        "classification", "extraction", "summarization",
        "qa", "reasoning", "generation",
    }
    assert SUPPORTED_TASKS == expected