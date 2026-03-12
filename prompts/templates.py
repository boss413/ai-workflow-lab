"""
prompts/templates.py

Generate standardized prompts for each benchmark capability category.
Prompts simulate realistic enterprise automation scenarios.

The classification template accepts dynamic categories injected from the
dataset config (datasets.yaml label_map), so the same prompt template
works for AG News, support tickets, or any other classification dataset.
"""

from __future__ import annotations

import logging
from string import Template
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_TASKS: frozenset[str] = frozenset(
    {"classification", "extraction", "summarization", "qa", "reasoning", "generation"}
)

# ---------------------------------------------------------------------------
# Prompt templates
# $variable placeholders substituted via string.Template.safe_substitute().
# classification uses $categories — injected at runtime from dataset config.
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, str] = {
    "classification": """\
You are a classifier.

Classify the following text into exactly one of these categories:
$categories

Rules:
- Reply with only the category label. No explanation.
- Use lowercase.

Text:
$input

Category:""",

    "extraction": """\
You are an enterprise data extraction assistant.

Extract the named entities from the following text and return them as a JSON object.

Output format (JSON only, no explanation):
{
  "persons": ["<name>", ...],
  "organizations": ["<name>", ...],
  "locations": ["<name>", ...]
}

If a category has no entities, return an empty list.

Text:
$input

JSON:""",

    "summarization": """\
You are an enterprise document summarization assistant.

Summarize the following document in 2-3 concise sentences suitable for an executive brief.
Focus on key facts, decisions, and outcomes. Do not include opinions.

Document:
$input

Summary:""",

    "qa": """\
You are an enterprise knowledge base assistant.

Answer the question using only the information provided in the context below.
If the answer is not present in the context, reply with: "Not found in context."

Context:
$input

Question:
$question

Answer:""",

    "reasoning": """\
You are an enterprise analytical reasoning assistant.

Solve the following problem step by step.
Show your reasoning, then state the final answer on a new line in this format:
Answer: <value>

Problem:
$input

Solution:""",

    "generation": """\
You are an enterprise content generation assistant.

Write a short, professional sentence or paragraph that naturally incorporates all of the \
following concepts: $input

Requirements:
- Use all listed concepts.
- Professional tone.
- Two to four sentences maximum.

Output:""",
}

# Required substitution variables per task (excluding auto-injected ones like $categories)
_REQUIRED_FIELDS: dict[str, set[str]] = {
    "classification": {"input"},
    "extraction": {"input"},
    "summarization": {"input"},
    "qa": {"input", "question"},
    "reasoning": {"input"},
    "generation": {"input"},
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate_prompt(
    task: str,
    example: dict[str, Any],
    categories: list[str] | None = None,
) -> str:
    """
    Generate a standardized benchmark prompt for the given task and example.

    Args:
        task: Capability category. One of: classification, extraction,
              summarization, qa, reasoning, generation.
        example: Dictionary containing the example fields required by the task.
                 All tasks require at minimum an "input" key. The "qa" task
                 additionally requires a "question" key.
        categories: For classification only — list of valid label strings to
                    inject into the prompt. Loaded from datasets.yaml label_map.
                    Falls back to a generic placeholder if not provided.

    Returns:
        A fully rendered prompt string ready to be sent to a model.

    Raises:
        ValueError: If the task is unsupported or required example fields are missing.
    """
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task '{task}'. Supported tasks: {sorted(SUPPORTED_TASKS)}"
        )

    required = _REQUIRED_FIELDS[task]
    missing = required - example.keys()
    if missing:
        raise ValueError(
            f"Example is missing required fields for task '{task}': {sorted(missing)}"
        )

    empty_fields = {k for k in required if not str(example.get(k, "")).strip()}
    if empty_fields:
        raise ValueError(
            f"Example has empty required fields for task '{task}': {sorted(empty_fields)}"
        )

    substitutions = {key: str(example[key]) for key in required}

    if task == "classification":
        category_list = categories or ["(no categories configured)"]
        substitutions["categories"] = "\n".join(f"- {c}" for c in category_list)

    template = Template(_TEMPLATES[task])
    prompt = template.safe_substitute(substitutions)

    logger.debug("Generated prompt for task='%s' (%d chars).", task, len(prompt))
    return prompt


def get_template(task: str) -> str:
    """Return the raw template string for the given task."""
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task '{task}'. Supported tasks: {sorted(SUPPORTED_TASKS)}"
        )
    return _TEMPLATES[task]