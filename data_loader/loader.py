"""
data_loader/loader.py

Load and normalize HuggingFace benchmark datasets for each capability category.

The HuggingFace `datasets` library is imported lazily inside `_fetch_hf_dataset()`
so that the rest of the codebase (and unit tests) can import this module
without requiring the library to be installed in the test environment.
"""

from __future__ import annotations

import logging
import random
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical label maps
# ---------------------------------------------------------------------------

_AG_NEWS_LABELS: dict[int, str] = {
    0: "world",
    1: "sports",
    2: "business",
    3: "science/technology",
}

_NER_TAGS: dict[int, str] = {
    0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC",
}

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

_TASK_REGISTRY: dict[str, dict[str, Any]] = {
    "classification": {
        "hf_path": "ag_news",
        "hf_split": "test",
        "hf_config": None,
    },
    "extraction": {
        "hf_path": "conll2003",
        "hf_split": "test",
        "hf_config": None,
    },
    "summarization": {
        "hf_path": "xsum",
        "hf_split": "test",
        "hf_config": None,
    },
    "qa": {
        "hf_path": "rajpurkar/squad",
        "hf_split": "validation",
        "hf_config": None,
    },
    "reasoning": {
        "hf_path": "gsm8k",
        "hf_split": "test",
        "hf_config": "main",
    },
    "generation": {
        "hf_path": "common_gen",
        "hf_split": "validation",
        "hf_config": None,
    },
}

SUPPORTED_TASKS: frozenset[str] = frozenset(_TASK_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Normalizers — one per dataset
# ---------------------------------------------------------------------------

def _normalize_classification(row: dict[str, Any]) -> dict[str, Any]:
    """ag_news: text -> input, integer label -> label string."""
    return {
        "input": str(row["text"]).strip(),
        "label": _AG_NEWS_LABELS[int(row["label"])],
    }


def _normalize_extraction(row: dict[str, Any]) -> dict[str, Any]:
    """conll2003: tokens + ner_tags -> input sentence, entity annotations -> label."""
    tokens: list[str] = row["tokens"]
    ner_tags: list[int] = row["ner_tags"]

    entities: list[str] = [
        f"{token}/{_NER_TAGS.get(tag, 'O')}"
        for token, tag in zip(tokens, ner_tags)
        if _NER_TAGS.get(tag, "O") != "O"
    ]

    return {
        "input": " ".join(tokens),
        "label": ", ".join(entities) if entities else "none",
    }


def _normalize_summarization(row: dict[str, Any]) -> dict[str, Any]:
    """xsum: document -> input, summary -> label."""
    return {
        "input": str(row["document"]).strip(),
        "label": str(row["summary"]).strip(),
    }


def _normalize_qa(row: dict[str, Any]) -> dict[str, Any]:
    """squad: context + question -> input/question, first answer -> label."""
    answer_texts: list[str] = row.get("answers", {}).get("text", [])
    return {
        "input": str(row["context"]).strip(),
        "question": str(row["question"]).strip(),
        "label": answer_texts[0].strip() if answer_texts else "",
    }


def _normalize_reasoning(row: dict[str, Any]) -> dict[str, Any]:
    """gsm8k: question -> input, answer -> label."""
    return {
        "input": str(row["question"]).strip(),
        "label": str(row["answer"]).strip(),
    }


def _normalize_generation(row: dict[str, Any]) -> dict[str, Any]:
    """common_gen: concept list -> comma-joined input, target -> label."""
    concepts: list[str] = row.get("concepts", [])
    return {
        "input": ", ".join(concepts),
        "label": str(row.get("target", "")).strip(),
    }


_NORMALIZERS: dict[str, Any] = {
    "classification": _normalize_classification,
    "extraction": _normalize_extraction,
    "summarization": _normalize_summarization,
    "qa": _normalize_qa,
    "reasoning": _normalize_reasoning,
    "generation": _normalize_generation,
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def load_dataset(task: str, sample_size: int, seed: int = 42) -> list[dict[str, Any]]:
    """
    Load a HuggingFace dataset for the given task and return normalized examples.

    Args:
        task: Capability category. One of: classification, extraction,
              summarization, qa, reasoning, generation.
        sample_size: Number of examples to return. Must be a positive integer.
        seed: Random seed for reproducible sampling.

    Returns:
        List of normalized dicts. Every example contains at least:
            - "input" (str)
            - "label" (str)
        The "qa" task additionally includes:
            - "question" (str)

    Raises:
        ValueError: If task is unsupported or sample_size is invalid.
        RuntimeError: If the dataset cannot be fetched from HuggingFace.
    """
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task '{task}'. Supported tasks: {sorted(SUPPORTED_TASKS)}"
        )
    if not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError(
            f"sample_size must be a positive integer, got: {sample_size!r}"
        )

    entry = _TASK_REGISTRY[task]
    normalizer = _NORMALIZERS[task]

    logger.info(
        "Loading HuggingFace dataset for task='%s' path='%s' split=%s config=%s.",
        task, entry["hf_path"], entry["hf_split"], entry["hf_config"],
    )

    dataset = _fetch_hf_dataset(entry)

    total = len(dataset)
    actual = min(sample_size, total)

    if actual < sample_size:
        logger.warning(
            "Requested sample_size=%d exceeds dataset size=%d. Using all %d examples.",
            sample_size, total, actual,
        )

    indices = random.Random(seed).sample(range(total), actual)

    examples: list[dict[str, Any]] = []
    for idx in indices:
        try:
            examples.append(normalizer(dataset[idx]))
        except Exception as exc:
            logger.warning("Skipping row %d due to normalization error: %s", idx, exc)

    logger.info("Loaded %d examples for task='%s'.", len(examples), task)
    return examples


def _fetch_hf_dataset(entry: dict[str, Any]) -> Any:
    """
    Fetch a HuggingFace dataset split. Kept as a separate function to allow
    straightforward mocking in unit tests without requiring network access.
    """
    try:
        from datasets import load_dataset as hf_load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' library is required. Install it with: pip install datasets"
        ) from exc

    try:
        return hf_load_dataset(
            entry["hf_path"],
            entry["hf_config"],
            split=entry["hf_split"],
            trust_remote_code=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load HuggingFace dataset '{entry['hf_path']}': {exc}"
        ) from exc