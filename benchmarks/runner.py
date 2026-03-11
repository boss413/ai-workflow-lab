"""
benchmarks/runner.py

Execute benchmark runs across datasets and models.
Writes raw results to results/raw/{task}_{model}.jsonl.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from data_loader.loader import load_dataset
from models.base_model import ModelResponse
from prompts.templates import generate_prompt

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/raw")

# ---------------------------------------------------------------------------
# Model registry — maps "provider:model" strings to adapter instances
# ---------------------------------------------------------------------------

def _build_adapter(model_id: str) -> Any:
    """
    Instantiate the correct adapter for a given model identifier.

    Args:
        model_id: Format is "provider:model_name" (e.g. "openai:gpt-4o-mini").

    Returns:
        An instantiated BaseModel subclass.

    Raises:
        ValueError: If the provider is unknown or the format is invalid.
    """
    if ":" not in model_id:
        raise ValueError(
            f"Invalid model_id '{model_id}'. Expected format: 'provider:model_name'."
        )

    provider, model_name = model_id.split(":", maxsplit=1)

    if provider == "openai":
        from models.openai_adapter import OpenAIAdapter  # noqa: PLC0415
        return OpenAIAdapter(model_name=model_name)

    if provider == "anthropic":
        from models.anthropic_adapter import AnthropicAdapter  # noqa: PLC0415
        return AnthropicAdapter(model_name=model_name)

    if provider == "google":
        from models.gemini_adapter import GeminiAdapter  # noqa: PLC0415
        return GeminiAdapter(model_name=model_name)

    raise ValueError(
        f"Unknown provider '{provider}' in model_id '{model_id}'. "
        f"Supported providers: openai, anthropic, google."
    )


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------

def _make_record(
    task: str,
    model_id: str,
    prompt: str,
    response: ModelResponse,
) -> dict[str, Any]:
    """Build a JSONL record from a single benchmark result."""
    return {
        "task": task,
        "model": model_id,
        "prompt": prompt,
        "response": response.text,
        "latency": round(response.latency, 4),
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost": round(response.cost, 8),
    }


def _append_record(output_path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON record to a JSONL file, creating the file if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _output_path(task: str, model_id: str, results_dir: Path) -> Path:
    """Return the JSONL output path for a task/model pair."""
    safe_model = model_id.replace(":", "_").replace("/", "_")
    return results_dir / f"{task}_{safe_model}.jsonl"


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

def _run_single_model(
    task: str,
    model_id: str,
    examples: list[dict[str, Any]],
    adapter: Any,
    output_path: Path,
) -> list[dict[str, Any]]:
    """
    Run one model over all examples for a task and write results to disk.

    Returns the list of result records produced.
    """
    records: list[dict[str, Any]] = []

    for i, example in enumerate(examples):
        try:
            prompt = generate_prompt(task, example)
        except Exception as exc:
            logger.warning(
                "Skipping example %d for task='%s': prompt generation failed: %s",
                i, task, exc,
            )
            continue

        try:
            response = adapter.run(prompt)
        except Exception as exc:
            logger.warning(
                "Skipping example %d for task='%s' model='%s': model call failed: %s",
                i, task, model_id, exc,
            )
            continue

        record = _make_record(task, model_id, prompt, response)
        _append_record(output_path, record)
        records.append(record)

        logger.debug(
            "task='%s' model='%s' example=%d latency=%.3fs cost=$%.6f",
            task, model_id, i, response.latency, response.cost,
        )

    logger.info(
        "Completed task='%s' model='%s': %d/%d examples written to %s.",
        task, model_id, len(records), len(examples), output_path,
    )
    return records


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_benchmark(
    task: str,
    models: list[str],
    sample_size: int = 100,
    results_dir: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Execute benchmark runs for a task across one or more models.

    Args:
        task: Capability category to benchmark
              (classification, extraction, summarization, qa, reasoning, generation).
        models: List of model identifiers in "provider:model_name" format.
        sample_size: Number of dataset examples to evaluate per model.
        results_dir: Directory for JSONL output files. Defaults to results/raw/.

    Returns:
        Dict mapping model_id -> list of result records written for that model.

    Raises:
        ValueError: If task or model identifiers are invalid.
        RuntimeError: If the dataset cannot be loaded.
    """
    if not models:
        raise ValueError("'models' list must not be empty.")

    output_dir = results_dir or RESULTS_DIR

    logger.info(
        "Starting benchmark: task='%s' models=%s sample_size=%d.",
        task, models, sample_size,
    )

    examples = load_dataset(task, sample_size)
    logger.info("Loaded %d examples for task='%s'.", len(examples), task)

    all_results: dict[str, list[dict[str, Any]]] = {}

    for model_id in models:
        logger.info("Running model='%s' on task='%s'.", model_id, task)

        try:
            adapter = _build_adapter(model_id)
        except (ValueError, EnvironmentError) as exc:
            logger.error("Skipping model='%s': %s", model_id, exc)
            all_results[model_id] = []
            continue

        out_path = _output_path(task, model_id, output_dir)
        records = _run_single_model(task, model_id, examples, adapter, out_path)
        all_results[model_id] = records

    logger.info("Benchmark complete for task='%s'.", task)
    return all_results