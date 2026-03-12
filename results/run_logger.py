"""
results/run_logger.py

Write structured JSON run artifacts for each completed benchmark run.

Each artifact captures everything needed to reproduce and audit a benchmark result:
timestamp, git commit, task, dataset, model, prompt version, latency, tokens,
cost, and task metrics.

Artifacts are saved to: results/<run_id>.json
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _get_git_commit() -> str:
    """Return the current HEAD commit hash, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------

def _make_run_id(task: str, model: str, timestamp: datetime) -> str:
    """
    Build a filesystem-safe run identifier.

    Format: YYYY-MM-DD_{task}_{safe_model}
    Example: 2026-03-11_classification_openai_gpt-4.1-mini
    """
    date_str = timestamp.strftime("%Y-%m-%d")
    safe_model = model.replace(":", "_").replace("/", "_")
    return f"{date_str}_{task}_{safe_model}"


# ---------------------------------------------------------------------------
# Artifact schema
# ---------------------------------------------------------------------------

def _build_artifact(
    run_id: str,
    task: str,
    dataset: str,
    model: str,
    prompt_version: str,
    samples: int,
    metrics: dict[str, float],
    latency_avg: float,
    cost_total: float,
    tokens_input_avg: float,
    tokens_output_avg: float,
    timestamp: datetime,
    git_commit: str,
) -> dict[str, Any]:
    """Assemble the full run artifact dict."""
    return {
        "run_id": run_id,
        "timestamp": timestamp.isoformat(),
        "git_commit": git_commit,
        "task": task,
        "dataset": dataset,
        "model": model,
        "prompt_version": prompt_version,
        "samples": samples,
        "metrics": metrics,
        "latency_avg": round(latency_avg, 4),
        "cost_total": round(cost_total, 8),
        "tokens_input_avg": round(tokens_input_avg, 1),
        "tokens_output_avg": round(tokens_output_avg, 1),
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def log_run(
    task: str,
    dataset: str,
    model: str,
    records: list[dict[str, Any]],
    metrics: dict[str, float],
    prompt_version: str = "v1",
    results_dir: Path | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """
    Build and persist a structured JSON artifact for a completed benchmark run.

    Args:
        task: Capability category (e.g. 'classification').
        dataset: HuggingFace dataset name used (e.g. 'ag_news').
        model: Model identifier in 'provider:model_name' format.
        records: Raw result records produced by the benchmark runner.
                 Each record must contain latency, cost, input_tokens, output_tokens.
        metrics: Aggregated task metrics (e.g. {'accuracy': 0.91, 'f1': 0.90}).
        prompt_version: Version tag for the prompt template used.
        results_dir: Directory to write the JSON artifact. Defaults to results/.
        git_commit: Git commit hash. Auto-detected from git if not provided.

    Returns:
        The artifact dict that was written to disk.

    Raises:
        ValueError: If records is empty or missing required fields.
        IOError: If the artifact cannot be written.
    """
    if not records:
        raise ValueError("Cannot log a run with zero records.")

    _validate_records(records)

    output_dir = results_dir or RESULTS_DIR
    now = datetime.now(tz=timezone.utc)
    commit = git_commit if git_commit is not None else _get_git_commit()
    run_id = _make_run_id(task, model, now)

    samples = len(records)
    latency_avg = sum(r["latency"] for r in records) / samples
    cost_total = sum(r["cost"] for r in records)
    tokens_input_avg = sum(r["input_tokens"] for r in records) / samples
    tokens_output_avg = sum(r["output_tokens"] for r in records) / samples

    artifact = _build_artifact(
        run_id=run_id,
        task=task,
        dataset=dataset,
        model=model,
        prompt_version=prompt_version,
        samples=samples,
        metrics=metrics,
        latency_avg=latency_avg,
        cost_total=cost_total,
        tokens_input_avg=tokens_input_avg,
        tokens_output_avg=tokens_output_avg,
        timestamp=now,
        git_commit=commit,
    )

    _write_artifact(artifact, output_dir, run_id)
    return artifact


def _validate_records(records: list[dict[str, Any]]) -> None:
    required = {"latency", "cost", "input_tokens", "output_tokens"}
    for i, record in enumerate(records):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Record {i} missing required fields: {sorted(missing)}")


def _write_artifact(artifact: dict[str, Any], output_dir: Path, run_id: str) -> Path:
    """Write the artifact JSON to disk and return the output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}.json"

    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)
    except OSError as exc:
        raise IOError(f"Failed to write run artifact to {output_path}: {exc}") from exc

    logger.info("Run artifact saved: %s", output_path)
    return output_path