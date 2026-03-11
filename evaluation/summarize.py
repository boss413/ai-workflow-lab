"""
evaluation/summarize.py

Aggregate raw benchmark JSONL results into summary tables grouped by model and task.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Required fields every JSONL record must contain.
REQUIRED_FIELDS: frozenset[str] = frozenset(
    {"task", "model", "latency", "cost", "input_tokens", "output_tokens"}
)

# Optional scoring fields — included in aggregation when present.
SCORE_FIELDS: tuple[str, ...] = ("correct", "score")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of parsed records."""
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Results file is empty: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {exc}"
                ) from exc

    if not records:
        raise ValueError(f"No valid records found in {path}")
    return records


def _load_results_dir(results_path: Path) -> list[dict[str, Any]]:
    """Load all JSONL files from a directory, merging into one record list."""
    jsonl_files = sorted(results_path.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(
            f"No JSONL files found in results directory: {results_path}"
        )

    all_records: list[dict[str, Any]] = []
    for jsonl_file in jsonl_files:
        try:
            records = _load_jsonl(jsonl_file)
            all_records.extend(records)
            logger.info("Loaded %d records from %s.", len(records), jsonl_file)
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("Skipping %s: %s", jsonl_file, exc)

    if not all_records:
        raise ValueError(f"No valid records loaded from {results_path}")
    return all_records


def _load_records(results_path: str | Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file or a directory of JSONL files."""
    path = Path(results_path)
    if path.is_dir():
        return _load_results_dir(path)
    return _load_jsonl(path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_records(records: list[dict[str, Any]]) -> None:
    """Raise ValueError if any record is missing required fields."""
    for i, record in enumerate(records):
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(
                f"Record {i} is missing required fields: {sorted(missing)}"
            )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _build_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert records to a DataFrame, coercing numeric columns."""
    df = pd.DataFrame(records)

    numeric_cols = ["latency", "cost", "input_tokens", "output_tokens"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for field in SCORE_FIELDS:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")

    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics grouped by task and model.

    Always computed: mean latency, mean cost, total input/output tokens.
    Computed when present: accuracy (from 'correct'), mean score.
    """
    agg_spec: dict[str, Any] = {
        "latency": ("latency", "mean"),
        "cost_mean": ("cost", "mean"),
        "cost_total": ("cost", "sum"),
        "input_tokens_mean": ("input_tokens", "mean"),
        "output_tokens_mean": ("output_tokens", "mean"),
        "n_examples": ("latency", "count"),
    }

    if "correct" in df.columns:
        agg_spec["accuracy"] = ("correct", "mean")

    if "score" in df.columns:
        agg_spec["score_mean"] = ("score", "mean")

    summary = (
        df.groupby(["task", "model"])
        .agg(**agg_spec)
        .reset_index()
    )

    # Round for readability
    float_cols = [c for c in summary.columns if summary[c].dtype == float]
    summary[float_cols] = summary[float_cols].round(4)

    return summary.sort_values(["task", "model"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def summarize(results_path: str | Path) -> pd.DataFrame:
    """
    Load raw benchmark JSONL results and return an aggregated summary DataFrame.

    Groups results by task and model, computing mean latency, mean/total cost,
    token counts, accuracy (if 'correct' field present), and mean score
    (if 'score' field present).

    Args:
        results_path: Path to a single JSONL file or a directory of JSONL files.

    Returns:
        DataFrame with one row per (task, model) pair and aggregated metrics
        as columns.

    Raises:
        FileNotFoundError: If the path does not exist or contains no JSONL files.
        ValueError: If records are malformed or missing required fields.
    """
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results path does not exist: {path}")

    logger.info("Summarizing results from: %s", path)

    records = _load_records(path)
    _validate_records(records)

    df = _build_dataframe(records)
    summary = _aggregate(df)

    logger.info(
        "Summary complete: %d (task, model) combinations across %d records.",
        len(summary),
        len(records),
    )
    return summary


def summarize_to_markdown(results_path: str | Path) -> str:
    """
    Return the aggregated summary as a Markdown table string.

    Args:
        results_path: Path to a JSONL file or directory of JSONL files.

    Returns:
        Markdown-formatted table string.
    """
    df = summarize(results_path)
    return df.to_markdown(index=False)