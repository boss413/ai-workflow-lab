"""
data_loader/hf_loader.py

Unified dataset loader driven by config/datasets.yaml.

Supports:
  source: huggingface  — fetched from HuggingFace Hub
  source: local_jsonl  — loaded from a local .jsonl file
  source: local_csv    — loaded from a local .csv file

Public interface:
  load_task_dataset(task_name, sample_size, seed, config_path) -> list[dict]

Each returned record has:
  {
    "input": str,
    "label": str | int,
    "metadata": {}
  }
"""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.yaml"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_dataset_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or _CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_task_config(task_name: str, config_path: Path | None = None) -> dict[str, Any]:
    registry = _load_dataset_config(config_path)
    if task_name not in registry:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {sorted(registry.keys())}"
        )
    return registry[task_name]


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def _load_from_huggingface(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch a dataset split from the HuggingFace Hub."""
    try:
        from datasets import load_dataset as hf_load  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' library is required: pip install datasets"
        ) from exc

    kwargs: dict[str, Any] = {"split": config["hf_split"]}
    if config.get("hf_config"):
        kwargs["name"] = config["hf_config"]

    logger.info("Fetching HuggingFace dataset '%s' split='%s'.", config["hf_name"], config["hf_split"])
    dataset = hf_load(config["hf_name"], **kwargs)
    return list(dataset)


def _load_from_local_jsonl(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(config["path"])
    if not path.exists():
        raise FileNotFoundError(f"Local dataset not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d records from local JSONL: %s", len(records), path)
    return records


def _load_from_local_csv(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(config["path"])
    if not path.exists():
        raise FileNotFoundError(f"Local dataset not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    logger.info("Loaded %d records from local CSV: %s", len(records), path)
    return records


_SOURCE_LOADERS = {
    "huggingface": _load_from_huggingface,
    "local_jsonl":  _load_from_local_jsonl,
    "local_csv":    _load_from_local_csv,
}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_record(
    raw: dict[str, Any],
    input_field: str,
    label_field: str,
    label_map: dict[int, str] | None,
) -> dict[str, Any]:
    """Extract input and label from a raw record, applying label_map if present."""
    raw_input = raw.get(input_field, "")
    raw_label = raw.get(label_field)

    # Join token lists (e.g. conll2003 tokens field)
    if isinstance(raw_input, list):
        raw_input = " ".join(str(t) for t in raw_input)

    # Apply integer → string label map if configured
    if label_map and isinstance(raw_label, int):
        raw_label = label_map.get(raw_label, str(raw_label))

    return {
        "input": str(raw_input),
        "label": raw_label,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def load_task_dataset(
    task_name: str,
    sample_size: int | None = None,
    seed: int = 42,
    config_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Load and normalize a dataset for a given benchmark task.

    Reads source and field mappings from config/datasets.yaml. Supports
    HuggingFace Hub datasets and local JSONL/CSV files.

    Args:
        task_name:   Key in datasets.yaml (e.g. 'classification').
        sample_size: Number of examples to return. Returns all if None.
                     Sampling is shuffled with the given seed so results
                     are representative rather than just the first N rows.
        seed:        Random seed for reproducible shuffling.
        config_path: Override path to datasets.yaml (useful for tests).

    Returns:
        List of normalized dicts: {"input": str, "label": ..., "metadata": {}}

    Raises:
        ValueError: If task_name is not in the registry.
        FileNotFoundError: If a local dataset path does not exist.
        RuntimeError: If a required library is missing.
    """
    config = _get_task_config(task_name, config_path)
    source = config.get("source", "huggingface")

    if source not in _SOURCE_LOADERS:
        raise ValueError(f"Unknown dataset source '{source}'. Valid: {sorted(_SOURCE_LOADERS)}")

    raw_records = _SOURCE_LOADERS[source](config)

    # Shuffle before sampling so we get a representative cross-section,
    # not just the first N rows of the dataset.
    rng = random.Random(seed)
    rng.shuffle(raw_records)

    if sample_size is not None:
        raw_records = raw_records[:sample_size]

    input_field = config["input_field"]
    label_field = config["label_field"]
    label_map = config.get("label_map")

    # label_map keys come from YAML as ints already; ensure they are
    if label_map:
        label_map = {int(k): str(v) for k, v in label_map.items()}

    normalized = [
        _normalize_record(raw, input_field, label_field, label_map)
        for raw in raw_records
    ]

    logger.info(
        "Loaded %d '%s' examples from source='%s'.",
        len(normalized), task_name, source,
    )
    return normalized