"""
config/config_loader.py

Loads and validates runtime configuration for models and tasks.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

VALID_TASKS = {"classification", "extraction", "summarization", "qa", "reasoning", "generation"}
REQUIRED_TASK_FIELDS = {"dataset", "sample_size"}
REQUIRED_TOP_LEVEL_KEYS = {"models", "tasks"}

CONFIG_DIR = Path(__file__).parent


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    if not isinstance(content, dict):
        raise ValueError(f"Config file must contain a YAML mapping at root: {path}")
    return content


def _validate_models(models: Any) -> list[str]:
    """Validate the models list."""
    if not isinstance(models, list):
        raise ValueError("'models' must be a list of model identifiers.")
    if len(models) == 0:
        raise ValueError("'models' list must not be empty.")
    for entry in models:
        if not isinstance(entry, str):
            raise ValueError(f"Each model entry must be a string, got: {type(entry)}")
        if ":" not in entry:
            raise ValueError(
                f"Model entry '{entry}' must follow the format 'provider:model_name'."
            )
    return models


def _validate_tasks(tasks: Any) -> dict[str, dict[str, Any]]:
    """Validate the tasks configuration."""
    if not isinstance(tasks, dict):
        raise ValueError("'tasks' must be a mapping of task names to task configurations.")
    if len(tasks) == 0:
        raise ValueError("'tasks' mapping must not be empty.")
    for task_name, task_config in tasks.items():
        if task_name not in VALID_TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Valid tasks are: {sorted(VALID_TASKS)}"
            )
        if not isinstance(task_config, dict):
            raise ValueError(f"Configuration for task '{task_name}' must be a mapping.")
        missing_fields = REQUIRED_TASK_FIELDS - task_config.keys()
        if missing_fields:
            raise ValueError(
                f"Task '{task_name}' is missing required fields: {sorted(missing_fields)}"
            )
        sample_size = task_config["sample_size"]
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError(
                f"Task '{task_name}' has invalid 'sample_size': must be a positive integer."
            )
        dataset = task_config["dataset"]
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError(
                f"Task '{task_name}' has invalid 'dataset': must be a non-empty string."
            )
    return tasks


def _merge_configs(models_config: dict[str, Any], tasks_config: dict[str, Any]) -> dict[str, Any]:
    """Merge models and tasks configs into a single configuration dictionary."""
    if "models" not in models_config:
        raise ValueError("models.yaml is missing required top-level key: 'models'")
    if "tasks" not in tasks_config:
        raise ValueError("tasks.yaml is missing required top-level key: 'tasks'")
    return {
        "models": models_config["models"],
        "tasks": tasks_config["tasks"],
    }


def load_config(
    models_path: Path | None = None,
    tasks_path: Path | None = None,
) -> dict[str, Any]:
    """
    Load and validate runtime configuration for models and tasks.

    Args:
        models_path: Path to models.yaml. Defaults to config/models.yaml.
        tasks_path: Path to tasks.yaml. Defaults to config/tasks.yaml.

    Returns:
        A validated configuration dictionary with keys 'models' and 'tasks'.

    Raises:
        FileNotFoundError: If a config file does not exist.
        ValueError: If the config fails validation.
    """
    models_path = models_path or CONFIG_DIR / "models.yaml"
    tasks_path = tasks_path or CONFIG_DIR / "tasks.yaml"

    logger.info("Loading models config from: %s", models_path)
    models_raw = _load_yaml(models_path)

    logger.info("Loading tasks config from: %s", tasks_path)
    tasks_raw = _load_yaml(tasks_path)

    config = _merge_configs(models_raw, tasks_raw)

    config["models"] = _validate_models(config["models"])
    config["tasks"] = _validate_tasks(config["tasks"])

    logger.info(
        "Config loaded successfully: %d models, %d tasks.",
        len(config["models"]),
        len(config["tasks"]),
    )
    return config


def _validate_generation_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate generation_config.yaml structure."""
    missing = {"defaults", "tasks"} - config.keys()
    if missing:
        raise ValueError(f"generation_config.yaml missing required keys: {sorted(missing)}")

    defaults = config["defaults"]
    missing_defaults = {"temperature", "top_p"} - defaults.keys()
    if missing_defaults:
        raise ValueError(
            f"generation_config.yaml 'defaults' missing required keys: {sorted(missing_defaults)}"
        )

    tasks = config["tasks"]
    for task_name, task_cfg in tasks.items():
        if task_name not in VALID_TASKS:
            raise ValueError(f"Unknown task '{task_name}' in generation_config.yaml")
        if "max_tokens" not in task_cfg:
            raise ValueError(
                f"Task '{task_name}' in generation_config.yaml is missing 'max_tokens'"
            )
        if not isinstance(task_cfg["max_tokens"], int) or task_cfg["max_tokens"] <= 0:
            raise ValueError(
                f"Task '{task_name}' has invalid 'max_tokens': must be a positive integer"
            )
    return config


def load_generation_config(
    generation_config_path: Path | None = None,
) -> dict[str, Any]:
    """
    Load and validate generation_config.yaml.

    Returns a dict with keys 'defaults' (temperature, top_p) and
    'tasks' (per-task max_tokens).

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config fails validation.
    """
    path = generation_config_path or CONFIG_DIR / "generation_config.yaml"
    logger.info("Loading generation config from: %s", path)
    raw = _load_yaml(path)
    config = _validate_generation_config(raw)
    logger.info("Generation config loaded: defaults=%s", config["defaults"])
    return config


def get_generation_params(task: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Return normalized generation parameters for a given task.

    Args:
        task: Capability category name.
        config: Pre-loaded generation config dict. Loaded from disk if not provided.

    Returns:
        Dict with keys: temperature, top_p, max_tokens.
    """
    if config is None:
        config = load_generation_config()

    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task '{task}'. Valid tasks: {sorted(VALID_TASKS)}")

    task_cfg = config.get("tasks", {}).get(task, {})
    if "max_tokens" not in task_cfg:
        raise ValueError(f"No max_tokens configured for task '{task}'")

    return {
        "temperature": config["defaults"]["temperature"],
        "top_p": config["defaults"]["top_p"],
        "max_tokens": task_cfg["max_tokens"],
    }