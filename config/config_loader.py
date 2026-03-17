"""
config/config_loader.py

Loads and validates runtime configuration for models, tasks, and generation params.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

VALID_TASKS = {"classification", "extraction", "summarization", "qa", "reasoning", "generation"}
VALID_PROVIDERS = {"openai", "anthropic", "google", "ollama"}
VALID_COST_TIERS = {"low", "medium", "high"}
REQUIRED_TASK_FIELDS = {"dataset", "sample_size"}

CONFIG_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    if not isinstance(content, dict):
        raise ValueError(f"Config file must contain a YAML mapping at root: {path}")
    return content


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------

def _validate_model_entry(alias: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Validate a single model entry and return it with defaults applied."""
    required = {"provider", "model_name"}
    missing = required - entry.keys()
    if missing:
        raise ValueError(f"Model '{alias}' is missing required fields: {sorted(missing)}")

    if entry["provider"] not in VALID_PROVIDERS:
        raise ValueError(
            f"Model '{alias}' has unknown provider '{entry['provider']}'. "
            f"Valid: {sorted(VALID_PROVIDERS)}"
        )
    if not isinstance(entry["model_name"], str) or not entry["model_name"].strip():
        raise ValueError(f"Model '{alias}' has invalid model_name.")

    cost_tier = entry.get("cost_tier", "medium")
    if cost_tier not in VALID_COST_TIERS:
        raise ValueError(
            f"Model '{alias}' has unknown cost_tier '{cost_tier}'. "
            f"Valid: {sorted(VALID_COST_TIERS)}"
        )

    return {
        "alias": alias,
        "provider": entry["provider"],
        "model_name": entry["model_name"],
        "model_id": f"{entry['provider']}:{entry['model_name']}",
        "cost_tier": cost_tier,
        "strengths": entry.get("strengths", []),
        "enabled": entry.get("enabled", True),
    }


def _validate_models(models_dict: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(models_dict, dict):
        raise ValueError("'models' must be a mapping of alias → model config.")
    if not models_dict:
        raise ValueError("'models' must not be empty.")
    return {
        alias: _validate_model_entry(alias, entry)
        for alias, entry in models_dict.items()
    }


# ---------------------------------------------------------------------------
# Task validation
# ---------------------------------------------------------------------------

def _validate_tasks(tasks: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(tasks, dict):
        raise ValueError("'tasks' must be a mapping of task names to task configurations.")
    if not tasks:
        raise ValueError("'tasks' mapping must not be empty.")
    for task_name, task_config in tasks.items():
        if task_name not in VALID_TASKS:
            raise ValueError(f"Unknown task '{task_name}'. Valid: {sorted(VALID_TASKS)}")
        if not isinstance(task_config, dict):
            raise ValueError(f"Configuration for task '{task_name}' must be a mapping.")
        missing_fields = REQUIRED_TASK_FIELDS - task_config.keys()
        if missing_fields:
            raise ValueError(
                f"Task '{task_name}' is missing required fields: {sorted(missing_fields)}"
            )
        if not isinstance(task_config["sample_size"], int) or task_config["sample_size"] <= 0:
            raise ValueError(f"Task '{task_name}' has invalid 'sample_size'.")
        if not isinstance(task_config["dataset"], str) or not task_config["dataset"].strip():
            raise ValueError(f"Task '{task_name}' has invalid 'dataset'.")
    return tasks


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_models(models_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """
    Load and validate models.yaml.

    Returns a dict keyed by alias, each value containing:
      alias, provider, model_name, model_id, cost_tier, strengths, enabled
    """
    path = models_path or CONFIG_DIR / "models.yaml"
    raw = _load_yaml(path)
    if "models" not in raw:
        raise ValueError("models.yaml is missing required top-level key: 'models'")
    return _validate_models(raw["models"])


def load_config(
    models_path: Path | None = None,
    tasks_path: Path | None = None,
) -> dict[str, Any]:
    """Load and validate models.yaml and tasks.yaml into a merged config dict."""
    tasks_path = tasks_path or CONFIG_DIR / "tasks.yaml"
    tasks_raw = _load_yaml(tasks_path)
    if "tasks" not in tasks_raw:
        raise ValueError("tasks.yaml is missing required top-level key: 'tasks'")

    models = load_models(models_path)
    tasks = _validate_tasks(tasks_raw["tasks"])

    logger.info("Config loaded: %d models, %d tasks.", len(models), len(tasks))
    return {"models": models, "tasks": tasks}


# ---------------------------------------------------------------------------
# Model selection helpers
# ---------------------------------------------------------------------------

def get_enabled_models(models: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return only models where enabled=True."""
    return {k: v for k, v in models.items() if v["enabled"]}


def get_models_for_task(
    task: str,
    models: dict[str, dict[str, Any]],
    enabled_only: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Return models that list the given task in their strengths.

    Args:
        task: Capability category to filter by.
        models: Full model registry from load_models().
        enabled_only: If True (default), exclude disabled models.
    """
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task '{task}'. Valid: {sorted(VALID_TASKS)}")
    return {
        alias: m for alias, m in models.items()
        if task in m["strengths"]
        and (not enabled_only or m["enabled"])
    }


def get_models_by_cost_tier(
    tier: str,
    models: dict[str, dict[str, Any]],
    enabled_only: bool = True,
) -> dict[str, dict[str, Any]]:
    """Return models matching a cost tier (low | medium | high)."""
    if tier not in VALID_COST_TIERS:
        raise ValueError(f"Unknown cost tier '{tier}'. Valid: {sorted(VALID_COST_TIERS)}")
    return {
        alias: m for alias, m in models.items()
        if m["cost_tier"] == tier
        and (not enabled_only or m["enabled"])
    }


def get_model_ids(models: dict[str, dict[str, Any]]) -> list[str]:
    """Return list of 'provider:model_name' strings for use with _build_adapter."""
    return [m["model_id"] for m in models.values()]


# ---------------------------------------------------------------------------
# Generation config
# ---------------------------------------------------------------------------

def _validate_generation_config(config: dict[str, Any]) -> dict[str, Any]:
    missing = {"defaults", "tasks"} - config.keys()
    if missing:
        raise ValueError(f"generation_config.yaml missing required keys: {sorted(missing)}")
    missing_defaults = {"temperature", "top_p"} - config["defaults"].keys()
    if missing_defaults:
        raise ValueError(
            f"generation_config.yaml 'defaults' missing: {sorted(missing_defaults)}"
        )
    for task_name, task_cfg in config["tasks"].items():
        if task_name not in VALID_TASKS:
            raise ValueError(f"Unknown task '{task_name}' in generation_config.yaml")
        if "max_tokens" not in task_cfg:
            raise ValueError(f"Task '{task_name}' in generation_config.yaml missing 'max_tokens'")
        if not isinstance(task_cfg["max_tokens"], int) or task_cfg["max_tokens"] <= 0:
            raise ValueError(f"Task '{task_name}' has invalid 'max_tokens'")
    return config


def load_generation_config(generation_config_path: Path | None = None) -> dict[str, Any]:
    """Load and validate generation_config.yaml."""
    path = generation_config_path or CONFIG_DIR / "generation_config.yaml"
    raw = _load_yaml(path)
    config = _validate_generation_config(raw)
    logger.info("Generation config loaded: defaults=%s", config["defaults"])
    return config


def get_generation_params(task: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized {temperature, top_p, max_tokens} for a task."""
    if config is None:
        config = load_generation_config()
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task '{task}'. Valid: {sorted(VALID_TASKS)}")
    task_cfg = config.get("tasks", {}).get(task, {})
    if "max_tokens" not in task_cfg:
        raise ValueError(f"No max_tokens configured for task '{task}'")
    return {
        "temperature": config["defaults"]["temperature"],
        "top_p": config["defaults"]["top_p"],
        "max_tokens": task_cfg["max_tokens"],
    }