"""
tests/test_config_loader.py

Unit tests for config/config_loader.py.
"""

import pytest
import yaml
from pathlib import Path

from config.config_loader import load_config, _validate_models, _validate_tasks, _load_yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def valid_models_yaml(tmp_path: Path) -> Path:
    content = {"models": ["openai:gpt-4.1", "anthropic:claude-3-sonnet", "google:gemini-pro"]}
    path = tmp_path / "models.yaml"
    path.write_text(yaml.dump(content))
    return path


@pytest.fixture()
def valid_tasks_yaml(tmp_path: Path) -> Path:
    content = {
        "tasks": {
            "classification": {"dataset": "ag_news", "sample_size": 200},
            "summarization": {"dataset": "xsum", "sample_size": 100},
        }
    }
    path = tmp_path / "tasks.yaml"
    path.write_text(yaml.dump(content))
    return path


# ---------------------------------------------------------------------------
# load_config — happy path
# ---------------------------------------------------------------------------

def test_load_config_returns_dict(valid_models_yaml, valid_tasks_yaml):
    config = load_config(models_path=valid_models_yaml, tasks_path=valid_tasks_yaml)
    assert isinstance(config, dict)


def test_load_config_has_models_key(valid_models_yaml, valid_tasks_yaml):
    config = load_config(models_path=valid_models_yaml, tasks_path=valid_tasks_yaml)
    assert "models" in config


def test_load_config_has_tasks_key(valid_models_yaml, valid_tasks_yaml):
    config = load_config(models_path=valid_models_yaml, tasks_path=valid_tasks_yaml)
    assert "tasks" in config


def test_load_config_models_list(valid_models_yaml, valid_tasks_yaml):
    config = load_config(models_path=valid_models_yaml, tasks_path=valid_tasks_yaml)
    assert isinstance(config["models"], list)
    assert len(config["models"]) == 3


def test_load_config_tasks_dict(valid_models_yaml, valid_tasks_yaml):
    config = load_config(models_path=valid_models_yaml, tasks_path=valid_tasks_yaml)
    assert isinstance(config["tasks"], dict)
    assert "classification" in config["tasks"]
    assert "summarization" in config["tasks"]


def test_load_config_task_fields_present(valid_models_yaml, valid_tasks_yaml):
    config = load_config(models_path=valid_models_yaml, tasks_path=valid_tasks_yaml)
    task = config["tasks"]["classification"]
    assert task["dataset"] == "ag_news"
    assert task["sample_size"] == 200


# ---------------------------------------------------------------------------
# load_config — file not found
# ---------------------------------------------------------------------------

def test_load_config_missing_models_file(tmp_path, valid_tasks_yaml):
    missing = tmp_path / "nonexistent_models.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(models_path=missing, tasks_path=valid_tasks_yaml)


def test_load_config_missing_tasks_file(tmp_path, valid_models_yaml):
    missing = tmp_path / "nonexistent_tasks.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(models_path=valid_models_yaml, tasks_path=missing)


# ---------------------------------------------------------------------------
# load_config — missing top-level keys
# ---------------------------------------------------------------------------

def test_load_config_models_yaml_missing_models_key(tmp_path, valid_tasks_yaml):
    bad = tmp_path / "models.yaml"
    bad.write_text(yaml.dump({"not_models": []}))
    with pytest.raises(ValueError, match="models.yaml is missing required top-level key"):
        load_config(models_path=bad, tasks_path=valid_tasks_yaml)


def test_load_config_tasks_yaml_missing_tasks_key(tmp_path, valid_models_yaml):
    bad = tmp_path / "tasks.yaml"
    bad.write_text(yaml.dump({"not_tasks": {}}))
    with pytest.raises(ValueError, match="tasks.yaml is missing required top-level key"):
        load_config(models_path=valid_models_yaml, tasks_path=bad)


# ---------------------------------------------------------------------------
# _validate_models
# ---------------------------------------------------------------------------

def test_validate_models_valid():
    result = _validate_models(["openai:gpt-4.1", "anthropic:claude-3-sonnet"])
    assert result == ["openai:gpt-4.1", "anthropic:claude-3-sonnet"]


def test_validate_models_not_a_list():
    with pytest.raises(ValueError, match="'models' must be a list"):
        _validate_models("openai:gpt-4.1")


def test_validate_models_empty_list():
    with pytest.raises(ValueError, match="must not be empty"):
        _validate_models([])


def test_validate_models_entry_not_string():
    with pytest.raises(ValueError, match="must be a string"):
        _validate_models([123])


def test_validate_models_missing_colon():
    with pytest.raises(ValueError, match="provider:model_name"):
        _validate_models(["gpt41-no-provider"])


# ---------------------------------------------------------------------------
# _validate_tasks
# ---------------------------------------------------------------------------

def test_validate_tasks_valid():
    tasks = {"classification": {"dataset": "ag_news", "sample_size": 100}}
    result = _validate_tasks(tasks)
    assert result == tasks


def test_validate_tasks_not_a_dict():
    with pytest.raises(ValueError, match="'tasks' must be a mapping"):
        _validate_tasks(["classification"])


def test_validate_tasks_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        _validate_tasks({})


def test_validate_tasks_unknown_task():
    with pytest.raises(ValueError, match="Unknown task"):
        _validate_tasks({"unknown_task": {"dataset": "ds", "sample_size": 10}})


def test_validate_tasks_missing_dataset_field():
    with pytest.raises(ValueError, match="missing required fields"):
        _validate_tasks({"classification": {"sample_size": 100}})


def test_validate_tasks_missing_sample_size_field():
    with pytest.raises(ValueError, match="missing required fields"):
        _validate_tasks({"classification": {"dataset": "ag_news"}})


def test_validate_tasks_invalid_sample_size_zero():
    with pytest.raises(ValueError, match="invalid 'sample_size'"):
        _validate_tasks({"classification": {"dataset": "ag_news", "sample_size": 0}})


def test_validate_tasks_invalid_sample_size_negative():
    with pytest.raises(ValueError, match="invalid 'sample_size'"):
        _validate_tasks({"classification": {"dataset": "ag_news", "sample_size": -5}})


def test_validate_tasks_invalid_sample_size_string():
    with pytest.raises(ValueError, match="invalid 'sample_size'"):
        _validate_tasks({"classification": {"dataset": "ag_news", "sample_size": "big"}})


def test_validate_tasks_invalid_dataset_empty_string():
    with pytest.raises(ValueError, match="invalid 'dataset'"):
        _validate_tasks({"classification": {"dataset": "  ", "sample_size": 100}})


# ---------------------------------------------------------------------------
# _load_yaml — edge cases
# ---------------------------------------------------------------------------

def test_load_yaml_non_mapping_root(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        _load_yaml(bad)


def test_load_yaml_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_yaml(tmp_path / "does_not_exist.yaml")