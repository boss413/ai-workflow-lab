# Task 02: Implement Config Loader

## Objective
Load runtime configuration for models and tasks.

## Files
config/models.yaml
config/tasks.yaml
config/config_loader.py

## Example models.yaml

models:
  - openai:gpt-4.1
  - anthropic:claude-3-sonnet
  - google:gemini-pro

## Example tasks.yaml

tasks:
  classification:
    dataset: ag_news
    sample_size: 200
  summarization:
    dataset: xsum
    sample_size: 100

## Interface

load_config() -> dict

## Requirements
- YAML parsing
- Validate required fields
- Return structured configuration dictionary

## Tests
tests/test_config_loader.py

Test cases:
- YAML loads successfully
- Missing keys throw validation error