# Task 04: Prompt Template Generator

## Objective
Generate standardized prompts for benchmark tasks.

## Files
prompts/templates.py

## Interface

generate_prompt(task: str, example: dict) -> str

## Example

Input:
task = "classification"

Output prompt:

You are an AI assistant for customer support.

Classify the message into:

billing
technical
account

Message:
{input}

## Requirements
- Prompt templates per task
- Variable substitution
- Deterministic formatting

## Tests
tests/test_prompts.py

Test cases
- prompt renders correctly
- fields substituted