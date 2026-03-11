# Task 06: OpenAI Model Adapter

## Objective
Implement adapter for OpenAI models.

## Files
models/openai_adapter.py

## Interface

run(prompt: str, model_name: str)

## Requirements
- call OpenAI API
- track token usage
- calculate cost
- return ModelResponse

## Dependencies
openai python SDK

## Tests
tests/test_openai_adapter.py

Mock API responses.