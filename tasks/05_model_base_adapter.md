# Task 05: Base Model Adapter

## Objective
Create abstract interface for model adapters.

## Files
models/base_model.py

## Interface

class BaseModel:

    def run(prompt: str) -> ModelResponse

## ModelResponse schema

{
 text: str
 input_tokens: int
 output_tokens: int
 cost: float
 latency: float
}

## Requirements
- Abstract base class
- Shared response format

## Tests
tests/test_base_model.py