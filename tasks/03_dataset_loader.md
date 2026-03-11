# Task 03: Dataset Loader

## Objective
Load benchmark datasets from HuggingFace and normalize schema.

## Files
datasets/loader.py

## Interface

load_dataset(task: str, sample_size: int) -> list[dict]

## Example Output

{
 "input": "...",
 "label": "billing"
}

## Requirements
- Use HuggingFace datasets library
- Support sampling
- Normalize schema across datasets

## Dependencies
datasets library

## Tests
tests/test_dataset_loader.py

Test cases:
- dataset loads
- sample size correct
- schema normalized