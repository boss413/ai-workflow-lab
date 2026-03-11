# Task 11: Result Aggregation

## Objective
Aggregate raw benchmark results.

## Files
evaluation/summarize.py

## Interface

summarize(results_path: str)

## Output

table of:

accuracy
latency
cost

grouped by model + task.

## Dependencies
pandas

## Tests
tests/test_summary.py