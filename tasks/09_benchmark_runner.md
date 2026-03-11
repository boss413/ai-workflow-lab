# Task 09: Benchmark Runner

## Objective
Execute benchmark runs across datasets and models.

## Files
benchmarks/runner.py

## Interface

run_benchmark(task: str, models: list[str])

## Responsibilities
- load dataset
- generate prompt
- call model
- collect results

## Output

results/raw/{task}.jsonl

Each record:

{
 task
 model
 prompt
 response
 latency
 tokens
 cost
}

## Tests
tests/test_runner.py