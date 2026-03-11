LLM Benchmarking Framework

This document converts the design architecture into concrete module interfaces and responsibilities.

System Overview

The benchmarking framework consists of the following modules:

datasets
prompts
models
benchmarks
evaluation
results
report

Each module has:

responsibility

interface

dependencies

tests

Module 1 — Dataset Loader
Purpose

Load and normalize datasets used for benchmarking tasks.

Supports:

HuggingFace datasets

local datasets

sampling subsets for faster tests

Interface
load_dataset(task: str, sample_size: int) -> List[Example]

Example output:

{
    "input": str,
    "label": str
}

Some tasks may include additional fields:

{
    "input": str,
    "question": str,
    "label": str
}
Responsibilities

fetch dataset

sample subset

normalize schema

Dependencies
datasets (huggingface library)
Tests
tests/test_dataset_loader.py

Test cases:

dataset loads successfully

sample size respected

schema normalized correctly

Module 2 — Prompt Templates
Purpose

Generate prompts used in benchmarking tasks.

Prompts simulate enterprise automation scenarios rather than academic benchmarks.

Interface
generate_prompt(task: str, example: dict) -> str

Example:

generate_prompt("classification", example)
Responsibilities

inject dataset inputs into prompt template

enforce consistent prompt formatting

Dependencies

None.

Tests
tests/test_prompts.py

Test cases:

correct prompt structure

correct variable substitution

Module 3 — Model Adapters
Purpose

Provide a unified interface for interacting with multiple model providers.

Adapters normalize:

API calls

token tracking

cost calculation

response formatting

Interface
run_model(model_name: str, prompt: str) -> ModelResponse

Response schema:

{
    "text": str,
    "input_tokens": int,
    "output_tokens": int,
    "cost": float,
    "latency": float
}
Implementations
OpenAIAdapter
AnthropicAdapter
GeminiAdapter
OpenWeightAdapter

Each adapter implements:

run(prompt)
Dependencies

Provider SDKs:

openai
anthropic
google-generativeai
Tests
tests/test_model_adapter.py

Test cases:

correct response schema

token accounting works

latency recorded

Mock API responses.

Module 4 — Benchmark Runner
Purpose

Execute benchmark tests across tasks and models.

Interface
run_benchmark(task: str, models: List[str], sample_size: int)
Responsibilities

load dataset

generate prompt

call model

collect results

Output

Writes raw results:

results/raw/{task}.jsonl

Example entry:

{
  "task": "classification",
  "model": "gpt4",
  "prompt": "...",
  "response": "...",
  "correct": true,
  "latency": 1.4,
  "cost": 0.003
}
Dependencies
datasets
prompts
models
evaluation
Tests
tests/test_runner.py

Test cases:

runner executes without failure

results file generated

results schema correct

Module 5 — Evaluation Metrics
Purpose

Score model outputs relative to expected answers.

Different tasks use different metrics.

Interface
score(task: str, prediction: str, truth: dict) -> dict

Example output:

{
    "correct": True,
    "score": 0.92
}
Task Metrics
Task	Metric
classification	accuracy
extraction	schema match
summarization	ROUGE
QA	exact match
reasoning	final answer correctness
generation	LLM judge rubric
Dependencies
evaluate (HF metrics)
Tests
tests/test_metrics.py
Module 6 — Result Storage
Purpose

Store raw benchmark outputs for reproducibility.

Format:

JSONL

Example path:

results/raw/classification_openai.jsonl

Each record contains:

prompt

model output

latency

tokens

score

Responsibilities

append results safely

maintain schema consistency

Module 7 — Report Generator
Purpose

Convert raw benchmark data into executive-facing results.

Outputs:

results/summary/benchmark_table.md
Interface
generate_report(results_path: str) -> Report
Output
Summary table
Model	Accuracy	Cost	Latency
Capability table

| Task | Model | Accuracy | Cost | Latency |

Dependencies
pandas
Testing Strategy

Testing occurs at three levels.

Unit Tests

For each module.

Examples:

dataset loader
prompt generator
metrics
Integration Tests

Runner + adapters.

Verifies:

dataset → prompt → model → evaluation
Benchmark Validation

Manual validation to ensure:

prompts behave as expected

metrics correctly reflect results