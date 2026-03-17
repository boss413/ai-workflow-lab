"""
scripts/run_reasoning_benchmark.py

Reasoning benchmark runner using GSM8K (grade school math word problems).

GSM8K ground truth format:  "...step by step...\\n#### 18"
Model output format:        "...reasoning...\\nAnswer: 18"

Scoring: exact numeric match after normalizing both to plain numbers.
Handles integers and decimals; strips $, commas, trailing zeros.

Output: results/raw/reasoning_<timestamp>.json

Usage:
  python -m scripts.run_reasoning_benchmark
  python -m scripts.run_reasoning_benchmark --models gpt41_mini claude_haiku
  python -m scripts.run_reasoning_benchmark --cost-tier low
  python -m scripts.run_reasoning_benchmark --sample-size 20
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from config.config_loader import (
    load_models,
    load_generation_config,
    get_generation_params,
    get_enabled_models,
    get_models_for_task,
    get_models_by_cost_tier,
)
from data_loader.hf_loader import load_task_dataset
from prompts.templates import generate_prompt
from benchmarks.runner import _build_adapter

load_dotenv()
logging.basicConfig(level=logging.WARNING)

TASK = "reasoning"


# ---------------------------------------------------------------------------
# Answer extraction and scoring
# ---------------------------------------------------------------------------

def extract_ground_truth(raw_answer: str) -> str | None:
    """
    Extract the numeric answer from a GSM8K label string.
    GSM8K format: '...working...\\n#### 18'
    Returns the number as a normalised string, or None if not found.
    """
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', raw_answer)
    if not match:
        return None
    return _normalize_number(match.group(1))


def extract_model_answer(text: str) -> str | None:
    """
    Extract the final answer from model output.
    Looks for 'Answer: <number>' as instructed by the prompt.
    Falls back to the last number in the text if not found.
    """
    # Primary: explicit Answer: tag
    match = re.search(r'Answer:\s*\$?(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if match:
        return _normalize_number(match.group(1))

    # Fallback: last standalone number in the text
    numbers = re.findall(r'(?<!\d)(-?[\d,]+\.?\d*)(?!\d)', text)
    if numbers:
        return _normalize_number(numbers[-1])

    return None


def _normalize_number(s: str) -> str:
    """
    Normalize a numeric string for comparison.
    Strips commas, leading $, converts to float then back to remove
    trailing zeros, returns as string.
    e.g. '1,234.50' -> '1234.5', '18' -> '18', '$42.00' -> '42'
    """
    s = s.replace(',', '').strip().lstrip('$')
    try:
        f = float(s)
        # Return as int string if whole number, else float string
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def score_reasoning(
    model_output: str,
    ground_truth_raw: str,
) -> dict:
    """
    Score a reasoning response by comparing extracted numeric answers.

    Returns:
        predicted:   normalized model answer string (or None)
        actual:      normalized ground truth string (or None)
        correct:     1 if answers match exactly, 0 otherwise
        parse_error: True if either answer could not be extracted
    """
    actual    = extract_ground_truth(ground_truth_raw)
    predicted = extract_model_answer(model_output)

    if actual is None or predicted is None:
        return {
            "predicted":   predicted,
            "actual":      actual,
            "correct":     0,
            "parse_error": True,
        }

    return {
        "predicted":   predicted,
        "actual":      actual,
        "correct":     int(predicted == actual),
        "parse_error": False,
    }


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_models(all_models, aliases, cost_tier, task_match):
    if aliases:
        unknown = [a for a in aliases if a not in all_models]
        if unknown:
            raise ValueError(f"Unknown model alias(es): {unknown}. "
                             f"Available: {sorted(all_models.keys())}")
        selected = {a: all_models[a] for a in aliases}
        disabled = [a for a, m in selected.items() if not m["enabled"]]
        if disabled:
            print(f"  Warning: these models are disabled in models.yaml: {disabled}")
        return selected
    if cost_tier:
        return get_models_by_cost_tier(cost_tier, all_models)
    if task_match:
        return get_models_for_task(TASK, all_models)
    return get_enabled_models(all_models)


# ---------------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------------

_FATAL_FRAGMENTS = ("404", "400", "not_found", "invalid_api_key",
                    "invalid_request", "authentication")


def _is_fatal(exc: Exception) -> bool:
    return any(f in str(exc).lower() for f in _FATAL_FRAGMENTS)


def run_one_model(
    model_entry: dict,
    dataset: list[dict],
    generation_params: dict,
) -> tuple[list[dict], dict]:
    model_id = model_entry["model_id"]
    alias    = model_entry["alias"]

    try:
        adapter = _build_adapter(model_id)
    except Exception as exc:
        print(f"  [SKIP] {alias} ({model_id}): {exc}")
        return [], {}

    records = []
    for i, row in enumerate(dataset):
        if not str(row.get("input", "")).strip():
            print(f"    [{i+1:02d}/{len(dataset)}] SKIP  (empty input)")
            continue

        try:
            prompt   = generate_prompt(TASK, row)
            response = adapter.run(prompt, generation_params=generation_params)
        except Exception as exc:
            if _is_fatal(exc):
                print(f"  [ABORT] {alias}: {exc}")
                return [], {}
            logging.warning("Error on row %d for %s: %s", i, model_id, exc)
            print(f"    [{i+1:02d}/{len(dataset)}] SKIP  (error: {exc})")
            continue

        scores = score_reasoning(response.text, str(row["label"]))

        records.append({
            "input":        row["input"],
            "predicted":    scores["predicted"],
            "actual":       scores["actual"],
            "correct":      scores["correct"],
            "parse_error":  scores["parse_error"],
            "model_output": response.text,
            "confidence":   response.confidence,
            "confidence_source": response.confidence_source,
            "latency":      round(response.latency, 4),
            "input_tokens":  response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost":          response.cost,
        })

        status = "OK  " if scores["correct"] else "MISS"
        pred   = scores["predicted"] or "?"
        actual = scores["actual"] or "?"
        parse_flag = " [parse error]" if scores["parse_error"] else ""
        print(f"    [{i+1:02d}/{len(dataset)}] {status}  "
              f"predicted={pred:>8}  actual={actual:>8}{parse_flag}")

    if not records:
        return records, {}

    n           = len(records)
    n_parseable = sum(1 for r in records if not r["parse_error"])
    summary = {
        "alias":          alias,
        "model_id":       model_id,
        "cost_tier":      model_entry["cost_tier"],
        "samples":        n,
        "accuracy":       round(sum(r["correct"]    for r in records) / n, 4),
        "parse_errors":   n - n_parseable,
        "avg_latency":    round(sum(r["latency"]    for r in records) / n, 4),
        "total_cost":     round(sum(r["cost"]       for r in records), 6),
        "avg_output_tokens": round(sum(r["output_tokens"] for r in records) / n, 1),
    }
    return records, summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run reasoning benchmark")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--models", nargs="+", metavar="ALIAS")
    group.add_argument("--cost-tier", choices=["low", "medium", "high"])
    group.add_argument("--task-match", action="store_true")
    p.add_argument("--sample-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    all_models = load_models()
    selected   = select_models(all_models, args.models, args.cost_tier, args.task_match)

    if not selected:
        print("No models selected. Check models.yaml or your filter flags.")
        return

    gen_config        = load_generation_config()
    generation_params = get_generation_params(TASK, gen_config)

    print(f"Task:        {TASK}")
    print(f"Dataset:     gsm8k")
    print(f"Scoring:     exact numeric match on final answer")
    model_list = ", ".join(
        f'{m["alias"]} ({m["model_id"]})' for m in selected.values()
    )
    print(f"Models:      {model_list}")
    print(f"Samples:     {args.sample_size}  seed={args.seed}")
    print(f"Gen params:  {generation_params}")
    print()

    dataset = load_task_dataset(TASK, sample_size=args.sample_size, seed=args.seed)

    all_results: dict[str, list[dict]] = {}
    summaries:   list[dict] = []

    for alias, model_entry in selected.items():
        print(f"── {alias}  ({model_entry['model_id']})")
        records, summary = run_one_model(model_entry, dataset, generation_params)
        if records:
            all_results[alias] = records
            summaries.append(summary)
            parse_note = (f"  [{summary['parse_errors']} parse errors]"
                          if summary["parse_errors"] else "")
            print(f"   accuracy={summary['accuracy']:.0%}  "
                  f"avg_latency={summary['avg_latency']:.2f}s  "
                  f"avg_output_tokens={summary['avg_output_tokens']:.0f}  "
                  f"cost=${summary['total_cost']:.4f}{parse_note}")
        print()

    if len(summaries) > 1:
        print("── Comparison ──────────────────────────────────────────────────")
        header = (f"{'Alias':<20} {'Model':<35} {'Accuracy':>10} "
                  f"{'Latency':>10} {'Tokens':>8} {'Cost':>10}")
        print(header)
        print("-" * len(header))
        for s in sorted(summaries, key=lambda x: x["accuracy"], reverse=True):
            print(f"{s['alias']:<20} {s['model_id']:<35} "
                  f"{s['accuracy']:>9.0%} {s['avg_latency']:>9.2f}s "
                  f"{s['avg_output_tokens']:>7.0f} ${s['total_cost']:>8.4f}")

    output_dir  = Path("results/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{TASK}_{timestamp}.json"

    with output_path.open("w") as f:
        json.dump({
            "task":              TASK,
            "timestamp":         timestamp,
            "sample_size":       args.sample_size,
            "seed":              args.seed,
            "generation_params": generation_params,
            "summaries":         summaries,
            "results":           all_results,
        }, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()