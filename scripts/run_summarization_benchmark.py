"""
scripts/run_summarization_benchmark.py

Summarization benchmark runner using XSum dataset.

XSum reference summaries are single sentences. The prompt asks for 2-3 sentences
which intentionally tests whether models add value beyond the bare minimum —
ROUGE scores will be lower than academic baselines but comparisons between
models remain valid.

Scoring:
  ROUGE-1  unigram overlap (vocabulary coverage)
  ROUGE-2  bigram overlap  (phrase accuracy)
  ROUGE-L  longest common subsequence (fluency)
  correct  = ROUGE-1 >= 0.2  (low bar — rewards any meaningful overlap)

Install: pip install rouge-score

Output: results/raw/summarization_<timestamp>.json

Usage:
  python -m scripts.run_summarization_benchmark
  python -m scripts.run_summarization_benchmark --models gpt41_mini claude_haiku
  python -m scripts.run_summarization_benchmark --cost-tier low
  python -m scripts.run_summarization_benchmark --sample-size 20
"""

from __future__ import annotations

import argparse
import json
import logging
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

TASK = "summarization"
ROUGE1_THRESHOLD = 0.2   # minimum ROUGE-1 to count as "correct"


# ---------------------------------------------------------------------------
# ROUGE scoring
# ---------------------------------------------------------------------------

def _make_scorer():
    """Lazy-import rouge_scorer so import error surfaces at runtime with a clear message."""
    try:
        from rouge_score import rouge_scorer  # noqa: PLC0415
        return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    except ImportError:
        raise RuntimeError(
            "rouge-score is required for summarization scoring.\n"
            "Install with: pip install rouge-score"
        )


def score_summary(prediction: str, reference: str) -> dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Returns {"rouge1": float, "rouge2": float, "rougeL": float, "correct": int}
    """
    if not prediction.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "correct": 0}

    scorer = _make_scorer()
    scores = scorer.score(reference, prediction)
    r1 = round(scores["rouge1"].fmeasure, 4)
    r2 = round(scores["rouge2"].fmeasure, 4)
    rl = round(scores["rougeL"].fmeasure, 4)
    return {
        "rouge1":  r1,
        "rouge2":  r2,
        "rougeL":  rl,
        "correct": int(r1 >= ROUGE1_THRESHOLD),
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

        reference  = str(row["label"])
        prediction = response.text.strip()
        scores     = score_summary(prediction, reference)

        records.append({
            "input":       row["input"][:200] + "..." if len(row["input"]) > 200 else row["input"],
            "predicted":   prediction,
            "reference":   reference,
            "rouge1":      scores["rouge1"],
            "rouge2":      scores["rouge2"],
            "rougeL":      scores["rougeL"],
            "correct":     scores["correct"],
            "confidence":  response.confidence,
            "confidence_source": response.confidence_source,
            "latency":     round(response.latency, 4),
            "input_tokens":  response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost":          response.cost,
        })

        status    = "OK  " if scores["correct"] else "MISS"
        pred_preview = prediction[:80].replace("\n", " ")
        print(f"    [{i+1:02d}/{len(dataset)}] {status}  "
              f"r1={scores['rouge1']:.2f}  r2={scores['rouge2']:.2f}  "
              f"rL={scores['rougeL']:.2f}  | {pred_preview}")

    if not records:
        return records, {}

    n = len(records)
    summary = {
        "alias":        alias,
        "model_id":     model_id,
        "cost_tier":    model_entry["cost_tier"],
        "samples":      n,
        "accuracy":     round(sum(r["correct"] for r in records) / n, 4),
        "avg_rouge1":   round(sum(r["rouge1"]  for r in records) / n, 4),
        "avg_rouge2":   round(sum(r["rouge2"]  for r in records) / n, 4),
        "avg_rougeL":   round(sum(r["rougeL"]  for r in records) / n, 4),
        "avg_latency":  round(sum(r["latency"] for r in records) / n, 4),
        "total_cost":   round(sum(r["cost"]    for r in records), 6),
    }
    return records, summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run summarization benchmark")
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
    print(f"Dataset:     xsum")
    print(f"Scoring:     ROUGE-1/2/L  (correct = ROUGE-1 >= {ROUGE1_THRESHOLD})")
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
            print(f"   r1={summary['avg_rouge1']:.2f}  "
                  f"r2={summary['avg_rouge2']:.2f}  "
                  f"rL={summary['avg_rougeL']:.2f}  "
                  f"latency={summary['avg_latency']:.2f}s  "
                  f"cost=${summary['total_cost']:.4f}")
        print()

    if len(summaries) > 1:
        print("── Comparison ──────────────────────────────────────────────────")
        header = (f"{'Alias':<20} {'Model':<35} {'R1':>6} {'R2':>6} "
                  f"{'RL':>6} {'Latency':>10} {'Cost':>10}")
        print(header)
        print("-" * len(header))
        for s in sorted(summaries, key=lambda x: x["avg_rouge1"], reverse=True):
            print(f"{s['alias']:<20} {s['model_id']:<35} "
                  f"{s['avg_rouge1']:>6.2f} {s['avg_rouge2']:>6.2f} "
                  f"{s['avg_rougeL']:>6.2f} {s['avg_latency']:>9.2f}s "
                  f"${s['total_cost']:>8.4f}")

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
            "rouge1_threshold":  ROUGE1_THRESHOLD,
            "summaries":         summaries,
            "results":           all_results,
        }, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()