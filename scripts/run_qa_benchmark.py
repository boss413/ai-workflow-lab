"""
scripts/run_qa_benchmark.py

Question Answering benchmark runner using SQuAD validation set.

SQuAD answers are extractive spans from the context. Each question has
multiple valid answer strings (annotator variations). Scoring uses:

  Exact Match (EM)  — prediction matches any reference answer exactly
                      after normalization (lowercase, strip punctuation/articles)
  Token F1          — token overlap between prediction and best reference answer
  correct           — token F1 >= 0.5

Output: results/raw/qa_<timestamp>.json

Usage:
  python -m scripts.run_qa_benchmark
  python -m scripts.run_qa_benchmark --models gpt41_mini claude_haiku
  python -m scripts.run_qa_benchmark --cost-tier low
  python -m scripts.run_qa_benchmark --sample-size 20
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import string
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

TASK = "qa"
F1_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# SQuAD-style normalization and scoring
# ---------------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    """
    Normalize an answer string for comparison.
    Lowercases, removes punctuation, strips articles (a/an/the),
    and collapses whitespace. Standard SQuAD evaluation normalization.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def _token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between normalized prediction and reference strings."""
    pred_tokens = _normalize_answer(prediction).split()
    ref_tokens  = _normalize_answer(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    tp = sum(min(pred_counts.get(t, 0), ref_counts[t]) for t in ref_counts)
    precision = tp / len(pred_tokens)
    recall    = tp / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def score_qa(prediction: str, answers: dict | list | str) -> dict[str, float]:
    """
    Score a QA prediction against SQuAD-style answers.

    answers can be:
      - dict {"text": [...], "answer_start": [...]}   (SQuAD raw format)
      - list of strings
      - single string

    Returns {"exact_match": float, "f1": float, "correct": int}
    """
    # Extract list of reference strings
    if isinstance(answers, dict):
        refs = answers.get("text", [])
    elif isinstance(answers, list):
        refs = [str(a) for a in answers]
    else:
        refs = [str(answers)]

    if not refs:
        return {"exact_match": 0.0, "f1": 0.0, "correct": 0}

    pred_norm = _normalize_answer(prediction)

    # Exact match: 1 if prediction matches any reference after normalization
    exact = int(any(pred_norm == _normalize_answer(r) for r in refs))

    # F1: best F1 across all references
    best_f1 = max(_token_f1(prediction, r) for r in refs)

    return {
        "exact_match": float(exact),
        "f1":          best_f1,
        "correct":     int(best_f1 >= F1_THRESHOLD),
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

        # SQuAD needs both context (input) and question
        if not row.get("question"):
            print(f"    [{i+1:02d}/{len(dataset)}] SKIP  (missing question)")
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

        prediction = response.text.strip()
        scores     = score_qa(prediction, row["label"])

        # Get first reference answer for display
        label = row["label"]
        if isinstance(label, dict):
            ref_display = label.get("text", ["?"])[0]
        elif isinstance(label, list):
            ref_display = label[0] if label else "?"
        else:
            ref_display = str(label)

        records.append({
            "question":    row.get("question", ""),
            "input":       row["input"][:200] + "..." if len(row["input"]) > 200 else row["input"],
            "predicted":   prediction,
            "reference":   ref_display,
            "exact_match": scores["exact_match"],
            "f1":          scores["f1"],
            "correct":     scores["correct"],
            "confidence":  response.confidence,
            "confidence_source": response.confidence_source,
            "latency":     round(response.latency, 4),
            "input_tokens":  response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost":          response.cost,
        })

        status = "OK  " if scores["correct"] else "MISS"
        print(f"    [{i+1:02d}/{len(dataset)}] {status}  "
              f"f1={scores['f1']:.2f}  em={int(scores['exact_match'])}  "
              f"pred={prediction[:50]!r}")

    if not records:
        return records, {}

    n = len(records)
    summary = {
        "alias":         alias,
        "model_id":      model_id,
        "cost_tier":     model_entry["cost_tier"],
        "samples":       n,
        "accuracy":      round(sum(r["correct"]     for r in records) / n, 4),
        "avg_f1":        round(sum(r["f1"]          for r in records) / n, 4),
        "avg_em":        round(sum(r["exact_match"] for r in records) / n, 4),
        "avg_latency":   round(sum(r["latency"]     for r in records) / n, 4),
        "total_cost":    round(sum(r["cost"]        for r in records), 6),
    }
    return records, summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run QA benchmark")
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
    print(f"Dataset:     squad (validation)")
    print(f"Scoring:     token F1 + exact match  (correct = F1 >= {F1_THRESHOLD})")
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
            print(f"   f1={summary['avg_f1']:.2f}  "
                  f"em={summary['avg_em']:.2f}  "
                  f"accuracy={summary['accuracy']:.0%}  "
                  f"latency={summary['avg_latency']:.2f}s  "
                  f"cost=${summary['total_cost']:.4f}")
        print()

    if len(summaries) > 1:
        print("── Comparison ──────────────────────────────────────────────────")
        header = (f"{'Alias':<20} {'Model':<35} {'F1':>6} {'EM':>6} "
                  f"{'Accuracy':>10} {'Latency':>10} {'Cost':>10}")
        print(header)
        print("-" * len(header))
        for s in sorted(summaries, key=lambda x: x["avg_f1"], reverse=True):
            print(f"{s['alias']:<20} {s['model_id']:<35} "
                  f"{s['avg_f1']:>6.2f} {s['avg_em']:>6.2f} "
                  f"{s['accuracy']:>9.0%} {s['avg_latency']:>9.2f}s "
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
            "f1_threshold":      F1_THRESHOLD,
            "summaries":         summaries,
            "results":           all_results,
        }, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()