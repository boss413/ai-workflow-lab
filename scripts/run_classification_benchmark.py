"""
scripts/run_classification_benchmark.py

Generic classification benchmark runner.

Model selection (choose one, checked in this order):
  --models gpt41_mini gpt4o     run specific model aliases from models.yaml
  --cost-tier low               run all enabled models of a given cost tier
  --task-match                  run all enabled models that list 'classification' as a strength
  (none)                        run all enabled models

To switch datasets (e.g. AG News → support tickets), update
config/datasets.yaml classification entry. No changes to this file needed.

Output: results/raw/classification_<timestamp>.json

Usage:
  python -m scripts.run_classification_benchmark
  python -m scripts.run_classification_benchmark --models gpt41_mini gpt4o
  python -m scripts.run_classification_benchmark --cost-tier low
  python -m scripts.run_classification_benchmark --task-match
  python -m scripts.run_classification_benchmark --sample-size 100
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
    get_model_ids,
)
from data_loader.hf_loader import load_task_dataset, _load_dataset_config
from prompts.templates import generate_prompt
from benchmarks.runner import _build_adapter

load_dotenv()
logging.basicConfig(level=logging.WARNING)

TASK = "classification"
CONFIG_DIR = Path("config")


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_models(
    all_models: dict,
    aliases: list[str] | None,
    cost_tier: str | None,
    task_match: bool,
) -> dict:
    """
    Return a filtered model dict based on selection flags.

    Priority: --models > --cost-tier > --task-match > all enabled
    """
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

def normalize(text: str) -> str:
    return text.lower().strip().rstrip(".")


def run_one_model(
    model_entry: dict,
    dataset: list[dict],
    categories: list[str],
    generation_params: dict,
) -> tuple[list[dict], dict]:
    model_id = model_entry["model_id"]
    alias = model_entry["alias"]

    try:
        adapter = _build_adapter(model_id)
    except Exception as exc:
        print(f"  [SKIP] {alias} ({model_id}): {exc}")
        return [], {}

    records = []
    for i, row in enumerate(dataset):
        try:
            prompt = generate_prompt(TASK, row, categories=categories)
            response = adapter.run(prompt, generation_params=generation_params)
        except Exception as exc:
            logging.warning("Error on row %d for %s: %s", i, model_id, exc)
            continue

        predicted = normalize(response.text)
        actual = str(row["label"])
        correct = int(predicted == actual)

        records.append({
            "input": row["input"],
            "predicted": predicted,
            "actual": actual,
            "correct": correct,
            "confidence": response.confidence,
            "confidence_source": response.confidence_source,
            "latency": round(response.latency, 4),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost": response.cost,
        })

        status = "OK  " if correct else "MISS"
        conf_str = f"  conf={response.confidence:.0%}" if response.confidence is not None else ""
        print(f"    [{i+1:02d}/{len(dataset)}] {status}  actual={actual:14s}  predicted={predicted}{conf_str}")

    if not records:
        return records, {}

    n = len(records)
    summary = {
        "alias": alias,
        "model_id": model_id,
        "cost_tier": model_entry["cost_tier"],
        "samples": n,
        "accuracy": round(sum(r["correct"] for r in records) / n, 4),
        "avg_latency": round(sum(r["latency"] for r in records) / n, 4),
        "total_cost": round(sum(r["cost"] for r in records), 6),
        "avg_confidence": round(
            sum(r["confidence"] for r in records if r["confidence"] is not None)
            / max(sum(1 for r in records if r["confidence"] is not None), 1), 4
        ) if any(r["confidence"] is not None for r in records) else None,
    }
    return records, summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run classification benchmark")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--models", nargs="+", metavar="ALIAS",
                       help="Model aliases from models.yaml (e.g. gpt41_mini gpt4o)")
    group.add_argument("--cost-tier", choices=["low", "medium", "high"],
                       help="Run all enabled models of this cost tier")
    group.add_argument("--task-match", action="store_true",
                       help="Run all enabled models that list 'classification' as a strength")
    p.add_argument("--sample-size", type=int, default=20,
                   help="Number of examples per model (default: 20)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for dataset sampling (default: 42)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    all_models = load_models()
    selected = select_models(all_models, args.models, args.cost_tier, args.task_match)

    if not selected:
        print("No models selected. Check models.yaml or your filter flags.")
        return

    gen_config = load_generation_config()
    generation_params = get_generation_params(TASK, gen_config)

    dataset_config = _load_dataset_config()[TASK]
    label_map = dataset_config.get("label_map", {})
    categories = list(label_map.values()) if label_map else []

    print(f"Task:         {TASK}")
    print(f"Dataset:      {dataset_config['hf_name']}")
    print(f"Categories:   {', '.join(categories)}")
    model_list = ", ".join(f'{m["alias"]} ({m["model_id"]})' for m in selected.values())
    print(f"Models:       {model_list}")
    print(f"Samples:      {args.sample_size}  seed={args.seed}")
    print(f"Gen params:   {generation_params}")
    print()

    dataset = load_task_dataset(TASK, sample_size=args.sample_size, seed=args.seed)

    all_results: dict[str, list[dict]] = {}
    summaries: list[dict] = []

    for alias, model_entry in selected.items():
        print(f"── {alias}  ({model_entry['model_id']})")
        records, summary = run_one_model(model_entry, dataset, categories, generation_params)
        if records:
            all_results[alias] = records
            summaries.append(summary)
            conf_str = f"  avg_conf={summary['avg_confidence']:.0%}" if summary["avg_confidence"] else ""
            print(f"   accuracy={summary['accuracy']:.0%}  "
                  f"avg_latency={summary['avg_latency']:.2f}s  "
                  f"total_cost=${summary['total_cost']:.4f}{conf_str}")
        print()

    # Comparison table
    if len(summaries) > 1:
        print("── Comparison ──────────────────────────────────────────────────")
        header = f"{'Alias':<20} {'Model':<30} {'Tier':<8} {'Accuracy':>10} {'Latency':>10} {'Cost':>10}"
        print(header)
        print("-" * len(header))
        for s in sorted(summaries, key=lambda x: x["accuracy"], reverse=True):
            print(f"{s['alias']:<20} {s['model_id']:<30} {s['cost_tier']:<8} "
                  f"{s['accuracy']:>9.0%} {s['avg_latency']:>9.2f}s ${s['total_cost']:>8.4f}")

    # Save output
    output_dir = Path("results/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{TASK}_{timestamp}.json"

    with output_path.open("w") as f:
        json.dump({
            "task": TASK,
            "timestamp": timestamp,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "generation_params": generation_params,
            "categories": categories,
            "summaries": summaries,
            "results": all_results,
        }, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()