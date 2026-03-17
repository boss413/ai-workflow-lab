"""
scripts/run_generation_benchmark.py

Generation benchmark runner using CommonGen dataset.

CommonGen requires generating a plausible sentence that uses all given concepts.
Input: comma-separated concept words (e.g. "ski, mountain, skier, slope, race")
Reference: one human-written example sentence

Scoring:
  Concept coverage  — fraction of input concepts present in the output (stem-matched)
  ROUGE-L           — longest common subsequence overlap with reference
  correct           — concept_coverage >= 0.8 AND rouge_l >= 0.1

ROUGE is secondary here — a model can write a perfectly good sentence
that scores low on ROUGE simply by using different phrasing than the reference.
Concept coverage is the primary signal.

Install: pip install rouge-score nltk
         python -c "import nltk; nltk.download('punkt_tab')"  (for stemming)

Output: results/raw/generation_<timestamp>.json

Usage:
  python -m scripts.run_generation_benchmark
  python -m scripts.run_generation_benchmark --models gpt41_mini claude_haiku
  python -m scripts.run_generation_benchmark --cost-tier low
  python -m scripts.run_generation_benchmark --sample-size 20
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

TASK = "generation"
COVERAGE_THRESHOLD = 0.8   # fraction of concepts that must appear in output
ROUGE_L_THRESHOLD  = 0.1   # minimum ROUGE-L (very low — penalises empty output only)


# ---------------------------------------------------------------------------
# Concept coverage scoring
# ---------------------------------------------------------------------------

def _simple_stem(word: str) -> str:
    """
    Lightweight stemmer: strips common suffixes so 'skiing' matches 'ski',
    'races' matches 'race', etc. Avoids requiring NLTK at runtime.
    """
    word = word.lower()
    for suffix in ("ing", "tion", "ed", "er", "ers", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def score_concept_coverage(output: str, concepts: list[str]) -> float:
    """
    Return the fraction of concepts that appear in the output text.
    Uses simple suffix stemming so 'skiing' satisfies concept 'ski'.
    """
    if not output.strip() or not concepts:
        return 0.0

    output_words = set(_simple_stem(w) for w in re.findall(r'\w+', output.lower()))
    matched = sum(
        1 for c in concepts
        if _simple_stem(c) in output_words or c.lower() in output.lower()
    )
    return round(matched / len(concepts), 4)


def _rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 via rouge-score library. Returns 0.0 if library unavailable."""
    try:
        from rouge_score import rouge_scorer  # noqa: PLC0415
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return round(scorer.score(reference, prediction)["rougeL"].fmeasure, 4)
    except ImportError:
        return 0.0


def score_generation(
    output: str,
    concepts_str: str,
    reference: str,
) -> dict[str, float]:
    """
    Score a generation output.

    Args:
        output:       Model's generated sentence.
        concepts_str: Comma-separated concept string (the 'input' field).
        reference:    Reference sentence (the 'label' field).

    Returns dict with coverage, rouge_l, correct.
    """
    concepts = [c.strip() for c in concepts_str.split(",") if c.strip()]
    coverage = score_concept_coverage(output, concepts)
    rl       = _rouge_l(output, reference)
    correct  = int(coverage >= COVERAGE_THRESHOLD and rl >= ROUGE_L_THRESHOLD)
    return {
        "concept_coverage": coverage,
        "rouge_l":          rl,
        "correct":          correct,
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

        output    = response.text.strip()
        reference = str(row["label"])
        scores    = score_generation(output, row["input"], reference)

        records.append({
            "concepts":        row["input"],
            "predicted":       output,
            "reference":       reference,
            "concept_coverage": scores["concept_coverage"],
            "rouge_l":         scores["rouge_l"],
            "correct":         scores["correct"],
            "confidence":      response.confidence,
            "confidence_source": response.confidence_source,
            "latency":         round(response.latency, 4),
            "input_tokens":    response.input_tokens,
            "output_tokens":   response.output_tokens,
            "cost":            response.cost,
        })

        status = "OK  " if scores["correct"] else "MISS"
        print(f"    [{i+1:02d}/{len(dataset)}] {status}  "
              f"cov={scores['concept_coverage']:.2f}  "
              f"rL={scores['rouge_l']:.2f}  "
              f"| {output[:70]!r}")

    if not records:
        return records, {}

    n = len(records)
    summary = {
        "alias":            alias,
        "model_id":         model_id,
        "cost_tier":        model_entry["cost_tier"],
        "samples":          n,
        "accuracy":         round(sum(r["correct"]          for r in records) / n, 4),
        "avg_coverage":     round(sum(r["concept_coverage"] for r in records) / n, 4),
        "avg_rouge_l":      round(sum(r["rouge_l"]          for r in records) / n, 4),
        "avg_latency":      round(sum(r["latency"]          for r in records) / n, 4),
        "total_cost":       round(sum(r["cost"]             for r in records), 6),
    }
    return records, summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run generation benchmark")
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
    print(f"Dataset:     common_gen (validation)")
    print(f"Scoring:     concept coverage + ROUGE-L")
    print(f"             correct = coverage >= {COVERAGE_THRESHOLD} AND rouge_l >= {ROUGE_L_THRESHOLD}")
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
            print(f"   accuracy={summary['accuracy']:.0%}  "
                  f"avg_coverage={summary['avg_coverage']:.2f}  "
                  f"avg_rougeL={summary['avg_rouge_l']:.2f}  "
                  f"latency={summary['avg_latency']:.2f}s  "
                  f"cost=${summary['total_cost']:.4f}")
        print()

    if len(summaries) > 1:
        print("── Comparison ──────────────────────────────────────────────────")
        header = (f"{'Alias':<20} {'Model':<35} {'Acc':>6} "
                  f"{'Cov':>6} {'RL':>6} {'Latency':>10} {'Cost':>10}")
        print(header)
        print("-" * len(header))
        for s in sorted(summaries, key=lambda x: x["avg_coverage"], reverse=True):
            print(f"{s['alias']:<20} {s['model_id']:<35} "
                  f"{s['accuracy']:>5.0%} "
                  f"{s['avg_coverage']:>6.2f} {s['avg_rouge_l']:>6.2f} "
                  f"{s['avg_latency']:>9.2f}s ${s['total_cost']:>8.4f}")

    output_dir  = Path("results/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{TASK}_{timestamp}.json"

    with output_path.open("w") as f:
        json.dump({
            "task":               TASK,
            "timestamp":          timestamp,
            "sample_size":        args.sample_size,
            "seed":               args.seed,
            "generation_params":  generation_params,
            "coverage_threshold": COVERAGE_THRESHOLD,
            "rouge_l_threshold":  ROUGE_L_THRESHOLD,
            "summaries":          summaries,
            "results":            all_results,
        }, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()