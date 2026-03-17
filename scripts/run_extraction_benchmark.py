"""
scripts/run_extraction_benchmark.py

Extraction benchmark runner — Named Entity Recognition using CoNLL2003.

The model receives token-joined text and must return a JSON object with:
  {"persons": [...], "organizations": [...], "locations": [...]}

Scoring uses token-level F1 across all three entity types combined.
MISC entities in CoNLL are intentionally excluded (prompt doesn't request them).

Output: results/raw/extraction_<timestamp>.json

Usage:
  python -m scripts.run_extraction_benchmark
  python -m scripts.run_extraction_benchmark --models gpt41_mini claude_haiku
  python -m scripts.run_extraction_benchmark --cost-tier low
  python -m scripts.run_extraction_benchmark --sample-size 50
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

TASK = "extraction"
CONFIG_DIR = Path("config")

# CoNLL2003 NER tag int → BIO label
_NER_TAGS: dict[int, str] = {
    0: "O",
    1: "B-PER", 2: "I-PER",
    3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC",
    7: "B-MISC", 8: "I-MISC",
}

# Map CoNLL entity types to the keys the prompt uses
_TYPE_MAP = {"PER": "persons", "ORG": "organizations", "LOC": "locations"}


# ---------------------------------------------------------------------------
# Ground truth extraction from CoNLL NER tag list
# ---------------------------------------------------------------------------

def decode_ner_tags(tokens_str: str, tag_ids: list[int]) -> dict[str, list[str]]:
    """
    Convert a joined token string + list of CoNLL int tags into
    {persons, organizations, locations} entity lists.

    MISC entities are excluded — the prompt doesn't ask for them.
    """
    tokens = tokens_str.split()
    entities: dict[str, list[str]] = {"persons": [], "organizations": [], "locations": []}
    current: list[str] = []
    current_type: str | None = None

    for token, tag_id in zip(tokens, tag_ids):
        tag = _NER_TAGS.get(tag_id, "O")
        if tag == "O":
            if current and current_type:
                entities[current_type].append(" ".join(current))
            current, current_type = [], None
        elif tag.startswith("B-"):
            if current and current_type:
                entities[current_type].append(" ".join(current))
            ent_type = _TYPE_MAP.get(tag[2:])
            if ent_type:
                current, current_type = [token], ent_type
            else:
                current, current_type = [], None  # MISC — skip
        elif tag.startswith("I-") and current:
            current.append(token)

    if current and current_type:
        entities[current_type].append(" ".join(current))

    return entities


# ---------------------------------------------------------------------------
# Model output parsing
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> dict[str, list[str]] | None:
    """
    Extract and parse a JSON object from model output.

    Handles:
    - Raw JSON
    - JSON wrapped in ```json ... ``` fences
    - Partial matches where only a JSON object is present

    Returns None if no valid JSON with the expected keys is found.
    """
    # Strip markdown fences if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Try parsing directly
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return _normalize_entity_dict(data)
    except json.JSONDecodeError:
        pass

    # Try to extract first {...} block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return _normalize_entity_dict(data)
        except json.JSONDecodeError:
            pass

    return None


def _normalize_entity_dict(data: dict) -> dict[str, list[str]]:
    """Ensure all three keys exist and values are lists of strings."""
    result: dict[str, list[str]] = {"persons": [], "organizations": [], "locations": []}
    for key in result:
        val = data.get(key, [])
        if isinstance(val, list):
            result[key] = [str(v).strip() for v in val if v]
    return result


# ---------------------------------------------------------------------------
# Scoring — token-level F1
# ---------------------------------------------------------------------------

def _entity_set(entities: dict[str, list[str]]) -> set[tuple[str, str]]:
    """Return a set of (entity_text_lower, entity_type) tuples for F1 comparison."""
    result = set()
    for etype, elist in entities.items():
        for e in elist:
            result.add((e.lower().strip(), etype))
    return result


def score_extraction(
    predicted: dict[str, list[str]] | None,
    actual: dict[str, list[str]],
) -> dict[str, float]:
    """
    Compute precision, recall, and F1 over entity (text, type) pairs.

    A predicted entity is correct if it exactly matches (case-insensitive)
    an entity in the ground truth for the same type.

    Returns {"precision": float, "recall": float, "f1": float, "correct": int}
    where correct=1 if F1 >= 0.5, else 0.
    """
    if predicted is None:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "correct": 0}

    pred_set = _entity_set(predicted)
    true_set = _entity_set(actual)

    if not true_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "correct": 1}

    tp = len(pred_set & true_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall    = tp / len(true_set) if true_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "correct":   int(f1 >= 0.5),
    }


# ---------------------------------------------------------------------------
# Model selection (shared with classification runner)
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
        try:
            prompt   = generate_prompt(TASK, row)
            response = adapter.run(prompt, generation_params=generation_params)
        except Exception as exc:
            if _is_fatal(exc):
                print(f"  [ABORT] {alias}: {exc}")
                return [], {}
            logging.warning("Error on row %d for %s: %s", i, model_id, exc)
            continue

        # Decode ground truth from CoNLL tag list
        actual_entities = decode_ner_tags(row["input"], row["label"])

        # Parse model JSON output
        predicted_entities = parse_json_response(response.text)

        scores = score_extraction(predicted_entities, actual_entities)

        records.append({
            "input":          row["input"],
            "predicted_json": predicted_entities,
            "actual_entities": actual_entities,
            "precision":      scores["precision"],
            "recall":         scores["recall"],
            "f1":             scores["f1"],
            "correct":        scores["correct"],
            "confidence":     response.confidence,
            "confidence_source": response.confidence_source,
            "latency":        round(response.latency, 4),
            "input_tokens":   response.input_tokens,
            "output_tokens":  response.output_tokens,
            "cost":           response.cost,
        })

        status = "OK  " if scores["correct"] else "MISS"
        pred_str = json.dumps(predicted_entities) if predicted_entities else "[parse error]"
        print(f"    [{i+1:02d}/{len(dataset)}] {status}  f1={scores['f1']:.2f}  {pred_str}")

    if not records:
        return records, {}

    n = len(records)
    summary = {
        "alias":        alias,
        "model_id":     model_id,
        "cost_tier":    model_entry["cost_tier"],
        "samples":      n,
        "accuracy":     round(sum(r["correct"]   for r in records) / n, 4),
        "avg_f1":       round(sum(r["f1"]        for r in records) / n, 4),
        "avg_precision":round(sum(r["precision"] for r in records) / n, 4),
        "avg_recall":   round(sum(r["recall"]    for r in records) / n, 4),
        "avg_latency":  round(sum(r["latency"]   for r in records) / n, 4),
        "total_cost":   round(sum(r["cost"]      for r in records), 6),
    }
    return records, summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run extraction benchmark")
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
    print(f"Dataset:     conll2003")
    print(f"Entities:    persons, organizations, locations")
    print(f"Scoring:     token-level F1  (correct = F1 >= 0.5)")
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
                  f"precision={summary['avg_precision']:.2f}  "
                  f"recall={summary['avg_recall']:.2f}  "
                  f"latency={summary['avg_latency']:.2f}s  "
                  f"cost=${summary['total_cost']:.4f}")
        print()

    if len(summaries) > 1:
        print("── Comparison ──────────────────────────────────────────────────")
        header = f"{'Alias':<20} {'Model':<30} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Latency':>10} {'Cost':>10}"
        print(header)
        print("-" * len(header))
        for s in sorted(summaries, key=lambda x: x["avg_f1"], reverse=True):
            print(f"{s['alias']:<20} {s['model_id']:<30} "
                  f"{s['avg_f1']:>6.2f} {s['avg_precision']:>6.2f} "
                  f"{s['avg_recall']:>6.2f} {s['avg_latency']:>9.2f}s "
                  f"${s['total_cost']:>8.4f}")

    output_dir = Path("results/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
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