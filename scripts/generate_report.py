"""
scripts/generate_report.py

Generate benchmark summary tables from results/raw/*.json files.

Reads all benchmark output files, aggregates across multiple seeds/runs
per (task, model), and produces:

  results/summary/benchmark_summary.md   — full markdown table per task
  results/summary/benchmark_summary.csv  — same data as CSV for Excel/Sheets

Output columns vary by task:
  All tasks:      model, tier, samples, accuracy, avg_latency, total_cost
  extraction/qa:  + f1, precision/recall or em
  summarization:  + rouge1, rouge2, rougeL
  reasoning:      + parse_errors, avg_output_tokens
  generation:     + coverage, rouge_l

Usage:
  python -m scripts.generate_report
  python -m scripts.generate_report --results-dir results/raw
  python -m scripts.generate_report --tasks classification reasoning
  python -m scripts.generate_report --seeds 42 100       # filter to specific seeds
  python -m scripts.generate_report --models gpt41_mini claude_haiku
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


RESULTS_DIR = Path("results/raw")
SUMMARY_DIR = Path("results/summary")

# Primary metric used for ranking within each task
PRIMARY_METRIC: dict[str, str] = {
    "classification": "accuracy",
    "extraction":     "avg_f1",
    "summarization":  "avg_rouge1",
    "qa":             "avg_f1",
    "reasoning":      "accuracy",
    "generation":     "avg_coverage",
}

# Columns to include per task, in display order
TASK_COLUMNS: dict[str, list[str]] = {
    "classification": ["accuracy", "avg_confidence", "avg_latency", "total_cost"],
    "extraction":     ["accuracy", "avg_f1", "avg_precision", "avg_recall", "avg_latency", "total_cost"],
    "summarization":  ["accuracy", "avg_rouge1", "avg_rouge2", "avg_rougeL", "avg_latency", "total_cost"],
    "qa":             ["accuracy", "avg_f1", "avg_em", "avg_latency", "total_cost"],
    "reasoning":      ["accuracy", "parse_errors", "avg_output_tokens", "avg_latency", "total_cost"],
    "generation":     ["accuracy", "avg_coverage", "avg_rouge_l", "avg_latency", "total_cost"],
}

DISPLAY_NAMES: dict[str, str] = {
    "accuracy":          "Accuracy",
    "avg_f1":            "F1",
    "avg_precision":     "Precision",
    "avg_recall":        "Recall",
    "avg_rouge1":        "ROUGE-1",
    "avg_rouge2":        "ROUGE-2",
    "avg_rougeL":        "ROUGE-L",
    "avg_em":            "Exact Match",
    "avg_coverage":      "Coverage",
    "avg_rouge_l":       "ROUGE-L",
    "avg_confidence":    "Avg Conf",
    "avg_latency":       "Latency(s)",
    "avg_output_tokens": "Avg Tokens",
    "parse_errors":      "Parse Errs",
    "total_cost":        "Cost($)",
    "samples":           "N",
    "cost_tier":         "Tier",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_result_files(
    results_dir: Path,
    tasks: list[str] | None = None,
    seeds: list[int] | None = None,
    models: list[str] | None = None,
) -> list[dict]:
    """Load all JSON result files, applying optional filters."""
    files = sorted(results_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No result files found in {results_dir}")

    records = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  Warning: skipping {f.name}: {exc}")
            continue

        task = data.get("task")
        seed = data.get("seed")

        if tasks and task not in tasks:
            continue
        if seeds and seed not in seeds:
            continue

        for summary in data.get("summaries", []):
            alias = summary.get("alias")
            if models and alias not in models:
                continue
            records.append({
                "file":      f.name,
                "task":      task,
                "seed":      seed,
                "timestamp": data.get("timestamp"),
                "sample_size": data.get("sample_size"),
                **summary,
            })

    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(records: list[dict]) -> dict[str, list[dict]]:
    """
    Aggregate records by (task, alias), averaging numeric metrics across
    multiple seeds/runs. Returns dict keyed by task.
    """
    # Group by (task, alias)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["task"], r["alias"])].append(r)

    # Identify numeric columns (exclude meta fields)
    meta = {"file", "task", "seed", "timestamp", "alias", "model_id",
            "cost_tier", "sample_size", "samples"}

    task_rows: dict[str, list[dict]] = defaultdict(list)

    for (task, alias), rows in sorted(groups.items()):
        # Base row with non-numeric fields from most recent entry
        rows_sorted = sorted(rows, key=lambda r: r.get("timestamp", ""))
        latest = rows_sorted[-1]

        agg: dict = {
            "alias":      alias,
            "model_id":   latest.get("model_id", ""),
            "cost_tier":  latest.get("cost_tier", ""),
            "runs":       len(rows),
            "samples":    sum(r.get("samples", 0) for r in rows),
        }

        # Average all numeric fields
        numeric_keys = [k for k in latest if k not in meta
                        and isinstance(latest[k], (int, float))
                        and latest[k] is not None]

        for key in numeric_keys:
            vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            if vals:
                agg[key] = round(sum(vals) / len(vals), 4)

        # total_cost should be summed not averaged
        cost_vals = [r.get("total_cost", 0) for r in rows]
        agg["total_cost"] = round(sum(cost_vals), 6)

        task_rows[task].append(agg)

    # Sort each task by primary metric descending
    for task, rows in task_rows.items():
        metric = PRIMARY_METRIC.get(task, "accuracy")
        task_rows[task] = sorted(
            rows,
            key=lambda r: r.get(metric, 0),
            reverse=True,
        )

    return dict(task_rows)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt(val, key: str) -> str:
    if val is None:
        return "-"
    if key in ("accuracy", "avg_f1", "avg_precision", "avg_recall",
               "avg_rouge1", "avg_rouge2", "avg_rougeL", "avg_em",
               "avg_coverage", "avg_rouge_l", "avg_confidence"):
        return f"{val:.0%}"
    if key == "total_cost":
        return f"${val:.4f}"
    if key == "avg_latency":
        return f"{val:.2f}s"
    if key in ("avg_output_tokens",):
        return f"{val:.0f}"
    if key == "parse_errors":
        return str(int(val))
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _model_display(row: dict) -> str:
    """Short display name: alias + model_id in parens."""
    return f"{row['alias']} ({row['model_id']})"


def task_to_markdown(task: str, rows: list[dict]) -> str:
    cols = TASK_COLUMNS.get(task, ["accuracy", "avg_latency", "total_cost"])
    # Only include cols that have data
    active_cols = [c for c in cols if any(r.get(c) is not None for r in rows)]

    headers = ["Model", "Tier", "N", "Runs"] + [DISPLAY_NAMES.get(c, c) for c in active_cols]

    table_rows = []
    for r in rows:
        row_data = [
            _model_display(r),
            r.get("cost_tier", "-"),
            str(r.get("samples", "-")),
            str(r.get("runs", 1)),
        ] + [_fmt(r.get(c), c) for c in active_cols]
        table_rows.append(row_data)

    # Column widths
    widths = [max(len(h), max((len(row[i]) for row in table_rows), default=0))
              for i, h in enumerate(headers)]

    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"

    lines = [
        f"## {task.capitalize()}",
        "",
        fmt_row(headers),
        sep,
        *[fmt_row(r) for r in table_rows],
        "",
    ]
    return "\n".join(lines)


def generate_markdown(aggregated: dict[str, list[dict]], meta: dict) -> str:
    lines = [
        "# LLM Benchmark Results",
        "",
        f"Generated: {meta['generated_at']}  ",
        f"Tasks: {', '.join(aggregated.keys())}  ",
        f"Total result files: {meta['file_count']}",
        "",
        "---",
        "",
    ]
    for task in sorted(aggregated.keys()):
        lines.append(task_to_markdown(task, aggregated[task]))
    return "\n".join(lines)


def generate_csv(aggregated: dict[str, list[dict]]) -> str:
    """Flat CSV with task column, one row per (task, model)."""
    all_keys: list[str] = []
    for rows in aggregated.values():
        for r in rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)

    import csv, io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["task"] + all_keys, extrasaction="ignore")
    writer.writeheader()
    for task, rows in sorted(aggregated.items()):
        for r in rows:
            writer.writerow({"task": task, **r})
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate benchmark summary report")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--output-dir",  type=Path, default=SUMMARY_DIR)
    p.add_argument("--tasks",  nargs="+",
                   choices=["classification","extraction","summarization",
                            "qa","reasoning","generation"])
    p.add_argument("--seeds",  nargs="+", type=int,
                   help="Only include results from these seeds")
    p.add_argument("--models", nargs="+", metavar="ALIAS",
                   help="Only include these model aliases")
    p.add_argument("--no-csv", action="store_true",
                   help="Skip CSV output")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading results from {args.results_dir}...")
    records = load_result_files(
        args.results_dir,
        tasks=args.tasks,
        seeds=args.seeds,
        models=args.models,
    )

    if not records:
        print("No matching result records found.")
        return

    print(f"Loaded {len(records)} model-run records across "
          f"{len({r['task'] for r in records})} tasks, "
          f"{len({r['alias'] for r in records})} models, "
          f"{len({r['seed'] for r in records})} seed(s).")

    aggregated = aggregate(records)

    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_count":   len({r["file"] for r in records}),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Markdown
    md = generate_markdown(aggregated, meta)
    md_path = args.output_dir / "benchmark_summary.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"Markdown saved: {md_path}")
    print()
    print(md)

    # CSV
    if not args.no_csv:
        csv_path = args.output_dir / "benchmark_summary.csv"
        csv_path.write_text(generate_csv(aggregated), encoding="utf-8")
        print(f"CSV saved:      {csv_path}")


if __name__ == "__main__":
    main()