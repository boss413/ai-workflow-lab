"""
scripts/run_all_benchmarks.py

Run all six benchmark tasks in sequence.

Results are saved to results/raw/ per the individual benchmark scripts.
A summary of pass/fail per task is printed at the end.

Usage:
  python -m scripts.run_all_benchmarks
  python -m scripts.run_all_benchmarks --sample-size 50
  python -m scripts.run_all_benchmarks --cost-tier low
  python -m scripts.run_all_benchmarks --models gpt41_mini claude_haiku
  python -m scripts.run_all_benchmarks --tasks classification extraction reasoning
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime

TASKS = ["classification", "extraction", "summarization", "qa", "reasoning", "generation"]

TASK_SCRIPTS = {
    "classification": "scripts.run_classification_benchmark",
    "extraction":     "scripts.run_extraction_benchmark",
    "summarization":  "scripts.run_summarization_benchmark",
    "qa":             "scripts.run_qa_benchmark",
    "reasoning":      "scripts.run_reasoning_benchmark",
    "generation":     "scripts.run_generation_benchmark",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all benchmark tasks in sequence")
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS,
                   metavar="TASK", help="Tasks to run (default: all)")
    p.add_argument("--sample-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=100,
                   help="Random seed for dataset sampling (default: 100). "
                        "Use different seeds across runs to reduce sample bias.")

    group = p.add_mutually_exclusive_group()
    group.add_argument("--models", nargs="+", metavar="ALIAS")
    group.add_argument("--cost-tier", choices=["low", "medium", "high"])
    group.add_argument("--task-match", action="store_true",
                       help="Per task, run only models that list it as a strength")
    return p.parse_args()


def build_cmd(task: str, args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "-m", TASK_SCRIPTS[task],
           "--sample-size", str(args.sample_size),
           "--seed", str(args.seed)]
    if args.models:
        cmd += ["--models"] + args.models
    elif args.cost_tier:
        cmd += ["--cost-tier", args.cost_tier]
    elif args.task_match:
        cmd += ["--task-match"]
    return cmd


def main() -> None:
    args = parse_args()
    tasks = args.tasks

    start_time = datetime.now()
    print(f"{'='*60}")
    print(f"  Full benchmark run started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Tasks:       {', '.join(tasks)}")
    print(f"  Sample size: {args.sample_size}  seed={args.seed}")
    if args.models:
        print(f"  Models:      {', '.join(args.models)}")
    elif args.cost_tier:
        print(f"  Cost tier:   {args.cost_tier}")
    elif args.task_match:
        print(f"  Selection:   task-match")
    else:
        print(f"  Selection:   all enabled models")
    print(f"{'='*60}\n")

    results: list[dict] = []

    for i, task in enumerate(tasks, 1):
        print(f"\n{'─'*60}")
        print(f"  [{i}/{len(tasks)}] {task.upper()}")
        print(f"{'─'*60}")

        cmd = build_cmd(task, args)
        task_start = time.time()

        proc = subprocess.run(cmd, text=True)

        elapsed = time.time() - task_start
        success = proc.returncode == 0

        results.append({
            "task":    task,
            "success": success,
            "elapsed": round(elapsed, 1),
        })

        status = "✓ done" if success else "✗ failed"
        print(f"\n  {status} in {elapsed:.1f}s")

    # Final summary
    total_elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*60}")
    print(f"  Run complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total time:  {total_elapsed/60:.1f} min")
    print(f"{'='*60}")
    print(f"  {'Task':<20} {'Status':<10} {'Time':>8}")
    print(f"  {'-'*40}")
    for r in results:
        status = "done" if r["success"] else "FAILED"
        print(f"  {r['task']:<20} {status:<10} {r['elapsed']:>6.1f}s")

    failed = [r["task"] for r in results if not r["success"]]
    if failed:
        print(f"\n  Failed tasks: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\n  All tasks completed. Results in results/raw/")


if __name__ == "__main__":
    main()