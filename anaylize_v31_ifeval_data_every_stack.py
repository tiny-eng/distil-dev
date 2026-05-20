#!/usr/bin/env python3
"""
Analyze IFEval result JSON by stack_depth.

For each stack_depth, this script calculates:
- n
- correct
- pass_frac
- completion_tokens
- avg_tokens

Example:

python -m distil-dev.anaylize_v31_ifeval_data_every_stack \
  --input distil-dev/results/chutes_ifeval_results_parallel_2000.json \
  --output distil-dev/results/anaylize_chutes_ifeval_every_stack.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any


def get_tokens(item: dict[str, Any]) -> int:
    """
    Safely read token count from an item.
    """
    try:
        return int(item.get("tokens") or 0)
    except Exception:
        return 0


def get_stack_depth(item: dict[str, Any]) -> int | str:
    """
    Use stack_depth only.

    If stack_depth is missing, return 'unknown'.
    """
    if item.get("stack_depth") is not None:
        return item.get("stack_depth")

    return "unknown"


def analyze_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Calculate metrics for one stack_depth group.
    """
    n = len(items)
    correct = sum(1 for item in items if bool(item.get("ok")))
    pass_frac = correct / n if n > 0 else 0.0

    completion_tokens = sum(get_tokens(item) for item in items)
    avg_tokens = completion_tokens / n if n > 0 else 0.0

    return {
        "n": n,
        "correct": correct,
        "pass_frac": pass_frac,
        "completion_tokens": completion_tokens,
        "avg_tokens": avg_tokens,
    }


def analyze_by_stack_depth(data: dict[str, Any]) -> dict[str, Any]:
    """
    Group items by stack_depth and analyze each group.
    """
    items = data.get("items") or []

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in items:
        stack_depth = get_stack_depth(item)
        key = str(stack_depth)
        groups[key].append(item)

    by_stack_depth: dict[str, Any] = {}

    def sort_key(x: str) -> tuple[int, str]:
        if x.isdigit():
            return (0, f"{int(x):04d}")
        return (1, x)

    for stack_key in sorted(groups.keys(), key=sort_key):
        group_items = groups[stack_key]
        by_stack_depth[f"stack{stack_key}"] = analyze_group(group_items)

    overall = analyze_group(items)

    result: dict[str, Any] = {
        "overall": overall,
        "by_stack_depth": by_stack_depth,
    }

    metadata_keys = [
        "axis",
        "model",
        "base_url",
        "block_seed",
        "requested_n_items",
        "max_tokens",
        "temperature",
        "timeout",
        "concurrency",
        "save_every",
        "status",
    ]

    metadata: dict[str, Any] = {}

    for key in metadata_keys:
        if key in data:
            metadata[key] = data[key]

    result["metadata"] = metadata

    result["original_summary"] = {
        "n": data.get("n"),
        "correct": data.get("correct"),
        "pass_frac": data.get("pass_frac"),
        "completion_tokens": data.get("completion_tokens"),
    }

    return result


def print_report(result: dict[str, Any]) -> None:
    """
    Print readable report.
    """
    print()
    print("=" * 80)
    print("IFEval Analysis by stack_depth")
    print("=" * 80)

    overall = result["overall"]

    print("Overall")
    print("-" * 80)
    print(f"n:                 {overall['n']}")
    print(f"correct:           {overall['correct']}")
    print(f"pass_frac:         {overall['pass_frac']:.4f}")
    print(f"completion_tokens: {overall['completion_tokens']}")
    print(f"avg_tokens:        {overall['avg_tokens']:.2f}")
    print()

    print("By stack_depth")
    print("-" * 80)
    print(f"{'stack_depth':<12} {'n':>8} {'correct':>10} {'pass_frac':>12} {'avg_tokens':>12}")
    print("-" * 80)

    by_stack_depth = result["by_stack_depth"]

    for stack_name, stats in by_stack_depth.items():
        print(
            f"{stack_name:<12} "
            f"{stats['n']:>8} "
            f"{stats['correct']:>10} "
            f"{stats['pass_frac']:>12.4f} "
            f"{stats['avg_tokens']:>12.2f}"
        )

    print("=" * 80)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze IFEval JSON results grouped by stack_depth."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input IFEval result JSON file.",
    )

    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON file for analysis.",
    )

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = analyze_by_stack_depth(data)

    print_report(result)

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Saved stack_depth analysis JSON to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
