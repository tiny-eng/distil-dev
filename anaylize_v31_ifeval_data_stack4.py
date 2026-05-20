#!/usr/bin/env python3
"""
Filter IFEval result JSON.

This script selects only items where:
- stack_depth == 4 OR src contains "stack4"
- tokens < 1024

Then it recalculates:
- n
- correct
- pass_frac
- completion_tokens

Example:

python -m distil-dev.anaylize_v31_ifeval_data_stack4 \
  --input distil-dev/results/chutes_ifeval_results_parallel_stack4_100.json \
  --output distil-dev/results/anaylize_chutes_ifeval_stack4_tokens_lt1024.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any


def get_tokens(item: dict[str, Any]) -> int:
    """
    Safely get token count from item.
    """
    try:
        return int(item.get("tokens") or 0)
    except Exception:
        return 0


def filter_items(
    items: list[dict[str, Any]],
    stack_depth: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """
    Keep only items with exact stack_depth and tokens below max_tokens.
    """
    filtered: list[dict[str, Any]] = []

    for item in items:
        item_stack_depth = item.get("stack_depth")
        item_tokens = get_tokens(item)

        if item_stack_depth != stack_depth:
            continue

        if item_tokens >= max_tokens:
            continue

        filtered.append(item)

    return filtered


def build_filtered_result(
    original: dict[str, Any],
    filtered_items: list[dict[str, Any]],
    stack_depth: int,
    max_tokens: int,
) -> dict[str, Any]:
    """
    Build output JSON with recalculated metrics.
    """
    n = len(filtered_items)
    correct = sum(1 for item in filtered_items if bool(item.get("ok")))
    pass_frac = correct / n if n > 0 else 0.0
    completion_tokens = sum(get_tokens(item) for item in filtered_items)

    result: dict[str, Any] = {}

    # Keep useful original metadata if present.
    keep_keys = [
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

    for key in keep_keys:
        if key in original:
            result[key] = original[key]

    result["filter"] = {
        "stack_depth": stack_depth,
        "tokens": f"< {max_tokens}",
    }

    result["original_summary"] = {
        "n": original.get("n"),
        "correct": original.get("correct"),
        "pass_frac": original.get("pass_frac"),
        "completion_tokens": original.get("completion_tokens"),
    }

    result["n"] = n
    result["correct"] = correct
    result["pass_frac"] = pass_frac
    result["completion_tokens"] = completion_tokens
    result["items"] = filtered_items

    return result


def analyze_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analyze filtered items.
    """
    n = len(items)
    correct = sum(1 for item in items if bool(item.get("ok")))
    pass_frac = correct / n if n > 0 else 0.0
    completion_tokens = sum(get_tokens(item) for item in items)
    avg_tokens = completion_tokens / n if n > 0 else 0.0

    failed_items = [item for item in items if not item.get("ok")]
    passed_items = [item for item in items if item.get("ok")]

    failed_instruction_counts: dict[str, int] = {}
    instruction_stats: dict[str, dict[str, int]] = {}

    for item in items:
        instruction_ids = item.get("instruction_ids") or []
        per_instruction = item.get("per_instruction") or []

        for instruction_id, passed in zip(instruction_ids, per_instruction):
            if instruction_id not in instruction_stats:
                instruction_stats[instruction_id] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                }

            instruction_stats[instruction_id]["total"] += 1

            if bool(passed):
                instruction_stats[instruction_id]["passed"] += 1
            else:
                instruction_stats[instruction_id]["failed"] += 1
                failed_instruction_counts[instruction_id] = (
                    failed_instruction_counts.get(instruction_id, 0) + 1
                )

    instruction_pass_rates: dict[str, dict[str, float | int]] = {}

    for instruction_id, stats in instruction_stats.items():
        total = stats["total"]
        passed = stats["passed"]
        failed = stats["failed"]

        instruction_pass_rates[instruction_id] = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_frac": passed / total if total > 0 else 0.0,
        }

    most_failed_instructions = sorted(
        failed_instruction_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "n": n,
        "correct": correct,
        "pass_frac": pass_frac,
        "completion_tokens": completion_tokens,
        "avg_tokens": avg_tokens,
        "passed_count": len(passed_items),
        "failed_count": len(failed_items),
        "instruction_pass_rates": instruction_pass_rates,
        "most_failed_instructions": most_failed_instructions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Keep only IFEval items with exact stack_depth == 4 and tokens < 1024."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input IFEval result JSON file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output filtered JSON file.",
    )

    parser.add_argument(
        "--stack-depth",
        type=int,
        default=4,
        help="Exact stack_depth to keep. Default: 4.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Keep only items with tokens lower than this. Default: 1024.",
    )

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_items = data.get("items") or []

    filtered_items = filter_items(
        original_items,
        stack_depth=args.stack_depth,
        max_tokens=args.max_tokens,
    )

    result = build_filtered_result(
        original=data,
        filtered_items=filtered_items,
        stack_depth=args.stack_depth,
        max_tokens=args.max_tokens,
    )

    result["analysis"] = analyze_items(filtered_items)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 80)
    print("Filtered IFEval Result")
    print("=" * 80)
    print(f"Input:              {args.input}")
    print(f"Output:             {args.output}")
    print()
    print("Filter:")
    print(f"  stack_depth:      == {args.stack_depth}")
    print(f"  tokens:           < {args.max_tokens}")
    print()
    print("Original:")
    print(f"  n:                {data.get('n')}")
    print(f"  correct:          {data.get('correct')}")
    print(f"  pass_frac:        {data.get('pass_frac')}")
    print(f"  completion_tokens:{data.get('completion_tokens')}")
    print()
    print("Filtered:")
    print(f"  n:                {result['n']}")
    print(f"  correct:          {result['correct']}")
    print(f"  pass_frac:        {result['pass_frac']:.4f}")
    print(f"  completion_tokens:{result['completion_tokens']}")
    print(f"  avg_tokens:       {result['analysis']['avg_tokens']:.2f}")
    print(f"  passed_count:     {result['analysis']['passed_count']}")
    print(f"  failed_count:     {result['analysis']['failed_count']}")
    print()

    print("Most failed instructions:")
    for instruction_id, count in result["analysis"]["most_failed_instructions"][:20]:
        print(f"  {count:4d}  {instruction_id}")

    print("=" * 80)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())