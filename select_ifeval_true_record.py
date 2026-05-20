#!/usr/bin/env python3
"""
Select only successful IFEval records with tokens < max_tokens.

Input JSON format:
{
  "n": 2000,
  "correct": 1491,
  "pass_frac": 0.7455,
  "completion_tokens": 1270611,
  "items": [
    {
      "ok": true,
      "prompt": "...",
      "response_cleaned": "...",
      "tokens": 123
    }
  ]
}

Output JSON format:
{
  "n": 1491,
  "correct": 1491,
  "pass_frac": 1.0,
  "filter": {
    "ok": true,
    "tokens": "< 1024"
  },
  "items": [
    {
      "ok": true,
      "prompt": "...",
      "response_cleaned": "...",
      "tokens": 123
    }
  ]
}

Example:

python -m distil-dev.select_ifeval_true_record \
  --input distil-dev/results/chutes_ifeval_results_parallel_2000.json \
  --output distil-dev/results/select_ok_true_ifeval_results_parallel_2000.json \
  --max-tokens 1024
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def get_tokens(item: dict[str, Any]) -> int:
    """
    Safely read token count from item.
    """
    try:
        return int(item.get("tokens") or 0)
    except Exception:
        return 0


def select_ok_records(
    data: dict[str, Any],
    max_tokens: int,
) -> list[dict[str, Any]]:
    """
    Select only items where:
    - ok == true
    - tokens < max_tokens

    Keep:
    - ok
    - prompt
    - response_cleaned
    - tokens
    """
    selected: list[dict[str, Any]] = []

    items = data.get("items") or []

    for item in items:
        if item.get("ok") is not True:
            continue

        tokens = get_tokens(item)

        if tokens >= max_tokens:
            continue

        prompt = item.get("prompt", "")
        response_cleaned = item.get("response_cleaned", "")

        if not isinstance(prompt, str):
            prompt = str(prompt)

        if not isinstance(response_cleaned, str):
            response_cleaned = str(response_cleaned)

        selected.append(
            {
                "ok": True,
                "prompt": prompt,
                "response_cleaned": response_cleaned,
                "tokens": tokens,
            }
        )

    return selected


def build_output(
    *,
    original: dict[str, Any],
    selected: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    """
    Build output JSON.
    """
    n = len(selected)
    completion_tokens = sum(int(item.get("tokens") or 0) for item in selected)

    result: dict[str, Any] = {
        "n": n,
        "correct": n,
        "pass_frac": 1.0 if n > 0 else 0.0,
        "completion_tokens": completion_tokens,
        "filter": {
            "ok": True,
            "tokens": f"< {max_tokens}",
        },
        "items": selected,
        "source_summary": {
            "n": original.get("n"),
            "correct": original.get("correct"),
            "pass_frac": original.get("pass_frac"),
            "completion_tokens": original.get("completion_tokens"),
        },
    }

    # Optional metadata copied from original.
    for key in [
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
    ]:
        if key in original:
            result[key] = original[key]

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select only ok == true records with tokens < max_tokens from IFEval JSON."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input IFEval result JSON file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Keep only records with tokens lower than this value. Default: 1024.",
    )

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    selected = select_ok_records(
        data=data,
        max_tokens=args.max_tokens,
    )

    result = build_output(
        original=data,
        selected=selected,
        max_tokens=args.max_tokens,
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 80)
    print("Selected ok == true and tokens < max_tokens Records")
    print("=" * 80)
    print(f"Input:        {args.input}")
    print(f"Output:       {args.output}")
    print()
    print("Filter:")
    print("  ok:         True")
    print(f"  tokens:     < {args.max_tokens}")
    print()
    print("Original:")
    print(f"  n:          {data.get('n')}")
    print(f"  correct:    {data.get('correct')}")
    print(f"  pass_frac:  {data.get('pass_frac')}")
    print(f"  tokens:     {data.get('completion_tokens')}")
    print()
    print("Selected:")
    print(f"  n:          {result['n']}")
    print(f"  correct:    {result['correct']}")
    print(f"  pass_frac:  {result['pass_frac']}")
    print(f"  tokens:     {result['completion_tokens']}")
    print("=" * 80)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
