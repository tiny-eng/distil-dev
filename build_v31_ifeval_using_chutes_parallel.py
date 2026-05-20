#!/usr/bin/env python3
"""
v31_ifeval_verifiable — Google IFEval verifiable-instructions axis.

This runner uses a Chutes / OpenAI-compatible API endpoint.

Features:
- Parallel requests with --concurrency
- Periodic partial saving with --save-every
- Overwrites one JSON result file during generation
- Scores each item immediately after generation
- Can stop midway and inspect saved JSON

Run example:

export CHUTES_API_KEY="your_key_here"

python -m distil-dev.build_v31_ifeval_using_chutes_parallel \
    --block-seed 123 \
    --n-items 2000 \
    --concurrency 10 \
    --save-every 10 \
    --timeout 300 \
    --out distil-dev/results/chutes_ifeval_results_parallel_2000.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any


import distil.pod.axes.v31.ifeval_verifiable as ifeval_verifiable
import distil.pod.axes.v31._ifeval_vendor as _ifeval_vendor


logger = logging.getLogger("ifeval")


MAX_TOKENS = 1024
AXIS_NAME = "v31_ifeval_verifiable"

DEFAULT_TIMEOUT = 180.0

CHUTE_URL = "https://llm.chutes.ai/v1/"
CHUTE_MODEL = "Qwen/Qwen3-235B-A22B-Thinking-2507"

# Do NOT hardcode real API keys here.
# Use:
#   export CHUTES_API_KEY="..."
API_KEY = "cpk_2b247fc4ec3d4aa08487bcaaa9142fd1.e8a69a1082c15c3fa53fe560fd4cbf27.WH3dFzzVDPN0geBsjCJN4G5T7ZCA8GHZ"


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_TRAIL_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)
_THINK_NARRATIVE_RE = re.compile(
    r"^\s*Thinking Process:.*?(?=\n\n[A-Z0-9]|\Z)",
    re.DOTALL,
)


def strip_thinking(text: str) -> str:
    """Drop <think>...</think> / Thinking Process leaders before extraction."""
    if not text:
        return ""

    if "<think>" in text:
        text = _THINK_BLOCK_RE.sub("", text, count=1)
    elif "</think>" in text:
        text = _THINK_TRAIL_RE.sub("", text, count=1)

    if text.lstrip().startswith("Thinking Process:"):
        text = _THINK_NARRATIVE_RE.sub("", text, count=1)

    return text.strip()


@dataclass
class BenchResult:
    n: int
    correct: int
    completion_tokens: int
    items: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        pass_frac = self.correct / self.n if self.n > 0 else 0.0

        return {
            "n": self.n,
            "correct": self.correct,
            "pass_frac": pass_frac,
            "completion_tokens": self.completion_tokens,
            "items": self.items,
        }


def atomic_write_json(path: str, data: dict[str, Any]) -> None:
    """
    Safely overwrite JSON file.

    Writes to path.tmp first, then replaces path.
    This prevents corrupted JSON if the process stops during write.
    """
    if not path:
        return

    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_path = path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    os.replace(tmp_path, path)


def _post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    api_key: str = "EMPTY",
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)

    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} from {url}: {err_body[:2000]}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to endpoint {url}: {exc}") from exc


def _post_json_with_retries(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    api_key: str = "EMPTY",
    retries: int = 3,
    retry_sleep_s: float = 2.0,
) -> dict[str, Any]:
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            return _post_json(
                url=url,
                payload=payload,
                timeout=timeout,
                api_key=api_key,
            )

        except Exception as exc:
            last_exc = exc

            logger.warning(
                "POST failed attempt %d/%d: %s",
                attempt,
                retries,
                exc,
            )

            if attempt < retries:
                time.sleep(retry_sleep_s * attempt)

    raise RuntimeError(f"POST failed after {retries} attempts: {last_exc}")


def generate_one_chutes_chat(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    retries: int = 3,
) -> tuple[str, int]:
    endpoint = base_url.rstrip("/") + "/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }

    data = _post_json_with_retries(
        endpoint,
        payload,
        timeout=timeout,
        api_key=api_key,
        retries=retries,
    )

    choices = data.get("choices") or []

    if not choices:
        raise RuntimeError(f"No choices returned by Chutes: {data}")

    choice = choices[0]
    message = choice.get("message") or {}
    text = message.get("content", "")

    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens", 0)

    if not isinstance(completion_tokens, int):
        completion_tokens = 0

    return text or "", completion_tokens


def score_one_item(
    *,
    idx: int,
    item: dict[str, Any],
    text: str,
    n_tok: int,
    include_prompts: bool,
    include_responses: bool,
) -> dict[str, Any]:
    cleaned = strip_thinking(text or "")

    try:
        all_pass, per = _ifeval_vendor.evaluate_item(
            cleaned,
            item["instruction_ids"],
            item.get("kwargs") or [],
        )
    except Exception as exc:
        logger.warning("ifeval evaluate_item crashed on item %d: %s", idx, exc)
        all_pass, per = False, []

    row: dict[str, Any] = {
        "index": idx,
        "src": item.get("src", ""),
        "topic": item.get("topic", ""),
        "instruction_ids": item.get("instruction_ids"),
        "kwargs": item.get("kwargs"),
        "per_instruction": per,
        "stack_depth": item.get("stack_depth"),
        "target_stack_depth": item.get("target_stack_depth"),
        "ok": bool(all_pass),
        "tokens": int(n_tok or 0),
        "tail": (text or "")[-120:],
    }

    if include_prompts:
        row["prompt"] = item.get("prompt", "")

    if include_responses:
        row["response_raw"] = text or ""
        row["response_cleaned"] = cleaned

    return row


def build_result_dict(
    *,
    items_scored: list[dict[str, Any]],
    axis: str,
    model: str,
    base_url: str,
    block_seed: int,
    requested_n_items: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
    concurrency: int,
    save_every: int,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    correct = sum(1 for row in items_scored if row.get("ok"))
    completion_tokens = sum(int(row.get("tokens") or 0) for row in items_scored)

    result = BenchResult(
        n=len(items_scored),
        correct=correct,
        completion_tokens=completion_tokens,
        items=items_scored,
    ).as_dict()

    result["axis"] = axis
    result["model"] = model
    result["base_url"] = base_url
    result["block_seed"] = block_seed
    result["requested_n_items"] = requested_n_items
    result["max_tokens"] = max_tokens
    result["temperature"] = temperature
    result["timeout"] = timeout
    result["concurrency"] = concurrency
    result["save_every"] = save_every
    result["status"] = status
    result["updated_at_unix"] = time.time()

    if error:
        result["error"] = error[:2000]

    return result


def run(
    *,
    base_url: str,
    api_key: str,
    model: str,
    block_seed: int,
    n_items: int,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.0,
    timeout: float = DEFAULT_TIMEOUT,
    concurrency: int = 4,
    save_every: int = 10,
    out_path: str = "",
    include_prompts: bool = True,
    include_responses: bool = True,
    retries: int = 3,
) -> dict[str, Any]:
    """
    Run the v31 IFEval verifiable axis against Chutes.

    Saves partial JSON results every `save_every` completed items.
    """
    items = ifeval_verifiable.generate_items(block_seed, n_items)

    if not items:
        result = {
            "axis": AXIS_NAME,
            "model": model,
            "base_url": base_url,
            "block_seed": block_seed,
            "requested_n_items": n_items,
            "n": 0,
            "correct": 0,
            "pass_frac": 0.0,
            "completion_tokens": 0,
            "items": [],
            "status": "empty",
        }

        if out_path:
            atomic_write_json(out_path, result)

        return result

    if concurrency < 1:
        concurrency = 1

    if save_every < 1:
        save_every = 1

    scored_by_index: dict[int, dict[str, Any]] = {}
    completed = 0

    logger.info(
        "Starting run: n_items=%d concurrency=%d save_every=%d model=%s",
        len(items),
        concurrency,
        save_every,
        model,
    )

    def worker(idx: int, item: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        prompt = item["prompt"]

        logger.info("Generating item %d/%d", idx + 1, len(items))

        try:
            text, n_tokens = generate_one_chutes_chat(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                retries=retries,
            )

            row = score_one_item(
                idx=idx,
                item=item,
                text=text,
                n_tok=n_tokens,
                include_prompts=include_prompts,
                include_responses=include_responses,
            )

            return idx, row

        except Exception as exc:
            logger.exception("Item %d failed: %s", idx, exc)

            row: dict[str, Any] = {
                "index": idx,
                "src": item.get("src", ""),
                "topic": item.get("topic", ""),
                "instruction_ids": item.get("instruction_ids"),
                "kwargs": item.get("kwargs"),
                "per_instruction": [],
                "stack_depth": item.get("stack_depth"),
                "target_stack_depth": item.get("target_stack_depth"),
                "ok": False,
                "tokens": 0,
                "tail": "",
                "error": str(exc)[:2000],
            }

            if include_prompts:
                row["prompt"] = item.get("prompt", "")

            if include_responses:
                row["response_raw"] = ""
                row["response_cleaned"] = ""

            return idx, row

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(worker, idx, item)
                for idx, item in enumerate(items)
            ]

            for future in as_completed(futures):
                idx, row = future.result()

                scored_by_index[idx] = row
                completed += 1

                logger.info(
                    "Finished item %d/%d: ok=%s stack=%s target=%s tokens=%s",
                    idx + 1,
                    len(items),
                    row.get("ok"),
                    row.get("stack_depth"),
                    row.get("target_stack_depth"),
                    row.get("tokens"),
                )

                # Ordered partial items.
                partial_items = [
                    scored_by_index[i]
                    for i in sorted(scored_by_index.keys())
                ]

                # Save every N completed items.
                if out_path and completed % save_every == 0:
                    partial_result = build_result_dict(
                        items_scored=partial_items,
                        axis=AXIS_NAME,
                        model=model,
                        base_url=base_url,
                        block_seed=block_seed,
                        requested_n_items=n_items,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=timeout,
                        concurrency=concurrency,
                        save_every=save_every,
                        status="running",
                    )

                    atomic_write_json(out_path, partial_result)

                    logger.info(
                        "Saved partial result: completed=%d/%d path=%s",
                        completed,
                        len(items),
                        out_path,
                    )

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received. Saving partial result before exit.")

        partial_items = [
            scored_by_index[i]
            for i in sorted(scored_by_index.keys())
        ]

        result = build_result_dict(
            items_scored=partial_items,
            axis=AXIS_NAME,
            model=model,
            base_url=base_url,
            block_seed=block_seed,
            requested_n_items=n_items,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            concurrency=concurrency,
            save_every=save_every,
            status="interrupted",
            error="KeyboardInterrupt",
        )

        if out_path:
            atomic_write_json(out_path, result)
            logger.info("Saved interrupted partial result to: %s", out_path)

        return result

    except Exception as exc:
        logger.exception("Run failed: %s", exc)

        partial_items = [
            scored_by_index[i]
            for i in sorted(scored_by_index.keys())
        ]

        result = build_result_dict(
            items_scored=partial_items,
            axis=AXIS_NAME,
            model=model,
            base_url=base_url,
            block_seed=block_seed,
            requested_n_items=n_items,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            concurrency=concurrency,
            save_every=save_every,
            status="failed",
            error=str(exc),
        )

        if out_path:
            atomic_write_json(out_path, result)
            logger.info("Saved failed partial result to: %s", out_path)

        return result

    final_items = [
        scored_by_index[i]
        for i in sorted(scored_by_index.keys())
    ]

    result = build_result_dict(
        items_scored=final_items,
        axis=AXIS_NAME,
        model=model,
        base_url=base_url,
        block_seed=block_seed,
        requested_n_items=n_items,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        concurrency=concurrency,
        save_every=save_every,
        status="complete",
    )

    if out_path:
        atomic_write_json(out_path, result)
        logger.info("Saved final result to: %s", out_path)

    return result


def print_summary(result: dict[str, Any]) -> None:
    print()
    print("=" * 80)
    print("IFEval Verifiable Axis Result")
    print("=" * 80)
    print(f"axis:              {result.get('axis')}")
    print(f"model:             {result.get('model')}")
    print(f"base_url:          {result.get('base_url')}")
    print(f"block_seed:        {result.get('block_seed')}")
    print(f"requested_n_items: {result.get('requested_n_items')}")
    print(f"n:                 {result.get('n')}")
    print(f"correct:           {result.get('correct')}")

    pass_frac = result.get("pass_frac", 0.0)
    print(f"pass_frac:         {pass_frac:.4f}")

    print(f"completion_tokens: {result.get('completion_tokens')}")
    print(f"max_tokens:        {result.get('max_tokens')}")
    print(f"temperature:       {result.get('temperature')}")
    print(f"timeout:           {result.get('timeout')}")
    print(f"concurrency:       {result.get('concurrency')}")
    print(f"save_every:        {result.get('save_every')}")
    print(f"status:            {result.get('status')}")

    if result.get("error"):
        print(f"error:             {result.get('error')}")

    print("=" * 80)
    print()

    for row in result.get("items", []):
        print(
            f"[{row.get('index')}] "
            f"ok={row.get('ok')} "
            f"stack={row.get('stack_depth')} "
            f"target={row.get('target_stack_depth')} "
            f"tokens={row.get('tokens')}"
        )
        print(f"    ids:  {row.get('instruction_ids')}")
        print(f"    per:  {row.get('per_instruction')}")

        if row.get("error"):
            print(f"    error: {row.get('error')}")

        print(f"    tail: {row.get('tail')!r}")
        print()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v31 IFEval verifiable axis against Chutes/OpenAI-compatible endpoint."
    )

    parser.add_argument(
        "--base-url",
        default=CHUTE_URL,
        help=f"OpenAI-compatible base URL. Default: {CHUTE_URL}",
    )

    parser.add_argument(
        "--api-key",
        default=API_KEY,
        help="API key. If empty, reads CHUTES_API_KEY from environment.",
    )

    parser.add_argument(
        "--model",
        default=CHUTE_MODEL,
        help=f"Served model name. Default: {CHUTE_MODEL}",
    )

    parser.add_argument(
        "--block-seed",
        type=int,
        default=42,
        help="Seed used to generate deterministic procedural items.",
    )

    parser.add_argument(
        "--n-items",
        type=int,
        default=8,
        help="Number of IFEval items to generate and evaluate.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"Max generation tokens per prompt. Default: {MAX_TOKENS}",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Default: 0.0 for greedy-style eval.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout per request in seconds. Default: {DEFAULT_TIMEOUT}",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of parallel requests. Default: 4.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Overwrite the output JSON after this many completed items. Default: 10.",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per request. Default: 3.",
    )

    parser.add_argument(
        "--out",
        default="",
        help="Optional path to save JSON results. This file is overwritten periodically.",
    )

    parser.add_argument(
        "--no-prompts",
        action="store_true",
        help="Do not include prompts in per-item output.",
    )

    parser.add_argument(
        "--no-responses",
        action="store_true",
        help="Do not include raw/cleaned responses in per-item output.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


    result = run(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        block_seed=args.block_seed,
        n_items=args.n_items,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        concurrency=args.concurrency,
        save_every=args.save_every,
        out_path=args.out,
        include_prompts=not args.no_prompts,
        include_responses=not args.no_responses,
        retries=args.retries,
    )

    print_summary(result)

    if args.out:
        print(f"Saved result JSON to: {args.out}")

    return 0 if result.get("status") in {"complete", "empty", "interrupted"} else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
