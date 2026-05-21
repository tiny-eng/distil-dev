#!/usr/bin/env python3
"""
v31_ifeval_verifiable — Google IFEval verifiable-instructions axis.

This runner uses an already-running vLLM OpenAI-compatible server with concurrent requests.

Expected server example:

    vllm serve /mnt/d/models/Qwen/Qwen3-4B \
      --served-model-name qwen3-4b \
      --host 0.0.0.0 \
      --port 8001

Expected endpoint:

    http://38.102.125.144:8001/v1/chat/completions

Expected local files:

    ifeval.py
    ifeval_verifiable.py
    _ifeval_vendor.py

Run example:

python -m distil-dev.build_eval_v31_ifeval_parallel \
    --base-url http://38.102.125.144:9008/v1 \
    --model qwen3-4b \
    --block-seed 123 \
    --n-items 1000 \
    --max-concurrent 10 \
    --out distil-dev/results/qwen3_4b_ifeval_1000_raw_results_parallel_7340.json
"""

from __future__ import annotations

import argparse
import asyncio
import aiohttp
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import distil.pod.axes.v31.ifeval_verifiable as ifeval_verifiable
import distil.pod.axes.v31._ifeval_vendor as _ifeval_vendor

logger = logging.getLogger("ifeval")

MAX_TOKENS = 1024
AXIS_NAME = "v31_ifeval_verifiable"

DEFAULT_BASE_URL = "http://38.102.125.144:8888/v1"
DEFAULT_MODEL = "qwen3-4b"
DEFAULT_TIMEOUT = 180.0
DEFAULT_MAX_CONCURRENT = 10

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_TRAIL_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)
_THINK_NARRATIVE_RE = re.compile(
    r"^\s*Thinking Process:.*?(?=\n\n[A-Z0-9]|\Z)",
    re.DOTALL,
)


def strip_thinking(text: str) -> str:
    """Drop ``<think>...</think>`` / "Thinking Process:" leaders before extraction."""
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


async def generate_one_vllm_completion_async(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    api_key: str = "EMPTY",
) -> tuple[str, int, int]:
    """
    Generate one completion asynchronously using vLLM's OpenAI-compatible completion endpoint.
    
    Returns:
        (generated_text, completion_token_count, index)
    """
    endpoint = base_url.rstrip("/") + "/completions"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    try:
        async with session.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"HTTP {response.status} from {endpoint}: {error_text[:1000]}")
            
            data = await response.json()
            
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"No choices returned by vLLM: {data}")
            
            choice = choices[0]
            text = choice.get("text", "")
            
            usage = data.get("usage") or {}
            completion_tokens = usage.get("completion_tokens", 0)
            
            if not isinstance(completion_tokens, int):
                completion_tokens = 0
            
            return text or "", completion_tokens
            
    except asyncio.TimeoutError:
        raise RuntimeError(f"Request timeout after {timeout}s for endpoint {endpoint}")
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"Client error connecting to {endpoint}: {exc}")


async def generate_concurrent_vllm_server(
    *,
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float = 0.0,
    timeout: float = DEFAULT_TIMEOUT,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
) -> list[tuple[str, int]]:
    """
    Generate responses concurrently from an already-running vLLM server.
    
    Uses asyncio with a semaphore to limit concurrent requests.
    """
    results: list[tuple[str, int]] = [("", 0)] * len(prompts)
    
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate_one_with_semaphore(idx: int, prompt: str) -> None:
        async with semaphore:
            logger.debug("Starting generation for item %d/%d", idx + 1, len(prompts))
            try:
                async with aiohttp.ClientSession() as session:
                    text, n_tokens = await generate_one_vllm_completion_async(
                        session=session,
                        base_url=base_url,
                        model=model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=timeout,
                    )
                    results[idx] = (text, n_tokens)
                    logger.debug("Completed generation for item %d/%d", idx + 1, len(prompts))
            except Exception as exc:
                logger.error("Failed to generate for item %d: %s", idx + 1, exc)
                results[idx] = ("", 0)
    
    # Create tasks for all prompts
    tasks = [
        generate_one_with_semaphore(idx, prompt)
        for idx, prompt in enumerate(prompts)
    ]
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    
    return results


def evaluate_item_concurrent(item_data: tuple) -> dict[str, Any]:
    """
    Evaluate a single item (for use in ThreadPoolExecutor).
    
    Args:
        item_data: Tuple of (index, item_dict, response_text)
    
    Returns:
        Dictionary with evaluation results
    """
    idx, it, text = item_data
    cleaned = strip_thinking(text or "")
    
    try:
        all_pass, per = _ifeval_vendor.evaluate_item(
            cleaned,
            it["instruction_ids"],
            it.get("kwargs") or [],
        )
    except Exception as exc:
        logger.warning("ifeval evaluate_item crashed on item %d: %s", idx, exc)
        all_pass, per = False, []
    
    return {
        "index": idx,
        "ok": bool(all_pass),
        "per_instruction": per,
        "cleaned_response": cleaned,
        "raw_response": text,
        "all_pass": all_pass,
    }


def evaluate_items_concurrent(
    items: list[dict[str, Any]],
    responses: list[str],
    max_workers: int = DEFAULT_MAX_CONCURRENT,
) -> list[dict[str, Any]]:
    """
    Evaluate multiple items concurrently using ThreadPoolExecutor.
    
    Args:
        items: List of item dictionaries
        responses: List of response texts
        max_workers: Maximum number of concurrent evaluation threads
    
    Returns:
        List of evaluation result dictionaries
    """
    # Prepare evaluation tasks
    eval_tasks = [(idx, items[idx], responses[idx]) for idx in range(len(items))]
    
    # Run evaluations concurrently
    eval_results = [None] * len(items)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(evaluate_item_concurrent, task): task[0]
            for task in eval_tasks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                eval_results[idx] = result
            except Exception as exc:
                logger.error("Evaluation failed for item %d: %s", idx, exc)
                eval_results[idx] = {
                    "index": idx,
                    "ok": False,
                    "per_instruction": [],
                    "cleaned_response": "",
                    "raw_response": responses[idx],
                    "all_pass": False,
                }
    
    return eval_results


def run(
    *,
    base_url: str,
    model: str,
    block_seed: int,
    n_items: int,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.0,
    timeout: float = DEFAULT_TIMEOUT,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    include_prompts: bool = True,
    include_responses: bool = True,
) -> dict[str, Any]:
    """
    Run the v31 IFEval verifiable axis against a running vLLM server with concurrency.
    """
    # Generate items
    items = ifeval_verifiable.generate_items(block_seed, n_items)
    
    if not items:
        return {
            "axis": AXIS_NAME,
            "model": model,
            "base_url": base_url,
            "n": 0,
            "correct": 0,
            "pass_frac": 0.0,
            "completion_tokens": 0,
            "items": [],
        }
    
    prompts = [it["prompt"] for it in items]
    
    # Concurrent generation
    logger.info(f"Starting concurrent generation with {max_concurrent} concurrent requests...")
    start_time = time.time()
    
    try:
        # Run async generation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        gens = loop.run_until_complete(
            generate_concurrent_vllm_server(
                base_url=base_url,
                model=model,
                prompts=prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                max_concurrent=max_concurrent,
            )
        )
        loop.close()
        
        gen_time = time.time() - start_time
        logger.info(f"Generation completed in {gen_time:.2f}s")
        
    except Exception as exc:
        logger.exception("ifeval vLLM server generation failed: %s", exc)
        return {
            "axis": AXIS_NAME,
            "model": model,
            "base_url": base_url,
            "n": 0,
            "correct": 0,
            "pass_frac": 0.0,
            "completion_tokens": 0,
            "items": [],
            "error": str(exc)[:1000],
        }
    
    # Extract responses and token counts
    responses = [text for text, _ in gens]
    completion_tokens = sum(n_tok for _, n_tok in gens)
    
    # Concurrent evaluation
    logger.info(f"Starting concurrent evaluation with {max_concurrent} workers...")
    eval_start = time.time()
    eval_results = evaluate_items_concurrent(items, responses, max_workers=max_concurrent)
    eval_time = time.time() - eval_start
    logger.info(f"Evaluation completed in {eval_time:.2f}s")
    
    # Build final results
    scored: list[dict[str, Any]] = []
    correct = 0
    
    for idx, (it, (text, n_tok), eval_result) in enumerate(zip(items, gens, eval_results)):
        correct += int(eval_result["all_pass"])
        
        row: dict[str, Any] = {
            "index": idx,
            "src": it.get("src", ""),
            "topic": it.get("topic", ""),
            "instruction_ids": it.get("instruction_ids"),
            "kwargs": it.get("kwargs"),
            "per_instruction": eval_result["per_instruction"],
            "stack_depth": it.get("stack_depth"),
            "target_stack_depth": it.get("target_stack_depth"),
            "ok": eval_result["all_pass"],
            "tokens": int(n_tok or 0),
            "tail": (text or "")[-120:],
        }
        
        if include_prompts:
            row["prompt"] = it.get("prompt", "")
        
        if include_responses:
            row["response_raw"] = eval_result["raw_response"]
            row["response_cleaned"] = eval_result["cleaned_response"]
        
        scored.append(row)
        
        logger.info(
            "Item %d/%d: ok=%s stack=%s target=%s per=%s",
            idx + 1,
            len(items),
            eval_result["all_pass"],
            it.get("stack_depth"),
            it.get("target_stack_depth"),
            eval_result["per_instruction"],
        )
    
    res = BenchResult(
        n=len(scored),
        correct=correct,
        completion_tokens=completion_tokens,
        items=scored,
    ).as_dict()
    
    res["axis"] = AXIS_NAME
    res["model"] = model
    res["base_url"] = base_url
    res["block_seed"] = block_seed
    res["requested_n_items"] = n_items
    res["max_tokens"] = max_tokens
    res["temperature"] = temperature
    res["max_concurrent"] = max_concurrent
    res["generation_time_seconds"] = gen_time
    res["evaluation_time_seconds"] = eval_time
    
    return res


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
    print(f"max_concurrent:    {result.get('max_concurrent')}")
    print(f"gen_time:          {result.get('generation_time_seconds', 0):.2f}s")
    print(f"eval_time:         {result.get('evaluation_time_seconds', 0):.2f}s")
    
    if result.get("error"):
        print(f"error:             {result.get('error')}")
    
    print("=" * 80)
    print()
    
    for row in result.get("items", [])[:10]:  # Show first 10 items only
        print(
            f"[{row.get('index')}] "
            f"ok={row.get('ok')} "
            f"stack={row.get('stack_depth')} "
            f"target={row.get('target_stack_depth')} "
            f"tokens={row.get('tokens')}"
        )
        print(f"    ids:  {row.get('instruction_ids')}")
        print(f"    per:  {row.get('per_instruction')}")
        print(f"    tail: {row.get('tail')!r}")
        print()
    
    if len(result.get("items", [])) > 10:
        print(f"... and {len(result['items']) - 10} more items")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v31 IFEval verifiable axis against a running vLLM server with concurrency."
    )
    
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"vLLM OpenAI-compatible base URL. Default: {DEFAULT_BASE_URL}",
    )
    
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Served model name. Default: {DEFAULT_MODEL}",
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
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help=f"Maximum number of concurrent requests. Default: {DEFAULT_MAX_CONCURRENT}",
    )
    
    parser.add_argument(
        "--out",
        default="",
        help="Optional path to save JSON results.",
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
        model=args.model,
        block_seed=args.block_seed,
        n_items=args.n_items,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent,
        include_prompts=not args.no_prompts,
        include_responses=not args.no_responses,
    )
    
    print_summary(result)
    
    if args.out:
        import os
        
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Saved result JSON to: {args.out}")
    
    return 0 if not result.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))