#!/usr/bin/env python3
"""
v31_ifeval_verifiable — Google IFEval verifiable-instructions axis.

This runner uses an already-running vLLM OpenAI-compatible server.

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

python -m distil-dev.build_eval_v31_ifeval \
    --base-url http://38.102.125.144:8001/v1 \
    --model qwen3-8b \
    --block-seed 201 \
    --n-items 100 \
    --out distil-dev/results/qwen3_8b_ifeval_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import distil.pod.axes.v31.ifeval_verifiable as ifeval_verifiable
import distil.pod.axes.v31._ifeval_vendor as _ifeval_vendor

logger = logging.getLogger("ifeval")

MAX_TOKENS = 1024
AXIS_NAME = "v31_ifeval_verifiable"

DEFAULT_BASE_URL = "http://38.102.125.144:8888/v1"
DEFAULT_MODEL = "qwen3-4b"
DEFAULT_TIMEOUT = 180.0

CHUTE_URL = "https://llm.chutes.ai/v1/"
CHUTE_MODEL = "qwen3-32b"

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
    
def _post_json(url: str, payload: dict[str, Any], timeout: float, api_key: str = "EMPTY") -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json",
            # vLLM usually ignores auth unless launched with auth enabled.
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
            f"HTTP {exc.code} from {url}: {err_body[:1000]}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to vLLM endpoint {url}: {exc}") from exc


# def generate_one_vllm_chat(
#     *,
#     base_url: str,
#     model: str,
#     prompt: str,
#     max_tokens: int,
#     temperature: float,
#     timeout: float,
# ) -> tuple[str, int]:
#     """
#     Generate one completion using vLLM's OpenAI-compatible chat endpoint.

#     Returns:
#         (generated_text, completion_token_count)

#     The OpenAI-compatible endpoint usually returns token counts in usage,
#     not actual token IDs.
#     """
#     endpoint = base_url.rstrip("/") + "/chat/completions"

#     payload = {
#         "model": model,
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt,
#             }
#         ],
#         "max_tokens": max_tokens,
#         "temperature": temperature,
#         "top_p": 1.0,
#     }

#     data = _post_json(endpoint, payload, timeout=timeout)

#     choices = data.get("choices") or []
#     if not choices:
#         raise RuntimeError(f"No choices returned by vLLM: {data}")

#     choice = choices[0]
#     message = choice.get("message") or {}

#     text = message.get("content")
#     if text is None:
#         text = choice.get("text", "")

#     usage = data.get("usage") or {}
#     completion_tokens = usage.get("completion_tokens", 0)

#     if not isinstance(completion_tokens, int):
#         completion_tokens = 0

#     return text or "", completion_tokens

def generate_one_vllm_completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> tuple[str, int]:
    """
    Generate one completion using vLLM's OpenAI-compatible completion endpoint.

    This mirrors the original local vLLM behavior:

        engine.generate(prompts, params)

    because both use raw prompt strings rather than chat messages.

    Returns:
        (generated_text, completion_token_count)

    The OpenAI-compatible endpoint usually returns token counts in usage,
    not actual token IDs.
    """
    endpoint = base_url.rstrip("/") + "/completions"

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }

    data = _post_json(endpoint, payload, timeout=timeout)

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

def generate_one_chutes_completion(
        *,
        api_key: str,
        base_url: str = "https://llm.chutes.ai/v1",
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
)  -> tuple[str, int]:
    endpoint = base_url.rstrip("/") + "/completions"

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }

    data = _post_json(
        endpoint,
        payload,
        timeout=timeout,
        api_key=api_key,
    )

    choices = data.get("choices") or []

    if not choices:
        raise RuntimeError(f"No choices returned by Chutes: {data}")

    choice = choices[0]

    text = choice.get("text", "")

    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens", 0)

    if not isinstance(completion_tokens, int):
        completion_tokens = 0

    return text or "", completion_tokens



def generate_greedy_vllm_server(
    *,
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float = 0.0,
    timeout: float = DEFAULT_TIMEOUT,
    sleep_s: float = 0.0,
) -> list[tuple[str, int]]:
    """
    Generate responses sequentially from an already-running vLLM server.

    This mirrors the original generate_greedy(...) behavior conceptually,
    but uses HTTP instead of engine.generate(...).
    """
    outs: list[tuple[str, int]] = []

    for i, prompt in enumerate(prompts, start=1):
        logger.info("Generating item %d/%d", i, len(prompts))

        # text, n_tokens = generate_one_vllm_chat(
        text, n_tokens = generate_one_vllm_completion(    
            base_url=base_url,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )

        outs.append((text, n_tokens))

        if sleep_s > 0:
            time.sleep(sleep_s)

    return outs

def generate_greedy_chutes_server(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float = 0.0,
    timeout: float = DEFAULT_TIMEOUT,
    sleep_s: float = 0.0,
) -> list[tuple[str, int]]:
    """
    Generate responses sequentially from an already-running vLLM server.

    This mirrors the original generate_greedy(...) behavior conceptually,
    but uses HTTP instead of engine.generate(...).
    """
    outs: list[tuple[str, int]] = []

    for i, prompt in enumerate(prompts, start=1):
        logger.info("Generating item %d/%d", i, len(prompts))

        # text, n_tokens = generate_one_vllm_chat(
        text, n_tokens = generate_one_chutes_completion(    
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )

        outs.append((text, n_tokens))

        if sleep_s > 0:
            time.sleep(sleep_s)

    return outs

def run(
    *,
    base_url: str,
    model: str,
    block_seed: int,
    n_items: int,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.0,
    timeout: float = DEFAULT_TIMEOUT,
    include_prompts: bool = True,
    include_responses: bool = True,
) -> dict[str, Any]:
    """
    Run the v31 IFEval verifiable axis against a running vLLM server.
    """
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

    try:
        gens = generate_greedy_vllm_server(
            base_url=base_url,
            model=model,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )

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

    scored: list[dict[str, Any]] = []
    correct = 0
    completion_tokens = 0

    for idx, (it, (text, n_tok)) in enumerate(zip(items, gens, strict=False)):
        completion_tokens += int(n_tok or 0)

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

        row: dict[str, Any] = {
            "index": idx,
            "src": it.get("src", ""),
            "topic": it.get("topic", ""),
            "instruction_ids": it.get("instruction_ids"),
            "kwargs": it.get("kwargs"),
            "per_instruction": per,
            "stack_depth": it.get("stack_depth"),
            "target_stack_depth": it.get("target_stack_depth"),
            "ok": bool(all_pass),
            "tokens": int(n_tok or 0),
            "tail": (text or "")[-120:],
        }

        if include_prompts:
            row["prompt"] = it.get("prompt", "")

        if include_responses:
            row["response_raw"] = text or ""
            row["response_cleaned"] = cleaned

        scored.append(row)
        correct += int(bool(all_pass))

        logger.info(
            "Item %d/%d: ok=%s stack=%s target=%s per=%s",
            idx + 1,
            len(items),
            bool(all_pass),
            it.get("stack_depth"),
            it.get("target_stack_depth"),
            per,
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
        print(f"    tail: {row.get('tail')!r}")
        print()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v31 IFEval verifiable axis against a running vLLM server."
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