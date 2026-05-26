"""
python local_eval_chutes_vllm.py \
  --student-model ./your-local-student-model \
  --n-prompts 256 \
  --block-hash 0000000000000000000abcdef123456789 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --out-dir eval_out
"""


from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from vllm import LLM, SamplingParams

from distil.pod.kl import average_kl, average_rkl, top_k_overlap
from distil.pod.progress import write_progress, record_bench_timing
from distil.eval.dataset import sample_prompts

CHUTE_URL = "https://llm.chutes.ai/v1/"
CHUTE_MODEL = "moonshotai/Kimi-K2.6-TEE"
API_KEY = "cpk_2b247fc4ec3d4aa08487bcaaa9142fd1.e8a69a1082c15c3fa53fe560fd4cbf27.WH3dFzzVDPN0geBsjCJN4G5T7ZCA8GHZ"
DATASET = "huggingface.co/"

DEFAULT_TIMEOUT = 180.0

EVAL_TOP_K = 20
TEACHER_MAX_GEN_TOKENS = 512

def atomic_json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, intent=2, ensure_ascii=False)

    os.replace(tmp, path)

def parse_chutes_top_logprobs(choice: dict[str, Any]) -> list[dict[str, float]]:
    logprobs_obj = choice.get("logprobs") or {}
    top_logprobs = logprobs_obj.get("top_logprobs") or []

    trace: list[dict[str, float]] = []

    for pos in top_logprobs:
        row: dict[str, float] = {}

        if isinstance(pos, dict):
            for token_text, logprob in pos.items():
                try:
                    row[str(token_text)] = float(logprob)
                except Exception:
                    continue

        trace.append(row)

    return trace

def generate_one_chutes_completion(
        *,
        base_url: str,
        api_key: str,
        model: str,
        prompt: str,
        timeout: float,
) -> tuple[str, int, list[dict[str, float]], dict[str, Any]]:
    """
    Generate one teacher completion using Chutes /completions.
    """
    endpoint = base_url.rstrip("/") + "/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": TEACHER_MAX_GEN_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "logprobs": EVAL_TOP_K,
    }

    resp = requests.post(
        endpoint,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    if resp.status_code >= 400:
        raise RuntimeError(
            f"Chutes completion failed: HTTP {resp.status_code}\n"
            f"{resp.text[:4000]}"
        )
    
    data = resp.json()
    choices = data.get("choices") or []

    if not choices:
        return "", 0, [], data
    
    choice = choices[0]

    text = choice.get("text") or ""
    trace_text = parse_chutes_top_logprobs(choice)

    usage = data.get("usage") or {}
    n_tokens = usage.get("completion_tokens")

    if n_tokens is None:
        n_tokens = len(trace_text)

    return text, int(n_tokens or 0), trace_text, data

def get_teacher_generated_traces(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompts: list[str],
    timeout: float,
    sleep_s: float,
) -> tuple[list[str], list[list[dict[str, float]]], int]:
    """
    For each sampled prompt:
        - generate teacher completion with Chutes
        - collect top-20 logprobs for generated tokens
    """
    completions: list[str] = []
    traces: list[list[dict[str, float]]] = []
    total_tokens = 0

    for i, prompt in enumerate(prompts):
        text, n_tokens, trace, _raw = generate_one_chutes_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            timeout=timeout,
        )

        completions.append(text)
        traces.append(trace)
        total_tokens += n_tokens

        print(
            f"[teacher] {i + 1}/{len(prompts)}"
            f"tokens={n_tokens} trace_positions={len(trace)}"
            f"top_k={EVAL_TOP_K}"
        )

        if sleep_s > 0:
            time.sleep(sleep_s)

    return completions, traces, total_tokens

def load_vllm_student(args: argparse.Namespace) -> LLM:
    """
    Load local student model with inline vLLM.
    """
    llm_kwargs: dict[str, Any] = {
        "model": args.student_model,
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }

    if args.dtype:
        llm_kwargs["dtype"] = args.dtype

    if args.max_model_len and args.max_model_len > 0:
        llm_kwargs["max_model_len"] = args.max_model_len

    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True

    return LLM(**llm_kwargs)

def get_vllm_tokenizer(llm: LLM):
    return llm.get_tokenizer()

def teacher_trace_text_to_student_ids_vllm(
    teacher_trace_text: list[list[dict[str, float]]],
    tokenizer: Any,
) -> list[list[dict[int, float]]]:
    """
    Convert Chutes teacher token strings to vLLM student token IDs.
    """
    out: list[list[dict[int, float]]] = []

    for prompt_trace in teacher_trace_text:
        converted_prompt: list[dict[int, float]] = []

        for pos in prompt_trace:
            row: dict[int, float] = {}

            for token_text, logprob in pos.items():
                ids = tokenizer.encode(token_text, add_special_tokens=False)

                if len(ids) != 1:
                    continue

                row[int(ids[0])] = float(logprob)

            converted_prompt.append(row)

        out.append(converted_prompt)

    return out

def _vllm_logprob_obj_to_dict(pos_obj: Any) -> dict[int, float]:
    """
    Convert one vLLM prompt_logprobs position into:

      {token_id: logprob}
    """
    if not pos_obj:
        return {}

    row: dict[int, float] = {}

    for token_id, lp_obj in pos_obj.items():
        try:
            token_id_int = int(token_id)
        except Exception:
            continue

        if hasattr(lp_obj, "logprob"):
            lp = float(lp_obj.logprob)
        else:
            lp = float(lp_obj)

        row[token_id_int] = lp

    return row

def count_tokens(
    *,
    tokenizer: Any,
    text: str,
) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def get_student_trace_vllm(
    *,
    llm: LLM,
    tokenizer: Any,
    prompts: list[str],
    completions: list[str],
) -> tuple[list[list[dict[int, float]]], int]:
    full_texts = [
        prompt + completion
        for prompt, completion in zip(prompts, completions)
    ]

    prompt_token_counts = [
        count_tokens(tokenizer=tokenizer, text=prompt)
        for prompt in prompts
    ]

    full_token_counts = [
        count_tokens(tokenizer=tokenizer, text=full_texts)
        for full_text in full_texts
    ]

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=EVAL_TOP_K,
    )

    outputs = llm.generate(
        prompts=full_texts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    traces: list[list[dict[int, float]]] = []
    total_completion_tokens = 0

    for i, output in enumerate(outputs):
        prompt_logprobs = output.prompt_logprobs or []

        prompt_len = prompt_token_counts[i]
        full_len = full_token_counts[i]

        completion_trace: list[dict[int, float]] = []

        for token_pos in range(prompt_len, full_len):
            if token_pos >= len(prompt_logprobs):
                continue

            row = _vllm_logprob_obj_to_dict(prompt_logprobs[token_pos])
            completion_trace.append(row)

        traces.append(completion_trace)
        total_completion_tokens += len(completion_trace)

        print(
            f"[student-vllm] {i + 1}/{len(outputs)}"
            f"prompt_tokens={prompt_len} full_tokens={full_len}"
            f"completion_tokens={len(completion_trace)}"
            f"top_k={EVAL_TOP_K}"
        )

    return traces, total_completion_tokens

def align_trace_lengths(
    teacher_trace: list[list[dict[int, float]]],
    student_trace: list[list[dict[int, float]]],
) -> tuple[list[list[dict[int, float]]], list[list[dict[int, float]]]]:
    """
    Trim teacher/student per prompt to same number of positions.
    """
    t_out: list[list[dict[int, float]]] = []
    s_out: list[list[dict[int, float]]] = []

    for t_prompt, s_prompt in zip(teacher_trace, student_trace, strict=False):
        n = min(len(t_prompt), len(s_prompt))
        t_out.append(t_prompt[:n])
        s_out.append(s_prompt[:n])

    return t_out, s_out

def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    progress_path = out_dir / "eval_progress.json"

    api_key = API_KEY

    write_progress(
        progress_path,
        phase="starting",
        teacher_model=args.teacher_model,
        student_model=args.stuende.model,
        n_prompts=args.n_prompts,
        block_hash=args.block_hash,
        top_k=EVAL_TOP_K,
        teacher_max_gen_tokens=TEACHER_MAX_GEN_TOKENS,
    )

    write_progress(progress_path, phase="sampling_prompts")

    prompts = sample_prompts(
        n=args.n_prompts,
        block_hash=args.block_hash,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )

    if not prompts:
        raise RuntimeError("sample_prompts returned no prompts")
    
    print(
        f"Sampled {len(prompts)} prompts "
        f"block_hash={args.block_hash!r} "
        f"min_chars={args.min_chars} max_chars={args.max_chars}"
    )

    write_progress(
        progress_path,
        phase="sampled_prompts",
        n_prompts=len(prompts)
    )

    atomic_json_write(out_dir / "prompts.json", prompts)

    write_progress(progress_path, phase="teacher_generation")

    llm = load_vllm_student(args)
    tokenizer = get_vllm_tokenizer(llm)

    write_progress(progress_path, phase="teacher_generation")

    teacher_t0 = time.time()

    completions, teacher_trace_text, teacher_tokens = get_teacher_generated_traces(
        base_url=args.base_url,
        api_key=api_key,
        model=args.teacher_model,
        prompts=prompts,
        timeout=args.timeoutm
        sleep_s=args.sleep_s,
    )

    teacher_wall_s = time.time() - teacher_t0

    record_bench_timing(
        progress_path,
        name="teacher_chutes",
        wall_s=teacher_wall_s,
        n_prompts=len(prompts),
        completion_tokens=teacher_tokens,
        extra={
            "model": args.teacher_model,
            "top_k": EVAL_TOP_K,
            "max_gen_tokens": TEACHER_MAX_GEN_TOKENS,
        },
    )

    atomic_json_write(out_dir / "teacher_comletions.json", completions)
    atomic_json_write(out_dir / "teacher_trace_text.json", teacher_trace_text)

    write_progress(progress_path, phase="converting_teacher_trace")

    teacher_trace = teacher_trace_text_to_student_ids_vllm(
        teacher_trace_text,
        tokenizer,
    )

    atomic_json_write(
        out_dir / "teacher_trace_student_ids_unaligned.json",
        teacher_trace,
    )

    write_progress(progress_path, phase="student_vllm_scoring")

    student_t0 = time.time()

    student_trace, student_tokens = get_student_trace_vllm(
        llm=llm,
        tokenizer=tokenizer,
        prompts=prompts,
        completions=completions,
    )

    student_wall_s = time.time() - student_t0

    record_bench_timing(
        progress_path,
        name="student_vllm_score",
        wall_s=student_wall_s,
        n_prompts=len(prompts),
        completion_tokens=student_tokens,
        extra={
            "model": args.student_model,
            "top_k": EVAL_TOP_K,
        },
    )

    atomic_json_write(
        out_dir / "student_trace_unaligned.json",
        student_trace,
    )

    teacher_trace, student_trace = align_trace_lengths(
        teacher_trace,
        student_trace,
    )

    atomic_json_write(out_dir / "teacher_trace_student_ids.json", teacher_trace)
    atomic_json_write(out_dir / "student_trace.json", student_trace)

    write_progress(progress_path, phase="computing_metrics")

    kl_loss = average_kl(teacher_trace, student_trace)
    on_policy_rkl = average_rkl(teacher_trace, student_trace)
    overlap = top_k_overlap(teacher_trace, student_trace, k=EVAL_TOP_K)
    
    result = {
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "n_prompts": len(prompts),
        "block_hash": args.block_hash,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "top_k": EVAL_TOP_K,
        "teacher_max_gen_tokens": TEACHER_MAX_GEN_TOKENS,
        "temperature": 0.0,
        "metrics": {
            "kl_loss": kl_loss,
            "on_policy_rkl": on_policy_rkl,
            "top_k_overlap": overlap,
        },
        "timing": {
            "teacher_wall_s": round(teacher_wall_s, 3),
            "student_wall_s": round(student_wall_s, 3),
            "teacher_tokens": teacher_tokens,
            "student_tokens": student_tokens,
            "teacher_tok_per_s": round(teacher_tokens / max(teacher_wall_s, 1e-6), 3),
            "student_tok_per_s": round(student_tokens / max(student_wall_s, 1e-6), 3),
        },
    }

    atomic_json_write(out_dir / "eval_result.json", result)

    write_progress(
        progress_path,
        phase="done",
        metrics=result["metrics"],
        result_path=str(out_dir / "eval_result.json"),
        top_k=EVAL_TOP_K,
        teacher_max_gen_tokens=TEACHER_MAX_GEN_TOKENS,
    )

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Chutes teacher.
    p.add_argument("--base-url", default=CHUTE_URL)
    p.add_argument("--teacher-model", default=CHUTE_MODEL)
    p.add_argument("--api-key", default=API_KEY)

    # vLLM student.
    p.add_argument("--student-model", required=True)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=0)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")

    # Dataset sampling from dataset.py.
    p.add_argument("--n-prompts", type=int, default=256)
    p.add_argument("--block-hash", default=None)
    p.add_argument("--min-chars", type=int, default=64)
    p.add_argument("--max-chars", type=int, default=4096)
    
    # Eval runtime.
    p.add_argument("--out-dir", default="local_eval_chutes_vllm_out")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    p.add_argument("--sleep-s", type=float, default=0.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_eval(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()










