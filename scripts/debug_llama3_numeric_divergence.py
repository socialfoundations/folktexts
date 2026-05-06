"""Reproduce + diagnose the Llama-3-8B base numeric divergence.

Sweep v3 showed `meta-llama--Meta-Llama-3-8B numeric` produces 99.1% of rows
exactly at 0.5 on vLLM, while transformers produces 101 unique values. The
Llama-3 tokenizer has 1100 multi-digit decimal tokens; the suspicion is that
vLLM's `allowed_token_ids` constraint does not select multi-digit tokens the
way transformers' digit-mask + softmax-argmax does.

Run after sweep v3 finishes (GPU must be free).

Usage:
    python scripts/debug_llama3_numeric_divergence.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_PATH = "/fast/groups/sf/huggingface-models/meta-llama--Meta-Llama-3-8B"
PROMPT = (
    "Below is information about a person. The probability that this person earns "
    "more than $50000/year is between 0 and 1. The probability is 0."
)


def _load_digit_token_ids(tokenizer, vocab_dim: int) -> tuple[list[int], dict[str, int]]:
    vocab = tokenizer.get_vocab()
    digit_ids = sorted({i for t, i in vocab.items() if t.isdecimal() and 0 <= i < vocab_dim})
    multi = {t: i for t, i in vocab.items() if t.isdecimal() and len(t) > 1 and i < vocab_dim}
    return digit_ids, multi


def transformers_argmax(prompt: str, vocab_dim: int, digit_ids: list[int]):
    """Transformers path: full softmax → mask non-digits → argmax."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    last_logits = out.logits[0, -1, :]                         # (vocab_dim,)
    last_probs = torch.softmax(last_logits.float(), dim=-1).cpu().numpy()

    masked = last_probs.copy()
    mask = [False] * vocab_dim
    for i in digit_ids:
        mask[i] = True
    import numpy as np
    masked[~np.array(mask)] = 0.0

    top_id = int(np.argmax(masked))
    top_token = tokenizer.decode([top_id])
    top_prob = float(last_probs[top_id])

    # Top-50 over digits, raw probability
    digit_probs = [(int(i), tokenizer.decode([int(i)]), float(last_probs[int(i)])) for i in digit_ids]
    digit_probs.sort(key=lambda x: x[2], reverse=True)
    return {
        "argmax_id": top_id,
        "argmax_token": top_token,
        "argmax_prob": top_prob,
        "top10_digits": digit_probs[:10],
    }


def vllm_argmax(prompt: str, vocab_dim: int, digit_ids: list[int]):
    """vLLM path: SamplingParams(allowed_token_ids=digit_ids) + greedy."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.6,
        max_model_len=512,
        max_logprobs=200,
        seed=42,
    )

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=2,
        logprobs=200,                # large enough to often include the chosen token
        allowed_token_ids=digit_ids,
        seed=42,
    )
    out = llm.generate([prompt], sampling)
    completion = out[0].outputs[0]

    chosen_ids = list(completion.token_ids)
    chosen_text = "".join(tokenizer.decode([i]) for i in chosen_ids)

    pos0_logprobs = completion.logprobs[0] if completion.logprobs else {}
    sorted_pos0 = sorted(
        [(int(tid), getattr(lp, "decoded_token", None), float(getattr(lp, "logprob", lp)))
         for tid, lp in pos0_logprobs.items()],
        key=lambda x: x[2], reverse=True,
    )

    # Filter to digit-only entries seen in the dict
    digit_in_dict = [t for t in sorted_pos0 if t[0] in set(digit_ids)]

    return {
        "chosen_token_ids": chosen_ids,
        "chosen_text": chosen_text,
        "pos0_topk_dict_size": len(pos0_logprobs),
        "pos0_top10": sorted_pos0[:10],
        "pos0_digit_top10": digit_in_dict[:10],
    }


def main() -> int:
    # Cluster runtime env (mirrors README / migration report)
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    vocab_dim = int(cfg.vocab_size)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    digit_ids, multi_digit_tokens = _load_digit_token_ids(tok, vocab_dim)
    print(f"vocab_dim = {vocab_dim}")
    print(f"digit token count = {len(digit_ids)}")
    print(f"  single-digit ids 15..24 present: "
          f"{[i for i in range(15, 25) if i in set(digit_ids)]}")
    print(f"  example multi-digit tokens: {list(multi_digit_tokens.items())[:8]}")
    print(f"\nprompt: {PROMPT!r}")

    print("\n" + "=" * 80)
    print("VLLM PATH (allowed_token_ids = digit ids, greedy):")
    print("=" * 80)
    vl = vllm_argmax(PROMPT, vocab_dim, digit_ids)
    for k, v in vl.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 80)
    print("TRANSFORMERS PATH (softmax → mask non-digits → argmax) — pass 1 only:")
    print("=" * 80)
    tf = transformers_argmax(PROMPT, vocab_dim, digit_ids)
    for k, v in tf.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)
    tf_argmax_token = tf["argmax_token"]
    vl_first_token = ""
    if vl["chosen_token_ids"]:
        vl_first_token = tok.decode([vl["chosen_token_ids"][0]])
    if tf_argmax_token == vl_first_token:
        print(f"  ✓ Both paths picked the same pass-1 token: {vl_first_token!r}")
    else:
        print(f"  ✗ DIVERGENCE: TF picked {tf_argmax_token!r} "
              f"(p={tf['argmax_prob']:.4g}); vLLM picked {vl_first_token!r}.")
        # Is TF's choice in vLLM's allowed_token_ids?
        tf_id = tf["argmax_id"]
        in_allowed = tf_id in set(digit_ids)
        in_topk = any(tid == tf_id for tid, *_ in vl["pos0_top10"])
        print(f"    TF choice id={tf_id} in vLLM allowed_token_ids: {in_allowed}")
        print(f"    TF choice id={tf_id} in vLLM pos0 top-K dict: {in_topk}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
