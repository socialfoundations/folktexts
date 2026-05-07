"""Verify vLLM rejects over-cap inputs cleanly (no silent truncation).

Construct a prompt with token count > max_model_len; assert generate raises
rather than truncating to fit.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL = "/fast/groups/sf/huggingface-models/meta-llama--Llama-3.2-1B-Instruct"


def main() -> int:
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    from folktexts.llm_utils import load_vllm_model

    MAX_LEN = 1024
    print(f"Loading {Path(MODEL).name} with max_model_len={MAX_LEN}...")
    llm, tok = load_vllm_model(MODEL, max_model_len=MAX_LEN, gpu_memory_utilization=0.4)

    fill = "Item: an unimportant detail. " * 400
    prompt = "Begin.\n" + fill + "\nQuestion: Is this prompt overlong?\nAnswer:"
    n_tokens = len(tok.encode(prompt))
    print(f"Prompt: {len(prompt)} chars, {n_tokens} tokens (cap={MAX_LEN}). Expecting clean error.")

    from vllm import SamplingParams

    try:
        outs = llm.generate(prompt, SamplingParams(temperature=0.0, max_tokens=4))
        print(f"⚠ Did NOT raise — generated text: {outs[0].outputs[0].text!r}")
        print(f"⚠ ATTENTION: silent acceptance of overlong input")
        return 1
    except Exception as exc:
        msg = str(exc)
        truncated = msg[:300] + ("…" if len(msg) > 300 else "")
        print(f"✓ Raised cleanly: {type(exc).__name__}: {truncated}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
