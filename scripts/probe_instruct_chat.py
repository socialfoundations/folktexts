"""Probe pos-1 on an Instruct model under the chat-numeric path.

Renders the chat-template prompt that production uses (assistant prefill =
`Answer (between 0 and 1): 0.`, trim trailing <|eot_id|>) on Llama-3.2-3B-Instruct.
Mask=digits, raw_logprobs, max_tokens=2 so we can see the unmasked pos-1
distribution after pos-0 is forced to a digit. If the instruct model already
naturally wants a digit at pos-1, then the current prompt is fine for chat
models — only base Llama-3 is the corner case.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_PATH = "/fast/groups/sf/huggingface-models/meta-llama--Llama-3.2-3B-Instruct"
DATA_DIR = "/fast/acruz/data/folktables"
N_ROWS = 16


def main() -> int:
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    from transformers import AutoConfig, AutoTokenizer

    from folktexts.acs.acs_columns import acs_income_numeric_qa
    from folktexts.acs.acs_dataset import ACSDataset
    from folktexts.acs.acs_tasks import ACSTaskMetadata
    from folktexts.prompting import encode_row_prompt_chat
    from folktexts.task import TaskMetadata

    task = TaskMetadata.get_task("ACSIncome")
    if not isinstance(task, ACSTaskMetadata):
        print(f"Unexpected task type: {type(task).__name__}")
        return 1
    dataset = ACSDataset.make_from_task(
        task=task, cache_dir=DATA_DIR, subsampling=0.005, seed=42,
    )
    rows = dataset.data.iloc[:N_ROWS]

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    vocab_dim = int(cfg.vocab_size)

    digit_token_ids = sorted({
        tid for tk, tid in tok.get_vocab().items()
        if tk.isdecimal() and 0 <= tid < vocab_dim
    })
    digit_id_set = set(digit_token_ids)

    prompts = [
        encode_row_prompt_chat(row, task, tok, question=acs_income_numeric_qa, numeric=True)
        for _, row in rows.iterrows()
    ]
    print(f"Sample prompt tail (last 200 chars):\n{prompts[0][-200:]!r}\n")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_PATH, dtype="bfloat16", gpu_memory_utilization=0.55,
        max_model_len=2048, max_logprobs=200, seed=42,
        logprobs_mode="raw_logprobs",
    )
    sampling = SamplingParams(
        temperature=0.0, max_tokens=2, logprobs=50,
        allowed_token_ids=digit_token_ids, seed=42,
    )
    outputs = llm.generate(prompts, sampling)

    NEWLINE_IDS = set(tok.encode("\n", add_special_tokens=False))
    EOT_ID = tok.eos_token_id
    PERIOD_ID = 13

    digit0_sum = digit1_sum = period1_sum = newline1_sum = eot1_sum = 0.0
    n = 0
    for i, out in enumerate(outputs):
        completion = out.outputs[0]
        n += 1
        positions = completion.logprobs or []
        for pos_idx, pos in enumerate(positions[:2]):
            digit_p = period_p = newline_p = eot_p = 0.0
            for tid, lp in pos.items():
                p = math.exp(float(getattr(lp, "logprob", lp)))
                if tid in digit_id_set:
                    digit_p += p
                if tid == PERIOD_ID:
                    period_p += p
                if tid in NEWLINE_IDS:
                    newline_p += p
                if tid == EOT_ID:
                    eot_p += p
            if pos_idx == 0:
                digit0_sum += digit_p
            else:
                digit1_sum += digit_p
                period1_sum += period_p
                newline1_sum += newline_p
                eot1_sum += eot_p

        if i == 0:
            sorted_pos1 = sorted(
                ((int(tid), getattr(lp, "decoded_token", None), float(getattr(lp, "logprob", lp)))
                 for tid, lp in positions[1].items()),
                key=lambda x: x[2], reverse=True,
            )
            print(f"Row 0 pos-1 unmasked top-15:")
            for tid, decoded, logp in sorted_pos1[:15]:
                p = math.exp(logp)
                is_digit = "D" if tid in digit_id_set else " "
                print(f"   {is_digit} id={tid:6d}  p={p:.4f}  decoded={decoded!r}")

    print(f"\n=== Aggregate over {n} rows (Llama-3.2-3B-Instruct, chat-numeric, mask ON, raw_logprobs) ===")
    print(f"  unmasked digit-mass at pos-0: {digit0_sum/n:.4f}")
    print(f"  unmasked digit-mass at pos-1: {digit1_sum/n:.4f}")
    print(f"  unmasked '.' at pos-1:        {period1_sum/n:.4f}")
    print(f"  unmasked '\\n' at pos-1:       {newline1_sum/n:.4f}")
    print(f"  unmasked EOT at pos-1:        {eot1_sum/n:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
