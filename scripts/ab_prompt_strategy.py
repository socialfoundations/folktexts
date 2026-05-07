"""A/B compare numeric strategies on Llama-3-8B base, ACSIncome.

Variants:
  A_control : "Answer (between 0 and 1): 0."  +  2 forward passes, mask=digits,
              processed_logprobs, decoder = argmax-of-numeric-tokens (current
              production path).
  B_free    : "Answer (between 0 and 1): "      + 5 forward passes, NO mask,
              greedy generation, regex-extract first probability from text
              (the user's suggested approach).
  C_dot     : "Answer (between 0 and 1): "      + 4 forward passes,
              mask=digits+'.', argmax-over-numeric-tokens decoder
              (structured "free" — keeps mask but allows decimal point).

Compute AUC vs ground-truth labels for all three on the same row set; report
how many unique scores each produces and the 0.5-collapse rate.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_PATH = "/fast/groups/sf/huggingface-models/meta-llama--Meta-Llama-3-8B"
DATA_DIR = "/fast/acruz/data/folktables"
SUBSAMPLING = 0.005
SEED = 42


def make_body(task, row, q_text: str) -> str:
    from folktexts.prompting import ACS_TASK_DESCRIPTION
    return (
        ACS_TASK_DESCRIPTION + "\n"
        + f"\nInformation:\n{task.get_row_description(row)}\n"
        + f"\nQuestion: {q_text}"
    )


def extract_prob_from_text(text: str) -> float | None:
    """Parse the first numeric expression that looks like a probability.

    Priority: 0.X / 0.XX / .X (decimal-leading-zero), then a bare 0–100
    number understood as percent. Anything outside [0, 1] after coercion
    is rejected.
    """
    # Decimal form (0.5, 0.567, .5, 0.0001, 1.0)
    m = re.search(r"(?<![.\d])([01]?\.\d+|1\.0)(?![\d])", text)
    if m:
        try:
            v = float(m.group(1))
            if 0 <= v <= 1:
                return v
        except ValueError:
            pass
    # Percent (50%, 50 percent, 50)
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        try:
            v = float(m.group(1)) / 100.0
            if 0 <= v <= 1:
                return v
        except ValueError:
            pass
    # Bare integer in [0, 100]
    m = re.search(r"(?<![.\d])(\d{1,3})(?![\d])", text)
    if m:
        try:
            v = float(m.group(1))
            if v <= 1:
                return v
            if v <= 100:
                return v / 100.0
        except ValueError:
            pass
    return None


def main() -> int:
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    import numpy as np
    from sklearn.metrics import roc_auc_score
    from transformers import AutoConfig, AutoTokenizer

    from folktexts.acs.acs_columns import acs_income_numeric_qa
    from folktexts.acs.acs_dataset import ACSDataset
    from folktexts.acs.acs_tasks import ACSTaskMetadata
    from folktexts.task import TaskMetadata

    task = TaskMetadata.get_task("ACSIncome")
    if not isinstance(task, ACSTaskMetadata):
        print(f"Unexpected task type: {type(task).__name__}")
        return 1
    dataset = ACSDataset.make_from_task(
        task=task, cache_dir=DATA_DIR, subsampling=SUBSAMPLING, seed=SEED,
    )

    test_X, test_y = dataset.get_test()
    rows = test_X
    labels = np.asarray(test_y).astype(int)
    print(f"Test rows: {len(rows)}  positive rate: {labels.mean():.3f}")

    q_text = acs_income_numeric_qa.text

    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    vocab_dim = int(cfg.vocab_size)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    digit_token_ids = sorted({
        tid for tk, tid in tok.get_vocab().items()
        if tk.isdecimal() and 0 <= tid < vocab_dim
    })
    PERIOD_ID = 13
    digit_plus_period = sorted(set(digit_token_ids) | {PERIOD_ID})
    print(f"|digits|={len(digit_token_ids)}  |digits+'.'|={len(digit_plus_period)}")

    # Build prompts ----------------------------------------------------
    bodies = [make_body(task, row, q_text) for _, row in rows.iterrows()]
    prompts_A = [b + "\nAnswer (between 0 and 1): 0." for b in bodies]
    prompts_B = [b + "\nAnswer (between 0 and 1): " for b in bodies]
    prompts_C = [b + "\nAnswer (between 0 and 1): " for b in bodies]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.55,
        max_model_len=2048,
        max_logprobs=200,
        seed=SEED,
        logprobs_mode="processed_logprobs",
    )

    # -------------- A: current production path --------------
    sampling_A = SamplingParams(
        temperature=0.0, max_tokens=2, logprobs=50,
        allowed_token_ids=digit_token_ids, seed=SEED,
    )
    out_A = llm.generate(prompts_A, sampling_A)

    scores_A: list[float] = []
    for request_output in out_A:
        from folktexts.llm_utils import decode_topk_logprobs_to_risk_estimate
        completion = request_output.outputs[0]
        per_pass = []
        for pos in completion.logprobs or []:
            per_pass.append({
                int(tid): float(getattr(lp, "logprob", lp))
                for tid, lp in pos.items()
            })
        score = decode_topk_logprobs_to_risk_estimate(
            per_pass,
            tokenizer_vocab=tok.get_vocab(),
            vocab_dim=vocab_dim,
            question=acs_income_numeric_qa,
        )
        scores_A.append(score)

    # -------------- B: free-generation + regex --------------
    sampling_B = SamplingParams(
        temperature=0.0, max_tokens=8, seed=SEED,
        # NO allowed_token_ids — model writes whatever it wants
    )
    out_B = llm.generate(prompts_B, sampling_B)

    scores_B: list[float] = []
    raw_B: list[str] = []
    fail_B = 0
    for request_output in out_B:
        text = request_output.outputs[0].text
        raw_B.append(text)
        v = extract_prob_from_text(text)
        if v is None:
            fail_B += 1
            v = 0.5
        scores_B.append(v)

    # -------------- C: bare prefill, mask=digits+'.' --------------
    sampling_C = SamplingParams(
        temperature=0.0, max_tokens=4, logprobs=50,
        allowed_token_ids=digit_plus_period, seed=SEED,
    )
    out_C = llm.generate(prompts_C, sampling_C)

    # Custom decoder for C: argmax over numeric tokens at each pass,
    # concat -> regex extract.
    scores_C: list[float] = []
    raw_C: list[str] = []
    fail_C = 0
    for request_output in out_C:
        completion = request_output.outputs[0]
        chosen_ids = list(completion.token_ids)
        text = "".join(tok.decode([t]) for t in chosen_ids)
        raw_C.append(text)
        v = extract_prob_from_text(text)
        if v is None:
            fail_C += 1
            v = 0.5
        scores_C.append(v)

    # -------------- Report --------------
    def _summary(name: str, s: list[float]) -> None:
        arr = np.asarray(s)
        auc = roc_auc_score(labels, arr) if len(set(s)) > 1 else float("nan")
        n_unique = len(set(s))
        rate_05 = float(np.mean(np.abs(arr - 0.5) < 1e-9))
        print(
            f"  {name:<14}  AUC={auc:.4f}  unique={n_unique:>4}  "
            f"P(score=0.5)={rate_05:.3%}  mean={arr.mean():.3f}  std={arr.std():.3f}"
        )

    print("\n=== Results ===")
    _summary("A (control)", scores_A)
    _summary("B (free)",    scores_B)
    _summary("C (dot+mask)", scores_C)
    print(f"\n  B regex-fail rate: {fail_B}/{len(scores_B)}")
    print(f"  C regex-fail rate: {fail_C}/{len(scores_C)}")

    print("\n=== First 10 raw outputs ===")
    print("Idx | A_score | B_text                        | B_score | C_text             | C_score | label")
    for i in range(min(10, len(scores_A))):
        print(
            f"{i:3d} | {scores_A[i]:.3f}   | {repr(raw_B[i])[:32]:<32} | {scores_B[i]:.3f}   | "
            f"{repr(raw_C[i])[:18]:<18} | {scores_C[i]:.3f}   | {labels[i]}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
