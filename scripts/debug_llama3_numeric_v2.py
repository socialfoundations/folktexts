"""Phase 2 debug: real ACS prompts + production vLLM config.

Use the actual ACSIncome prompt that the sweep produces and run the EXACT
same vLLM call the production VLLMClassifier makes (logprobs=50,
allowed_token_ids = digit ids, num_forward_passes=2). Dump pos-0 and pos-1
top-K dicts so we can see why the production sweep collapses to 0.5/0.0/0.2/0.1.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_PATH = "/fast/groups/sf/huggingface-models/meta-llama--Meta-Llama-3-8B"
DATA_DIR = "/fast/acruz/data/folktables"


def main() -> int:
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    from folktexts.acs.acs_dataset import ACSDataset
    from folktexts.acs.acs_tasks import ACSTaskMetadata
    from folktexts.task import TaskMetadata

    # Load ACSIncome the same way the benchmark does.
    task = TaskMetadata.get_task("ACSIncome")
    if not isinstance(task, ACSTaskMetadata):
        print(f"Unexpected task type: {type(task).__name__}")
        return 1
    dataset = ACSDataset.make_from_task(
        task=task, cache_dir=DATA_DIR, subsampling=0.005, seed=42,
    )
    print(f"Dataset rows: {len(dataset.data)}")

    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    vocab_dim = int(cfg.vocab_size)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    digit_token_ids = sorted({
        tok_id for token, tok_id in tok.get_vocab().items()
        if token.isdecimal() and 0 <= tok_id < vocab_dim
    })

    from folktexts.prompting import encode_row_prompt
    from folktexts.acs.acs_columns import acs_income_numeric_qa

    test_data = dataset.data.iloc[:10]
    print(f"\nRendering prompts for first 10 rows using encode_row_prompt.")

    numeric_q = acs_income_numeric_qa

    prompts: list[str] = []
    for _, row in test_data.iterrows():
        prompts.append(encode_row_prompt(row, task=task, question=numeric_q))

    print("Prompt[0] (first 500 chars):")
    print(prompts[0][:500])
    print("...")
    print("Prompt[0] (last 100 chars):")
    print(repr(prompts[0][-100:]))

    print("\n" + "=" * 80)
    print("VLLM PRODUCTION SETTINGS: logprobs=50, allowed_token_ids=digits, num_forward_passes=2")
    print("=" * 80)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.6,
        max_model_len=2048,
        max_logprobs=200,
        seed=42,
        logprobs_mode="processed_logprobs",  # FIX: returns post-mask distribution
    )

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=2,
        logprobs=50,
        allowed_token_ids=digit_token_ids,
        seed=42,
    )
    outputs = llm.generate(prompts, sampling)

    for i, request_output in enumerate(outputs):
        completion = request_output.outputs[0]
        chosen_ids = list(completion.token_ids)
        chosen_text = "".join(tok.decode([x]) for x in chosen_ids)
        print(f"\n--- Row {i} ---")
        print(f"Chosen token_ids: {chosen_ids} -> text: {chosen_text!r}")
        for pos_idx, pos in enumerate(completion.logprobs):
            sorted_pos = sorted(
                [(int(tid), getattr(lp, "decoded_token", None), float(getattr(lp, "logprob", lp)))
                 for tid, lp in pos.items()],
                key=lambda x: x[2], reverse=True,
            )
            digit_in_top = [t for t in sorted_pos if t[0] in set(digit_token_ids)]
            print(f"  pos {pos_idx}: top-K size={len(pos)}, top-10 = {sorted_pos[:10]}")
            print(f"           digit tokens in top-K (count={len(digit_in_top)}): "
                  f"top-5 = {digit_in_top[:5]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
