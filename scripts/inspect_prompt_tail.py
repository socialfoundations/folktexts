"""Inspect the EXACT last bytes of the prompts that hit the model.

For each of the four (model x mode) cells:
  - Llama-3-8B base, zero-shot numeric (encode_row_prompt)
  - Llama-3-8B base, chat-numeric (encode_row_prompt_chat — base models trip
    the system-role probe; we still render to see what comes out)
  - Llama-3.2-3B-Instruct, zero-shot numeric
  - Llama-3.2-3B-Instruct, chat-numeric

print:
  1. Full last 300 chars with repr() (surfaces special tokens, whitespace,
     newlines, BOMs)
  2. The token IDs of the trailing 8 tokens after `tokenizer.encode(prompt,
     add_special_tokens=False)` so we can confirm what the model actually
     sees as the "current cursor" position
  3. The decoded text of each trailing token, to spot anomalies like a
     stray `<|eot_id|>` after the prefill or whitespace right before `0.`
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer  # noqa: E402

from folktexts.acs.acs_columns import acs_income_numeric_qa  # noqa: E402
from folktexts.acs.acs_dataset import ACSDataset  # noqa: E402
from folktexts.acs.acs_tasks import ACSTaskMetadata  # noqa: E402
from folktexts.prompting import (  # noqa: E402
    encode_row_prompt,
    encode_row_prompt_chat,
    tokenizer_supports_system_prompt,
)
from folktexts.task import TaskMetadata  # noqa: E402

DATA_DIR = "/fast/acruz/data/folktables"
MODEL_PATHS = [
    ("/fast/groups/sf/huggingface-models/meta-llama--Meta-Llama-3-8B", "llama-3-8B-base"),
    ("/fast/groups/sf/huggingface-models/meta-llama--Llama-3.2-3B-Instruct", "llama-3.2-3B-instruct"),
]


def banner(s: str) -> None:
    line = "=" * 88
    print(f"\n{line}\n  {s}\n{line}")


def render_zero_shot(task, row, question) -> str:
    return encode_row_prompt(row, task=task, question=question)


def render_chat(tokenizer, task, row, question) -> str:
    return encode_row_prompt_chat(
        row, task=task, tokenizer=tokenizer,
        question=question, numeric=True,
    )


def dump_prompt(label: str, prompt: str, tokenizer) -> None:
    banner(f"{label} — len={len(prompt)} chars")
    print(f"\n[Last 300 chars (repr)]:\n{prompt[-300:]!r}\n")

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"[Token count]: {len(ids)}")
    print(f"[Last 12 token IDs]: {ids[-12:]}")
    decoded = [(tid, tokenizer.decode([tid])) for tid in ids[-12:]]
    print("[Last 12 tokens (id, decoded)]:")
    for tid, d in decoded:
        print(f"   id={tid:6d}  ->  {d!r}")


def main() -> int:
    task = TaskMetadata.get_task("ACSIncome")
    if not isinstance(task, ACSTaskMetadata):
        print(f"Unexpected task type: {type(task).__name__}")
        return 1
    dataset = ACSDataset.make_from_task(
        task=task, cache_dir=DATA_DIR, subsampling=0.001, seed=42,
    )
    row = dataset.data.iloc[0]
    question = acs_income_numeric_qa

    for path, label in MODEL_PATHS:
        tok = AutoTokenizer.from_pretrained(path)
        supports_sys = tokenizer_supports_system_prompt(tok)
        banner(f"MODEL: {label} (path={path})  supports_system={supports_sys}")

        # 1) Zero-shot numeric
        zs = render_zero_shot(task, row, question)
        dump_prompt(f"{label} — ZERO-SHOT NUMERIC (encode_row_prompt)", zs, tok)

        # 2) Chat-numeric
        try:
            ch = render_chat(tok, task, row, question)
            dump_prompt(f"{label} — CHAT NUMERIC (encode_row_prompt_chat)", ch, tok)
        except Exception as exc:
            print(f"\n[chat path failed for {label}]: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
