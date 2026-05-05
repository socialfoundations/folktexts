"""End-to-end prompt inspection for the `with_answer_prefill` fix.

Runs `Benchmark.make_acs_benchmark` twice on a tiny subsample — once with
`use_chat_template=True`, once without — and prints the rendered prompt for
the first row that gets fed to the model. The chat-template path should show
`Answer (between 0 and 1): 0.` exactly once (in the assistant turn at the
tail, supplied by the structural fix). The zero-shot path should show the
same suffix once at the very end of the user prompt (legacy behaviour).
"""
from __future__ import annotations

import gc
import logging
import os
import sys
import tempfile
from pathlib import Path

import torch

from folktexts.benchmark import Benchmark, BenchmarkConfig
from folktexts.llm_utils import load_model_tokenizer

MODEL_PATH = "/fast/groups/sf/huggingface-models/meta-llama--Llama-3.2-3B-Instruct"
DATA_DIR = "/fast/acruz/data/folktables"
SUBSAMPLING = 0.001         # ~16 rows on ACSIncome test
BATCH_SIZE = 8
TASK_NAME = "ACSIncome"


def _wrap_encode_row_to_print_first(bench: Benchmark, label: str) -> None:
    """Tee the classifier's `_encode_row` so the first invocation is printed."""
    original = bench.llm_clf._encode_row
    state = {"printed": False}

    def teed(row, *args, **kwargs):
        out = original(row, *args, **kwargs)
        if not state["printed"]:
            state["printed"] = True
            banner = f"  {label} — first rendered prompt  "
            sep = "=" * max(80, len(banner))
            print(f"\n{sep}\n{banner.center(len(sep), '=')}\n{sep}")
            print(out)
            print(f"{sep}\n  prompt length: {len(out)} chars\n  "
                  f"ends with: {out[-80:]!r}\n{sep}\n", flush=True)
        return out

    bench.llm_clf._encode_row = teed


def _run_one(use_chat_template: bool, results_dir: Path) -> None:
    label = "CHAT-TEMPLATE" if use_chat_template else "ZERO-SHOT"
    print(f"\n>>> Building benchmark — {label}, numeric, subsampling={SUBSAMPLING}", flush=True)

    model, tokenizer = load_model_tokenizer(MODEL_PATH)

    config = BenchmarkConfig(
        numeric_risk_prompting=True,
        use_chat_template=use_chat_template,
        batch_size=BATCH_SIZE,
        seed=42,
    )

    bench = Benchmark.make_acs_benchmark(
        task_name=TASK_NAME,
        model=model,
        tokenizer=tokenizer,
        data_dir=DATA_DIR,
        config=config,
        subsampling=SUBSAMPLING,
    )

    _wrap_encode_row_to_print_first(bench, label)
    out_dir = results_dir / label.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    bench.run(results_root_dir=out_dir, fit_threshold=False)

    print(f">>> {label} run complete; results in {out_dir}", flush=True)
    test_auc = bench.results.get("test_auc") or bench.results.get("auc")
    if test_auc is not None:
        print(f">>> {label} test AUC: {test_auc:.4f}", flush=True)

    # Free GPU memory before the next run
    del bench, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    with tempfile.TemporaryDirectory(prefix="folktexts_e2e_") as td:
        results_dir = Path(td)
        _run_one(use_chat_template=False, results_dir=results_dir)
        _run_one(use_chat_template=True, results_dir=results_dir)
    print("\n✓ Both end-to-end runs finished without errors.")


if __name__ == "__main__":
    main()
