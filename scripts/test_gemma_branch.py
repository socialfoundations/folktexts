"""End-to-end test of the Gemma-rejection branch in Benchmark.make_acs_benchmark.

We load Llama-3.2-1B-Instruct (which scores correctly) and overwrite its
chat_template with a Gemma-1/2-style template that explicitly rejects the
`system` role. This forces tokenizer_supports_system_prompt() to return False,
exercising the fallback in folktexts/benchmark.py:590-606.

Two scenarios:
  1. No user-supplied system_prompt -> INFO log "running without a system prompt."
  2. User-supplied system_prompt    -> WARNING log "user-supplied system_prompt will be discarded."

Both must complete the benchmark successfully with finite metrics.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from folktexts.acs import ACSDataset, ACSTaskMetadata
from folktexts.benchmark import Benchmark, BenchmarkConfig
from folktexts.prompting import tokenizer_supports_system_prompt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
logging.getLogger().setLevel(logging.INFO)

LLAMA = "/fast/groups/sf/huggingface-models/meta-llama--Llama-3.2-1B-Instruct"
DATA_DIR = "/fast/acruz/data"

# Gemma-1/2-style chat template that hard-fails on the system role.
GEMMA_LIKE_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ raise_exception('System role not supported') }}"
    "{% endif %}"
    "{% for m in messages %}"
    "<|{{ m['role'] }}|>{{ m['content'] }}<|end|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def make_patched_tokenizer():
    tok = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
    tok.chat_template = GEMMA_LIKE_TEMPLATE
    assert tokenizer_supports_system_prompt(tok) is False, (
        "Patched tokenizer should reject the system role."
    )
    return tok


def run_scenario(label: str, results_subdir: str, system_prompt=None):
    print(f"\n{'='*60}\n[{label}]\n{'='*60}", flush=True)
    tokenizer = make_patched_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA, local_files_only=True, torch_dtype="bfloat16", device_map="cuda"
    )

    task = ACSTaskMetadata.get_task("ACSIncome")
    dataset = ACSDataset.make_from_task(task=task, cache_dir=DATA_DIR, seed=42)
    dataset.subsample(0.001)

    config_kwargs = {
        "use_chat_template": True,
        "batch_size": 16,
    }
    if system_prompt is not None:
        config_kwargs["system_prompt"] = system_prompt
    config = BenchmarkConfig(**config_kwargs)

    bench = Benchmark.make_benchmark(
        task=task,
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
    out_dir = Path("results") / results_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    bench.run(results_root_dir=str(out_dir))
    metrics = bench.results
    print(f"[{label}] roc_auc={metrics.get('roc_auc'):.4f} n_samples={metrics.get('n_samples')}")
    return metrics


if __name__ == "__main__":
    # Scenario 1: no user-supplied system_prompt -> INFO log
    m1 = run_scenario("scenario_1_no_user_sysprompt", "gemma_branch_no_sysprompt")

    # Scenario 2: user-supplied system_prompt -> WARNING log
    m2 = run_scenario(
        "scenario_2_with_user_sysprompt",
        "gemma_branch_with_sysprompt",
        system_prompt="You are a careful classifier. Be calibrated.",
    )

    print("\nSUMMARY:")
    print(f"  scenario_1 AUC = {m1['roc_auc']:.4f}")
    print(f"  scenario_2 AUC = {m2['roc_auc']:.4f}")
    sys.exit(0)
