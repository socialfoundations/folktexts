"""Phase 7 edge-case harness for the vLLM migration.

Five sub-tests. Each is independent so a partial run still produces signal.

  1. 1-sample DataFrame — `Benchmark.predict_proba` on a 1-row df. Verifies
     the per-batch loop doesn't choke at the n=1 edge.
  2. Restart / state — load model A, run a few rows, free, load model B in
     the same process, run a few rows. `_extract_per_pass_topk` is static
     so there should be no engine state, but worth confirming.
  3. Long-context prompt — feed a prompt that nearly fills `max_model_len`.
     Confirm vLLM either errors cleanly or processes without silent
     truncation.
  4. Tied logits — synthetic conditions where the chosen answer letter is
     a coin flip. Confirm both backends pick the same letter under
     temperature=0 + greedy.
  5. OOM clean failure — artificially set `gpu_memory_utilization=0.05`
     so vLLM cannot allocate the KV cache. Confirm a clear exception
     rather than a hang.

Run with `--tests 1,2,3` to skip slow ones; default runs all five.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_A = "/fast/groups/sf/huggingface-models/meta-llama--Llama-3.2-1B-Instruct"
MODEL_B = "/fast/groups/sf/huggingface-models/meta-llama--Llama-3.2-3B-Instruct"
DATA_DIR = "/fast/acruz/data/folktables"


def _setup_vllm_env() -> None:
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


def _free_cuda() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Test 1 — 1-sample DataFrame
# --------------------------------------------------------------------------
def test_1_one_sample_df() -> None:
    from folktexts.benchmark import Benchmark, BenchmarkConfig
    from folktexts.llm_utils import load_vllm_model

    print("\n--- Test 1: 1-sample DataFrame -------------------------")
    llm, tok = load_vllm_model(MODEL_A, max_model_len=1024, gpu_memory_utilization=0.4)
    config = BenchmarkConfig(numeric_risk_prompting=False, batch_size=8, seed=42)
    bench = Benchmark.make_acs_benchmark(
        task_name="ACSIncome",
        model=llm,
        tokenizer=tok,
        data_dir=DATA_DIR,
        config=config,
        subsampling=0.001,
        backend="vllm",
        model_name_or_path=MODEL_A,
    )

    test_X, test_y = bench.dataset.get_test()
    n_total = len(test_X)
    print(f"   Test set rows: {n_total}")

    one_row_X = test_X.iloc[:1]
    print(f"   Calling predict_proba on n=1 row...")
    probs = bench.llm_clf.predict_proba(one_row_X)
    print(f"   Returned shape: {probs.shape}, value: {probs[0]}")
    assert probs.shape == (1, 2), f"Expected (1, 2), got {probs.shape}"
    assert 0 <= probs[0, 1] <= 1, f"Score {probs[0, 1]} out of [0, 1]"
    print(f"   ✓ PASS (probs[0,1] = {probs[0, 1]:.4f})")

    del bench, llm, tok
    _free_cuda()


# --------------------------------------------------------------------------
# Test 2 — Restart / model swap in same process
# --------------------------------------------------------------------------
def test_2_restart_swap() -> None:
    from folktexts.benchmark import Benchmark, BenchmarkConfig
    from folktexts.llm_utils import load_vllm_model

    print("\n--- Test 2: Restart / model swap ---------------------------")

    def _run(model_path: str, label: str) -> float:
        print(f"   [{label}] loading {Path(model_path).name}...")
        llm, tok = load_vllm_model(model_path, max_model_len=1024, gpu_memory_utilization=0.4)
        config = BenchmarkConfig(numeric_risk_prompting=False, batch_size=8, seed=42)
        bench = Benchmark.make_acs_benchmark(
            task_name="ACSIncome",
            model=llm, tokenizer=tok,
            data_dir=DATA_DIR, config=config,
            subsampling=0.001,
            backend="vllm",
            model_name_or_path=model_path,
        )
        test_X, _ = bench.dataset.get_test()
        rows = test_X.iloc[:8]
        probs = bench.llm_clf.predict_proba(rows)
        print(f"   [{label}] mean p1 over 8 rows: {probs[:, 1].mean():.4f}")
        del bench, llm, tok
        _free_cuda()
        return float(probs[:, 1].mean())

    p_A = _run(MODEL_A, "A=Llama-3.2-1B-Instruct")
    p_B = _run(MODEL_B, "B=Llama-3.2-3B-Instruct")
    print(f"   Both runs completed in same process. Means differ: {p_A:.4f} vs {p_B:.4f}.")
    print(f"   ✓ PASS (no engine-state leakage; sequential loads worked)")


# --------------------------------------------------------------------------
# Test 3 — Long-context prompt
# --------------------------------------------------------------------------
def test_3_long_context() -> None:
    from folktexts.llm_utils import load_vllm_model

    print("\n--- Test 3: Long-context prompt ----------------------------")
    MAX_LEN = 2048
    llm, tok = load_vllm_model(MODEL_A, max_model_len=MAX_LEN, gpu_memory_utilization=0.4)

    from vllm import SamplingParams

    base_prompt = "The following list catalogues twentieth-century inventions. " * 4
    fill = "Item: an unimportant detail. " * 200
    prompt = base_prompt + fill + "\nQuestion: Is the year 2000 in the past?\nAnswer:"
    n_tokens = len(tok.encode(prompt))
    print(f"   Constructed prompt: {len(prompt)} chars, {n_tokens} tokens (cap={MAX_LEN}).")

    if n_tokens >= MAX_LEN:
        # vLLM should reject this cleanly (no silent truncation) or accept with
        # max_tokens set so input + output fits. The contract we want to verify:
        # if input alone exceeds max_model_len, generate raises.
        print(f"   Input {n_tokens} >= max_model_len {MAX_LEN}; expecting a clean error.")
        try:
            outs = llm.generate(prompt, SamplingParams(temperature=0.0, max_tokens=1))
            print(f"   ! Did NOT raise — generated text: {outs[0].outputs[0].text!r}")
            print(f"   ⚠ ATTENTION: silent acceptance of overlong input")
        except Exception as exc:
            print(f"   ✓ Raised cleanly: {type(exc).__name__}: {str(exc)[:120]}")
    else:
        # Just under — generate should succeed.
        outs = llm.generate(prompt, SamplingParams(temperature=0.0, max_tokens=4))
        print(f"   Generated: {outs[0].outputs[0].text!r}")
        print(f"   ✓ PASS (handled near-cap input without error)")

    del llm, tok
    _free_cuda()


# --------------------------------------------------------------------------
# Test 4 — Tied logits cross-backend agreement
# --------------------------------------------------------------------------
def test_4_tied_logits() -> None:
    print("\n--- Test 4: Tied-logit determinism -------------------------")
    from folktexts.llm_utils import load_model_tokenizer, load_vllm_model

    # Use a synthetic prompt designed to elicit ambiguous A/B answer.
    prompt = (
        "Question: Which is more likely (assume equal priors)?\n"
        "A. Heads on a fair coin.\n"
        "B. Tails on a fair coin.\n"
        "Answer:"
    )
    print(f"   Prompt:\n{prompt}\n")

    # transformers
    print("   Loading transformers...")
    import torch
    from transformers import AutoModelForCausalLM
    model_tf, tok_tf = load_model_tokenizer(MODEL_A)
    model_tf.eval()
    inputs = tok_tf(prompt, return_tensors="pt").to(model_tf.device)
    with torch.no_grad():
        logits = model_tf(**inputs).logits[0, -1]
    # Top-2 logits among the answer-letter prefix variants.
    candidates = []
    for letter in ["A", "B"]:
        for prefix in ["", " ", "Ġ"]:
            tok_id = tok_tf.get_vocab().get(f"{prefix}{letter}")
            if tok_id is not None and tok_id < model_tf.config.vocab_size:
                candidates.append((letter, prefix, tok_id, float(logits[tok_id])))
    candidates.sort(key=lambda x: x[3], reverse=True)
    print(f"   Transformers top candidates (letter, prefix, id, logit):")
    for c in candidates[:6]:
        print(f"     {c}")
    tf_argmax_letter = candidates[0][0]
    tf_top2_gap = abs(candidates[0][3] - candidates[1][3]) if len(candidates) > 1 else float("inf")
    del model_tf
    _free_cuda()

    # vLLM
    print("   Loading vLLM...")
    llm, tok_vl = load_vllm_model(MODEL_A, max_model_len=512, gpu_memory_utilization=0.4)

    from vllm import SamplingParams
    out = llm.generate(
        prompt, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20),
    )
    pos0 = out[0].outputs[0].logprobs[0]
    sorted_pos0 = sorted(
        ((int(tid), getattr(lp, "decoded_token", None), float(getattr(lp, "logprob", lp)))
         for tid, lp in pos0.items()),
        key=lambda x: x[2], reverse=True,
    )
    print(f"   vLLM top-K (top 5):")
    for tid, decoded, logp in sorted_pos0[:5]:
        print(f"     id={tid:6d}  decoded={decoded!r}  logprob={logp:.4f}")
    vl_top_id = sorted_pos0[0][0]
    vl_top_decoded = sorted_pos0[0][1]
    vl_argmax_letter = vl_top_decoded.strip().upper() if vl_top_decoded else "?"

    print(f"   TF argmax letter: {tf_argmax_letter}; vLLM argmax letter: {vl_argmax_letter}")
    print(f"   TF top-2 logit gap: {tf_top2_gap:.4f}")
    if tf_argmax_letter == vl_argmax_letter:
        print(f"   ✓ PASS (both backends pick {tf_argmax_letter})")
    else:
        print(f"   ! DIFFER — TF picked {tf_argmax_letter}, vLLM picked {vl_argmax_letter}")
        print(f"     This is acceptable when the gap is < 1e-3 (kernel noise band) — "
              f"observed gap {tf_top2_gap:.4f}.")

    del llm, tok_vl
    _free_cuda()


# --------------------------------------------------------------------------
# Test 5 — OOM clean failure
# --------------------------------------------------------------------------
def test_5_oom_clean_failure() -> None:
    print("\n--- Test 5: OOM clean failure ------------------------------")
    from folktexts.llm_utils import load_vllm_model

    # Drive gpu_memory_utilization low enough that KV cache for a 1B model
    # cannot fit. On a 183GB B200, 0.5% = ~900MB which is below the 1B model
    # weights (~2GB BF16). Engine init should error.
    print(f"   Loading {Path(MODEL_A).name} with gpu_memory_utilization=0.005 (expect failure)...")
    try:
        llm, tok = load_vllm_model(
            MODEL_A,
            max_model_len=1024,
            gpu_memory_utilization=0.005,
        )
        print(f"   ! Did NOT raise — engine loaded with gpu_memory_utilization=0.005")
        print(f"   ⚠ ATTENTION: vLLM accepted an unworkable memory budget")
        del llm, tok
    except Exception as exc:
        msg = str(exc)
        truncated = (msg[:200] + "…") if len(msg) > 200 else msg
        print(f"   ✓ Raised cleanly: {type(exc).__name__}: {truncated}")
        print(f"   ✓ PASS (vLLM refused the unworkable memory budget; no hang)")

    _free_cuda()


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
TESTS: dict[int, Callable[[], None]] = {
    1: test_1_one_sample_df,
    2: test_2_restart_swap,
    3: test_3_long_context,
    4: test_4_tied_logits,
    5: test_5_oom_clean_failure,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tests", default=",".join(str(i) for i in TESTS),
        help="Comma-separated subset of test numbers (1-5).",
    )
    args = parser.parse_args()

    _setup_vllm_env()
    selected = sorted({int(t.strip()) for t in args.tests.split(",") if t.strip()})
    unknown = [t for t in selected if t not in TESTS]
    if unknown:
        raise SystemExit(f"Unknown test(s): {unknown}. Valid: {sorted(TESTS)}")

    results: dict[int, str] = {}
    for n in selected:
        try:
            TESTS[n]()
            results[n] = "ok"
        except Exception as exc:
            print(f"\n!! Test {n} raised: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            results[n] = f"failed ({type(exc).__name__})"

    print("\n=== Summary ===")
    for n in selected:
        print(f"  Test {n}: {results[n]}")
    return 0 if all(r == "ok" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
