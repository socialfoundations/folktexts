# Phase 7 — Edge cases (vLLM backend)

Optional-but-recommended robustness checks listed in
`VLLM_MIGRATION_REPORT.md` §3 Phase 7. Harness:
`scripts/phase7_edge_cases.py` + `scripts/phase7_overcap.py`.

All tests run on a single B200 GPU on 2026-05-07. Models used:
`meta-llama--Llama-3.2-1B-Instruct` (Test 1, 3, 4, 5, over-cap) and
`meta-llama--Llama-3.2-3B-Instruct` (Test 2 second load).

| # | Sub-test | Outcome |
|---|---|---|
| 1 | `Benchmark.predict_proba` on a 1-row DataFrame | ✓ returns `(1, 2)` shape, score 0.4621 |
| 2 | Sequential model load (A=1B → free → B=3B) in same Python process | ✓ both runs complete, distinct outputs (mean p1: 0.5156 vs 0.7211) — no engine state leakage |
| 3a | Near-cap input (1452 / 2048 tokens) | ✓ generates `' Yes, the year'` cleanly |
| 3b | Over-cap input (2814 / 1024 tokens) | ✓ raises `VLLMValidationError`: explicit input/output overflow message, no silent truncation |
| 4 | Tied-logit cross-backend agreement on synthetic A-vs-B | ✓ both backends pick `B` (TF top-2 logit gap 0.25; ` B` and ` A` are vLLM top-2) |
| 5 | OOM with `gpu_memory_utilization=0.005` | ✓ raises `ValueError: No available memory for the cache blocks` at engine init — no hang |

## Detail: Test 4 — tied logits

```
Prompt:
Question: Which is more likely (assume equal priors)?
A. Heads on a fair coin.
B. Tails on a fair coin.
Answer:

TF top candidates (letter, prefix, id, logit):
   ('B', 'Ġ', 426, 25.0)        <- argmax
   ('A', 'Ġ', 362, 24.75)
   ('B', '', 33, 13.75)
   ('A', '', 32, 13.125)

vLLM top-K (top 5):
   id=  426  decoded=' B'  logprob=-0.6328
   id=  362  decoded=' A'  logprob=-0.8828
   id=  356  decoded=' C'  logprob=-3.7578
   ...
```

Both backends agree on `B` despite the small gap (0.25 logits in TF; ~0.25
nat logprob in vLLM). The cross-backend kernel-noise band documented
elsewhere (~1e-3 logprob from FlashInfer vs eager attention) is below
this gap, so determinism holds for the synthetic case.

## Detail: Test 5 — OOM clean failure

`gpu_memory_utilization=0.005` cannot fit even the 1B-Instruct KV cache.
vLLM's engine init raises:

```
ValueError: No available memory for the cache blocks. Try increasing
`gpu_memory_utilization` when initializing the engine.
```

The Python process then sees a `RuntimeError: Engine core initialization
failed` (vLLM's wrapper around the spawn-process error). The pattern is:
init fails fast, no hung worker, actionable message.

## Conclusion

All six edge-case checks pass. No regressions or unexpected behaviour
surfaced. Phase 7 is complete.
