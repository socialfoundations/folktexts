# vLLM migration — validation status

Aggregates the artifacts under `results/`. Each phase has a gate that the migration must pass before flipping the CLI default.

## Phase 1 — Cross-backend Table 1 (vLLM vs transformers)

- Source: `results/paper-reproduction-vllm/TABLE1_BACKEND_COMPARISON.md`
- Coverage: **20** matched pairs (target: 20)
- AUC gate (|Δ| ≤ 0.01): **18/20** within tolerance
- ECE gate (|Δ| ≤ 0.02): **18/20** within tolerance

## Phase 2 — Paper Table 1 reproduction

- transformers vs paper: `results/paper-reproduction/TABLE1_REPRODUCTION.md`
- vLLM vs paper: `results/paper-reproduction-vllm/TABLE1_REPRODUCTION.md`

## Phase 3 — Multi-seed stability

- Source: `results/multi-seed-stability/REPORT.md`

## Phase 4 — Modern + thinking-model coverage

- Source: `results/extended-sweep/REPORT.md`
- AUC gate (|Δ| ≤ 0.015): **13/14** within tolerance
- ECE gate (|Δ| ≤ 0.025): **13/14** within tolerance

## Phase 5 — Reasoning sweep + failure-rate audit

- Source: `results/reasoning-sweep/REPORT.md`

## Phase 7 — Edge cases

- Source: `results/phase7-edge-cases/REPORT.md`
- 1-sample DataFrame, model swap in same process, near/over-cap inputs,
  tied-logit cross-backend agreement, OOM clean failure: **6/6 PASS**.

## Post-PR divergence fixes (2026-05-07)

- Source: `divergences/`
- **#01 Llama-3 multi-digit numeric** — root-caused: vLLM default
  `logprobs_mode="raw_logprobs"` returned top-K from the unmasked
  distribution, leaking `'.'` to the QA decoder which picked it over
  the only-allowed digit. Fix: `load_vllm_model` defaults to
  `logprobs_mode="processed_logprobs"`. Validated:
  - Llama-3-8B base numeric: AUC 0.5071 → 0.5756 (TF: 0.5591); 0.5
    collapse gone, 4 → 9 unique values.
  - Llama-3-70B-Instruct numeric: AUC 0.8476 → 0.8258 (TF: 0.8258 —
    matches to 4 dp).
  - Mistral-7B-v0.1 numeric (regression check): AUC 0.7417 → 0.7421
    (TF: 0.7363) — no regression on tokenizers without multi-digit.
- **#03 Qwen3-Thinking max_new_tokens** — root-caused: 5000-token
  budget too tight for thinking-on. Fix: bumped
  `ReasoningQA.max_new_tokens` to 8000; CLI `max_model_len` heuristic
  now derives from `ReasoningQA.max_new_tokens` symbolically. Validated:
  Qwen3-Thinking thinking-on AUC 0.7369 → 0.7990; regex 0.5-fallback
  rate 13.1% → 2.5% (5× reduction).
- **#02 Qwen3-1.7B chat-MCQ** — TF-side bug (vLLM was already better).
  Not addressed by these fixes; documented for follow-up.

