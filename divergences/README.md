# vLLM migration — characterised divergences (post-PR follow-up)

This folder documents the cross-backend divergences surfaced during the
Phase 1-6 sweeps on `vllm-backend`. The migration PR (#31) ships with these
known and bounded; this folder is for the post-merge investigation that
roots out the underlying causes.

**Audience**: future-me / fresh-context Claude. Each file is self-contained:
symptom, root-cause hypothesis, repro, investigation paths, decision criteria.

## Index

| # | Title | Cells affected | Status |
|---|---|---|---|
| 01 | [Llama-3 multi-digit numeric divergence](01_llama3_numeric_multidigit.md) | 2 cells (Llama-3-8B base, Llama-3-70B-Instruct) | **ROOT-CAUSED & FIXED** — vLLM `logprobs_mode="processed_logprobs"` |
| 02 | [Qwen3-1.7B chat-MCQ collapse on transformers](02_qwen3_chatmcq_tf_collapse.md) | 1 cell (Qwen3-1.7B) | needs root-cause (TF-side; vLLM is *better*) |
| 03 | [Qwen3-Thinking exceeds max_new_tokens with thinking-on](03_qwen3_thinking_max_tokens.md) | 1 model (Qwen3-Thinking, both backends) | **FIXED** — `ReasoningQA.max_new_tokens` bumped to 8000 |

## How to use

1. Read the index entry for the divergence you want to tackle.
2. Open the file; the **Repro** section gives a one-shot script that exercises the issue.
3. The **Investigation paths** section lists ordered hypotheses, each with a check command.
4. The **Disposition** section tells you what shipping decision to make if root-cause is irreducible.

## Branch state at PR open

Validation: 38 cross-backend cells, 35/38 within strict gates. Two of the
three outliers (01 and 03) have been root-caused and fixed in this PR;
the third (02 — TF-side bug, vLLM was already better) is documented for
follow-up.

## Post-fix validation (2026-05-07)

| Cell | Pre-fix | Post-fix |
|---|---|---|
| Llama-3-8B base numeric (TF AUC 0.5591) | vLLM AUC 0.5071 (Δ=-0.052), 99% rows = 0.5 | **AUC 0.5756** (Δ=+0.017), 9 unique values |
| Llama-3-70B-Instruct numeric (TF AUC 0.8258) | vLLM AUC 0.8476 (Δ=+0.022) | **AUC 0.8258** (Δ=+0.000, matches TF to 4 dp) |
| Mistral-7B-v0.1 numeric (regression check, TF AUC 0.7363) | vLLM 0.7417 | **0.7421** (no regression) |
| Llama-3.2-3B-Instruct chat-MC (regression check) | — | AUC 0.8206, 1226 unique (MC unaffected) |
| Qwen3-Thinking thinking-on reasoning | AUC 0.7369, regex fallback 13.1% | **AUC 0.7990, regex fallback 2.5%** |

89 unit tests pass.
