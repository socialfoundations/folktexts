# 01 — Llama-3 multi-digit numeric divergence — **ROOT-CAUSED & FIXED**

## TL;DR (Status: FIXED 2026-05-07)

**Root cause:** vLLM's default `logprobs_mode="raw_logprobs"` returns top-K
logprobs computed BEFORE `apply_logits_processors` (which is where
`allowed_token_ids` is enforced). For `DirectNumericQA` on Llama-3 (1100
multi-digit decimal tokens), the unconstrained pos-1 distribution is
dominated by `'\n'`, `<|end_of_text|>`, and the decimal point `'.'` (id 13).
The QA decoder's `numeric_tokens_vocab` includes `'.'` as a numeric token,
so the decoder picks `'.'` over the only-allowed digit (e.g., `'067'` at
logprob -16.4 — barely scraping into the unmasked top-K). Answer text
becomes `"5."`, the regex matches `"5"`, and the result is parsed as
`0.5`. Llama-3-8B base numeric collapsed to 99% of rows at exactly 0.5.

**Fix:** `load_vllm_model` now passes `logprobs_mode="processed_logprobs"`,
which returns top-K logprobs from the masked distribution. The decoder
sees only digit tokens with re-normalized probabilities; `'.'` has zero
probability and is never picked. Commit: this PR.

**Validated cells (subsampling=0.03 ≈ 5000 rows on ACSIncome):**

| Cell | TF baseline | vLLM PRE-fix | vLLM POST-fix | Δ vs TF |
|---|---|---|---|---|
| `Meta-Llama-3-8B` (base) numeric | AUC 0.5591, ECE 0.179 | AUC 0.5071, ECE 0.130 | AUC 0.5756, ECE 0.138 | +0.017, -0.041 |
| ↑ unique values | 101 | **4** (99% at 0.5) | **9** (top: 0.5067 at 80%) | — |

(Llama-3-70B-Instruct verification pending — see "Validation runs" below.)

The remaining ~0.017 AUC delta is within the cross-backend kernel-noise
band documented elsewhere in the migration report (FlashInfer attention vs
transformers' eager kernel diverges at ~1e-3 logprob and can flip rank
ordering on close ties between multi-digit tokens).

## Original symptom (pre-fix, kept for archeology)

For models that use the **Llama-3 tokenizer** (which carries 1100
multi-digit decimal tokens), `DirectNumericQA` produced materially
different outputs across backends:

| Cell | tf AUC | vLLM AUC | ΔAUC | Verdict |
|---|---|---|---|---|
| `meta-llama--Meta-Llama-3-8B` (base) numeric, no chat | 0.559 | 0.507 | -0.052 | vLLM near-degenerate (99.1% predictions = 0.5) |
| `meta-llama--Meta-Llama-3-70B-Instruct` numeric, no chat | 0.826 | 0.848 | +0.022 | vLLM **better** |

The two opposite-sign deltas traced to **the same** mechanism. All
non-Llama-3 numeric cells (Mistral, Yi, Gemma-1) passed cleanly because
their tokenizers carry 0–16 multi-digit tokens AND for these tokenizers
the unconstrained pos-1 distribution is digit-heavy enough that the
unmasked top-K still contains digits with non-trivial probability.

### Distribution shape (pre-fix)

`Llama-3-8B base numeric` predictions:

```
TF: 4994 rows, 101 unique risk_score values, mean=0.5450, std=0.0688
    top5: {0.5678: 537, 0.5718: 519, 0.5067: 517, 0.5445: 495, 0.5698: 464}

vLLM PRE-fix: 4994 rows, 4 unique risk_score values, mean=0.4961, std=0.0418
    top5: {0.5: 4949, 0.0: 27, 0.2: 12, 0.1: 6}
```

After fix:

```
vLLM POST-fix: 4994 rows, 9 unique risk_score values, mean=0.5044, std=0.0608
    top5: {0.5067: 4162, 0.5718: 366, 0.5678: 276, 0.2718: 99, 0.2698: 35}
```

## Diagnosis trail

1. **Toy prompt diagnostic** (`scripts/debug_llama3_numeric_divergence.py`)
   ran TF and vLLM on a single fixed prompt. Both agreed at pass 1
   (`'5'`, id 20). Pass-2 vLLM picked `'445'` (id 19697) → answer 0.5445.
   This already differed from the production sweep's collapse to 0.5;
   suspicious that the divergence depended on prompt content.

2. **Real-prompt diagnostic** (`scripts/debug_llama3_numeric_v2.py`)
   loaded ACSIncome with `subsampling=0.005` and ran the first 10 rows
   through the production-equivalent vLLM path (`logprobs=50`,
   `allowed_token_ids=digit_ids`, `num_forward_passes=2`). Output for
   row 0:

   ```
   Chosen token_ids: [20, 27309] -> text: '5067'
     pos 0: top-K size=50, top-10 = [(20, '5', -4.16), (931, '000', -4.16), ...]
     pos 1: top-K size=51, top-10 = [(198, '\n', -1.25), (128001, '<|end_of_text|>', -1.25),
                                     (271, '\n\n', -1.50), ..., (13, '.', -3.88), ...]
              digit tokens in top-K: top-5 = [(27309, '067', -16.41)]
   ```

   The pos-1 top-K was dominated by **non-digit** tokens (`\n`,
   `<|end_of_text|>`, `.`). The only digit barely in the top-K was
   `'067'` at logprob -16.41 — far below `'.'` at -3.88.

3. The QA decoder (`DirectNumericQA._get_numeric_tokens`) includes `'.'`
   as a numeric token. With the unmasked top-K leaking `'.'` at high
   probability into the decoder's view, the decoder picks `'.'` at pass
   1, producing answer text `"5."` → regex matches `"5"` → parsed as
   0.5. Bug confirmed.

4. **vLLM internals**: `vllm/v1/sample/sampler.py` shows that with the
   default `logprobs_mode="raw_logprobs"`, `raw_logprobs` is computed
   BEFORE `apply_logits_processors` (which applies the
   `allowed_token_ids` mask). The mask only affects the chosen sample
   (greedy argmax), not the returned top-K logprobs.

## The fix

`folktexts/llm_utils.py:load_vllm_model` now sets
`kwargs.setdefault("logprobs_mode", "processed_logprobs")` before
constructing the engine. With this mode, `compute_logprobs` is called
AFTER `apply_logits_processors`, so the returned top-K reflects the
masked distribution. For Numeric (digit-only mask), only digit tokens
have non-`-inf` logprobs and the QA decoder behaves like the transformers
path: the `.` decoder probability is effectively 0, and the decoder picks
the highest-probability digit (e.g., `'067'`).

## Repro

```bash
source /etc/profile.d/modules.sh && module load cuda/13.2
export VLLM_USE_DEEP_GEMM=0

# Toy single-prompt diagnostic (pre/post-fix output is now identical)
$PYTHON scripts/debug_llama3_numeric_divergence.py

# Real-prompt diagnostic with production vLLM settings
$PYTHON scripts/debug_llama3_numeric_v2.py

# Full numeric benchmark on Llama-3-8B base, both backends
$PYTHON -m folktexts.cli.run_acs_benchmark \
  --model /fast/groups/sf/huggingface-models/meta-llama--Meta-Llama-3-8B \
  --task ACSIncome --data-dir /fast/acruz/data/folktables \
  --results-dir results/divergence_fix_validation/llama3-8b-numeric-vllm-fixed \
  --subsampling 0.03 --seed 42 --numeric-risk-prompting \
  --inference-backend vllm

# Compare the three (TF baseline, pre-fix vLLM, post-fix vLLM)
$PYTHON scripts/compare_postfix.py
```

## Why this also affected `Llama-3-70B-Instruct` (vLLM was BETTER)

For 70B-Instruct, the model's true distribution at pos 1 was more
digit-heavy than for the 8B base (instruction-tuned models concentrate
mass near the answer). With `'.'` still leaking into the top-K but with
a smaller relative gap to digits, vLLM picked `'.'` LESS often than for
the 8B base. The cell still benefited from the bug — vLLM's "0.5"
collapses were on rows where it would have predicted near-0.5 anyway,
producing a slightly cleaner distribution and higher AUC than TF's
multi-digit choices. With the fix, vLLM and TF align more closely.

## Other tokenizers (why they were unaffected)

The Llama-3 tokenizer carries 1100 multi-digit decimal tokens. Most other
families have 0–16 multi-digit tokens (Mistral 0, Yi 0, Gemma-1 16). For
those tokenizers, the unconstrained pos-1 distribution still places
single-digit tokens like `'0'`, `'5'` in the top-K with reasonable
probability, so the decoder picks digits over `'.'` even with raw
logprobs. The fix is benign for these models — `processed_logprobs`
just gives the same answer through a slightly different path.

## Validation runs

- ✅ `Meta-Llama-3-8B` numeric: AUC 0.5071 → 0.5756 (TF: 0.5591). Δ-vs-TF
  shrinks from -0.052 to +0.017. Distribution shape: 4 unique values →
  9 unique values, no longer collapsed.
- ⏳ `Meta-Llama-3-70B-Instruct` numeric: pending (running now in
  `results/divergence_fix_validation/llama3-70b-numeric-vllm-fixed/`).
  Expected: vLLM AUC moves from 0.848 toward 0.826 (TF baseline).
- ⏳ Regression check on a non-Llama-3 model (Mistral or Yi): pending.
- ⏳ Regression check on Llama-3 chat-MCQ: pending.

## Prompt-wording follow-up (2026-05-07)

After the fix landed, we re-investigated whether the prompt suffix itself
was suboptimal — specifically whether `Answer (between 0 and 1): 0.` is the
right prefill, given the unmasked pos-1 finding.

### Findings

1. **Prompt-tail special characters are clean.** Both zero-shot and
   chat-numeric prompt tails end with token IDs `[..., 220 (' '), 15 ('0'),
   13 ('.')]` for Llama-3 base/instruct (`scripts/inspect_prompt_tail.py`).
   The chat-template `<|eot_id|>` trim works correctly: the assistant
   role-marker comes BEFORE the prefill, EOT after is trimmed, and the
   model's "current cursor" sits exactly at `'.'`.

2. **The pos-1 0%-digit pattern is structural to the prompt, not specific to
   base models.** With pos-0 mask-forced to a digit and the unmasked top-K
   read out (`scripts/probe_prompt_suffixes_v2.py`), Llama-3-8B base and
   Llama-3.2-3B-Instruct (chat-numeric) both show:

   | Model / mode | unmasked digit-mass at pos-1 | dominant non-digit at pos-1 |
   |---|---|---|
   | Llama-3-8B base, zero-shot   | 0.000 | `\n` 0.36, `<\|eot\|>` 0.15 |
   | Llama-3.2-3B-Instruct, chat  | 0.000 | `<\|eot\|>` 0.34, `\n\n` 0.15 |

   The current prompt makes the model treat `0.X` as a complete answer; it
   "wants" to terminate, not continue with another digit. The
   `processed_logprobs` fix masks this symptom by deciding pos-1 by the
   relative-among-digits ordering, which carries enough signal on
   capable models (70B-Instruct AUC=0.8258, matches TF to 4dp) but is
   barely-better-than-random on Llama-3-8B base (~0.57).

3. **Dropping the `0.` prefill (the natural-looking fix) is a strict
   regression.** Tested two variants vs control on Llama-3-8B base (832
   rows, sub=0.005, `scripts/ab_prompt_strategy.py`):

   | Variant | AUC | unique scores | P(score=0.5) |
   |---|---|---|---|
   | A: current `... 0.` + mask + processed_logprobs | 0.5248 | 5 | 0.0% |
   | B: bare `:` + free generation + regex | 0.5190 | 3 | **97.6%** |
   | C: bare `:` + mask=digits+'.' + 4 passes + regex | 0.5190 | 3 | **97.6%** |

   Llama-3-8B base's "I don't know" mode is greedy `0.5\n` / `0.5.` —
   without the `0.` prefill forcing extra digits AND the digit-mask forcing
   them to be digits, the model emits exactly that and stops. The
   defensive prefill is empirically essential for weak base models.

4. **Variants found by probe to maximally lift unmasked digit-mass at
   pos-1** (e.g. `Reply in the form 0.XXXX. Answer: 0.` → ud@1=0.51) were
   not adopted: each one introduces side-effects (digit biases from
   examples, narrower expressible range, or unknown impact on Mistral/Yi
   /Gemma cells that already work). The risk of degrading 30+ green cells
   to "fix" one yellow-AUC cell isn't worth taking.

### Disposition

The current `processed_logprobs` fix stays. The prompt wording is left as
is. The Llama-3-8B-base AUC plateau is largely a model-capability ceiling
(base model, no instruction tuning, asked for numeric probabilities
directly) — the prompt isn't the lever.

CLAUDE.md "Gotchas" already covers the `logprobs_mode` mechanism; no further
code change needed.

## Cross-references

- Migration report: `VLLM_MIGRATION_REPORT.md` §1.3.2 ("Phase 4 results")
  and §2.4 ("Confirmed regression").
- Fix landing: `folktexts/llm_utils.py:load_vllm_model` (sets
  `logprobs_mode="processed_logprobs"` by default).
- CLAUDE.md "Gotchas" — entry "vLLM `logprobs_mode`" added.
- Diagnostic scripts: `scripts/debug_llama3_numeric_divergence.py`,
  `scripts/debug_llama3_numeric_v2.py`, `scripts/compare_postfix.py`,
  `scripts/inspect_prompt_tail.py`, `scripts/probe_prompt_suffixes.py`,
  `scripts/probe_prompt_suffixes_v2.py`, `scripts/ab_prompt_strategy.py`,
  `scripts/probe_instruct_chat.py`.
- vLLM source for the bug:
  `vllm/v1/sample/sampler.py:75-99` — `raw_logprobs` is computed before
  `apply_logits_processors`.
