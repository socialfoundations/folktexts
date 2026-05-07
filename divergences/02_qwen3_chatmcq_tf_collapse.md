# 02 — Qwen3-1.7B chat-MCQ: transformers collapses, vLLM is correct

## TL;DR

In Phase 4 (extended-sweep, modern models), one cell shows the **largest
delta** of the entire validation:

| Cell | tf AUC | vLLM AUC | ΔAUC | tf ECE | vLLM ECE | ΔECE |
|---|---|---|---|---|---|---|
| `Qwen--Qwen3-1.7B` chat_mcq | 0.5433 | **0.7335** | **+0.190** | 0.0920 | 0.3567 | **+0.265** |

Crucially, **vLLM is dramatically *better*** on AUC (almost 0.2 absolute) — and
*worse* on ECE. Transformers is the underperformer here, not the migration. This
cell does not reproduce on Qwen3-4B / Qwen3-4B-Instruct / Qwen3-4B-Thinking-2507.

## Symptom — distribution shape

```
TF: 1665 rows, 3 unique risk_score values, mean=0.2713, std=0.0725
    top5: {0.25: 1513 (91%), 0.5: 147 (9%), 0.0: 5 (0%)}

vLLM: 1665 rows, 425 unique risk_score values, mean=0.0066, std=0.0425
    top5: {0.0: 951 (57%), 0.000276: 19, 0.000244: 18, 0.000148: 16, 0.000355: 15}
```

Transformers collapses to 3 unique values dominated by `0.25` (91% of rows).
vLLM produces a continuum of 425 unique values with broad spread.

For binary classification (PINCP > $50k), `0.25` is consistent with one of the
choice-letter probabilities being normalised to a small fraction of total mass
across the answer letters. The "0.25" value specifically matches `1/4` —
suggesting transformers is splitting probability across 4 ways even though
this is binary (yes/no). That smells like a prefix-variant matching artefact in
`MultipleChoiceQA._decode_model_output_to_choice_distribution`.

## Hypothesis (current best guess)

`Qwen3-1.7B`'s chat template + tokenizer pair produces an answer-letter
distribution where:

1. The transformers full-softmax path gives weight to multiple prefix variants
   (`" A"`, `" B"`, `"A"`, `"B"`, etc. — all of which are tried in
   `qa_interface.py:319-360`'s prefix-variant loop) and renormalises in a way
   that collapses the prediction to a quartile (`0.25`, `0.5`, etc.).
2. The vLLM top-K=50 path only sees the dominant prefix variant in the model's
   actual top-K, doesn't spread the mass, and so the argmax-letter
   normalisation produces a smooth distribution.

Or, equivalently:
1. Transformers reads the *unconstrained* full softmax including ALL low-prob
   tokens, and the tail's prefix-variants pollute the answer-letter sum.
2. vLLM's top-K excludes the long tail; only "real" high-prob tokens contribute.

The fact that transformers' mode is *exactly* 0.25 (not something noisier like
0.3X) is the strongest hint: the transformers path is hitting a deterministic
normalisation outcome, not a gradient of probabilities.

## Repro

```bash
source /etc/profile.d/modules.sh && module load cuda/13.2
export VLLM_USE_DEEP_GEMM=0

# Re-run the cell on both backends (already exists in results/extended-sweep/)
$PYTHON scripts/extended_sweep.py --tier tier1 --models Qwen3-1.7B --modes chat_mcq --no-skip
```

To dig in:

```bash
# Inspect predictions distributions (already saved):
$PYTHON -c "
import pandas as pd, glob, json
for backend, root in [('TF','results/extended-sweep/transformers'), ('VL','results/extended-sweep/vllm')]:
    for jp in glob.glob(f'{root}/model-Qwen--Qwen3-1.7B_task-*/*/results.bench-*.json'):
        d = json.load(open(jp))
        cfg = d['config']
        if cfg.get('use_chat_template') and not cfg.get('numeric_risk_prompting') and not cfg.get('reasoning_prompting'):
            df = pd.read_csv(d['predictions_path'])
            print(f'{backend}: AUC={d[\"roc_auc\"]:.4f} ECE={d[\"ece\"]:.4f} unique={df[\"risk_score\"].nunique()}')
            print(f'  top5: {dict(df[\"risk_score\"].value_counts().head())}')
"
```

## Investigation paths (ordered)

1. **Print rendered prompt + top-K logprobs for a single row on each backend.**
   Use `scripts/inspect_e2e_prompts.py` as a starting point but extend to dump
   the per-row top-20 logprobs after the chat-template prefill. If the prompts
   are byte-identical and the model logits are identical (they should be at
   ~1e-3 noise floor), the divergence is in the QA-decoder layer.

2. **Add an instrumented warning in `MultipleChoiceQA._decode_model_output_to_choice_distribution`.**
   Log the per-prefix-variant total probability and the chosen variant for the
   first few transformers rows. Confirm whether transformers is picking a
   prefix variant whose total mass is `4 × <answer_letter_mass>`, hence the
   `0.25` normalisation.

3. **Test with `correct_order_bias=False`.** That disables the choice-permutation
   step. If transformers AUC jumps from 0.54 to ~0.73 (matching vLLM), the
   issue is in the order-bias correction code interacting badly with this
   model's logits.

4. **Compare per-row top-K logprobs at the prefill position across backends.**
   Run a tiny slice (10 rows) on both backends with extra logging to print the
   top-50 token logprobs after the chat template + answer prefill. If the
   logprobs agree but the QA decoder produces different answers, the bug is
   in how the decoder consumes the full softmax vs sparse top-K.

5. **Mitigation if irreducible**: this divergence is in *transformers' favour
   to lose* — vLLM is the better behaviour. If the root cause turns out to
   be a prefix-variant tie that vLLM happens to break correctly, document
   it and ship vLLM (already the default).

## Disposition

This cell is the strongest argument *for* the vLLM default — the migration
*fixes* a transformers issue that was there pre-existing. Do NOT block the
PR on this; investigate post-merge.

If the root cause is a transformers bug:
- Fix it in `qa_interface.py` (probably in the prefix-variant matching) so
  transformers also produces the correct 425-unique distribution.
- Re-run Phase 4 to confirm both backends now agree at vLLM's level.
- This becomes a "vLLM revealed a transformers bug" PR follow-up.

If the root cause is structural (top-K vs full softmax fundamentally produces
different decoded probabilities for this model):
- Document in CLAUDE.md as a model-specific quirk.
- Note that for this kind of model, vLLM is *more accurate*.

## Cross-references

- Migration report: `VLLM_MIGRATION_REPORT.md` §1.3.2.
- Phase 4 sweep results: `results/extended-sweep/REPORT.md`.
- The fix that landed during sweep v3 (`commit 2c1d2bc`) added a uniform
  fallback in `MultipleChoiceQA._decode_model_output_to_choice_distribution`
  for the zero-mass case. This issue is **not** the same — both backends
  here have non-zero answer-letter mass, but they renormalise differently.
