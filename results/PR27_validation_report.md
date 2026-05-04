# PR #27 — End-to-end validation report

## TL;DR

- **Unit tests (18/18) pass** on the chat-template prompting module under transformers 5.7.0 + Llama-3.2-1B-Instruct.
- **Pre-existing zero-shot path is unchanged** — baseline run on Llama-3.2-1B-Instruct produces AUC 0.830 with no spurious warnings.
- **New chat-template path works** — same model with `--use-chat-template` reaches AUC 0.853; numeric variant runs and respects the documented `NUMERIC_CHAT_PROMPT` [0, 1) cap (no prediction ≥ 1.0).
- **Gemma-style fallback branches both fire correctly** — INFO log when no user system prompt was supplied, WARNING log when one was. Benchmark completes in both cases.
- **CLI ergonomics warning fires** when `--system-prompt` / `--chat-prompt` are passed without `--use-chat-template`; run falls through to zero-shot as expected.
- **Side-finding (not introduced by PR #27):** real Gemma-3-1b-it crashes on both baseline and chat-template paths with `IndexError: boolean index ... 262144 vs 262145` at `folktexts/llm_utils.py:125` — `len(tokenizer.vocab) ≠ model.config.vocab_size` for Gemma-3. Full root cause + suggested fix in [§ Pre-existing Gemma vocab-mismatch bug](#1-pre-existing-gemma-vocab-mismatch-bug-step-5a5b-failure) below.
- **Compatibility note:** `transformers 5.7.0` is a major-version jump from the PR's `>=4.42.4` floor; everything still works, with one harmless deprecation (`torch_dtype` → `dtype` in `AutoModelForCausalLM.from_pretrained`).

## Environment

| Component | Version / location |
|---|---|
| Conda env | `/home/acruz/miniconda3/envs/folktexts` (Python 3.10.20) |
| torch | 2.5.1+cu121 |
| transformers | 5.7.0 |
| GPU | NVIDIA A100-SXM4-80GB |
| CUDA driver | 545.23.08 (CUDA 12.3) |
| Llama snapshot | `/fast/groups/sf/huggingface-models/meta-llama--Llama-3.2-1B-Instruct` |
| Gemma snapshot | `/fast/groups/sf/huggingface-models/google--gemma-3-1b-it` |
| ACS data cache | `/fast/acruz/data/folktables/` (pre-populated) |

All runs used `--subsampling 0.001` (≈166 samples from the ACS test split) and `--batch-size 16`. No HuggingFace Hub access was used at any point.

## Results matrix

| # | Phase | Status | Wall-clock | Key signal |
|---|---|:---:|---|---|
| 1 | Unit tests (`pytest tests/test_prompting_chat.py`) | ✅ | 17.9 s | 18/18 pass |
| 2 | Baseline zero-shot (Llama-3.2-1B-Instruct) | ✅ | 3m 33 s | AUC 0.830, 166 samples, "Using zero-shot prompting." |
| 3 | Chat-template MC (same model, `--use-chat-template`) | ✅ | 1m 56 s | AUC 0.853, "Using chat template prompting." |
| 4 | Chat-template numeric (`--numeric-risk-prompting`) | ✅ | 1m 46 s | AUC 0.548; predictions in [0.0, ~0.50]; zero values ≥ 1.0 |
| 5a | Real Gemma-3-1b-it baseline | ❌ | 1m 26 s | Pre-existing vocab-mismatch bug — see [details below](#1-pre-existing-gemma-vocab-mismatch-bug-step-5a5b-failure) |
| 5b | Real Gemma-3-1b-it chat template | ❌ | 1m 30 s | Same pre-existing bug (reproduces on **both** paths → not from PR #27) |
| 5c | Forced-rejection branch (Llama + Gemma-like template, no user sys-prompt) | ✅ | ~1 min | `INFO: Tokenizer's chat template rejects the \`system\` role; running without a system prompt.` — AUC 0.836 |
| 5d | Forced-rejection branch (with `system_prompt="..."`) | ✅ | ~1 min | `WARNING: ... user-supplied \`system_prompt\` will be discarded. Consider folding the instruction into \`custom_prompt_prefix\` or the user message instead.` — AUC 0.836 |
| 6 | CLI: `--system-prompt`/`--chat-prompt` without `--use-chat-template` | ✅ | 1m 52 s | `WARNING: \`system_prompt\` / \`chat_prompt\` were provided but \`use_chat_template=False\`; ... will be ignored.` |

## Answers to the three explicit asks

1. **Does the new chat-template path work?** Yes — Step 3 produced sensible metrics on a real instruct model and the chat-template log fires; the numeric variant (Step 4) honors the documented [0, 1) probability cap.
2. **Does the pre-existing zero-shot path still work?** Yes — Step 2 baseline is unaffected (no warnings, correct log, finite metrics).
3. **Do Gemma-specific branches behave correctly?** Yes — both fallback variants (silent INFO without user prompt; loud WARNING with one) fire as designed and the benchmark completes through the rejection path. The real-Gemma failure is a pre-existing folktexts/transformers compatibility bug independent of PR #27 (see below).

## Side-findings out of scope for PR #27

### 1. Pre-existing Gemma vocab-mismatch bug (Step 5a/5b failure)

**Symptom.** Any folktexts run on `google/gemma-3-1b-it` (and likely other Gemma-3 variants — the cause is in the tokenizer, not the chat template) crashes during the very first scoring forward pass:

```
File "/lustre/home/acruz/folktexts/folktexts/classifier/transformers_classifier.py", line 129,
  in _query_prompt_risk_estimates_batch
    last_token_probs_batch = query_model_batch_multiple_passes(...)
File "/lustre/home/acruz/folktexts/folktexts/llm_utils.py", line 125,
  in query_model_batch_multiple_passes
    current_probs[:, ~allowed_tokens_filter] = 0
IndexError: boolean index did not match indexed array along axis 1;
  size of axis is 262144 but size of corresponding boolean axis is 262145
```

The same crash reproduces on **both** the new `--use-chat-template` path (Step 5b) and the pre-existing zero-shot path (Step 5a) — so it is not introduced by PR #27 and would block any Gemma-3 benchmark on `main` today.

**Root cause.** In `folktexts/llm_utils.py:104` and `:111`:

```python
allowed_tokens_filter = np.ones(len(tokenizer.vocab), dtype=bool)
...
allowed_tokens_filter = np.zeros(len(tokenizer.vocab), dtype=bool)
```

The filter is sized using `len(tokenizer.vocab)`, which calls `tokenizer.get_vocab()` and **counts added/special tokens** alongside the base vocabulary. The model's logits, however, are sized to `model.config.vocab_size`, which equals `tokenizer.vocab_size` (the *base* size). For Gemma-3-1b-it I observed:

| Source | Value |
|---|---|
| `len(tok.vocab)` (i.e. `len(get_vocab())`) | **262 145** |
| `tok.vocab_size` (base vocab property) | **262 144** |
| `model.config.vocab_size` (logits dim) | **262 144** |

The off-by-one comes from a single added/special token that lives in `get_vocab()` but is not part of the model's output head. When `current_probs` (shape `[batch, 262144]`) is then masked with `~allowed_tokens_filter` (shape `[262145]`), numpy raises the `IndexError` above. Llama-3.2 happens to have `len(vocab) == vocab_size` so this path silently works there; Gemma triggers it.

The same pattern appears in the trailing assertion at `folktexts/llm_utils.py:143`:

```python
assert last_token_probs_array.shape == (len(text_inputs), n_passes, len(tokenizer.vocab))
```

— which would fail with an off-by-one `AssertionError` for Gemma even if the earlier mask were fixed.

**Suggested fix (one line, plus the assertion).** Size against the model's logits dim, not the tokenizer's `get_vocab()`:

```python
# llm_utils.py:104 area — defer sizing until we know the model vocab
vocab_dim = current_probs.shape[-1]   # set after the first query_model_batch call
allowed_tokens_filter = np.ones(vocab_dim, dtype=bool)
if digits_only:
    allowed_token_ids = np.array(
        [tok_id for token, tok_id in tokenizer.get_vocab().items()
         if token.isdecimal() and tok_id < vocab_dim]
    )
    allowed_tokens_filter = np.zeros(vocab_dim, dtype=bool)
    allowed_tokens_filter[allowed_token_ids] = True
```

and at line 143:

```python
assert last_token_probs_array.shape == (len(text_inputs), n_passes, vocab_dim)
```

A simpler stop-gap that *also* works for Gemma is `tokenizer.vocab_size` instead of `len(tokenizer.vocab)`, because by HF convention `tokenizer.vocab_size == model.config.vocab_size`. The `current_probs.shape[-1]` form is more defensive against future tokenizer/model mismatches.

**Scope.** This is a separate bug, unrelated to PR #27. It should be fixed in its own commit/PR (and ideally followed up with a CI test that exercises a Gemma tokenizer end-to-end so the regression doesn't return).

### 2. transformers 5.x deprecation

`AutoModelForCausalLM.from_pretrained(..., torch_dtype="bfloat16")` emits:

```
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
```

Not blocking, but worth a one-line update wherever folktexts (and the reproducer script `scripts/test_gemma_branch.py`) loads models, once the transformers floor is bumped.

## Reproducer artifacts

All under `/lustre/home/acruz/folktexts/results/`:

- `pytest.log` — Step 1 output
- `baseline/run.log` + JSON + PDFs — Step 2
- `chat_mc/run.log` + JSON + PDFs — Step 3
- `chat_numeric/run.log` + JSON + PDFs — Step 4
- `gemma/run.log`, `gemma_baseline/` — Step 5a/b (failed runs, pre-existing bug)
- `gemma_branch_no_sysprompt/`, `gemma_branch_with_sysprompt/` — Step 5c/d outputs
- `gemma_branch.log` — combined log from `scripts/test_gemma_branch.py`
- `warning_test/run.log` + JSON + PDFs — Step 6

Reproducer script: `/lustre/home/acruz/folktexts/scripts/test_gemma_branch.py` (loads Llama with a Gemma-style chat-template override to force the rejection branch).
