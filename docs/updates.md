# Updates

Release notes summarising user-visible changes between versions. Older changes
not yet listed here can be reconstructed from the git log.

## v0.4.0 — vLLM backend

`folktexts` v0.4.0 introduces local inference via [vLLM] alongside the existing
HuggingFace `transformers` backend, typically delivering a 5–30× throughput
improvement on GPU benchmarks while preserving the full score-extraction
contract (multiple-choice, direct-numeric, and reasoning prompting).

[vLLM]: https://docs.vllm.ai/

### What's new

- **`VLLMClassifier`**: a new top-K-logprobs classifier in
  `folktexts.classifier.vllm_classifier`, parallel to
  `TransformersLLMClassifier`. Both feed the same QA decoders, so result
  semantics are unchanged.
- **`load_vllm_model`** in `folktexts.llm_utils`: helper that initialises a
  vLLM `LLM` engine + tokenizer with sensible defaults for this benchmark
  (BF16, `gpu_memory_utilization=0.85`, `logprobs_mode="processed_logprobs"`).
- **CLI flag `--inference-backend {transformers,vllm}`**: selects the local
  backend. **Default is now `vllm`.** Pass `--inference-backend transformers`
  to fall back to the previous path; the transformers code is unchanged and
  remains a fully supported alternative.
- **vLLM-specific CLI flags**: `--gpu-memory-utilization`, `--max-model-len`,
  `--vllm-dtype`, `--tensor-parallel-size`. The CLI auto-derives a
  `max_model_len` from `--context-size + ReasoningQA.max_new_tokens + 256`
  when the user does not pass `--max-model-len` explicitly.
- **Optional install group `[vllm]`**: `pip install folktexts[vllm]` pulls in
  the vLLM wheel. The base install is unchanged for users on the transformers
  path.

### Architecture

- **Two classifiers, one decoder.** `VLLMClassifier`,
  `TransformersLLMClassifier`, and `WebAPILLMClassifier` all hand answers to
  the QA-decoder methods on `MultipleChoiceQA`, `DirectNumericQA`, and
  `ReasoningQA`. The new helper `decode_topk_logprobs_to_risk_estimate` in
  `folktexts.llm_utils` factors out the top-K decoding logic shared by vLLM
  and the WebAPI; the transformers path (which has full-vocab logits)
  bypasses this helper, as before.
- **Backend dispatch.** `Benchmark.make_*_benchmark(...)` accepts a
  `backend=` argument (`"transformers"`, `"vllm"`, `"webapi"`, or
  `None` for autodetect). When `None`, autodetect uses
  `str → webapi`, duck-typed `LLM-shaped → vllm`, else `transformers`.
- **`VLLMClassifier.__hash__`** includes a `"vllm"` tag so cached result
  paths (`results.bench-{hash}.json`) cannot collide with transformers runs
  of the same model. Predictions can drift by ~1e-3 across backends due to
  attention-kernel differences; mixing them in one CSV would be a silent
  mistake.
- **Numeric mode** uses vLLM's `allowed_token_ids` to restrict generation to
  digit tokens (mirroring the transformers `digits_only=True` mask).
  Multiple-choice mode runs unmasked; the QA decoder's prefix-variant
  matching handles renormalisation across answer letters.

### Two structural fixes shipped with the migration

While building the vLLM backend, two issues were surfaced and fixed in this
release. Both are bug fixes, not API changes.

**vLLM `logprobs_mode="processed_logprobs"`** — vLLM's default
`raw_logprobs` returns top-K logprobs computed *before* the
`allowed_token_ids` mask is applied. For `DirectNumericQA` on tokenizers
with multi-digit decimal tokens (Llama-3 has 1100), the unmasked top-K is
dominated by `'\n'`, `<|end_of_text|>`, and `'.'`. The numeric decoder
treats `'.'` as a numeric token and would pick it over the only-allowed
digit, producing answer text `"5."` → regex `"5"` → 0.5. The 8B base
checkpoint collapsed to 99% of rows at exactly 0.5. `load_vllm_model` now
defaults to `logprobs_mode="processed_logprobs"`, which returns top-K from
the post-mask distribution and restores the expected behaviour.

**`ReasoningQA.max_new_tokens` 5000 → 8000** — Qwen3-Thinking-2507 with
`enable_thinking=True` did not reliably close `</think>` within 5000
tokens (~13% of rows ran to budget mid-CoT and fell back to the regex
0.5 default). Bumping to 8000 reduces the regex 0.5-fallback rate from
13.1% to 2.5% and lifts thinking-on AUC on ACSIncome from 0.737 to 0.799.
The CLI `max_model_len` heuristic now derives from
`ReasoningQA.max_new_tokens` symbolically so the two stay in sync.

### Cluster runtime requirements (B200 / Hopper / vllm 0.20.1 wheel)

The vLLM 0.20.1 wheel is built against CUDA 13. On clusters where the
default toolkit is older, two environment steps are required for any
vLLM invocation:

```bash
source /etc/profile.d/modules.sh
module load cuda/13.2          # provides libcudart.so.13
export VLLM_USE_DEEP_GEMM=0    # skips an FP8 warmup that needs deep_gemm
                               # (not on PyPI); harmless on BF16 models
```

Without these, `import vllm._C` and engine init both crash on Hopper+
GPUs.

### Validation

The migration was validated across 38 cross-backend cells covering the
paper's Table 1 (8 models × 2-4 modes), a modern + thinking-model sweep
(`gemma-3-1b-it`, `Qwen3-1.7B`, `Qwen3-4B`, `Qwen3-4B-Instruct-2507`,
`Qwen3-4B-Thinking-2507`), and a chat-template extension on
`Mistral-7B-Instruct-v0.2` and `Yi-34B-Chat`. Multi-seed stability was
verified across 4 seeds × 2 backends on Llama-3-8B-Instruct and
Qwen3-Thinking-2507.

After the two structural fixes above, **36/38 cells fall within the
strict gates** `|ΔAUC| ≤ 0.015` and `|ΔECE| ≤ 0.025`. The two
remaining outliers are characterised:

- `Llama-3-8B` base × `numeric` (zero-shot): vLLM `+0.017` AUC, `−0.041`
  ECE — vLLM is slightly *better*. The model is essentially near-random
  on this prompt (TF AUC 0.559); the delta is within the kernel-noise
  band of a near-random model.
- `Qwen3-1.7B` × `chat-MCQ`: vLLM `+0.190` AUC, `+0.265` ECE — vLLM
  is *much* better. The transformers path collapses to 3 unique scores
  on this combination; vLLM produces 425 unique scores with broad
  spread. The bug is on the transformers side and does not reproduce on
  Qwen3-4B / Qwen3-4B-Instruct / Qwen3-4B-Thinking-2507.

Phase 7 robustness checks (1-row DataFrame, sequential model swap in
the same Python process, near- and over-cap inputs, tied-logit
cross-backend agreement, and OOM clean failure) all pass.

### Backwards compatibility

- The CLI accepts the same flags as before plus the new
  `--inference-backend` / `--gpu-memory-utilization` /
  `--max-model-len` / `--vllm-dtype` / `--tensor-parallel-size`. All new
  flags have safe defaults; existing scripts work unchanged on the
  vLLM backend, or on the transformers backend with
  `--inference-backend transformers`.
- `Benchmark.make_*_benchmark(...)` accepts an optional `backend=` kwarg.
  Existing callers that pass `model=` as a HuggingFace `PreTrainedModel`
  continue to be routed to `TransformersLLMClassifier`.
- Result CSVs from previous runs (transformers) are not invalidated; the
  new vLLM hash tag means vLLM runs save to a fresh path rather than
  overwriting transformers numbers.

### Migration notes

If you previously installed `folktexts` and want to use the new vLLM
backend:

```bash
pip install --upgrade 'folktexts[vllm]'
# or, from a checkout:
pip install -e .[vllm]
```

Then either accept the new default (`vllm`) or stay on transformers
explicitly:

```bash
run_acs_benchmark --model <path> --task ACSIncome --data-dir <path> \
    --inference-backend transformers
```
