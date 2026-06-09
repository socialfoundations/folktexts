# Updates

Release notes summarising user-visible changes between versions. Older changes
not yet listed here can be reconstructed from the git log.

## v0.6.0 — typed, composable prompt configuration

Replaces the scattered prompt-related keyword arguments with two frozen
configuration objects — `PromptConfig` (how a row is rendered) and
`FewShotConfig` (in-context examples) — built once and threaded through the call
stack. The defaults reproduce the original paper's prompts exactly, so a run
that does not touch these knobs is unchanged. See the
{doc}`prompt-configuration guide <configuring_prompts>` for the full reference.

### What's new

- **`PromptConfig`** and the `Vary*` pipeline stages. A prompt is built from a
  task prefix, a feature `[INFO]` block, and a question suffix; the `[INFO]`
  block runs through a typed pipeline (`VaryValueMap → VaryOrder →
  VaryConnector → VaryFormat`) whose order is fixed by the stages' return types.
- **CLI `--variation key=value …`** to change how the feature block renders.
  Keys: `format`, `connector`, `granularity`, `order`, `custom_prompt_prefix`,
  `custom_prompt_suffix`, `show_question`.
- **Low-granularity value maps** (`--variation granularity=low`): coarsen ACS
  feature values into broader bins (age ranges, grouped occupations).
- **`FewShotConfig`** consolidates the few-shot knobs, with new CLI flags
  `--compose-few-shot-examples` (`random` / `balanced` / per-class counts),
  `--example-order`, and `--few-shot-hide-question`.
- **`PROMPT_DEFAULT` sentinel** distinguishes "use the question type's default
  system/chat prompt" (`PROMPT_DEFAULT`) from "disable the role entirely"
  (`None`).

### Breaking changes (hard cut, no aliases)

Prompt configuration moved off scattered keyword arguments. Passing a removed
keyword to a constructor or `encode_row_prompt*` now raises `TypeError`.

| Before | After |
|---|---|
| `custom_prompt_prefix="..."` (classifier / `encode_row_prompt*`) | `prompt_config=PromptConfig.from_dict({"custom_prompt_prefix": "..."}, task)` or CLI `--variation custom_prompt_prefix=...` |
| `add_task_description=False` (`encode_row_prompt`) | `PromptConfig.from_dict(..., add_task_description=False)` |
| `few_shot=N`, `reuse_few_shot_examples=`, `balance_few_shot_examples=` | `few_shot_config=FewShotConfig(n_shots=N, reuse_examples=…, compose="balanced")` |
| `class_balancing=True` / CLI `--balance-few-shot-examples` | `compose="balanced"` / CLI `--compose-few-shot-examples balanced` |
| `numeric=True` (chat path) | removed — the question type derives it (`DirectNumericQA`) |
| `encode_row_prompt(row, task, question_obj)` (positional question) | `question=` is now keyword-only |
| `system_prompt=None` / `chat_prompt=None` meaning "default" | `PROMPT_DEFAULT` means "default"; `None` now disables the role |

Saved benchmark configs from before the change still load:
`BenchmarkConfig.load_from_disk` translates the legacy few-shot keys and ignores
any other unknown keys with a warning.

### Notes

- **Defaults unchanged on the token-scoring paths.** The zero-shot and few-shot
  text prompts are byte-identical to the previous release (v0.5.0), and the
  top-level public API (`Benchmark`, `BenchmarkConfig`, the classifiers, the
  `QAInterface` subclasses, `TaskMetadata`, `ACSDataset`) is the same.
- **Chat system prompt refined.** The default multiple-choice *chat* system
  prompt now ends with "Respond with a single answer choice." (previously it
  stopped at "…based on the information provided."). This affects only the
  chat-template / web-API path; the zero-shot and few-shot last-token scoring
  paths do not use a system prompt and are unaffected.
- **Stable results-file names.** `PromptConfig` / `FewShotConfig` are hashable
  and process-stable, so each distinct configuration writes to its own
  `results.bench-{hash}.json` — runs never silently overwrite one another.
- **Fix.** `WebAPILLMClassifier` no longer raises on multiple-choice / numeric
  questions when the system prompt is disabled, and now threads `--system-prompt`
  through the web-API chain-of-thought path.

## v0.5.0 — rename `reasoning_*` → `cot_*` / `chain_of_thought_*`

Two distinct concepts shared the word "reasoning" in the public API: the
free-form chain-of-thought prompt template and the HF chat-template
`enable_thinking` kwarg. They're orthogonal — `enable_thinking=True`
requires the CoT prompt path, but the CoT prompt path runs on any model
with no thinking-mode support — so we renamed the CoT side to drop the
ambiguity. `enable_thinking` is unchanged because that name matches the HF
kwarg and stays aligned with it.

### Renames (hard cut, no aliases)

| Before                                | After                                  |
|---------------------------------------|----------------------------------------|
| `ReasoningQA` class                   | `ChainOfThoughtQA`                     |
| `BenchmarkConfig.reasoning_prompting` | `BenchmarkConfig.cot_prompting`        |
| CLI flag `--reasoning-prompting`      | `--cot-prompting`                      |
| `TaskMetadata.reasoning_qa` / `use_reasoning_qa` | `TaskMetadata.cot_qa` / `use_cot_qa` |

The previous symbols raise `AttributeError` / `TypeError` rather than
warning + forwarding — update callsites in one commit.

### Migration

```py
# Before
from folktexts.qa_interface import ReasoningQA
config = BenchmarkConfig(reasoning_prompting=True, enable_thinking=True)

# After
from folktexts.qa_interface import ChainOfThoughtQA
config = BenchmarkConfig(cot_prompting=True, enable_thinking=True)
```

```bash
# Before
run_acs_benchmark --model <m> --task ACSIncome --data-dir <d> --reasoning-prompting
# After
run_acs_benchmark --model <m> --task ACSIncome --data-dir <d> --cot-prompting
```

### Notes

- **`enable_thinking` is unchanged** (dataclass field, CLI flag, and class
  attribute on `ChainOfThoughtQA`). It still requires `cot_prompting=True`
  and warns + auto-enables CoT mode if you forget.
- **`ChainOfThoughtQA.max_new_tokens=8000`** value is preserved.
- **Result JSON files** (`results.bench-*.json`) written before the rename
  carry `"config": {"reasoning_prompting": true, ...}`. They remain
  readable: sweep helpers (`scripts/cot_e2e_sweep.py`, `cot_sweep.py`,
  `audit_cot_failures.py`, `extended_sweep.py`, `multi_seed_stability.py`,
  `validate_pr26.py`) accept either key when scanning existing results.
- **Hash stability**: not preserved. `BenchmarkConfig.__hash__` uses
  `dataclasses.asdict(self)`, so the hash includes the field name. New
  runs write to fresh `results.bench-{hash}.json` paths; pre-rename
  cached paths stay readable but won't be short-circuited by a hash
  match.

## v0.4.0 — vLLM backend

`folktexts` v0.4.0 introduces local inference via [vLLM] alongside the existing
HuggingFace `transformers` backend, typically delivering a 5–30× throughput
improvement on GPU benchmarks while preserving the full score-extraction
contract (multiple-choice, direct-numeric, and chain-of-thought prompting).

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
  `max_model_len` from `--context-size + ChainOfThoughtQA.max_new_tokens + 256`
  when the user does not pass `--max-model-len` explicitly.
- **Optional install group `[vllm]`**: `pip install folktexts[vllm]` pulls in
  the vLLM wheel. The base install is unchanged for users on the transformers
  path.

### Architecture

- **Two classifiers, one decoder.** `VLLMClassifier`,
  `TransformersLLMClassifier`, and `WebAPILLMClassifier` all hand answers to
  the QA-decoder methods on `MultipleChoiceQA`, `DirectNumericQA`, and
  `ChainOfThoughtQA`. The new helper `decode_topk_logprobs_to_risk_estimate` in
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

**36/38 cells fall within the strict gates** `|ΔAUC| ≤ 0.015` and
`|ΔECE| ≤ 0.025`. The two remaining outliers are characterised:

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
