# PR #26 verification — post-rebase

This document is the verification checklist for the squash-rebase of the
`reasoning-models` branch onto `main`. It is intentionally portable so it can
travel to a GPU compute node for Phase B.

## What this branch does after the rebase

A single commit on top of `main` adds the `ReasoningQA` interface for
chain-of-thought / thinking models, integrated with main's chat-template
work (PR #27) and vocab-mismatch + prefill-duplication fixes (PR #29).

Key surface added:
- `ReasoningQA` (in `folktexts/qa_interface.py`) — free-text generation +
  regex extraction of "Probability: X%". Accepts the `with_answer_prefill`
  kwarg added by PR #29 for LSP, but ignores it (no prefill to strip).
- `_apply_chat_template_batch`, `_postprocess_generated_text`,
  `generate_text_batch` (in `folktexts/llm_utils.py`) — generation pipeline
  with `</think>` block stripping when `enable_thinking=True`.
- `BenchmarkConfig.reasoning_prompting`, `BenchmarkConfig.enable_thinking`.
- CLI flags `--reasoning-prompting`, `--enable-thinking`.
- `Benchmark.make_benchmark` raises `ValueError` if `use_chat_template=True`
  is combined with reasoning, since the reasoning path applies the chat
  template internally and would otherwise double-wrap.
- `tests/test_reasoning_qa.py` covering regex extraction and the LSP-fix.

## Phase A — Local (run before pushing the branch)

1. Clean editable install:
   ```
   pip install -e ".[apis,tests]"
   ```
2. Full test suite — every test passes, including main's three new files
   (`test_prompting_chat.py`, `test_qa_interface.py`, `test_llm_utils.py`)
   and the new `test_reasoning_qa.py`:
   ```
   pytest tests/ -v
   ```
3. Lint and type:
   ```
   tox -e lint
   tox -e type
   ```
4. CLI sanity:
   ```
   python -m folktexts.cli.run_acs_benchmark --help
   ```
   Confirm all five new flags are listed: `--use-chat-template`,
   `--chat-prompt`, `--system-prompt`, `--reasoning-prompting`,
   `--enable-thinking`.
5. Mutual-exclusion check (should raise `ValueError`):
   ```
   python -m folktexts.cli.run_acs_benchmark \
     --model models/meta-llama--Llama-3.2-1B-Instruct \
     --task ACSIncome --results-dir results --data-dir data \
     --subsampling 0.001 --use-chat-template --reasoning-prompting
   ```
   Expected: process exits with the message about reasoning + chat-template
   double-wrapping.
6. Tiny-model smokes against the local `Llama-3.2-1B-Instruct` checkpoint
   (already in `results/model-meta-llama--Llama-3.2-1B-Instruct_task-ACSIncome/`).
   For each, run with `--subsampling 0.001 --batch-size 4`. Pass criteria:
   process exits 0, writes a results dir, AUC value present in `results.json`.
   - default multiple-choice: no extra flags
   - `--numeric-risk-prompting`
   - `--use-chat-template`
   - `--use-chat-template --numeric-risk-prompting` (PR #29 chat-numeric path)
   - `--reasoning-prompting` (probability extraction may fall back to 0.5
     for many samples on a 1B model — that is expected; we only require no
     crash and a non-NaN AUC)
7. First-N generation logging works:
   ```
   FOLKTEXTS_LOG_GENERATIONS_FIRST_N=3 \
   python -m folktexts.cli.run_acs_benchmark \
     --model models/meta-llama--Llama-3.2-1B-Instruct \
     --task ACSIncome --results-dir results --data-dir data \
     --subsampling 0.001 --reasoning-prompting --logger-level INFO
   ```
   Expected: at least three prompt/generation pairs printed at INFO level.
8. Reasoning integration script:
   ```
   python scripts/test_reasoning_qa.py
   ```

## Phase B — GPU node (real reasoning model, gating the merge)

9. Pull a thinking-capable model:
   ```
   download_models --model "Qwen/Qwen3-4B" --save-dir models
   ```
10. End-to-end with thinking mode enabled:
    ```
    FOLKTEXTS_LOG_GENERATIONS_FIRST_N=5 \
    python -m folktexts.cli.run_acs_benchmark \
      --model models/Qwen--Qwen3-4B \
      --task ACSIncome --results-dir results --data-dir data \
      --subsampling 0.01 --batch-size 1 \
      --reasoning-prompting --enable-thinking \
      --logger-level INFO
    ```
    Verify in stdout / results:
    - generated text contains `<think>` ... `</think>` markers
    - `_postprocess_generated_text` strips the thinking block
      (visible only at `--logger-level DEBUG`)
    - the first 5 generations show real reasoning followed by
      `Probability: X%`
    - ≥ 80 % of samples produce a non-fallback probability
      (i.e. `extract_probability_from_text` returns non-None)
    - final AUC > 0.6 on the 1 % ACSIncome sample
11. Same model, no thinking, plain reasoning:
    ```
    python -m folktexts.cli.run_acs_benchmark \
      --model models/Qwen--Qwen3-4B \
      --task ACSIncome --results-dir results --data-dir data \
      --subsampling 0.01 --batch-size 1 --reasoning-prompting
    ```
    Confirm regex extraction still works without `<think>` blocks.
12. Sanity-check main's PR #29 fixes on the GPU node:
    - **Gemma vocab-mismatch**: download `google/gemma-2-2b-it`, run
      `--numeric-risk-prompting --use-chat-template`. Pre-PR #29 this would
      crash on `IndexError` in token lookup; expected: clean run.
    - **Mistral chat-numeric prefill**: download `mistralai/Mistral-7B-Instruct-v0.3`,
      run `--numeric-risk-prompting --use-chat-template`. Expected AUC ~0.808
      on ACSIncome (PR #29's reported number).
13. Cross-check, same model, same task: `--reasoning-prompting` vs default
    multiple-choice (no flags). AUCs should be in the same ballpark
    (within ~0.05) on a non-tiny sample. Big divergence would indicate
    something is wrong with the reasoning extraction.

## Phase C — Pre-merge gate

14. Read the squashed diff one more time:
    ```
    git diff main...reasoning-models -- ':!REBASE_VERIFICATION.md'
    ```
15. Force-push (history was rewritten):
    ```
    git push --force-with-lease origin reasoning-models
    ```
16. On GitHub, confirm PR #26 now shows `MERGEABLE` and CI is green.
17. Squash-merge PR #26 into `main`.

## Cleanup after merge

```
git branch -d reasoning-models-backup
git stash pop  # restore the pre-rebase notebooks/results
```

## If something goes wrong

The rebase is reversible while `reasoning-models-backup` exists locally:
```
git reset --hard reasoning-models-backup
```
This restores the original 11-commit branch tip exactly.
