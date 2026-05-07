# 03 — Qwen3-Thinking with `enable_thinking=True` exceeds `max_new_tokens=5000` — **FIXED & VALIDATED**

## TL;DR (Status: FIXED & VALIDATED 2026-05-07)

**Fix landed:** `ReasoningQA.max_new_tokens` bumped from 5000 to 8000
(`folktexts/qa_interface.py:507`). The CLI's `max_model_len` heuristic
now derives from `ReasoningQA.max_new_tokens` symbolically so the two
stay in sync. CLAUDE.md gotcha added.

**End-to-end validation** (Qwen3-4B-Thinking-2507, ACSIncome, n=832,
subsampling=0.005):

| Metric | Pre-fix (5k) | Post-fix (8k) |
|---|---|---|
| AUC | 0.7369 | **0.7990** (+0.062) |
| ECE | 0.1539 | **0.1393** |
| Brier | 0.2221 | **0.1910** |
| Regex 0.5-fallback rate | **13.1%** (109/832) | **2.5%** (21/832) — 5× reduction |

Post-fix thinking-on AUC (0.7990) now slightly exceeds pre-fix
thinking-off AUC (0.7848): the model's CoT actually helps once it
can finish.

## Original symptom (kept for reference)


Qwen3-4B-Thinking-2507 with `enable_thinking=True` (chat template's thinking
mode on) emits a CoT trace that does NOT close the `</think>` block within
`max_new_tokens=5000` for ~13% of ACSIncome rows. The regex extractor falls
back to 0.5 on those rows, dragging AUC from `0.785` (thinking-off) to
`0.737` (thinking-on).

This is **NOT** a backend regression — both transformers and vLLM see the
same model and the same generation budget. It's a model+budget configuration
issue.

## Symptom

From Phase 5 reasoning sweep (vLLM only, subsampling=0.005):

| Cell | AUC | ECE | regex 0.5-rate |
|---|---|---|---|
| Qwen3-4B-Thinking-2507 reasoning **with** thinking | 0.737 | 0.154 | **13.1%** |
| Qwen3-4B-Thinking-2507 reasoning **without** thinking | 0.785 | 0.142 | 3.1% |
| Llama-3-8B-Instruct reasoning (no thinking template) | 0.785 | 0.169 | 0.2% |

Log warnings during the sweep:

```
WARNING:root:</think> marker not found in output (thinking mode was enabled).
Using full generated text (20021 chars).
```

20k chars ≈ 5k tokens — exactly at the `max_new_tokens` budget. So the model
hits the budget mid-thinking, never closes `</think>`, never reaches the
final answer, and the regex finds no probability to extract.

## Root cause

`max_new_tokens` for `ReasoningQA` is set in
`folktexts/qa_interface.py:504-509` (or thereabouts — search for the
`ReasoningQA` class). The default is `5000` tokens. For Qwen3-Thinking, that
budget is too tight: the model uses 5k+ tokens just for the thinking trace.

In the vLLM CLI, `max_model_len` is sized as
`context_size + 5000 + 256` for `ReasoningQA` runs (`run_acs_benchmark.py`),
which limits how much we can lift `max_new_tokens` without also lifting
`max_model_len`.

## Repro

```bash
source /etc/profile.d/modules.sh && module load cuda/13.2
export VLLM_USE_DEEP_GEMM=0

# Existing result already shows the issue:
$PYTHON scripts/reasoning_sweep.py --report-only

# To verify the </think> warning in real time:
FOLKTEXTS_LOG_GENERATIONS_FIRST_N=5 \
  $PYTHON scripts/reasoning_sweep.py --backends vllm --no-skip
# Inspect the printed generations — count how many contain "</think>".
```

## Fix (concrete, low-risk)

1. **Bump `ReasoningQA.max_new_tokens` from 5000 to 8000** in `qa_interface.py`.
   The model needs ~5-6k for thinking + ~50 for the final answer; 8k gives
   comfortable headroom.

2. **Adjust the CLI `max_model_len` heuristic** in `cli/run_acs_benchmark.py`:
   change `5000 + 256` to `8000 + 256` for ReasoningQA, OR make it follow
   `ReasoningQA.max_new_tokens` symbolically so the two stay in sync.

3. **Re-run the reasoning sweep on both backends** to confirm the regex
   fail-rate drops on the thinking-on cell. Expected outcome: thinking-on AUC
   ≈ 0.785 (matches thinking-off) once the model actually finishes its
   thinking block.

## Investigation paths

The fix above is straightforward. Things to check first to validate:

1. **Inspect actual generation lengths.** Set `FOLKTEXTS_LOG_GENERATIONS_FIRST_N=20`
   on a Qwen3-Thinking thinking-on run and read the generated outputs. Confirm
   `</think>` is absent and the text ends mid-thought.

2. **Run with `max_new_tokens=8000` on a single batch** to confirm the
   `</think>` does appear with the larger budget. If a 16-row batch at 8k
   tokens gives 0/16 "</think> not found" warnings, the fix works.

3. **Watch out for OOM** at higher `max_model_len` on the 30B-A3B-Thinking
   model (Tier 3). The KV cache scales linearly with `max_model_len`. If
   8k pushes a B200 over the edge for the 30B-Thinking model, the heuristic
   should clamp to whatever the GPU memory budget allows.

## Disposition

**Ship the fix in this PR or a quick follow-up.** Cost: 2 lines + a re-run on
the dedicated reasoning model. Benefit: thinking-on AUC restored to parity
with thinking-off, and a clearer signal for "does the migration handle
thinking-models correctly".

## Cross-references

- Migration report: `VLLM_MIGRATION_REPORT.md` §1.3.4 ("Phase 5 results").
- Reasoning sweep: `results/reasoning-sweep/REPORT.md`.
- Qwen3-Thinking on tier1 extended-sweep also produced
  `chat_numeric` = 0.785 (vLLM) which uses standard chat-template (no
  reasoning-prompting); that mode is unaffected.
- CLAUDE.md gotcha to add: "Qwen3-Thinking-2507 needs `max_new_tokens >= 8000`
  to reliably close `</think>`; default 5000 leaves 13% of rows unfinished."
