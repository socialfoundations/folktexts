# Configuring prompts

## Overview

You configure every prompt through two frozen dataclasses, built once and passed
down the call stack unchanged (they replace the older approach of threading
individual keyword arguments through every call):

- **`PromptConfig`** — how one row is rendered: value mapping, ordering, the
  label↔value connector, the final layout, an optional custom prefix/suffix, and
  the system prompt.
- **`FewShotConfig`** — whether and how in-context examples are prepended.

Every prompt is then composed of three independently-built parts:

```
[PREFIX]  task description / system context   (constant across rows)
[INFO]    serialized feature-value pairs       (row-specific)
[SUFFIX]  question text + answer prefill        (constant)
```

The *answer prefill* is the short lead-in the prompt ends on, so the model's next
token is the answer we score; its exact text depends on the question type
(multiple-choice ends on `Answer:`, numeric on `Answer (between 0 and 1): 0.`,
chain-of-thought generates free-form). In chat mode it becomes the assistant's
opening turn instead.

Both configs are hashable, so each distinct configuration gets its own
results-file name and runs never silently overwrite one another. The defaults
reproduce the original paper's prompts exactly; read on only to change how
prompts are rendered. The command-line equivalents are summarized in the
{doc}`README <readme>`.

## Question modes

A run asks the model one of three kinds of question. The mode is a single choice
that determines what the model is asked to produce and how that output becomes a
probability:

| Mode | Activate with | What the model does |
|:---|:---|:---|
| **Multiple-choice** (default) | *(nothing — it is the default)* | Answers a multiple-choice question; we score the answer-letter tokens and read the probability off them. Order-bias correction is on by default (`correct_order_bias`). |
| **Numeric** | `numeric_risk_prompting=True` · `--numeric-risk-prompting` | Reports the probability directly — the prompt ends on `Answer (between 0 and 1): 0.` and we read the digit tokens. |
| **Chain-of-thought** | `cot_prompting=True` · `--cot-prompting` | Generates free-form reasoning and ends with a `Probability: X%` line, recovered by regex. Works on any model, with or without a chat template. |

`enable_thinking` / `--enable-thinking` is a sub-option of chain-of-thought: it
turns on a tokenizer's native thinking mode (e.g. Qwen3) via
`apply_chat_template(..., enable_thinking=True)`, and the resulting
`<think>…</think>` block is stripped before extraction. It only applies when CoT
is on — setting it alone implicitly enables CoT (and warns). If both
`numeric_risk_prompting` and `cot_prompting` are set, chain-of-thought wins.

The mode is separate from *how the prompt is delivered* — zero-shot (the
default), few-shot (`FewShotConfig` / `--few-shot`), or chat-template formatting
(`use_chat_template` / `--use-chat-template`). The allowed pairings:

| | zero-shot | few-shot | chat-template |
|:---|:---:|:---:|:---:|
| **Multiple-choice** | ✓ | ✓ | ✓ |
| **Numeric** | ✓ | ✓ | ✓ |
| **Chain-of-thought** | ✓ | – | ✗ |

✓ supported · ✗ raises `ValueError` · – not a supported combination. Two pairings
are rejected at config time: **few-shot + chat-template**, and **chain-of-thought
+ chat-template** (CoT already applies the chat template internally, so an outer
one would double-wrap the prompt). Few-shot and chat-template are themselves
mutually exclusive — a run uses exactly one delivery path. Chain-of-thought is a
standalone path: it runs zero-shot and is not combined with few-shot.

## The variation pipeline

The `[INFO]` block is produced by a pipeline of `Vary*` stages whose order is
enforced by their return types — each stage's output type is the next stage's
input type, so they compose in exactly one order:

```
VaryValueMap → VaryOrder → VaryConnector → VaryFormat
(list→list)    (list→list)  (list→list)    (list→str)
```

`VaryFormat` collapses the feature list into a single string, which is why no
per-item stage can run after it. You don't instantiate these stages yourself —
set the keys in the last column below (via `--variation` or
`PromptConfig.from_dict`).

| INFO-pipeline stage | Controls | `--variation` key |
|:---|:---|:---|
| `VaryValueMap` | How raw column values render as text; `low` granularity coarsens ACS values into broader bins (age ranges, grouped occupations). | `granularity` |
| `VaryOrder` | Feature ordering (named columns first, the rest appended). | `order` |
| `VaryConnector` | The label↔value separator (`is:`, `is`, `=`, `:`, …). | `connector` |
| `VaryFormat` | Final layout (`textbullet`, `bullet`, `comma`, `text`). | `format` |

The `[PREFIX]` and `[SUFFIX]` are built separately: `VaryPrefix` and `VarySuffix`
each return their `str` directly, and `VarySystemPrompt` holds the optional
system-role string for the chat path.

| Prompt part | Controls | Key |
|:---|:---|:---|
| `VaryPrefix` | Task description + optional custom prefix. | `custom_prompt_prefix` |
| `VarySuffix` | Question text / answer prefill. | `custom_prompt_suffix`, `show_question` |
| `VarySystemPrompt` | Optional system-role string (chat path). | `system_prompt=` / `--system-prompt` (not a `--variation` key) |

## `PromptConfig`

`PromptConfig` holds one instance of each `Vary*` stage — one each for the
prefix, suffix, and system prompt, plus the four-stage pipeline for the feature
block. Build one from a dictionary of overrides whose keys are the seven
`--variation` keys from the tables above (`format`, `connector`, `granularity`,
`order`, `custom_prompt_prefix`, `custom_prompt_suffix`, `show_question`),
validated against `DEFAULT_PROMPT_STYLE` — an unknown key raises `ValueError`.
The `task` argument must be a `TaskMetadata` object; resolve a task name with
`TaskMetadata.get_task`:

```py
from folktexts import TaskMetadata
from folktexts.prompting import PromptConfig

task = TaskMetadata.get_task("ACSIncome")
prompt_config = PromptConfig.from_dict(
    {
        "format": "bullet",
        "connector": "=",
        "order": "AGEP,SCHL,COW",
        "custom_prompt_prefix": "Consider the following person.",
    },
    task=task,
)
```

Pass it straight to any classifier:

```py
from folktexts.classifier import VLLMClassifier

clf = VLLMClassifier(
    llm=llm, tokenizer=tokenizer, task="ACSIncome",
    prompt_config=prompt_config,
)
```

### The `PROMPT_DEFAULT` sentinel

`system_prompt` (and `chat_prompt`) have three modes: omit the argument for the
built-in default, pass `None` to remove the role entirely (needed for
Gemma-style templates that reject a system turn), or pass your own string. The
"built-in default" mode is spelled with the public sentinel `PROMPT_DEFAULT` —
which is **distinct from `None`**:

```py
from folktexts.prompting import PROMPT_DEFAULT, PromptConfig

PromptConfig.from_dict({}, task=task)                       # default system prompt
PromptConfig.from_dict({}, task=task, system_prompt=None)   # no system role at all
PromptConfig.from_dict({}, task=task, system_prompt="...")  # custom system prompt
```

These defaults are `ClassVar`s on the `QAInterface` hierarchy: multiple-choice
questions use the base `QAInterface` defaults, `DirectNumericQA` overrides them
with numeric-specific prompts, and `ChainOfThoughtQA` sets them to `None`
(free-form generation). The question type therefore supplies the right default,
which is why there is no longer a `numeric` flag to pass — pick the mode as
described under **Question modes** above.

## `FewShotConfig`

Few-shot prompting is configured with a single frozen dataclass:

```py
from folktexts.prompting import FewShotConfig
from folktexts.benchmark import Benchmark

bench = Benchmark.make_acs_benchmark(
    "ACSIncome", model=llm, tokenizer=tokenizer, data_dir="~/data",
    few_shot_config=FewShotConfig(
        n_shots=4,
        compose="balanced",        # "random" | "balanced" | per-class counts in label order, e.g. (2, 2) = 2 of class 0 + 2 of class 1
        example_order=(3, 2, 1, 0),  # optional permutation of the example indices
        reuse_examples=True,         # reuse the same examples for every row
        show_question_in_examples=True,
    ),
)
```

Few-shot prompting cannot be combined with the chat-template path
(`use_chat_template=True`) — that combination raises `ValueError`.

## Migrating from the flat-keyword API

Earlier versions configured prompts through scattered keyword arguments. Those
have been consolidated into `PromptConfig` / `FewShotConfig`. Passing a removed
keyword to a constructor or `encode_row_prompt*` now raises `TypeError` instead
of being silently ignored. Saved benchmark configs from before the change still
load: `BenchmarkConfig.load_from_disk` translates the legacy few-shot keys and
ignores any other unknown keys with a warning.

| Old | New |
|:---|:---|
| `custom_prompt_prefix="..."` (classifier / `encode_row_prompt*`) | `prompt_config=PromptConfig.from_dict({"custom_prompt_prefix": "..."}, task)` or CLI `--variation custom_prompt_prefix=...` |
| `add_task_description=False` | now an argument to `PromptConfig.from_dict(...)` |
| `few_shot=N`, `reuse_few_shot_examples=...`, `balance_few_shot_examples=...` (`BenchmarkConfig`) | `few_shot_config=FewShotConfig(n_shots=N, reuse_examples=..., compose="balanced")` |
| `class_balancing=True` (`sample_n_train_examples` / `encode_row_prompt_few_shot`) | `compose="balanced"` / CLI `--compose-few-shot-examples balanced` |
| CLI `--balance-few-shot-examples` | CLI `--compose-few-shot-examples balanced` |
| `numeric=True` (`encode_row_prompt_chat` / `resolve_chat_defaults`) | removed — the default is derived from the `QAInterface` subclass (`DirectNumericQA`) |
| `encode_row_prompt(row, task, question_obj)` (positional question) | `question=` is now keyword-only |
| `system_prompt=None` / `chat_prompt=None` to *mean* "default" | `PROMPT_DEFAULT` means "default"; `None` now means "explicitly disable" |

The top-level public API (`Benchmark`, `BenchmarkConfig`, the classifiers, the
`QAInterface` subclasses, `TaskMetadata`, `ACSDataset`) is unchanged.
