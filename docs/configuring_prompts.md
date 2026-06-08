# Configuring prompts

`folktexts` builds every prompt from three independent parts:

```
[PREFIX]  task description / system context   (constant across rows)
[INFO]    serialized feature-value pairs       (row-specific)
[SUFFIX]  question text + answer prefill        (constant)
```

The **defaults reproduce the prompts used in the original paper exactly** — this
page is only needed if you want to *change* how prompts are rendered. The
command-line equivalents of everything here are summarized in the
{doc}`README <readme>`; this page documents the Python API and
how to migrate from the older flat-keyword interface.

## The variation pipeline

The `[INFO]` block is produced by a small, typed pipeline of frozen
dataclasses. Each stage has a fixed input/output type, which makes the order
unambiguous:

```
VaryValueMap → VaryOrder → VaryConnector → VaryFormat
(list→list)    (list→list)  (list→list)    (list→str)
```

| Stage | Controls | `--variation` key |
|:---|:---|:---|
| `VaryValueMap` | How raw column values map to readable text (`original` vs `low` granularity). | `granularity` |
| `VaryOrder` | Feature ordering (named columns first, the rest appended). | `order` |
| `VaryConnector` | The label↔value separator (`is:`, `is`, `=`, `:`, …). | `connector` |
| `VaryFormat` | Final layout (`textbullet`, `bullet`, `comma`, `text`). | `format` |
| `VaryPrefix` | Task description and optional custom prefix. | `custom_prompt_prefix` |
| `VarySuffix` | Question text / answer prefill. | `custom_prompt_suffix`, `show_question` |
| `VarySystemPrompt` | Optional system-role string (chat path). | *(via `system_prompt`)* |

`VaryFormat` collapses the feature list to a string, so per-item stages cannot
run after it — the types enforce a correct pipeline.

## `PromptConfig`

`PromptConfig` is a frozen dataclass holding one instance of each stage. Build
one from a dictionary of overrides — the keys are validated against
`DEFAULT_PROMPT_STYLE`, and an unknown key raises `ValueError`:

```py
from folktexts.prompting import PromptConfig

prompt_config = PromptConfig.from_dict(
    {"format": "bullet", "connector": "=", "order": "AGEP,SCHL,COW"},
    task="ACSIncome",
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

`PromptConfig` is hashable and deterministic across processes, so each distinct
configuration gets its own stable results-file name (no accidental overwrites).

### The `PROMPT_DEFAULT` sentinel

System- and chat-prompt fields default to the public sentinel `PROMPT_DEFAULT`,
which means *"use the QA subclass's `ClassVar` default"*. This is **distinct
from `None`**, which means *"explicitly disable this role"*:

```py
from folktexts.prompting import PROMPT_DEFAULT, PromptConfig

PromptConfig.from_dict({}, task="ACSIncome")                       # default system prompt
PromptConfig.from_dict({}, task="ACSIncome", system_prompt=None)   # no system role at all
PromptConfig.from_dict({}, task="ACSIncome", system_prompt="...")  # custom system prompt
```

Per-question defaults live on the `QAInterface` subclasses
(`MultipleChoiceQA`, `DirectNumericQA`, `ChainOfThoughtQA`) as the
`default_system_prompt` / `default_chat_prompt` class variables.

## `FewShotConfig`

Few-shot prompting is configured with a single frozen dataclass:

```py
from folktexts.prompting import FewShotConfig
from folktexts.benchmark import Benchmark

bench = Benchmark.make_acs_benchmark(
    "ACSIncome", model=llm, tokenizer=tokenizer, data_dir="~/data",
    few_shot_config=FewShotConfig(
        n_shots=4,
        compose="balanced",        # "random" | "balanced" | per-class counts e.g. (2, 2)
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
have been consolidated into `PromptConfig` / `FewShotConfig`. Removed keywords
now raise a `TypeError` (instead of being silently ignored), and saved benchmark
configs from before the change are still loaded — `BenchmarkConfig.load_from_disk`
translates the legacy keys automatically.

| Old | New |
|:---|:---|
| `custom_prompt_prefix="..."` (classifier / `encode_row_prompt*`) | `prompt_config=PromptConfig.from_dict({"custom_prompt_prefix": "..."}, task)` or CLI `--variation custom_prompt_prefix=...` |
| `add_task_description=False`, `with_answer_prefill=...` | folded into `PromptConfig` (via `VaryPrefix` / `VarySuffix`) |
| `few_shot=N`, `reuse_few_shot_examples=...`, `balance_few_shot_examples=...` (`BenchmarkConfig`) | `few_shot_config=FewShotConfig(n_shots=N, reuse_examples=..., compose="balanced")` |
| `class_balancing=True` (`sample_n_train_examples` / `encode_row_prompt_few_shot`) | `compose="balanced"` / CLI `--compose-few-shot-examples balanced` |
| CLI `--balance-few-shot-examples` | CLI `--compose-few-shot-examples balanced` |
| `numeric=True` (`encode_row_prompt_chat` / `resolve_chat_defaults`) | removed — the default is derived from the `QAInterface` subclass (`DirectNumericQA`) |
| `encode_row_prompt(row, task, question_obj)` (positional question) | `question=` is now keyword-only |
| `system_prompt=None` / `chat_prompt=None` to *mean* "default" | `PROMPT_DEFAULT` means "default"; `None` now means "explicitly disable" |

The top-level public API (`Benchmark`, `BenchmarkConfig`, the classifiers, the
`QAInterface` subclasses, `TaskMetadata`, `ACSDataset`) is unchanged.
