# Configuring prompts

`folktexts` builds every prompt from three parts:

```
[PREFIX]  task description / system context   (constant across rows)
[INFO]    serialized feature-value pairs       (row-specific)
[SUFFIX]  question text + answer prefill        (constant)
```

The *answer prefill* is the fixed lead-in the prompt ends on (e.g. `Answer:`), so
the model's next token is the answer we score; in chat mode it becomes the
assistant's opening turn instead.

The defaults reproduce the original paper's prompts exactly; this page is only
needed if you want to *change* how they are rendered. It documents the Python
API and the migration from the older flat-keyword interface — the command-line
equivalents are summarized in the {doc}`README <readme>`.

## The variation pipeline

The `[INFO]` block is produced by a typed pipeline of frozen dataclasses. You
don't instantiate these stages yourself — you set the keys in the last column
below (via `--variation` or `PromptConfig.from_dict`). Each stage has a fixed
input/output type, so the order is fixed: the final stage, `VaryFormat`,
collapses the feature list to a string, which is why no per-item stage can run
after it.

```
VaryValueMap → VaryOrder → VaryConnector → VaryFormat
(list→list)    (list→list)  (list→list)    (list→str)
```

| Stage | Controls | `--variation` key |
|:---|:---|:---|
| `VaryValueMap` | How raw column values render as text; `low` granularity coarsens ACS values into broader bins (age ranges, grouped occupations). | `granularity` |
| `VaryOrder` | Feature ordering (named columns first, the rest appended). | `order` |
| `VaryConnector` | The label↔value separator (`is:`, `is`, `=`, `:`, …). | `connector` |
| `VaryFormat` | Final layout (`textbullet`, `bullet`, `comma`, `text`). | `format` |
| `VaryPrefix` | Task description and optional custom prefix. | `custom_prompt_prefix` |
| `VarySuffix` | Question text / answer prefill. | `custom_prompt_suffix`, `show_question` |
| `VarySystemPrompt` | Optional system-role string (chat path). | *(via `system_prompt`)* |

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
configuration gets its own stable results-file name.

### The `PROMPT_DEFAULT` sentinel

`system_prompt` (and `chat_prompt`) have three modes: omit the argument for the
built-in default, pass `None` to remove the role entirely (needed for
Gemma-style templates that reject a system turn), or pass your own string. The
"built-in default" mode is spelled with the public sentinel `PROMPT_DEFAULT` —
which is **distinct from `None`**:

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
have been consolidated into `PromptConfig` / `FewShotConfig`. Removed keywords
now raise a `TypeError` (instead of being silently ignored), and saved benchmark
configs from before the change are still loaded — `BenchmarkConfig.load_from_disk`
translates the legacy keys automatically.

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
