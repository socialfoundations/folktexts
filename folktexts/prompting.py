"""Prompt construction utilities for risk-estimation tasks.

This module maps risk-estimation questions to different prompting techniques
and supports systematic prompt variations for benchmarking and evaluation.

Each prompt (corresponding to a tabular data row) is represented
as composition of three parts:

    [PREFIX]  Shared task description and/or system context.
              This section is constant across all rows.
    [INFO]    Row-specific serialized feature-value pairs.
    [SUFFIX]  Question text defining the prediction task.

Within INFO the prompt variation pipeline is fixed by semantics:
VaryValueMap → VaryOrder → VaryConnector → VaryFormat
Return types enforce the order: per-item stages share list→list;
VaryFormat collapses the list to str, making it impossible to apply
a per-item stage after it.

The module implements multiple prompting strategies, including:

    - Multiple-choice Q&A vs direct numeric Q&A
    - Zero-shot prompting
    - Few-shot prompting
    - Chain-of-thought (CoT) prompting
"""

from __future__ import annotations

import dataclasses
import logging
from copy import copy
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd
from jinja2 import TemplateError
from transformers import AutoTokenizer

from folktexts.acs import ACS_TASK_DESCRIPTION, ACS_TASK_DESCRIPTION_DEFAULTS

from .dataset import Dataset
from .qa_interface import (
    MultipleChoiceQA,
    QAInterface,
)
from .task import TaskMetadata

# Sentinel distinguishing "use the mode-appropriate default" from `None`
# ("explicitly disable the role").
PROMPT_DEFAULT = object()


DEFAULT_PROMPT_STYLE: dict[str, Any] = {
    "format": "textbullet",
    "connector": "is",
    "granularity": "original",
    "order": None,
    "custom_prompt_prefix": None,
    "custom_prompt_suffix": None,
    "show_question": True,
}


# ---------------------------------------------------------------------------
# Intermediate representation for prompt construction
# ---------------------------------------------------------------------------


@dataclass
class FeatureItem:
    col: str  # pandas column name
    label: str  # human-readable name from ColumnToText.short_description
    raw_value: Any  # original value from the DataFrame
    text_value: str = ""  # filled by VaryValueMap
    connected: str = ""  # filled by VaryConnector


# ---------------------------------------------------------------------------
# Variation stages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VaryPrefix:
    task_description: str
    add_task_description: bool = True
    custom_prefix: str | None = None

    """
    A stage for adding a prefix to the prompt.
    Parameters
    ----------
    task_description : str
        The description of the task to include in the prefix.
    add_task_description : bool, optional
        Whether to include the task description in the prefix. Default is True.
    custom_prefix : str | None, optional
        A custom string to include in the prefix after the task description and before 
        the encoded features. If None, no custom prefix is added. Default is None.
    """

    def __call__(self) -> str:
        parts = []
        if self.add_task_description:
            parts.append(self.task_description)
        if self.custom_prefix:
            cp = (
                self.custom_prefix
                if self.custom_prefix.endswith("\n")
                else self.custom_prefix + "\n"
            )
            parts.append(cp)
        prefix = "".join(parts)
        return prefix + ("\n" if prefix else "") + "Information:\n"


@dataclass(frozen=True)
class VarySuffix:
    question: QAInterface
    show_question: bool = True
    with_answer_prefill: bool = (
        True  # set False for chat mode (prefill is assistant turn)
    )
    show_label: bool = False
    label: Any = None  # only used when show_label=True
    custom_suffix: str | None = None

    """
    A stage for adding a suffix to the prompt, typically containing the question.
    Parameters
    ----------
    question : QAInterface
        The question interface to use for generating the question prompt or answer prefix.
    show_question : bool, optional
        Whether to include the full question prompt (True) or just the answer prefix (False). Default is True.
    with_answer_prefill : bool, optional
        Whether to include the answer prefill in the question prompt. Default is True. Ignored if show_question is False.
    show_label : bool, optional
        Whether to include the label in the suffix. Default is False.
    label : Any, optional
        The label to include in the suffix if show_label is True. Ignored otherwise.
    custom_suffix : str | None, optional
        Custom string to include in the suffix after the question. If None, no custom suffix is added. Default is None.
    """

    def __post_init__(self):
        if self.show_label and self.label is None:
            raise ValueError("show_label=True requires label to be set.")

    def __call__(self) -> str:
        base = (
            self.question.get_question_prompt(
                with_answer_prefill=self.with_answer_prefill
            )
            if self.show_question
            else self.question.get_answer_prefix()
        )
        label_part = f" {self.label}\n\n" if self.show_label else ""
        return f"\n{base}{label_part}{self.custom_suffix or ''}"


@dataclass(frozen=True)
class VaryValueMap:
    cols_to_text: dict = field(hash=False, compare=False)

    """
    A stage for mapping raw feature values to human-readable text.
    Parameters
    ----------
    cols_to_text : dict
        A mapping from column names to ColumnToText objects, which provide the logic for converting raw values to text.
    """

    def __call__(self, items: list[FeatureItem]) -> list[FeatureItem]:
        return [
            dataclasses.replace(
                item, text_value=self.cols_to_text[item.col][item.raw_value]
            )
            for item in items
        ]

    @classmethod
    def with_low_granularity(
        cls, cols_to_text: dict, simplified_value_maps: dict
    ) -> "VaryValueMap":
        """Return a VaryValueMap with simplified (low-granularity) value maps.

        Shallow-copies each ColumnToText that has a simplified map available,
        leaving the original task object untouched.
        """
        modified = {}
        for col, c2t in cols_to_text.items():
            if col in simplified_value_maps:
                c2t_copy = copy(c2t)
                c2t_copy._value_map = simplified_value_maps[col]
                modified[col] = c2t_copy
            else:
                modified[col] = c2t
        return cls(cols_to_text=modified)


@dataclass(frozen=True)
class VaryOrder:
    order: tuple | list | str | None = None  # column names; None → keep original

    """
    A stage for reordering the feature items.
    Parameters
    ----------
    order : tuple | list | str | None, optional
        Column names specifying the desired order of features in the prompt (a tuple/list,
        or a comma-separated string). If None, the original order is preserved. Default is None.
    """

    def __post_init__(self):
        # Frozen dataclasses must be hashable: PromptConfig.__hash__ reaches hash(VaryOrder),
        # and a list field raises "unhashable type: 'list'". Normalize to a tuple
        # (as FewShotConfig already does for `compose`).
        if isinstance(self.order, str):
            object.__setattr__(self, "order", tuple(c.strip() for c in self.order.split(",")))
        elif self.order is not None:
            object.__setattr__(self, "order", tuple(self.order))

    def __call__(self, items: list[FeatureItem]) -> list[FeatureItem]:
        if not self.order:
            return items
        index = {item.col: item for item in items}
        return [index[col] for col in self.order if col in index]


VaryFeatureOrder = VaryOrder  # alias for backward compatibility


@dataclass(frozen=True)
class VaryConnector:
    connector: str = "is"

    """
    A stage for connecting feature labels to their values in the prompt.
    Parameters
    ----------
    connector : str, optional
        The string to use for connecting feature labels to their values. 
        For example, "is" would produce prompts like "Age is 30", while ":" would produce "Age: 30". Default is "is".
    """

    def __call__(self, items: list[FeatureItem]) -> list[FeatureItem]:
        sep = ": " if self.connector == ":" else f" {self.connector} "
        return [
            dataclasses.replace(item, connected=f"{item.label}{sep}{item.text_value}")
            for item in items
        ]


@dataclass(frozen=True)
class VaryFormat:
    format: str = "textbullet"

    """
    A stage for formatting the connected feature strings into the final prompt.
    Parameters
    ----------
    format : str, optional
        The format to use for the final prompt. Options include:
        - "bullet": Each feature on a new line, prefixed with "- " \
            (e.g. "- Age is 30\n- Occupation is Engineer").
        - "comma": All features on the same line, separated by commas \
            (e.g. "Age is 30, Occupation is Engineer").
        - "text": All features in a single line, each prefixed with "The "and suffixed with a period \
            (e.g. "The Age is 30. The Occupation is Engineer.").
        - "textbullet": Each feature on a new line, prefixed with "- The " and suffixed with a period \
            (e.g. "- The Age is 30.\n- The Occupation is Engineer."). 
        Default is "textbullet".
    """

    _TEMPLATES: ClassVar[dict] = {
        "bullet": lambda s: f"- {s}\n",
        "comma": lambda s: f"{s}, ",
        "text": lambda s: f"The {s}. ",
        "textbullet": lambda s: f"- The {s}.\n",
    }

    def __post_init__(self):
        if self.format not in self._TEMPLATES:
            raise ValueError(
                f"Unknown format {self.format!r}. Choose from {list(self._TEMPLATES)}"
            )

    def __call__(self, items: list[FeatureItem]) -> str:
        template = self._TEMPLATES[self.format]
        return "".join(template(item.connected) for item in items).rstrip(", \n")


@dataclass(frozen=True)
class VarySystemPrompt:
    system_prompt: str

    """
    A stage for adding a system prompt to the chat context.
    Parameters
    ----------
    system_prompt : str
        The system prompt string to include in the chat context. This provides instructions or 
        context to the model before the user prompt. """

    def __call__(self) -> str:
        return self.system_prompt


# ---------------------------------------------------------------------------
# FewShotConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FewShotConfig:
    """Configuration for few-shot prompting.

    Parameters
    ----------
    n_shots : int
        Number of example questions and answers to prepend.
    example_order : list[int] | str | None, optional
        Integer permutation to reorder examples (e.g. ``[2, 0, 1]`` or ``"2,0,1"``). ``None`` keeps the sampled order.
    compose : str | list, optional
        How to select few-shot samples: ``"random"`` (default), ``"balanced"`` (equal draws per class), or a list of
        per-class counts summing to ``n_shots``.
    reuse_examples : bool, optional
        Whether to reuse the same examples across calls, by default False.
    """

    n_shots: int
    example_order: list[int] | str | None = None
    compose: str | list = "random"
    reuse_examples: bool = False

    def __post_init__(self):
        if self.n_shots < 1:
            raise ValueError(f"n_shots must be >= 1; got {self.n_shots}.")

        # Normalize example_order to a tuple: a frozen dataclass must be hashable, and a
        # list field breaks hash(FewShotConfig) (reached via BenchmarkConfig.__hash__).
        if isinstance(self.example_order, str):
            object.__setattr__(
                self,
                "example_order",
                tuple(int(i) for i in self.example_order.split(",")),
            )
        elif self.example_order is not None:
            object.__setattr__(self, "example_order", tuple(self.example_order))
        if self.example_order is not None:
            if sorted(self.example_order) != list(range(self.n_shots)):
                raise ValueError(
                    f"example_order must be a permutation of [0, ..., n_shots-1]; "
                    f"got {self.example_order} for n_shots={self.n_shots}."
                )

        if isinstance(self.compose, str) and "," in self.compose:
            object.__setattr__(
                self, "compose", tuple(int(c) for c in self.compose.split(","))
            )
        elif isinstance(self.compose, list):
            object.__setattr__(self, "compose", tuple(self.compose))
        if isinstance(self.compose, tuple):
            if any(c < 0 for c in self.compose):
                raise ValueError(
                    f"compose counts must be non-negative; got {self.compose}."
                )
            if sum(self.compose) != self.n_shots:
                raise ValueError(
                    f"compose counts must sum to n_shots={self.n_shots}; got sum={sum(self.compose)}."
                )
        elif self.compose not in ("random", "balanced"):
            raise ValueError(
                f"compose must be 'random', 'balanced', or a list of counts; got {self.compose!r}."
            )


# ---------------------------------------------------------------------------
# PromptConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptConfig:
    prefix: VaryPrefix
    value_map: VaryValueMap
    order: VaryOrder
    connector: VaryConnector
    format: VaryFormat
    suffix: VarySuffix
    system_prompt: VarySystemPrompt | None = None

    @classmethod
    def default(cls, task: TaskMetadata) -> "PromptConfig":
        return cls.from_dict({}, task=task)

    @classmethod
    def from_dict(
        cls,
        pv: dict[str, Any],
        task: TaskMetadata,
        question: QAInterface | None = None,
        add_task_description: bool = True,
        system_prompt: str | None = PROMPT_DEFAULT,  # type: ignore[assignment]
    ) -> "PromptConfig":
        """Build a PromptConfig from a prompt-variation dict and a task.

        Parameters
        ----------
        pv : dict
            Prompt style overrides; see ``DEFAULT_PROMPT_STYLE`` for valid keys.
        task : TaskMetadata
            The task that defines features, column mappings, and the question.
        question : QAInterface, optional
            Override the task's default question interface.
        add_task_description : bool, optional
            Whether to include the task description in the prefix.
        system_prompt : str | None, optional
            System prompt string; wrapped in ``VarySystemPrompt`` when provided.
            Defaults to ``question.default_system_prompt`` (set per QA subclass).
            Pass ``None`` explicitly to disable the system role (e.g. for
            Gemma-style templates).
        """
        unknown = set(pv) - set(DEFAULT_PROMPT_STYLE)
        if unknown:
            raise ValueError(
                f"Unknown prompt_variation keys: {sorted(unknown)}. Valid keys: {sorted(DEFAULT_PROMPT_STYLE)}."
            )

        granularity = pv.get("granularity", DEFAULT_PROMPT_STYLE["granularity"])
        if granularity not in ("original", "low"):
            raise ValueError(
                f"Unknown granularity {granularity!r}. Choose 'original' or 'low'."
            )

        order = pv.get("order", DEFAULT_PROMPT_STYLE["order"])
        if isinstance(order, str):
            order = [col.strip() for col in order.split(",")]

        question = question or task.question
        if system_prompt is PROMPT_DEFAULT:
            system_prompt = question.default_system_prompt
        value_map = (
            VaryValueMap.with_low_granularity(
                task.cols_to_text, cls._get_simplified_value_maps(task)
            )
            if granularity == "low"
            else VaryValueMap(task.cols_to_text)
        )
        return cls(
            prefix=VaryPrefix(
                task_description=cls._get_task_description(task),
                add_task_description=add_task_description,
                custom_prefix=pv.get(
                    "custom_prompt_prefix", DEFAULT_PROMPT_STYLE["custom_prompt_prefix"]
                ),
            ),
            value_map=value_map,
            order=VaryOrder(order=order),
            connector=VaryConnector(
                connector=pv.get("connector", str(DEFAULT_PROMPT_STYLE["connector"]))
            ),
            format=VaryFormat(
                format=pv.get("format", str(DEFAULT_PROMPT_STYLE["format"]))
            ),
            suffix=VarySuffix(
                question=question,
                show_question=pv.get(
                    "show_question", bool(DEFAULT_PROMPT_STYLE["show_question"])
                ),
                custom_suffix=pv.get(
                    "custom_prompt_suffix", DEFAULT_PROMPT_STYLE["custom_prompt_suffix"]
                ),
            ),
            system_prompt=VarySystemPrompt(system_prompt)
            if system_prompt is not None
            else None,
        )

    @staticmethod
    def _get_task_description(task: TaskMetadata) -> str:
        descriptions = {
            "ACS": ACS_TASK_DESCRIPTION.substitute(ACS_TASK_DESCRIPTION_DEFAULTS),
        }

        for key, desc in descriptions.items():
            if key in task.name:
                return desc
        raise ValueError(f"Cannot determine task description for task '{task.name}'")

    @staticmethod
    def _get_simplified_value_maps(task: TaskMetadata) -> dict:
        if task.name.startswith("ACS"):
            from folktexts.acs.acs_columns_simplified import simplified_value_maps

            return simplified_value_maps
        raise NotImplementedError(
            f"Low-granularity value maps are not available for task '{task.name}'."
        )

    @staticmethod
    def _get_few_shot_task_description(task: TaskMetadata) -> str | None:
        overrides = {
            "respondent": "different survey respondents",
            "suffix": " for each person",
        }
        if task.name.startswith("ACS"):
            return ACS_TASK_DESCRIPTION.substitute(
                {**ACS_TASK_DESCRIPTION_DEFAULTS, **overrides}
            )
        return None


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class PromptBuilder:
    def __init__(self, task: TaskMetadata):
        self.task = task

    def _extract_items(self, row: pd.Series) -> list[FeatureItem]:
        return [
            FeatureItem(
                col=col,
                label=self.task.cols_to_text[col].short_description,
                raw_value=row[col],
            )
            for col in self.task.features
            if col in row.index
        ]

    def build(
        self,
        row: pd.Series,
        config: PromptConfig,
        question: QAInterface | None = None,
    ) -> str:
        if question is not None:
            config = dataclasses.replace(
                config, suffix=dataclasses.replace(config.suffix, question=question)
            )
        items = self._extract_items(row)
        items = config.value_map(items)
        items = config.order(items)
        items = config.connector(items)
        info_block = config.format(items)
        info_block += "\n"
        return config.prefix() + info_block + config.suffix()

    def build_few_shot(
        self,
        row: pd.Series,
        config: PromptConfig,
        examples: list[tuple],  # list of (pd.Series, label)
        question: QAInterface | None = None,
        example_order: list[int] | None = None,
    ) -> str:
        if example_order is not None:
            assert len(example_order) == len(examples)
            examples = [examples[i] for i in example_order]

        few_shot_desc = config._get_few_shot_task_description(self.task)

        parts = []
        for i, (ex_row, ex_label) in enumerate(examples):
            if i == 0:
                prefix = config.prefix
                if few_shot_desc is not None:
                    prefix = dataclasses.replace(prefix, task_description=few_shot_desc)
            else:
                prefix = dataclasses.replace(config.prefix, add_task_description=False)
            ex_config = dataclasses.replace(
                config,
                prefix=prefix,
                suffix=dataclasses.replace(
                    config.suffix,
                    show_question=False,
                    show_label=True,
                    label=ex_label,
                ),
            )
            parts.append(self.build(ex_row, ex_config))

        target_config = dataclasses.replace(
            config,
            prefix=dataclasses.replace(config.prefix, add_task_description=False),
        )
        parts.append(self.build(row, target_config, question=question))
        return "".join(parts)

    def build_chat(
        self,
        row: pd.Series,
        config: PromptConfig,
        tokenizer: AutoTokenizer,
        question: QAInterface | None = None,
        chat_prompt: str | None = PROMPT_DEFAULT,  # type: ignore[assignment]
        **kwargs,
    ) -> str:
        resolved_question = question or config.suffix.question
        if chat_prompt is PROMPT_DEFAULT:
            chat_prompt = resolved_question.default_chat_prompt
        # Always strip the answer prefill from the user turn: the chat template
        # supplies it as a separate assistant turn (chat_prompt).
        chat_config = dataclasses.replace(
            config,
            suffix=dataclasses.replace(
                config.suffix,
                with_answer_prefill=False,
                question=resolved_question,
            ),
        )
        user_content = self.build(row, chat_config)
        system_content = config.system_prompt() if config.system_prompt else None
        return apply_chat_template(
            tokenizer,
            user_prompt=user_content,
            system_prompt=system_content,
            chat_prompt=chat_prompt,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


def encode_row_prompt(
    row: pd.Series,
    task: TaskMetadata,
    question: QAInterface = None,
    prompt_config: PromptConfig | None = None,
) -> str:
    """
    Encode a question regarding a given row into a natural-language prompt.

    Parameters
    ----------
    row : pd.Series
        The data row to encode.
    task : TaskMetadata
        The task that defines features, column mappings, and the question.
    question : QAInterface, optional
        Override the question interface. Defaults to ``task.question``.
        When ``prompt_config`` is provided, only the suffix question is
        replaced (used for order-bias correction). Otherwise a default
        config is built with this question.
    prompt_config : PromptConfig, optional
        A pre-built config; all style parameters are taken from it.
        Build once at classifier/benchmark init and pass here to avoid
        rebuilding on every row.

    `with_answer_prefill` is forwarded to `question.get_question_prompt`. The
    chat-template path passes `False` so the prefill is supplied as a separate
    assistant turn rather than baked into the user message.

    Returns
    -------
    str
        The fully formatted prompt string.
    """
    if prompt_config is not None:
        return PromptBuilder(task).build(
            row[task.features], prompt_config, question=question
        )
    config = PromptConfig.from_dict({}, task=task, question=question)
    return PromptBuilder(task).build(row[task.features], config)


def encode_row_prompt_few_shot(
    row: pd.Series,
    task: TaskMetadata,
    dataset: Dataset,
    n_shots: int = None,
    question: QAInterface = None,
    reuse_examples: bool = False,
    compose_few_shot_examples: str | list = "random",
    example_order: list[int] | str | None = None,
    prompt_config: PromptConfig | None = None,
    few_shot_config: FewShotConfig | None = None,
) -> str:
    """Encode a question regarding a given row using few-shot prompting.

    Parameters
    ----------
    row : pd.Series
        The row that the question will be about.
    task : TaskMetadata
        The task that the row belongs to.
    dataset : Dataset
        The dataset to draw few-shot examples from (sampled from the train split).
    n_shots : int, optional
        The number of example questions and answers to prepend. Ignored when
        ``few_shot_config`` is provided.
    question : QAInterface, optional
        The question interface to use; defaults to ``task.question``.
    reuse_examples : bool, optional
        Whether to reuse the same examples for consistency. By default will
        resample new examples each time (`reuse_examples=False`). Ignored
        when ``few_shot_config`` is provided.
    compose_few_shot_examples : str or list, optional
        How to select few-shot samples: ``"random"`` (default), ``"balanced"``
        (equal draws per class), or a list of per-class counts summing to
        ``n_shots``. Ignored when ``few_shot_config`` is provided.
    example_order : list[int] | str | None, optional
        Integer permutation to reorder examples before building the prompt
        (e.g. ``[2, 0, 1]`` for 3 shots). ``None`` keeps the sampled order.
        Ignored when ``few_shot_config`` is provided.
    prompt_config : PromptConfig, optional
        A pre-built config object. When provided, all other style parameters
        are ignored.
    few_shot_config : FewShotConfig, optional
        Typed few-shot configuration. When provided, the individual
        ``n_shots``, ``reuse_examples``, ``compose_few_shot_examples``, and
        ``example_order`` params are ignored.

    Returns
    -------
    prompt : str
        The encoded few-shot prompt.
    """
    if few_shot_config is None:
        if n_shots is None:
            raise ValueError("Either `few_shot_config` or `n_shots` must be provided.")
        few_shot_config = FewShotConfig(
            n_shots=n_shots,
            example_order=example_order,
            compose=compose_few_shot_examples,
            reuse_examples=reuse_examples,
        )

    assert few_shot_config.example_order is None or isinstance(
        few_shot_config.example_order, list
    )  # mypy
    logging.debug(f"Composition of few shot examples: {few_shot_config.compose}")

    # Take `n_shots` random samples from the train set
    X_examples, y_examples = dataset.sample_n_train_examples(
        few_shot_config.n_shots,
        reuse_examples=few_shot_config.reuse_examples,
        composition=few_shot_config.compose,
    )

    X_examples = X_examples.sort_index()
    y_examples = y_examples.sort_index()
    logging.debug(f"ys index: {y_examples.index.tolist()}")
    logging.debug(f"ys: {y_examples.values.tolist()}")

    # Get the question to ask
    question = question or task.question

    # Collect `n_shots` example rows with respective labels
    examples = []
    for i in range(few_shot_config.n_shots):
        if isinstance(question, MultipleChoiceQA):
            label = question.get_answer_key_from_value(y_examples.iloc[i])
            if label is None:
                raise ValueError(
                    f"Could not find answer key for few-shot label '{y_examples.iloc[i]}' in question choices."
                )
        else:
            label = y_examples.iloc[i]
        logging.debug(f"shot {i}: label={label}\tindex={y_examples.index[i]}")
        examples.append((X_examples.iloc[i], label))

    # prompt_config takes precedence over individual style parameters
    config = prompt_config or PromptConfig.from_dict({}, task=task, question=question)
    prompt = PromptBuilder(task).build_few_shot(
        row=row,
        config=config,
        examples=examples,
        question=question if prompt_config is not None else None,
        example_order=few_shot_config.example_order,
    )
    logging.debug(prompt)
    return prompt


def tokenizer_supports_system_prompt(tokenizer: AutoTokenizer) -> bool:
    """Check whether the tokenizer's chat template supports system messages.

    Some models (e.g. Gemma) raise a TemplateError when a system role is used.
    Other templates surface this with different exception types depending on
    transformers / Jinja versions (e.g. `RuntimeError`, `KeyError`, or a
    template-defined exception macro), so we treat any failure of the probe
    as "system role not supported" rather than letting it propagate and
    crash the benchmark.
    """
    test_conversation = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "test"},
    ]
    try:
        tokenizer.apply_chat_template(test_conversation, tokenize=False)
        return True
    except (TemplateError, ValueError):
        return False
    except Exception:
        # Defensive fallback for unexpected template-rendering failures —
        # safer to skip the system prompt than to hard-fail the benchmark.
        return False


def resolve_chat_defaults(
    question: QAInterface,
    system_prompt: str | None = PROMPT_DEFAULT,  # type: ignore[assignment]
    chat_prompt: str | None = PROMPT_DEFAULT,  # type: ignore[assignment]
) -> tuple[str | None, str | None]:
    """Resolve default system_prompt / chat_prompt for chat-template prompting.

    Defaults are read from ``question.default_system_prompt`` and
    ``question.default_chat_prompt`` (``ClassVar``s defined on each
    ``QAInterface`` subclass). Pass ``PROMPT_DEFAULT`` (or omit the argument)
    to use the question's ClassVar default. Pass ``None`` explicitly to disable
    a role entirely (e.g. for Gemma-style tokenizers that reject the system role).
    """
    if system_prompt is PROMPT_DEFAULT:
        system_prompt = question.default_system_prompt
    if chat_prompt is PROMPT_DEFAULT:
        chat_prompt = question.default_chat_prompt
    return system_prompt, chat_prompt


def encode_row_prompt_chat(
    row: pd.Series,
    task: TaskMetadata,
    tokenizer: AutoTokenizer,
    system_prompt: str | None = PROMPT_DEFAULT,  # type: ignore[assignment]
    chat_prompt: str | None = PROMPT_DEFAULT,  # type: ignore[assignment]
    question: QAInterface | None = None,
    prompt_config: PromptConfig | None = None,
) -> str:
    """Encode a row prompt using the tokenizer's chat template.

    Parameters
    ----------
    row : pd.Series
        The row that the question will be about.
    task : TaskMetadata
        The task metadata object.
    tokenizer : AutoTokenizer
        The tokenizer whose chat template will be applied.
    system_prompt : str | None, optional
        System prompt text. Only used when ``prompt_config`` is not provided;
        passed straight to ``PromptConfig.from_dict`` which selects the
        mode-appropriate default when omitted. Pass ``None`` explicitly to
        disable the system role (e.g. for Gemma-style templates that reject
        it). When ``prompt_config`` is provided, system_prompt is ignored —
        patch the config directly instead.
    chat_prompt : str | None, optional
        Assistant prefill text. If omitted, the mode-appropriate default is
        selected from the question type. Pass ``None`` explicitly to skip the
        assistant prefill — note that this routes inference through
        ``add_generation_prompt=True`` and breaks the last-token scoring
        assumption used by ``LLMClassifier``, so it is not appropriate for the
        benchmark path.
    question : QAInterface, optional
        The question interface to use. When ``prompt_config`` is provided this
        overrides only the suffix question (used for order-bias correction).
    prompt_config : PromptConfig, optional
        A pre-built config object. When provided, all style parameters are
        ignored. ``question`` still overrides the suffix question when given.

    Returns
    -------
    str
        The fully formatted chat-template prompt.
    """
    if prompt_config is not None:
        return PromptBuilder(task).build_chat(
            row[task.features],
            prompt_config,
            tokenizer,
            question=question,
            chat_prompt=chat_prompt,
        )
    config = PromptConfig.from_dict(
        {}, task=task, question=question, system_prompt=system_prompt
    )
    return PromptBuilder(task).build_chat(
        row[task.features],
        config,
        tokenizer,
        chat_prompt=chat_prompt,
    )


def apply_chat_template(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    system_prompt: str | None = None,
    chat_prompt: str | None = None,
    **kwargs,
) -> str:
    """Apply the tokenizer's chat template to assemble a single prompt string.

    Notes
    -----
    `system_prompt` is treated as "include" iff it is not `None`. This means an
    empty string `""` will inject an empty system message rather than be
    treated as "no system role" — pass `None` (or omit the argument) to skip
    the system role entirely.

    `chat_prompt` is the assistant prefill. When provided, the returned prompt
    is trimmed so it ends exactly with `chat_prompt`, preserving the
    last-token scoring contract relied on by `LLMClassifier`. If the chat
    template mutates or strips the prefill (so it cannot be located verbatim
    in the rendered output), a `ValueError` is raised rather than silently
    returning a corrupted prompt.

    When `chat_prompt is None`, `add_generation_prompt=True` is used and the
    model is left to generate freely; this is **not** appropriate for the
    benchmark scoring path (the last token will be a template-emitted role
    header, not the prefill).
    """
    # Add system prompt
    conversation = (
        [{"role": "system", "content": system_prompt}]
        if system_prompt is not None
        else []
    )

    # Add user prompt
    conversation.append({"role": "user", "content": user_prompt})

    if chat_prompt is not None:
        # Using the Anthropic-style chat prompt
        conversation.append({"role": "assistant", "content": chat_prompt})
        kwargs.setdefault("add_generation_prompt", False)
    else:
        # No assistant prefill; let the model generate freely
        kwargs.setdefault("add_generation_prompt", True)

    # Apply prompt template
    if kwargs.pop("tokenize", False):
        raise ValueError(
            "apply_chat_template always returns a string (tokenize=False); pass tokenize=False or omit it."
        )
    filled_prompt = tokenizer.apply_chat_template(  # ignore[attr-defined]
        conversation=conversation,
        tokenize=False,
        **kwargs,
    )

    if chat_prompt is not None:
        # Trim any special tokens that the template appended after the prefill
        # (e.g. a trailing newline or `<end_of_turn>`) so the last token of the
        # returned prompt is the last token of `chat_prompt` itself — this is
        # what `LLMClassifier` assumes when it reads answer-token probabilities.
        idx = filled_prompt.rfind(chat_prompt)
        if idx == -1:
            raise ValueError(
                "Assistant prefill not found verbatim in the templated output; "
                "the tokenizer's chat template likely transforms it (e.g. "
                "stripping or escaping). Cannot safely trim trailing tokens — "
                "pass a `chat_prompt` that survives templating, or run without "
                "an assistant prefill."
            )
        filled_prompt = filled_prompt[: idx + len(chat_prompt)]

    return filled_prompt
