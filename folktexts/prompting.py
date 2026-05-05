"""Module to map risk-estimation questions to different prompting techniques.

e.g.,
- multiple-choice Q&A vs direct numeric Q&A;
- zero-shot vs few-shot vs CoT;
"""
from __future__ import annotations

import pandas as pd
from jinja2 import TemplateError
from transformers import AutoTokenizer

from .dataset import Dataset
from .qa_interface import QAInterface
from .task import TaskMetadata

# Sentinel distinguishing "use the mode-appropriate default" from `None`
# ("explicitly disable the role"). Module-private; not part of the public API.
_DEFAULT = object()

SYSTEM_PROMPT = """\
You are a helpful assistant. You answer multiple-choice questions based on the information provided.
"""

NUMERIC_SYSTEM_PROMPT = """\
You are a helpful assistant. You provide numeric probability \
estimates based on the information provided.
"""

ACS_TASK_DESCRIPTION = """\
The following data corresponds to a survey respondent. \
The survey was conducted among US residents in 2018. \
Please answer the question based on the information provided. \
The data provided is enough to reach an approximate answer.
"""

ACS_FEW_SHOT_TASK_DESCRIPTION = """\
The following data corresponds to different survey respondents. \
The survey was conducted among US residents in 2018. \
Please answer each question based on the information provided. \
The data provided is enough to reach an approximate answer for each person.
"""

ANTHROPIC_CHAT_PROMPT = """If had to select one of the options, my answer would be"""
GEMMA_CHAT_PROMPT = """The provided information suggests that the answer is"""
# NOTE: The leading `0.` is part of the prefill, so the model only generates
# the digits after the decimal point. This caps the expressible probability
# at the open interval [0, 1) — true posteriors at or near 1.0 cannot be
# emitted exactly. If you need full [0, 1] coverage, override `chat_prompt`
# with e.g. `"Answer (between 0 and 1): "` and let the model produce the
# leading digit itself (note that this also widens the digit-scoring search
# space and may degrade calibration for low-probability cases).
NUMERIC_CHAT_PROMPT = """Answer (between 0 and 1): 0."""


def encode_row_prompt(
    row: pd.Series,
    task: TaskMetadata,
    question: QAInterface = None,
    custom_prompt_prefix: str = None,
    add_task_description: bool = True,
    with_answer_prefill: bool = True,
) -> str:
    """Encode a question regarding a given row.

    `with_answer_prefill` is forwarded to `question.get_question_prompt`. The
    chat-template path passes `False` so the prefill is supplied as a separate
    assistant turn rather than baked into the user message.
    """
    # Get the question to ask
    question = question or task.question
    return (
        (ACS_TASK_DESCRIPTION + "\n" if add_task_description else "")
        + (f"\n{custom_prompt_prefix}\n" if custom_prompt_prefix else "")
        + f"""\
Information:
{task.get_row_description(row)}

{question.get_question_prompt(with_answer_prefill=with_answer_prefill)}""")


def encode_row_prompt_few_shot(
    row: pd.Series,
    task: TaskMetadata,
    dataset: Dataset,
    n_shots: int,
    question: QAInterface = None,
    reuse_examples: bool = False,
    class_balancing: bool = False,
    custom_prompt_prefix: str = None,
) -> str:
    """Encode a question regarding a given row using few-shot prompting.

    Parameters
    ----------
    row : pd.Series
        The row that the question will be about.
    task : TaskMetadata
        The task that the row belongs to.
    n_shots : int, optional
        The number of example questions and answers to use before prompting
        about the given row, by default 3.
    reuse_examples : bool, optional
        Whether to reuse the same examples for consistency. By default will
        resample new examples each time (`reuse_examples=False`).

    Returns
    -------
    prompt : str
        The encoded few-shot prompt.
    """
    # Take `n_shots` random samples from the train set
    X_examples, y_examples = dataset.sample_n_train_examples(
        n_shots,
        reuse_examples=reuse_examples,
        class_balancing=class_balancing,
    )

    # Start with task description
    prompt = ACS_FEW_SHOT_TASK_DESCRIPTION + "\n"

    # Get the question to ask
    question = question or task.question

    # Add `n` example rows with respective labels
    for i in range(n_shots):
        prompt += (
            encode_row_prompt(
                X_examples.iloc[i],
                task=task,
                add_task_description=False,
                custom_prompt_prefix=custom_prompt_prefix,
            )
            + f" {question.get_answer_key_from_value(y_examples.iloc[i])}"
            + "\n\n"
        )

    # Add the target row without its label
    prompt += encode_row_prompt(
        row,
        task=task,
        add_task_description=False,
        custom_prompt_prefix=custom_prompt_prefix,
        question=question,
    )
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
    numeric: bool,
    system_prompt: str | None = None,
    chat_prompt: str | None = None,
) -> tuple[str, str]:
    """Resolve default system_prompt / chat_prompt for chat-template prompting.

    A `None` value means "use the default for this mode". To explicitly disable
    a role downstream, override the resolved value with `None` after calling
    this function (which is what `Benchmark.make_benchmark` does for tokenizers
    that reject the system role).
    """
    if system_prompt is None:
        system_prompt = NUMERIC_SYSTEM_PROMPT if numeric else SYSTEM_PROMPT
    if chat_prompt is None:
        chat_prompt = NUMERIC_CHAT_PROMPT if numeric else ANTHROPIC_CHAT_PROMPT
    return system_prompt, chat_prompt


def encode_row_prompt_chat(
    row: pd.Series,
    task: TaskMetadata,
    tokenizer: AutoTokenizer,
    system_prompt: str | None = _DEFAULT,    # type: ignore[assignment]
    chat_prompt: str | None = _DEFAULT,      # type: ignore[assignment]
    numeric: bool = False,
    question: QAInterface | None = None,
    custom_prompt_prefix: str | None = None,
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
        System prompt text. If omitted, the mode-appropriate default selected
        by `numeric` is used. Pass `None` explicitly to disable the system
        role (e.g. for Gemma-style templates that reject it).
    chat_prompt : str | None, optional
        Assistant prefill text. If omitted, the mode-appropriate default
        selected by `numeric` is used. Pass `None` explicitly to skip the
        assistant prefill — note that this routes inference through
        `add_generation_prompt=True` and breaks the last-token scoring
        assumption used by `LLMClassifier`, so it is not appropriate for the
        benchmark path.
    numeric : bool, optional
        Whether numeric risk prompting is being used. Selects which default
        prompts are applied when `system_prompt` / `chat_prompt` are omitted.
    question : QAInterface, optional
        The question interface to use.
    custom_prompt_prefix : str, optional
        A custom prompt prefix to prepend.

    Returns
    -------
    str
        The fully formatted chat-template prompt.
    """
    if system_prompt is _DEFAULT:
        system_prompt = NUMERIC_SYSTEM_PROMPT if numeric else SYSTEM_PROMPT
    if chat_prompt is _DEFAULT:
        chat_prompt = NUMERIC_CHAT_PROMPT if numeric else ANTHROPIC_CHAT_PROMPT

    # Skip the answer prefill in the user message: the chat path supplies it
    # as the assistant turn (`chat_prompt`). Including it in both turns would
    # duplicate the string in the rendered prompt and silently degrade scoring.
    user_content = encode_row_prompt(
        row, task, question=question,
        custom_prompt_prefix=custom_prompt_prefix,
        with_answer_prefill=False,
    )

    return apply_chat_template(
        tokenizer,
        user_prompt=user_content,
        system_prompt=system_prompt,
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
    conversation = ([
        {"role": "system", "content": system_prompt}
    ] if system_prompt is not None else [])

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
    filled_prompt = tokenizer.apply_chat_template(
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
