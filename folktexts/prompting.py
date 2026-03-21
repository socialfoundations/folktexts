"""Module to map risk-estimation questions to different prompting techniques.

e.g.,
- multiple-choice Q&A vs direct numeric Q&A;
- zero-shot vs few-shot vs CoT;
"""
from __future__ import annotations

import logging

import pandas as pd
from transformers import AutoTokenizer

from .dataset import Dataset
from .qa_interface import QAInterface
from .task import TaskMetadata

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
NUMERIC_CHAT_PROMPT = """Answer (between 0 and 1): 0."""


def encode_row_prompt(
    row: pd.Series,
    task: TaskMetadata,
    question: QAInterface = None,
    custom_prompt_prefix: str = None,
    add_task_description: bool = True,
) -> str:
    """Encode a question regarding a given row."""
    # Get the question to ask
    question = question or task.question
    return (
        (ACS_TASK_DESCRIPTION + "\n" if add_task_description else "")
        + (f"\n{custom_prompt_prefix}\n" if custom_prompt_prefix else "")
        + f"""\
Information:
{task.get_row_description(row)}

{question.get_question_prompt()}""")


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
    """
    test_conversation = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "test"},
    ]
    try:
        tokenizer.apply_chat_template(test_conversation, tokenize=False)
        return True
    except Exception:
        return False


def encode_row_prompt_chat(
    row: pd.Series,
    task: TaskMetadata,
    tokenizer: AutoTokenizer,
    question: QAInterface = None,
    custom_prompt_prefix: str = None,
    chat_prompt: str = None,
    supports_system_prompt: bool = True,
    system_prompt: str = None,
    numeric: bool = False,
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
    question : QAInterface, optional
        The question interface to use.
    custom_prompt_prefix : str, optional
        A custom prompt prefix to prepend.
    chat_prompt : str, optional
        The assistant prefill text. If None, defaults to ANTHROPIC_CHAT_PROMPT
        for multiple-choice mode, or NUMERIC_CHAT_PROMPT ("0.") for numeric
        mode.
    supports_system_prompt : bool, optional
        Whether the model supports a system role, by default True.
    system_prompt : str, optional
        Custom system prompt text. If None, defaults to SYSTEM_PROMPT for
        multiple-choice mode, or NUMERIC_SYSTEM_PROMPT for numeric mode.
    numeric : bool, optional
        Whether numeric risk prompting is being used, by default False.
        Controls which defaults are used for chat_prompt and system_prompt.

    Returns
    -------
    str
        The fully formatted chat-template prompt.
    """
    user_content = encode_row_prompt(row, task, question=question, custom_prompt_prefix=custom_prompt_prefix)

    # Resolve chat_prompt default based on prompting mode
    if chat_prompt is None:
        chat_prompt = NUMERIC_CHAT_PROMPT if numeric else ANTHROPIC_CHAT_PROMPT

    # Resolve system_prompt default based on prompting mode
    if system_prompt is None:
        system_prompt = NUMERIC_SYSTEM_PROMPT if numeric else SYSTEM_PROMPT

    resolved_system_prompt = system_prompt if supports_system_prompt else None

    return apply_chat_template(
        tokenizer,
        user_prompt=user_content,
        system_prompt=resolved_system_prompt,
        chat_prompt=chat_prompt,
    )


def apply_chat_template(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    system_prompt: str = None,
    chat_prompt: str = None,
    **kwargs,
) -> str:
    # Add system prompt
    conversation = ([
        {"role": "system", "content": system_prompt}
    ] if system_prompt else [])

    # Add user prompt
    conversation.append({"role": "user", "content": user_prompt})

    if chat_prompt:
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

    if chat_prompt:
        # Make sure no special tokens follow the `CHAT_PROMPT`;
        # > some models add a newline character and/or a <end_of_turn> token
        filled_prompt = filled_prompt[: len(chat_prompt) + filled_prompt.rfind(chat_prompt)]

    return filled_prompt
