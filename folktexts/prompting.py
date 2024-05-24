"""Module to map risk-estimation questions to different prompting techniques.

e.g.,
- multiple-choice Q&A vs direct numeric Q&A;
- zero-shot vs few-shot vs CoT;
"""
import pandas as pd
from transformers import AutoTokenizer

from .datasets import Dataset


SYSTEM_PROMPT = """\
You are a helpful assistant. You answer multiple-choice questions based on the information provided.
"""

ACS_TASK_DESCRIPTION = """\
The following data corresponds to a survey respondent. \
Please answer the question based on the information provided. \
The data provided is enough to reach an approximate answer.
"""

ACS_FEW_SHOT_TASK_DESCRIPTION = """\
The following data corresponds to different survey respondents. \
Please answer each question based on the information provided. \
The data provided is enough to reach an approximate answer for each person.
"""

ANTHROPIC_CHAT_PROMPT = """If had to select one of the options, my answer would be"""
GEMMA_CHAT_PROMPT = """The provided information suggests that the answer is"""


def encode_row_prompt(
    row: pd.Series,
    dataset: Dataset,
    randomize: bool = False,
    add_task_description: bool = True,
) -> str:
    """Encode a question regarding a given row."""
    return (
        (ACS_TASK_DESCRIPTION + "\n" if add_task_description else "")
        + f"""\
Information:
{dataset.get_row_description(row)}

{dataset.question.get_question_and_answer_key(randomize=randomize)}""")


def encode_row_prompt_few_shot(
    row: pd.Series,
    dataset: Dataset,
    n_shots: int = 10,
    reuse_examples: bool = False,
) -> str:
    """Encode a question regarding a given row using few-shot prompting.

    Parameters
    ----------
    row : pd.Series
        The row that the question will be about.
    dataset : Dataset
        The dataset that the row belongs to.
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
    X_examples, y_examples = dataset.sample_n_train_examples(n_shots, reuse_examples=reuse_examples)

    # Start with task description
    prompt = ACS_FEW_SHOT_TASK_DESCRIPTION + "\n"

    # Add `n` example rows with respective labels
    for i in range(n_shots):
        prompt += (
            encode_row_prompt(X_examples.iloc[i], dataset=dataset, add_task_description=False)
            + f" {dataset.question.get_answer_key_from_value(y_examples.iloc[i])}"
            + "\n\n"
        )

    # Add the target row without its label
    prompt += encode_row_prompt(row, dataset=dataset, add_task_description=False)
    return prompt


def encode_row_prompt_chat(
    row: pd.Series,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    randomize: bool = False,
    **chat_template_kwargs,
) -> str:
    # TODO: implement two functions
    # - one for gemma-like models that are not compatible with system prompts
    # - and another for regular models compatible with system prompts
    return apply_chat_template(
        tokenizer,
        (
            SYSTEM_PROMPT
            + encode_row_prompt(row, dataset, randomize=randomize)
        ),
        **chat_template_kwargs,
    )


def apply_chat_template(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    system_prompt: str = None,
    chat_prompt: str = ANTHROPIC_CHAT_PROMPT,
    **kwargs,
) -> str:
    # Add system prompt
    conversation = ([
        {"role": "system", "content": system_prompt}
    ] if system_prompt else [])

    # Add user prompt
    conversation.append({"role": "user", "content": user_prompt})

    # Using the Anthropic-style chat prompt
    conversation.append({"role": "assistant", "content": chat_prompt})

    # Default kwargs
    kwargs.setdefault("add_generation_prompt", False)

    # Apply prompt template
    filled_prompt = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        **kwargs,
    )

    # Make sure no special tokens follow the `CHAT_PROMPT`;
    # > some models add a newline character and/or a <end_of_turn> token
    filled_prompt = filled_prompt[: len(chat_prompt) + filled_prompt.find(chat_prompt)]
    return filled_prompt
