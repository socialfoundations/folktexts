"""Module to map risk-estimation questions to different prompting techniques.

e.g.,
- multiple-choice Q&A vs direct numeric Q&A;
- zero-shot vs few-shot vs CoT;
"""
import math
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


def sample_n_shots(dataset: Dataset, n: int, seed: int = None) -> tuple[pd.DataFrame, pd.Series]:
    """Return a balanced set of samples from the training set."""
    X_train, y_train = dataset.get_train()  # TODO: speed this up!

    n_samples = X_train.sample(n=n, random_state=seed)
    return n_samples, y_train.loc[n_samples.index]

    # # Sample `n` examples in total, stratified per class
    # n_per_class = [
    #     math.floor(n / len(y_train.unique())),
    #     math.ceil(n / len(y_train.unique())),
    # ]
    # assert sum(n_per_class) == n
    # n_samples = pd.concat([
    #     X_train[y_train == label].sample(n=curr_n, random_state=seed)
    #     for label, curr_n in zip(y_train.unique(), n_per_class)
    # ])

    # # Return samples and corresponding labels
    # return n_samples, y_train.loc[n_samples.index]


def encode_row_prompt_few_shot(
    row: pd.Series,
    dataset: Dataset,
    n_shots: int = 3,
    seed: int = 42,
) -> str:
    """Encode a question regarding a given row using few-shot prompting.

    NOTE: the seed should be changed for each new model query, to ensure that
    we don't reuse the same examples over and over.

    Parameters
    ----------
    row : pd.Series
        The row that the question will be about.
    dataset : Dataset
        The dataset that the row belongs to.
    n_shots : int, optional
        The number of example questions and answers to use before prompting
        about the given row, by default 3.
    seed : int, optional
        The random seed to use for sampling the few-shot examples, by default 42.
        Using the same seed will result in the same examples being sampled.

    Returns
    -------
    prompt : str
        The encoded few-shot prompt.
    """
    # Take `n_shots` random samples from the train set
    X_examples, y_examples = sample_n_shots(dataset, n=n_shots, seed=seed)

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
