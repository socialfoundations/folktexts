"""Module to map LLM outputs to risk-score estimates.

That is, different ways to decode LLM outputs into risk scores.
"""
import logging

import torch
from transformers import AutoTokenizer

from .questions import Question, Choice


# Minimum probability density assigned to all valid answers
# > small models will be worse at using valid answers...
ANSWER_PROB_THRESHOLD = 0.1


def get_answer_to_question(
    question: Question,
    last_token_probs: torch.Tensor,
    tokenizer: AutoTokenizer,
) -> dict[Choice: float]:
    """Decodes the model's last token probabilities into a choice for the given question.

    Parameters
    ----------
    question : Question
        The question to answer.
    last_token_probs : torch.Tensor
        The model's last token probabilities for the question.
    tokenizer : AutoTokenizer
        The tokenizer used to encode the question.

    Returns
    -------
    answers : dict[Choice, float]
        How much probability the model places on each answer choice.
    """
    # NOTE!
    # Most tokenizers will prefix output tokens with a space, unless the prompt
    # ends with a "\n" character.

    # Check which answer-key template has higher probability density
    # > e.g., "A" vs " A"

    def get_choice_token_id(choice: Choice, prefix: str = " ") -> int:
        return tokenizer.encode(
            f"{prefix}{question.choice_to_key[choice]}",
            return_tensors="pt",
            add_special_tokens=False,       # don't add BOS or EOS tokens
        ).flatten()[-1]

    prefixes = ["", " "]  # check both "A" and " A" templates

    # Map probabilities to choice values
    answers_per_prefix = {
        prf: {
            choice: last_token_probs[get_choice_token_id(choice, prefix=prf)].item()
            for choice in question.choices
        }
        for prf in prefixes
    }

    # Choose the prefix with the highest probability density
    best_prefix = max(answers_per_prefix, key=lambda prf: sum(answers_per_prefix[prf].values()))
    answers = answers_per_prefix[best_prefix]

    # Log prefix information in debug mode
    for prefix, choice_probs in answers_per_prefix.items():
        logging.debug(f"prefix='{prefix}' has density {sum(choice_probs.values()):.2%}")

    # Normalize probabilities to sum to 1
    answers_sum_prob = sum(answers.values())

    # Log total probability density assigned to answers
    (logging.warning if answers_sum_prob < ANSWER_PROB_THRESHOLD else logging.debug)(
        f"Answers have {answers_sum_prob:.2%} probability assigned."
    )

    return {
        choice: prob / answers_sum_prob
        for choice, prob in answers.items()
    }


def get_risk_estimate_from_answers(
    answers: dict[Choice, float],
) -> float:
    """Computes a risk estimate from the model's answer probabilities.

    Parameters
    ----------
    answers : dict[Choice, float]
        How much probability the model places on each answer choice.

    Returns
    -------
    float
        The risk estimate for the question.
    """
    sorted_choices_by_value = sorted(
        answers.keys(),
        key=lambda choice: choice.get_numeric_value(),
    )

    # If binary question, return probability of positive answer
    # > positive answer always has the highest numeric value
    if len(answers) == 2:
        positive_choice = sorted_choices_by_value[-1]
        return answers[positive_choice]

    # Compute risk estimate by summing weighted choices
    risk_estimate = sum(
        choice.get_numeric_value() * prob
        for choice, prob in answers.items()
    )

    logging.info(f"Risk estimate: {risk_estimate:.2f}")
    return risk_estimate
