"""Interface for question-answering with LLMs.

- Create different types of questions (direct numeric, multiple-choice).
- Encode questions and decode model outputs.
- Compute risk-estimate from model outputs.
"""
from __future__ import annotations

import dataclasses
import itertools
import logging
import re
from abc import ABC
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ._utils import hash_dict

# Minimum probability density assigned to all valid answers
# > small models will be worse at using valid answers...
ANSWER_PROB_THRESHOLD = 0.1

# Default answer keys for multiple-choice questions
_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class QAInterface(ABC):
    """An interface for a question-answering system."""

    column: str
    text: str
    num_forward_passes: int

    def get_question_prompt(self) -> str:
        """Returns a question and answer key."""
        raise NotImplementedError

    def get_answer_from_model_output(
        self,
        last_token_probs: np.ndarray,
        tokenizer_vocab: dict[str, int],
    ) -> float:
        """Decodes the model's output into an answer for the given question.

        Parameters
        ----------
        last_token_probs : np.ndarray
            The model's last token probabilities for the question. The first
            dimension corresponds to the number of forward passes as specified
            by `self.num_forward_passes`.
        tokenizer : dict[str, int]
            The tokenizer's vocabulary.

        Returns
        -------
        answer : float
            The answer to the question.
        """
        raise NotImplementedError

    def __hash__(self) -> int:
        return int(hash_dict(dataclasses.asdict(self)), 16)


@dataclass(frozen=True)
class DirectNumericQA(QAInterface):
    """Represents a direct numeric question.

    Notes
    -----
    For example, the prompt could be "
    Q: What is 2 + 2?
    A: "
    With the expected answer being "4".

    If looking for a direct numeric probability, the answer prompt will be
    framed as so: "
    Q: What is the probability, between 0 and 1, of getting heads on a coin flip?
    A: 0."
    So that we can extract a numeric answer with at most 2 forward passes.
    This is done automatically by passing the kwarg `answer_probability=True`.

    Note that some models have multi-digit tokens in their vocabulary, so we
    need to correctly assess which tokens in the vocabulary correspond to valid
    numeric answers.
    """

    num_forward_passes: int = 2     # NOTE: overrides superclass default
    answer_probability: bool = True

    def get_question_prompt(self) -> str:
        question_prompt = f"""Question: {self.text}\n"""
        if self.answer_probability:
            question_prompt += "Answer (between 0 and 1): 0."
        else:
            question_prompt += "Answer: "

        return question_prompt

    def _get_numeric_tokens(self, tokenizer_vocab: dict[str, int]) -> dict[str, int]:
        """Returns the indices of tokens that correspond to numbers.

        This can include digits ("0"-"9"), multi-digit tokens (e.g., "100"), and
        the decimal point (".").
        """
        numeric_tokens = {
            key: token_id for key, token_id in tokenizer_vocab.items()
            if key.isdigit()
        }

        if "." in tokenizer_vocab:
            numeric_tokens["."] = tokenizer_vocab["."]

        return numeric_tokens

    def get_answer_from_model_output(
        self,
        last_token_probs: np.ndarray,
        tokenizer_vocab: dict[str, int],
    ) -> float | int:
        """Outputs a numeric answer inferred from the model's output.

        Parameters
        ----------
        last_token_probs : np.ndarray
            The last token probabilities of the model for the question.
            The first dimension must correspond to the number for forward passes
            as specified by `num_forward_passes`.
        tokenizer_vocab: dict[str, int],
            The tokenizer's vocabulary.

        Returns
        -------
        answer : float | int
            The numeric answer to the question.

        Notes
        -----
        Eventually we could run a search algorithm to find the most likely
        answer over multiple forward passes, but for now we'll just take the
        argmax on each forward pass.
        """
        numeric_tokens_vocab = self._get_numeric_tokens(tokenizer_vocab)

        if len(last_token_probs) < self.num_forward_passes:
            logging.info(
                f"Expected {self.num_forward_passes} forward passes, got "
                f"{len(last_token_probs)}.")

        answer_text = ""
        for ltp in last_token_probs:
            # Get the probability of each numeric token
            num_tokens_probs = {
                num_token: ltp[token_id] if isinstance(ltp[token_id], float) else ltp[token_id].item()
                for num_token, token_id in numeric_tokens_vocab.items()
            }

            # Get the most likely numeric token
            most_likely_numeric_token = max(num_tokens_probs, key=num_tokens_probs.get)
            answer_text += str(most_likely_numeric_token)

            logging.debug(f"Total prob. assigned to numeric tokens: {sum(num_tokens_probs.values()):.2%}")

        # Filter out any non-numeric characters
        match_ = re.match(r"[-+]?\d*\.\d+|\d+", answer_text)
        assert match_, f"Could not find numeric answer in '{answer_text}'."
        numeric_answer_text = match_.group()

        if self.answer_probability and "." not in numeric_answer_text:
            return float(f"0.{numeric_answer_text}")
        else:
            return float(numeric_answer_text)


@dataclass(frozen=True, eq=True)
class Choice:
    """Represents a choice in multiple-choice Q&A.

    Attributes
    ----------
    text : str
        The text of the choice. E.g., "25-34 years old".
    data_value : object
        The categorical value corresponding to this choice in the data.
    numeric_value : float, optional
        A meaningful numeric value for the choice. E.g., if the choice is "25-34
        years old", the numeric value could be 30. The choice with the highest
        numeric value can be used as a proxy for the positive class. If not
        provided, will try to use the `choice.value`.
    """
    text: str
    data_value: object
    numeric_value: float = None

    def get_numeric_value(self) -> float:
        """Returns the numeric value of the choice."""
        return self.numeric_value if self.numeric_value is not None else float(self.data_value)


@dataclass(frozen=True, eq=True)    # NOTE: kw_only=True requires Python 3.10
class MultipleChoiceQA(QAInterface):
    """Represents a multiple-choice question and its answer keys."""

    num_forward_passes: int = 1     # NOTE: overrides superclass default
    choices: tuple[Choice] = dataclasses.field(default_factory=tuple)
    _answer_keys_source: tuple[str] = dataclasses.field(default_factory=lambda: tuple(_ALPHABET))

    def __post_init__(self):
        if not self.choices:
            raise ValueError("Choices must be provided.")
        if len(self.choices) > len(self._answer_keys_source):
            raise ValueError("Number of choices must be less than or equal to the number of answer keys.")

    def __hash__(self) -> int:
        return int(hash_dict(dataclasses.asdict(self)), 16)

    @classmethod
    def create_question_from_value_map(
        cls,
        column: str,
        value_map: dict[str, str],
        attribute: str,
        **kwargs,
    ) -> "MultipleChoiceQA":
        """Constructs a question from a value map."""
        choices = tuple(Choice(text, str(value)) for value, text in value_map.items())

        # Set default question text
        kwargs.setdefault("text", f"What is this person's {attribute}?")

        return cls(
            column=column,
            choices=choices,
            **kwargs,
        )

    @classmethod
    def create_answer_keys_permutations(cls, question: "MultipleChoiceQA") -> Iterator["MultipleChoiceQA"]:
        """Yield questions with all permutations of answer keys.

        Parameters
        ----------
        question : Question
            The template question whose answer keys will be permuted.

        Returns
        -------
        permutations : Iterator[Question]
            A generator of questions with all permutations of answer keys.
        """
        for perm in itertools.permutations(question.choices):
            yield dataclasses.replace(question, choices=perm)

    @property
    def answer_keys(self) -> tuple[str, ...]:
        return self._answer_keys_source[:len(self.choices)]

    @property
    def key_to_choice(self) -> dict[str, Choice]:
        return dict(zip(self.answer_keys, self.choices))

    @property
    def choice_to_key(self) -> dict[Choice, str]:
        return {choice: key for key, choice in self.key_to_choice.items()}

    def get_value_to_text_map(self) -> dict[object, str]:
        """Returns the map from choice data value to choice textual representation."""
        return {choice.data_value: choice.text for choice in self.choices}

    def get_answer_key_from_value(self, value: object) -> str:
        """Returns the answer key corresponding to the given data value.
        """
        for choice in self.choices:
            if choice.data_value == value:
                return self.choice_to_key[choice]

        logging.error(f"Could not find choice for value: {value}")
        return None

    def get_answer_from_text(self, text: str) -> Choice:
        text = text.strip().upper()
        if text in self.key_to_choice:
            return self.key_to_choice[text]

        logging.error(f"Could not find answer for text: {text}")
        return None

    def get_question_prompt(self) -> str:
        choice_str = "\n".join(
            f"{key}. {choice.text}."
            for key, choice in self.key_to_choice.items()
        )

        return (f"""\
Question: {self.text}
{choice_str}
Answer:""")

    def _decode_model_output_to_choice_distribution(
        self,
        last_token_probs: np.ndarray,
        tokenizer_vocab: dict[str, int],
    ) -> float:
        """Decodes the model's output into an answer distribution.

        Parameters
        ----------
        last_token_probs : np.ndarray
            The model's last token probabilities for the question.
        tokenizer_vocab: dict[str, int],
            The tokenizer's vocabulary.

        Returns
        -------
        answers : dict[Choice, float]
            How much probability the model places on each answer choice.

        Notes
        -----
        Answer-key tokens may be prefixed with a space, so we need to check
        both "A" and " A" templates.
        """
        def _get_choice_token_id(choice: Choice, prefix: str = " ") -> int:
            choice_answer_text = f"{prefix}{self.choice_to_key[choice]}"
            if choice_answer_text in tokenizer_vocab:
                return tokenizer_vocab[choice_answer_text]
            else:
                return None

        # Different models may use different prefixes to represent white space
        # or word boundaries; here we try a few common ones
        prefixes = ["", " ", "_", "\u2581", "\u0120", "\u010A"]

        # Map probabilities to choice values
        answers_per_prefix = {
            prf: {
                choice: last_token_probs[choice_token_id].item()
                for choice in self.choices
                if (choice_token_id := _get_choice_token_id(choice, prefix=prf)) is not None
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
        msg = f"Answers have {answers_sum_prob:.2%} probability assigned."
        if answers_sum_prob < ANSWER_PROB_THRESHOLD:
            id_to_tok = {v: k for k, v in tokenizer_vocab.items()}
            argmax_token = id_to_tok[np.argmax(last_token_probs)]
            logging.warning(msg + f" Argmax token: '{argmax_token}'.")
        else:
            logging.debug(msg)

        return {
            choice: prob / answers_sum_prob
            for choice, prob in answers.items()
        }

    def get_answer_from_model_output(
        self,
        last_token_probs: np.ndarray,
        tokenizer_vocab: dict[str, int],
    ) -> float:
        """Decodes the model's output into an answer for the given question.

        Parameters
        ----------
        last_token_probs : np.ndarray
            The model's last token probabilities for the question. The first
            dimension corresponds to the number of forward passes as specified
            by `self.num_forward_passes`.
        tokenizer_vocab: dict[str, int],
            The tokenizer's vocabulary.

        Returns
        -------
        answer : float
            The answer to the question.
        """
        if last_token_probs.ndim > 1:
            if last_token_probs.shape[0] > 1:
                logging.warning(
                    f"Multiple ({last_token_probs.shape[0]}) forward passes "
                    f"detected: using only the first pass.")

            # Using only 1st forward pass results
            last_token_probs = last_token_probs[0]

        answers = self._decode_model_output_to_choice_distribution(
            last_token_probs=last_token_probs,
            tokenizer_vocab=tokenizer_vocab,
        )

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

        logging.debug(f"Risk estimate: {risk_estimate:.2f}")
        return risk_estimate
