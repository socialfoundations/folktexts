import logging
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Choice:
    """An answer choice in a question."""
    text: str
    value: str

    # A meaningful numeric value for the choice
    # e.g., if the choice is "25-34 years old", the numeric value could be 30
    numeric_value: float = None

    def get_numeric_value(self) -> float:
        """Returns the numeric value of the choice.
        May throw an error if the value was not provided and cannot be inferred.
        """
        return self.numeric_value if self.numeric_value is not None else float(self.value)


class Question:

    def __init__(
        self,
        text: str,
        choices: list[Choice],
        answer_keys: list[str] = None,
        use_numbers: bool = False,
        randomize_keys: bool = False,
        seed: int = 42,
    ):
        # Save question data
        self.text = text
        self.choices = choices

        # Construct rng
        self._rng = np.random.default_rng(seed)

        # Construct answer keys
        self.answer_keys = answer_keys
        if self.answer_keys is None:
            ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            n_choices = len(self.choices)
            self.answer_keys = (
                [str(i) for i in range(1, n_choices + 1)] if use_numbers
                else list(ALPHABET)[:n_choices]
            )

        if randomize_keys:
            self.answer_keys = self._rng.permutation(self.answer_keys)

        # Construct mapping from answer key to choice
        self._key_to_choice = dict(zip(self.answer_keys, self.choices))

    @property
    def key_to_choice(self) -> dict[str, Choice]:
        return self._key_to_choice

    @property
    def choice_to_key(self) -> dict[Choice, str]:
        return {choice: key for key, choice in self._key_to_choice.items()}

    @classmethod
    def make_question_from_value_map(
        cls,
        value_map: dict[str, str],
        attribute: str,
        **kwargs,
    ) -> "Question":
        """Constructs a question from a value map."""
        choices = [Choice(text, str(value)) for value, text in value_map.items()]

        # Set default question text
        kwargs.setdefault("text", f"What is this person's {attribute}?")

        return cls(
            choices=choices,
            **kwargs,
        )

    def get_value_to_text_map(self) -> dict[str, str]:
        """Returns the map from choice value to choice text."""
        return {choice.value: choice.text for choice in self.choices}

    def get_answer_key_from_value(self, value: str) -> Choice:
        """Returns the answer key corresponding to the given data value.
        """
        for choice in self.choices:
            if choice.value == value:
                return self.choice_to_key[choice]

        logging.error(f"Could not find choice for value: {value}")
        return None

    def get_answer_from_text(self, text: str) -> Choice:
        text = text.strip().upper()
        if text in self._key_to_choice:
            return self._key_to_choice[text]

        logging.error(f"Could not find answer for text: {text}")
        return None

    def get_question_and_answer_key(
        self,
        randomize: bool = False,
    ) -> str:
        ordered_keys = self.answer_keys if not randomize else self._rng.permutation(self.answer_keys)

        choice_str = "\n".join(
            f"{key}. {self._key_to_choice[key].text}."
            for key in ordered_keys
        )

        return (f"""\
Question: {self.text}
{choice_str}
Answer:""")
