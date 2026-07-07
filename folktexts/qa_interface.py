"""Interface for question-answering with LLMs.

- Create different types of questions (direct numeric, multiple-choice, chain-of-thought).
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
from typing import ClassVar, Iterator

import numpy as np

from ._utils import hash_dict

# Minimum probability density assigned to all valid answers
# > small models will be worse at using valid answers...
ANSWER_PROB_THRESHOLD = 0.1

# Default answer keys for multiple-choice questions
_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ---------------------------------------------------------------------------
# Default system / chat prompts — owned here so each QAInterface subclass can
# declare its own defaults without importing from prompting.py (which imports
# from this module, which would create a circular dependency).
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful assistant. You answer multiple-choice questions \
based on the information provided. Respond with a single answer choice.
"""

NUMERIC_SYSTEM_PROMPT = """\
You are a helpful assistant. You provide numeric probability \
estimates based on the information provided.
"""
ANTHROPIC_CHAT_PROMPT = "If had to select one of the options, my answer would be"
GEMMA_CHAT_PROMPT = "The provided information suggests that the answer is"


# NOTE: The leading `0.` is part of the prefill, so the model only generates
# the digits after the decimal point. This caps the expressible probability
# at the open interval [0, 1) — true posteriors at or near 1.0 cannot be
# emitted exactly. If you need full [0, 1] coverage, override `chat_prompt`
# with e.g. `"Answer (between 0 and 1): "` and let the model produce the
# leading digit itself (note that this also widens the digit-scoring search
# space and may degrade calibration for low-probability cases).
NUMERIC_CHAT_PROMPT = "Answer (between 0 and 1): 0."

# Alternative prefill used when `DirectNumericQA(percentage=True)` is active:
# no `0.` prefill; the model is asked to reply with an integer percentage.
# Rationale: OpenAI gpt-5 under `reasoning_effort='none'` collapses the
# decimal-prefill prompt to `'0'` regardless of input (AUC≈chance); reframing
# as an integer percentage restores discrimination because both digits become
# meaningful (probe on gpt-5.4-nano: HIGH-income row → '92', LOW → '12').
NUMERIC_PERCENTAGE_CHAT_PROMPT = "Probability (0-100): "


@dataclass(frozen=True)
class QAInterface(ABC):
    """An interface for a question-answering system."""

    column: str
    text: str
    num_forward_passes: int

    # Subclasses override these to declare their mode-appropriate defaults.
    # `None` means "no default" (i.e. no system prompt / no chat prefill).
    default_system_prompt: ClassVar[str | None] = SYSTEM_PROMPT
    default_chat_prompt: ClassVar[str | None] = ANTHROPIC_CHAT_PROMPT

    # Default sampling temperature for *text-generation* prompting, read via
    # `LLMClassifier._resolve_temperature`. Only meaningful for
    # `ChainOfThoughtQA` (which overrides it): token-probability methods
    # (multiple-choice, direct-numeric) read the untempered next-token
    # distribution on every backend and never sample, so temperature does not
    # apply to them. A `ClassVar` (not a dataclass field) so it does not
    # affect the frozen dataclass hash / result-cache identity.
    default_temperature: ClassVar[float] = 0.0

    def get_answer_prefix(self) -> str:
        """Returns the answer label that follows the question (e.g. 'Answer:')."""
        raise NotImplementedError

    def get_question_prompt(self, with_answer_prefill: bool = True) -> str:
        """Returns the question text.

        `with_answer_prefill=True` (the default) bakes the answer prefill into
        the returned string — required by the zero-shot / few-shot last-token
        scoring path, which reads probabilities from the very next token after
        the prefill. Set to `False` for chat-template prompting, where the
        prefill is supplied separately as the assistant turn (otherwise the
        same string ends up emitted twice and silently degrades scoring).
        """
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

    num_forward_passes: int = 2  # NOTE: overrides superclass default
    answer_probability: bool = True
    percentage: bool = False
    """Ask the model for an integer percentage (0-100) instead of a decimal
    with a `0.` prefill; the decoded value is divided by 100. Used to work
    around OpenAI gpt-5 models degenerating to `'0'` under the decimal-prefill
    prompt when `reasoning_effort='none'` is required for logprobs.
    Auto-enabled by `WebAPILLMClassifier` for gpt-5 family models."""

    default_system_prompt: ClassVar[str] = NUMERIC_SYSTEM_PROMPT
    default_chat_prompt: ClassVar[str] = NUMERIC_CHAT_PROMPT

    def get_answer_prefix(self) -> str:
        if self.percentage:
            return NUMERIC_PERCENTAGE_CHAT_PROMPT
        if self.answer_probability:
            return "Answer (between 0 and 1): 0."
        return "Answer: "

    def get_question_prompt(self, with_answer_prefill: bool = True) -> str:
        question_prompt = f"Question: {self.text}"
        if with_answer_prefill:
            question_prompt += f"\n{self.get_answer_prefix()}"

        return question_prompt

    def _get_numeric_tokens(
        self,
        tokenizer_vocab: dict[str, int],
        vocab_dim: int,
    ) -> dict[str, int]:
        """Returns the indices of tokens that correspond to numbers.

        This can include digits ("0"-"9"), multi-digit tokens (e.g., "100"), and
        the decimal point (".").

        Token ids are filtered to `< vocab_dim` (the model's logits axis); some
        tokenizer families place added/special tokens beyond the base vocab,
        and the caller indexes `last_token_probs` by these ids.

        Parameters
        ----------
        tokenizer_vocab : dict[str, int]
            The tokenizer vocabulary mapping token strings to token IDs.
        vocab_dim : int
            Size of the model's logits axis. Token IDs >= vocab_dim are excluded
            (some tokenizer families place added/special tokens beyond the base
            vocab, and the caller indexes ``last_token_probs`` by these IDs).

        Returns
        -------
        dict[str, int]
            Mapping from numeric token string to token ID, filtered to
            ``token_id < vocab_dim``.

        """
        numeric_tokens = {
            key: token_id
            for key, token_id in tokenizer_vocab.items()
            if key.isdigit() and token_id < vocab_dim
        }

        if "." in tokenizer_vocab and tokenizer_vocab["."] < vocab_dim:
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
        numeric_tokens_vocab = self._get_numeric_tokens(
            tokenizer_vocab,
            vocab_dim=last_token_probs.shape[-1],
        )

        if len(last_token_probs) < self.num_forward_passes:
            logging.info(
                f"Expected {self.num_forward_passes} forward passes, got "
                f"{len(last_token_probs)}."
            )

        # Percentage mode uses its own decoder: the model prefixes markdown
        # around the digit, so a per-position argmax over the entire response
        # captures noise. Return early — the classic sequential-digit decoder
        # below assumes a `0.<digit><digit>...` layout that percentage mode
        # violates.
        if self.percentage:
            return self._decode_percentage_expected_value(
                last_token_probs, numeric_tokens_vocab, tokenizer_vocab
            )

        answer_text = ""
        for ltp in last_token_probs:
            # Get the probability of each numeric token
            num_tokens_probs = {
                num_token: ltp[token_id]
                if isinstance(ltp[token_id], float)
                else ltp[token_id].item()
                for num_token, token_id in numeric_tokens_vocab.items()
            }

            # Get the most likely numeric token
            most_likely_numeric_token = max(num_tokens_probs, key=num_tokens_probs.get)
            answer_text += str(most_likely_numeric_token)

            logging.debug(
                f"Total prob. assigned to numeric tokens: {sum(num_tokens_probs.values()):.2%}"
            )

        # Filter out any non-numeric characters
        match_ = re.match(r"[-+]?\d*\.\d+|\d+", answer_text)
        assert match_, f"Could not find numeric answer in '{answer_text}'."
        numeric_answer_text = match_.group()

        if self.answer_probability and "." not in numeric_answer_text:
            return float(f"0.{numeric_answer_text}")
        else:
            return float(numeric_answer_text)

    # Preamble anchor for percentage-mode decoding. Matches the `probability`
    # keyword (case-insensitive) followed by a colon at any distance, which
    # is where all gpt-5-style responses (and the underlying prompt
    # `NUMERIC_PERCENTAGE_CHAT_PROMPT`) place their answer delimiter:
    #   - `**Probability (0-100): 28%**`      -> `Probability (0-100):`
    #   - `**Estimated probability: 18%**`    -> `probability:`
    #   - `**Probability (above $50,000): ~62%**` -> `Probability (above $50,000):`
    # `[^:]*` deliberately stops at the FIRST colon after `probability`,
    # so any digit tokens the model emits as prompt echo (the `0` and `100`
    # in `(0-100)`) precede the anchor and are correctly skipped.
    _ANSWER_ANCHOR: ClassVar[re.Pattern] = re.compile(
        r"probability[^:]*:", re.IGNORECASE
    )

    def _decode_percentage_expected_value(
        self,
        last_token_probs: np.ndarray,
        numeric_tokens_vocab: dict[str, int],
        tokenizer_vocab: dict[str, int],
    ) -> float:
        """Decode a percentage-mode response deterministically.

        Real gpt-5 responses either echo the prompt's `(0-100)` range
        label (`'**Probability (0-100): 28%**'`) or paraphrase it
        (`'**Estimated probability: 18%**'`) before the answer digit.
        Argmax-per-position over the whole response is noisy: the `0`
        and `100` from the range label carry high probability and can
        hijack the "best answer position" selection.

        Algorithm:
          1. Reconstruct the chosen-token stream via argmax per
             position (temperature=0 guarantees chosen == argmax).
          2. Walk positions accumulating text; once the accumulated
             text matches `_ANSWER_ANCHOR` (`probability<...>:`), mark
             the anchor as consumed.
          3. Return the mass-weighted expected value of integer-percent
             tokens at the first *post-anchor* position that carries
             meaningful numeric mass.

        Falls back to 0.0 when either no anchor appears (the model
        never framed the answer as a probability — usually a preamble
        that never reached the digit) or no post-anchor position
        carries digit mass.
        """
        # Restrict to tokens that could be an integer percentage answer.
        # Rejects: multi-digit tokens with a `.`, values >100, non-digit
        # strings that slipped through _get_numeric_tokens.
        valid_percent_tokens: dict[str, int] = {}
        for tok, tid in numeric_tokens_vocab.items():
            if not re.fullmatch(r"\d{1,3}", tok):
                continue
            value = int(tok)
            if 0 <= value <= 100:
                valid_percent_tokens[tok] = tid
        if not valid_percent_tokens:
            logging.warning(
                "No valid percentage tokens (integers 0-100) in the model's "
                "response; returning 0.0."
            )
            return 0.0

        # Reconstruct chosen tokens per position to walk the response text
        # deterministically. `tokenizer_vocab` on the WebAPI backend is a
        # synthetic map (token string → sequential id) built from the top-K
        # entries actually returned for this row, so the inverse is total
        # on the ids that appear as argmax.
        inverse_vocab = {tid: tok for tok, tid in tokenizer_vocab.items()}

        passed_anchor = False
        accumulated_text = ""
        for ltp in last_token_probs:
            chosen_id = int(np.argmax(ltp))
            chosen_tok = inverse_vocab.get(chosen_id, "")
            accumulated_text += chosen_tok

            if not passed_anchor:
                if self._ANSWER_ANCHOR.search(accumulated_text):
                    passed_anchor = True
                # Everything before/at the anchor is prompt echo (label
                # digits like `0`/`100`, punctuation, preamble words) —
                # never the answer, even when it's a valid percentage.
                continue

            probs_by_value: dict[int, float] = {}
            for tok, tid in valid_percent_tokens.items():
                p = ltp[tid]
                if not isinstance(p, float):
                    p = p.item()
                if p > 0:
                    probs_by_value[int(tok)] = probs_by_value.get(int(tok), 0.0) + p
            total_mass = sum(probs_by_value.values())
            # Positions between the label and the answer (whitespace,
            # `):`, `%`, etc.) carry no numeric mass; skip them.
            if total_mass < 0.1:
                continue

            expected = sum(v * p for v, p in probs_by_value.items()) / total_mass
            return min(expected / 100.0, 1.0)

        logging.warning(
            "No post-anchor digit position found in percentage-mode "
            "response (accumulated_text=%r); returning 0.0.",
            accumulated_text,
        )
        return 0.0


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
        return (
            self.numeric_value
            if self.numeric_value is not None
            else float(self.data_value)
        )


@dataclass(frozen=True, eq=True)  # NOTE: kw_only=True requires Python 3.10
class MultipleChoiceQA(QAInterface):
    """Represents a multiple-choice question and its answer keys."""

    num_forward_passes: int = 1  # NOTE: overrides superclass default
    choices: tuple[Choice] = dataclasses.field(default_factory=tuple)
    _answer_keys_source: tuple[str] = dataclasses.field(
        default_factory=lambda: tuple(_ALPHABET)
    )

    def __post_init__(self):
        if not self.choices:
            raise ValueError("Choices must be provided.")
        if len(self.choices) > len(self._answer_keys_source):
            raise ValueError(
                "Number of choices must be less than or equal to the number of answer keys."
            )

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
    def create_answer_keys_permutations(
        cls, question: "MultipleChoiceQA"
    ) -> Iterator["MultipleChoiceQA"]:
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
        return self._answer_keys_source[: len(self.choices)]

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
        """Returns the answer key corresponding to the given data value."""
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

    def get_answer_prefix(self) -> str:
        return "Answer:"

    def get_question_prompt(self, with_answer_prefill: bool = True) -> str:
        choice_str = "\n".join(
            f"{key}. {choice.text}." for key, choice in self.key_to_choice.items()
        )

        prompt = f"Question: {self.text}\n{choice_str}"
        if with_answer_prefill:
            prompt += f"\n{self.get_answer_prefix()}"
        return prompt

    def _decode_model_output_to_choice_distribution(
        self,
        last_token_probs: np.ndarray,
        tokenizer_vocab: dict[str, int],
    ) -> dict[Choice, float]:
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
        prefixes = ["", " ", "_", "\u2581", "\u0120", "\u010a"]

        # Map probabilities to choice values
        answers_per_prefix = {
            prf: {
                choice: last_token_probs[choice_token_id].item()
                for choice in self.choices
                if (choice_token_id := _get_choice_token_id(choice, prefix=prf))
                is not None
            }
            for prf in prefixes
        }

        # Choose the prefix with the highest probability density
        best_prefix = max(
            answers_per_prefix, key=lambda prf: sum(answers_per_prefix[prf].values())
        )
        answers = answers_per_prefix[best_prefix]

        # Log prefix information in debug mode
        for prefix, choice_probs in answers_per_prefix.items():
            logging.debug(
                f"prefix='{prefix}' has density {sum(choice_probs.values()):.2%}"
            )

        # Normalize probabilities to sum to 1
        answers_sum_prob = sum(answers.values())

        # Log total probability density assigned to answers
        msg = f"Answers have {answers_sum_prob:.2%} probability assigned."
        if answers_sum_prob < ANSWER_PROB_THRESHOLD:
            id_to_tok = {v: k for k, v in tokenizer_vocab.items()}
            argmax_id = int(np.argmax(last_token_probs))
            argmax_token = id_to_tok.get(argmax_id, f"<id={argmax_id}>")
            logging.warning(msg + f" Argmax token: '{argmax_token}'.")
        else:
            logging.debug(msg)

        # No mass on any choice token — happens when the top-K logprobs cap
        # excludes all answer-letter variants (vLLM/WebAPI), or with extreme
        # FP16 underflow on transformers. Fall back to uniform over the QA's
        # declared choices: same effect as the model saying "I don't know."
        if answers_sum_prob <= 0 or not answers:
            n = len(self.choices)
            return {choice: 1.0 / n for choice in self.choices}

        return {choice: prob / answers_sum_prob for choice, prob in answers.items()}

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
                    f"detected: using only the first pass."
                )

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
            choice.get_numeric_value() * prob for choice, prob in answers.items()
        )

        logging.debug(f"Risk estimate: {risk_estimate:.2f}")
        return risk_estimate


# Regex patterns for extracting probability from generated text
# Matches formats like: "Probability: 80%", "Probability: 0.80", "probability: 80 percent"
# Patterns are ordered by specificity - more specific patterns first
_PROBABILITY_PATTERNS = [
    # Match "Probability: X%" or "probability: X%" (with optional "is", "of", etc.)
    r"[Pp]robability(?:\s+(?:is|of|estimate)?)?[:\s]+(\d+(?:\.\d+)?)\s*%",
    # Match "Probability: 0.XX" or "probability: 0.XX" or "Probability: 1.0"
    r"[Pp]robability(?:\s+(?:is|of|estimate)?)?[:\s]+(\d*\.?\d+)(?![%\d])",
    # Match "X%" anywhere in text (prefer later matches in fallback)
    r"(\d+(?:\.\d+)?)\s*%",
    # Match "X percent" pattern
    r"(\d+(?:\.\d+)?)\s+percent",
    # Match standalone decimal (0.XX or .XX) that looks like probability
    r"(?<![.\d])(0?\.\d+)(?![.\d])",
]


@dataclass(frozen=True)
class ChainOfThoughtQA(QAInterface):
    """A chain-of-thought (CoT) question interface.

    The model is instructed to reason step-by-step in free-form text and end
    with an explicit `Probability: X%` line; the probability is recovered via
    regex. This works on any model regardless of chat template.

    Orthogonal to the tokenizer's `enable_thinking` chat-template kwarg: CoT
    prompting always uses free-form generation, and `enable_thinking=True`
    additionally activates the `<think>...</think>` block on tokenizers that
    support it (e.g., Qwen3-Thinking) — the block is stripped before regex
    extraction.

    Notes
    -----
    Unlike `DirectNumericQA` and `MultipleChoiceQA` which use token
    probabilities, this interface uses full text generation. The
    `num_forward_passes` is set to -1 to signal text-generation mode instead
    of token-probability extraction.

    The regex extraction is flexible and accepts multiple formats:
    - "Probability: 80%" -> 0.80
    - "Probability: 0.80" -> 0.80
    - "Probability: 80 percent" -> 0.80
    - "... 75%" (at end of text) -> 0.75

    Attributes
    ----------
    enable_thinking : bool
        Whether to enable thinking mode for tokenizers that support it (e.g.,
        Qwen3). When True, the tokenizer's `apply_chat_template` is called
        with `enable_thinking=True`. Default is False.
    """

    # CoT instructions are baked into get_question_prompt; no system prompt by default.
    # No chat prefill either — CoT generates free-form text, not a fixed-prefix token.
    default_system_prompt: ClassVar[str | None] = None
    default_chat_prompt: ClassVar[str | None] = None

    num_forward_passes: int = -1  # -1 signals text generation mode
    # Thinking-mode models (e.g., Qwen3-Thinking) need >= 8000 tokens to reliably
    # close `</think>` and emit the final answer; 5000 leaves ~13% of rows unfinished.
    max_new_tokens: int = 8000
    enable_thinking: bool = False

    @property
    def default_temperature(self) -> float:
        """Greedy (0.0) for plain CoT; 1.0 in thinking mode.

        Sampling at temperature 1.0 makes small/instruct models lose the
        'Probability: X%' output format far more often (42% vs 13% regex
        fallbacks on Llama-3.2-1B, collapsing AUC to ~0.5), so plain CoT stays
        greedy — matching the pre-existing behavior. Thinking-mode models are
        the exception: greedy decoding is explicitly discouraged for them
        (e.g. Qwen3-Thinking). Overridable via `LLMClassifier(temperature=...)`
        / `--temperature`.
        """
        return 1.0 if self.enable_thinking else 0.0

    def get_question_prompt(self, with_answer_prefill: bool = True) -> str:
        """Returns the CoT question prompt.

        The `with_answer_prefill` parameter is accepted for interface
        compatibility with `QAInterface` but has no effect: CoT prompts
        produce free-form text and have no answer prefill to strip.
        """
        return f"""\
Question: {self.text}

Think step-by-step about the factors that could influence the answer to this question. \
After reasoning through the relevant information, provide your final probability estimate.

Your response MUST end with your probability estimate in the following format:
Probability: X%
where X is a number between 0 and 100.

Reasoning:"""

    @staticmethod
    def extract_probability_from_text(generated_text: str) -> float | None:
        """Extract a probability value from generated text using regex patterns.

        The extraction prioritizes (in order): the explicit "Probability: X[%]"
        anchor, last loose percentage, "X percent", then a bare 0.XX decimal.
        Returns a float in [0, 1] or None if nothing matched.
        """
        # First, try the explicit "Probability: X[%]" anchor (most reliable).
        explicit_patterns = [
            (
                r"[Pp]robability(?:\s+(?:is|of|estimate)?)?[:\s]+(\d+(?:\.\d+)?)\s*%",
                True,
            ),
            (
                r"[Pp]robability(?:\s+(?:is|of|estimate)?)?[:\s]+(\d*\.?\d+)(?![%\d])",
                False,
            ),
        ]
        for pattern, percent_form in explicit_patterns:
            value = ChainOfThoughtQA._extract_last_probability(
                generated_text,
                pattern,
                percent_form=percent_form,
            )
            if value is not None:
                logging.debug(
                    f"Extracted probability {value:.2%} using pattern: {pattern}"
                )
                return value

        # Fallback ladder: any percentage, "X percent", or a bare 0.XX decimal.
        # `flags=0` for the percent forms; "percent" pattern is case-insensitive.
        for pattern, percent_form, flags in [
            (r"(\d+(?:\.\d+)?)\s*%", True, 0),
            (r"(\d+(?:\.\d+)?)\s+percent", True, re.IGNORECASE),
            (r"(?<![.\d])(0?\.\d+)(?![.\d])", False, 0),
        ]:
            value = ChainOfThoughtQA._extract_last_probability(
                generated_text,
                pattern,
                percent_form=percent_form,
                flags=flags,
            )
            if value is not None:
                logging.debug(f"Used fallback extraction: {value:.2%}")
                return value

        snippet = (
            generated_text[:250] + "..." + generated_text[-250:]
            if len(generated_text) > 500
            else generated_text
        )
        logging.error(f"Could not extract probability from text:\n{snippet}")
        return None

    @staticmethod
    def _extract_last_probability(
        text: str,
        pattern: str,
        *,
        percent_form: bool,
        flags: int = 0,
    ) -> float | None:
        """Apply `pattern` to `text`, take the last match, and return it as a
        probability in [0, 1] (dividing by 100 if `percent_form`) or None.

        The "last match" rule matters: models often revise their estimate
        mid-reasoning, and the final value is the one we want to trust.
        """
        matches = re.findall(pattern, text, flags=flags)
        if not matches:
            return None
        value = float(matches[-1])
        # If the pattern is the explicit `Probability: <number>(?!%)` form,
        # callers may still emit a value > 1 they meant as a percentage.
        if value > 1:
            value = value / 100.0
        elif percent_form:
            value = value / 100.0
        if 0 <= value <= 1:
            return value
        logging.warning(f"Extracted value {value} is out of range [0, 1]")
        return None

    def get_answer_from_model_output(
        self,
        generated_text: str,
        tokenizer_vocab: dict[str, int] = None,
    ) -> float:
        """Extract the probability answer from the model's generated text.

        Parameters
        ----------
        generated_text : str
            The full text generated by the model, including reasoning and
            the final probability estimate.
        tokenizer_vocab : dict[str, int], optional
            The tokenizer's vocabulary. Not used for ChainOfThoughtQA but
            included for interface compatibility.

        Returns
        -------
        answer : float
            The extracted probability as a float between 0 and 1.

        Raises
        ------
        ValueError
            If no valid probability could be extracted from the generated text.
        """
        probability = self.extract_probability_from_text(generated_text)

        if probability is None:
            logging.error(
                "Failed to extract probability from generated text. "
                "Returning default value of 0.5."
            )
            return 0.5

        logging.debug(f"Extracted probability: {probability:.2%}")
        return probability
