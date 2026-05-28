"""Unit tests for `qa_interface`.

Structural properties this file guards:

1. `get_question_prompt(with_answer_prefill: bool)` — `True` (default) keeps
   the legacy zero-shot string byte-for-byte; `False` removes the answer
   prefill so the chat-template path can supply it as the assistant turn
   without duplicating it in the user message.

2. `get_answer_prefix()` — the answer prefill string returned by each QA
   subclass independently of the full question prompt.

3. `_get_numeric_tokens(tokenizer_vocab, vocab_dim)` — filters digit / decimal
   tokens whose ids fall outside `[0, vocab_dim)`. The caller (`get_answer_from_model_output`)
   derives `vocab_dim` from the actual logits axis (`last_token_probs.shape[-1]`),
   so out-of-range vocab entries no longer trip an `IndexError` deep in the
   probability lookup.
"""

from __future__ import annotations

import numpy as np
import pytest
from folktexts.qa_interface import Choice, DirectNumericQA, MultipleChoiceQA

# ----------------------------------------------------------------------
# DirectNumericQA.get_question_prompt
# ----------------------------------------------------------------------


class TestDirectNumericQAGetQuestionPrompt:
    def _q(self, answer_probability: bool = True) -> DirectNumericQA:
        return DirectNumericQA(
            column="PINCP",
            text="What is this person's estimated yearly income?",
            answer_probability=answer_probability,
        )

    def test_default_matches_legacy_zero_shot_string(self):
        # Pin byte-for-byte equality with the pre-refactor zero-shot output
        # so the paper-reproducing path is provably untouched.
        expected = "Question: What is this person's estimated yearly income?\nAnswer (between 0 and 1): 0."
        assert self._q().get_question_prompt() == expected
        assert self._q().get_question_prompt(with_answer_prefill=True) == expected

    def test_with_answer_prefill_false_omits_prefill(self):
        out = self._q().get_question_prompt(with_answer_prefill=False)
        assert "Answer (between 0 and 1)" not in out
        assert "0." not in out
        # The bare question must still be present and the string must end at
        # the question text — that's exactly what the chat user-turn needs.
        assert out == "Question: What is this person's estimated yearly income?"

    def test_answer_probability_false_with_prefill_emits_open_answer(self):
        # `answer_probability=False` ⇒ open-ended numeric Q&A; the prefill is
        # `"Answer: "` (trailing space matters for tokenization in the
        # zero-shot path).
        out = self._q(answer_probability=False).get_question_prompt()
        assert out.endswith("\nAnswer: ")

    def test_answer_probability_false_no_prefill_omits_answer_line(self):
        out = self._q(answer_probability=False).get_question_prompt(
            with_answer_prefill=False,
        )
        assert "Answer" not in out
        assert out == "Question: What is this person's estimated yearly income?"


# ----------------------------------------------------------------------
# MultipleChoiceQA.get_question_prompt
# ----------------------------------------------------------------------


class TestMultipleChoiceQAGetQuestionPrompt:
    def _q(self) -> MultipleChoiceQA:
        return MultipleChoiceQA(
            column="PINCP",
            text="Is this person's income above $50k?",
            choices=(
                Choice(text="No", data_value=0, numeric_value=0.0),
                Choice(text="Yes", data_value=1, numeric_value=1.0),
            ),
        )

    def test_default_matches_legacy_zero_shot_string(self):
        expected = "Question: Is this person's income above $50k?\nA. No.\nB. Yes.\nAnswer:"
        assert self._q().get_question_prompt() == expected
        assert self._q().get_question_prompt(with_answer_prefill=True) == expected

    def test_with_answer_prefill_false_omits_answer_line(self):
        out = self._q().get_question_prompt(with_answer_prefill=False)
        # Choices must remain — they're the question content, not the prefill.
        assert "A. No." in out
        assert "B. Yes." in out
        # The trailing "Answer:" prefill is the only thing that should drop.
        assert not out.rstrip().endswith("Answer:")
        assert out == ("Question: Is this person's income above $50k?\nA. No.\nB. Yes.")


# ----------------------------------------------------------------------
# get_answer_prefix — the answer prefill string for each QA subclass
# ----------------------------------------------------------------------


class TestGetAnswerPrefix:
    def test_numeric_answer_probability_true(self):
        q = DirectNumericQA(column="x", text="dummy", answer_probability=True)
        assert q.get_answer_prefix() == "Answer (between 0 and 1): 0."

    def test_numeric_answer_probability_false(self):
        q = DirectNumericQA(column="x", text="dummy", answer_probability=False)
        assert q.get_answer_prefix() == "Answer: "

    def test_mc_answer_prefix(self):
        q = MultipleChoiceQA(
            column="x",
            text="dummy",
            choices=(
                Choice(text="No", data_value=0, numeric_value=0.0),
                Choice(text="Yes", data_value=1, numeric_value=1.0),
            ),
        )
        assert q.get_answer_prefix() == "Answer:"


# ----------------------------------------------------------------------
# DirectNumericQA._get_numeric_tokens — vocab_dim filter
# ----------------------------------------------------------------------


class TestGetNumericTokensVocabDimFilter:
    """The filter prevents an `IndexError` on tokenizers (e.g. Gemma-3) where
    digit-or-decimal-named tokens can sit at ids beyond `model.config.vocab_size`
    — the actual logits axis the caller indexes into.
    """

    def _q(self) -> DirectNumericQA:
        return DirectNumericQA(column="x", text="dummy")

    def test_drops_digit_token_whose_id_is_beyond_vocab_dim(self):
        vocab = {str(i): i for i in range(10)}
        vocab["888"] = 100  # out-of-range multi-digit
        nums = self._q()._get_numeric_tokens(vocab, vocab_dim=60)
        assert "888" not in nums
        # All single-digit base tokens (ids 0-9) are in-range and kept.
        assert all(str(i) in nums for i in range(10))

    def test_keeps_in_range_multi_digit_tokens(self):
        vocab = {str(i): i for i in range(10)}
        vocab["999"] = 50  # in-range
        nums = self._q()._get_numeric_tokens(vocab, vocab_dim=60)
        assert nums.get("999") == 50

    def test_keeps_decimal_when_in_range(self):
        vocab = {str(i): i for i in range(10)}
        vocab["."] = 7
        nums = self._q()._get_numeric_tokens(vocab, vocab_dim=60)
        assert nums.get(".") == 7

    def test_drops_decimal_when_out_of_range(self):
        vocab = {str(i): i for i in range(10)}
        vocab["."] = 100  # beyond vocab_dim
        nums = self._q()._get_numeric_tokens(vocab, vocab_dim=60)
        assert "." not in nums

    def test_no_decimal_in_vocab_is_handled(self):
        vocab = {str(i): i for i in range(10)}
        nums = self._q()._get_numeric_tokens(vocab, vocab_dim=10)
        assert "." not in nums
        # All digit tokens kept.
        assert len(nums) == 10

    def test_vocab_dim_zero_filters_everything(self):
        # Edge case: pathological vocab_dim. Should not crash, just drop all.
        vocab = {str(i): i for i in range(10)}
        vocab["."] = 7
        nums = self._q()._get_numeric_tokens(vocab, vocab_dim=0)
        assert nums == {}

    def test_returned_ids_are_all_within_vocab_dim(self):
        """Strong invariant: every returned id is a legal index into a
        `last_token_probs` array of length `vocab_dim`."""
        vocab = {str(i): i for i in range(20)}
        vocab["100"] = 30
        vocab["200"] = 70  # out-of-range
        vocab["."] = 50  # out-of-range
        vocab_dim = 40
        nums = self._q()._get_numeric_tokens(vocab, vocab_dim=vocab_dim)
        assert all(0 <= tid < vocab_dim for tid in nums.values())


# ----------------------------------------------------------------------
# DirectNumericQA.get_answer_from_model_output — derives vocab_dim from probs
# ----------------------------------------------------------------------


class TestGetAnswerFromModelOutputDerivesVocabDim:
    """The caller no longer has to thread `vocab_dim` in: it is derived from
    `last_token_probs.shape[-1]` (the logits axis the caller already gave us).
    This test pins that contract and that the filter is applied — i.e. an
    out-of-range digit-named token doesn't IndexError when scoring.
    """

    def _q(self) -> DirectNumericQA:
        return DirectNumericQA(column="x", text="dummy")

    def test_derives_vocab_dim_from_probs_and_drops_oor_tokens(self):
        # Vocab declares "5" at id 50 (in-range) and "9" at id 100 (out-of-range
        # for a probs array of width 60). Must not IndexError; "9" must not be
        # considered when picking the most-likely numeric token.
        vocab = {str(i): i for i in range(10)}  # "0"..."9" → ids 0..9 (in-range)
        vocab["99"] = 100  # out-of-range — the filter must drop this
        vocab["."] = 7
        # Probs array of width 60: assign all mass to id 5 on pass 0 ("5") and
        # id 3 on pass 1 ("3"). Expected answer: 0.53.
        probs = np.zeros((2, 60))
        probs[0, 5] = 1.0
        probs[1, 3] = 1.0
        ans = self._q().get_answer_from_model_output(probs, vocab)
        assert ans == pytest.approx(0.53)

    def test_no_indexerror_when_vocab_extends_past_logits_axis(self):
        # Gemma-style: digit-named tokens may exist at ids >= logits_dim. The
        # filter must drop them so `ltp[token_id]` doesn't IndexError.
        vocab = {str(i): i for i in range(10)}  # ids 0..9 — in-range
        vocab["77"] = 10  # digit-named, AT the logits axis — must be dropped
        probs = np.zeros((1, 10))
        probs[0, 4] = 1.0
        # `answer_probability=True` ⇒ "0." prefill + "4" → 0.4.
        ans = self._q().get_answer_from_model_output(probs, vocab)
        assert ans == pytest.approx(0.4)
