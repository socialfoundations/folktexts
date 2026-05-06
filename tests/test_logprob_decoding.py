"""Unit tests for `decode_topk_logprobs_to_risk_estimate`.

This helper sits between any "top-K logprobs" backend (vLLM, OpenAI-style web
APIs) and the QA-interface decoders. It scatters `exp(logprob)` from sparse
top-K dicts into a (n_passes, vocab_dim) array and dispatches to
`question.get_answer_from_model_output(probs, tokenizer_vocab)`.

Tests guard:
- Token-id keying (not string keying) for vLLM-style inputs.
- Out-of-range token ids are silently dropped (matches the digit-token vocab
  filter in qa_interface.DirectNumericQA._get_numeric_tokens).
- Both QA modes (MultipleChoiceQA / DirectNumericQA) round-trip correctly.
- Tokens absent from the top-K get probability 0 — nothing in the helper
  pretends those tokens have signal.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from folktexts.llm_utils import decode_topk_logprobs_to_risk_estimate
from folktexts.qa_interface import Choice, DirectNumericQA, MultipleChoiceQA


def _binary_mc_question() -> MultipleChoiceQA:
    return MultipleChoiceQA(
        column="PINCP",
        text="Is this person's income above $50k?",
        choices=(
            Choice(text="No", data_value=0, numeric_value=0.0),
            Choice(text="Yes", data_value=1, numeric_value=1.0),
        ),
    )


def _numeric_question() -> DirectNumericQA:
    return DirectNumericQA(column="PINCP", text="dummy")


# ----------------------------------------------------------------------
# MultipleChoiceQA: top-K logprobs → choice probability
# ----------------------------------------------------------------------

class TestMultipleChoiceDecoding:
    def test_binary_returns_positive_choice_probability(self):
        # Vocab pins ids for the answer-letter prefix variants. The QA decoder
        # tries multiple prefixes; here only " A" / " B" are populated.
        tokenizer_vocab = {" A": 1, " B": 2, " C": 3}
        # Single forward pass; logprobs ≈ p(A)=0.2, p(B)=0.8.
        per_pass_topk = [{
            1: math.log(0.2),
            2: math.log(0.8),
        }]
        risk = decode_topk_logprobs_to_risk_estimate(
            per_pass_topk,
            tokenizer_vocab=tokenizer_vocab,
            vocab_dim=10,
            question=_binary_mc_question(),
        )
        # B="Yes" is the positive choice (highest numeric_value); after
        # renormalising over {A, B}: 0.8 / (0.2 + 0.8) = 0.8.
        assert risk == pytest.approx(0.8, abs=1e-6)

    def test_decoder_finds_the_best_prefix_variant(self):
        # Provide " A" / " B" via the dominant " " prefix AND junk tokens for
        # other prefixes; decoder should pick the prefix variant with highest
        # total probability mass.
        tokenizer_vocab = {
            " A": 1, " B": 2,         # space prefix (dominant)
            "A": 11, "B": 12,         # bare prefix (no mass)
        }
        per_pass_topk = [{
            1: math.log(0.4),
            2: math.log(0.6),
        }]
        risk = decode_topk_logprobs_to_risk_estimate(
            per_pass_topk,
            tokenizer_vocab=tokenizer_vocab,
            vocab_dim=20,
            question=_binary_mc_question(),
        )
        assert risk == pytest.approx(0.6, abs=1e-6)

    def test_missing_tokens_count_as_zero_prob(self):
        # Only " A" appears in the top-K; " B" is absent ⇒ assumed 0 prob.
        # Renormalised: p(A)/(p(A)+0) = 1.0; positive = B ⇒ 0.0.
        tokenizer_vocab = {" A": 1, " B": 2}
        per_pass_topk = [{1: math.log(0.5)}]
        risk = decode_topk_logprobs_to_risk_estimate(
            per_pass_topk,
            tokenizer_vocab=tokenizer_vocab,
            vocab_dim=10,
            question=_binary_mc_question(),
        )
        assert risk == pytest.approx(0.0, abs=1e-6)

    def test_out_of_range_token_ids_are_silently_dropped(self):
        # If vocab_dim shrinks past a token's id, the helper must drop it
        # rather than write past the array end. Models with extra tokens
        # beyond model.config.vocab_size (Gemma-3, Llama-3.2) hit this.
        tokenizer_vocab = {" A": 1, " B": 99}  # 99 ≥ vocab_dim=10
        per_pass_topk = [{
            1: math.log(0.7),
            99: math.log(0.3),
        }]
        risk = decode_topk_logprobs_to_risk_estimate(
            per_pass_topk,
            tokenizer_vocab=tokenizer_vocab,
            vocab_dim=10,
            question=_binary_mc_question(),
        )
        # B's mass is dropped; only A remains; positive=B ⇒ 0.0.
        assert risk == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------------
# DirectNumericQA: top-K logprobs across multiple passes → float in [0,1]
# ----------------------------------------------------------------------

class TestNumericDecoding:
    def test_two_passes_concatenate_argmax_digits(self):
        # Pass 0 picks "2", pass 1 picks "5" → "0." + "25" = 0.25.
        tokenizer_vocab = {str(d): d for d in range(10)}
        per_pass_topk = [
            # Pass 0: "2" dominant
            {2: math.log(0.7), 1: math.log(0.2), 5: math.log(0.05)},
            # Pass 1: "5" dominant
            {5: math.log(0.6), 4: math.log(0.3), 6: math.log(0.05)},
        ]
        risk = decode_topk_logprobs_to_risk_estimate(
            per_pass_topk,
            tokenizer_vocab=tokenizer_vocab,
            vocab_dim=10,
            question=_numeric_question(),
        )
        assert risk == pytest.approx(0.25, abs=1e-6)

    def test_multi_digit_tokens_in_vocab_are_used_when_dominant(self):
        # Some tokenizers carry multi-digit tokens (e.g. "25"); when one
        # outranks the single-digit candidates, the QA decoder picks it.
        tokenizer_vocab = {str(d): d for d in range(10)}
        tokenizer_vocab["25"] = 25  # multi-digit token, in-range
        per_pass_topk = [
            # Pass 0: "25" outranks single digits
            {25: math.log(0.9), 2: math.log(0.05)},
            # Pass 1: irrelevant since the QA decoder uses argmax across
            # numeric_tokens but the regex extractor only keeps the leading
            # digits — provide a plausible filler.
            {0: math.log(0.5)},
        ]
        risk = decode_topk_logprobs_to_risk_estimate(
            per_pass_topk,
            tokenizer_vocab=tokenizer_vocab,
            vocab_dim=30,
            question=_numeric_question(),
        )
        # Pass 0 picks "25", pass 1 picks "0". Concatenated: "250"; the regex
        # extracts "250"; with answer_probability=True and no decimal, it
        # becomes 0.250.
        assert risk == pytest.approx(0.250, abs=1e-6)

    def test_decimal_token_is_used_when_present_and_in_range(self):
        # If "." is the argmax on a pass, the regex extracts a float directly
        # (e.g. "0." + "5" → 0.5).
        tokenizer_vocab = {str(d): d for d in range(10)}
        tokenizer_vocab["."] = 11
        per_pass_topk = [
            {5: math.log(0.9)},     # "5"
            {11: math.log(0.9)},    # "."
        ]
        risk = decode_topk_logprobs_to_risk_estimate(
            per_pass_topk,
            tokenizer_vocab=tokenizer_vocab,
            vocab_dim=20,
            question=_numeric_question(),
        )
        # Concatenation: "5."; the regex matches "5" (no fractional part),
        # giving 0.5 with answer_probability=True.
        assert risk == pytest.approx(0.5, abs=1e-6)

    def test_out_of_range_digit_token_is_dropped_by_qa_decoder(self):
        # The QA decoder filters to ids < vocab_dim. The helper additionally
        # never writes past vocab_dim, so out-of-range entries are doubly
        # safe.
        tokenizer_vocab = {str(d): d for d in range(10)}
        tokenizer_vocab["99"] = 99  # out-of-range
        per_pass_topk = [
            # "99" would be argmax if not dropped; "3" is the next best.
            {99: math.log(0.9), 3: math.log(0.05), 1: math.log(0.02)},
            {2: math.log(0.5)},
        ]
        risk = decode_topk_logprobs_to_risk_estimate(
            per_pass_topk,
            tokenizer_vocab=tokenizer_vocab,
            vocab_dim=10,
            question=_numeric_question(),
        )
        # "99" dropped (out-of-range), "3" wins pass 0, "2" wins pass 1
        # → "32" → 0.32.
        assert risk == pytest.approx(0.32, abs=1e-6)


# ----------------------------------------------------------------------
# Helper-level invariants (independent of QA mode)
# ----------------------------------------------------------------------

class TestHelperShapeInvariants:
    def test_zero_logprob_dicts_produce_a_finite_score(self):
        # Pathological input: every logprob is 0 (i.e. probability 1.0 for
        # every listed token). The QA decoder still has to return a finite
        # number, not crash.
        tokenizer_vocab = {" A": 1, " B": 2}
        per_pass_topk = [{1: 0.0, 2: 0.0}]
        risk = decode_topk_logprobs_to_risk_estimate(
            per_pass_topk,
            tokenizer_vocab=tokenizer_vocab,
            vocab_dim=10,
            question=_binary_mc_question(),
        )
        assert np.isfinite(risk)
        # Equal mass on the two choices ⇒ p(positive)=0.5.
        assert risk == pytest.approx(0.5, abs=1e-6)

    def test_negative_token_id_is_dropped(self):
        # Defensive: a backend that smuggles a negative id should not crash
        # nor be counted (numpy's negative indexing would otherwise wrap
        # around silently).
        tokenizer_vocab = {" A": 1, " B": 2}
        per_pass_topk = [{
            1: math.log(0.5),
            2: math.log(0.5),
            -1: math.log(0.99),
        }]
        risk = decode_topk_logprobs_to_risk_estimate(
            per_pass_topk,
            tokenizer_vocab=tokenizer_vocab,
            vocab_dim=10,
            question=_binary_mc_question(),
        )
        # Negative id ignored ⇒ A and B both at 0.5 ⇒ 0.5.
        assert risk == pytest.approx(0.5, abs=1e-6)
