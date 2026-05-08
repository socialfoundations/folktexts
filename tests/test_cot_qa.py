"""Unit tests for `ChainOfThoughtQA` — guards the regex extraction (the most
fragile part of the CoT path) and the LSP-compliant `get_question_prompt`
signature added during the rebase onto main's chat-template work.
"""
from __future__ import annotations

import pytest

from folktexts.benchmark import Benchmark, BenchmarkConfig
from folktexts.qa_interface import ChainOfThoughtQA


@pytest.fixture
def cot_qa() -> ChainOfThoughtQA:
    return ChainOfThoughtQA(
        column="PINCP",
        text="What is this person's estimated yearly income?",
    )


# ----------------------------------------------------------------------
# get_question_prompt — accepts the `with_answer_prefill` kwarg from
# `QAInterface` for LSP, but does not change the returned text (CoT
# prompts have no prefill to strip).
# ----------------------------------------------------------------------

class TestGetQuestionPrompt:
    def test_returns_non_empty_string(self, cot_qa: ChainOfThoughtQA):
        assert isinstance(cot_qa.get_question_prompt(), str)
        assert cot_qa.get_question_prompt().strip()

    def test_with_answer_prefill_kwarg_accepted(self, cot_qa: ChainOfThoughtQA):
        # The kwarg is purely for interface compatibility — both calls must
        # succeed and return identical output.
        with_prefill = cot_qa.get_question_prompt(with_answer_prefill=True)
        without_prefill = cot_qa.get_question_prompt(with_answer_prefill=False)
        assert with_prefill == without_prefill

    def test_prompt_includes_extraction_anchor(self, cot_qa: ChainOfThoughtQA):
        # The prompt must end with the "Probability: X%" anchor so the
        # regex extractor has a consistent target. If this anchor changes,
        # `_PROBABILITY_PATTERNS` must be updated in lockstep.
        prompt = cot_qa.get_question_prompt()
        assert "Probability: X%" in prompt


# ----------------------------------------------------------------------
# extract_probability_from_text — the regex pyramid
# ----------------------------------------------------------------------

class TestExtractProbability:
    @pytest.mark.parametrize("text,expected", [
        ("Probability: 75%", 0.75),
        ("probability: 75%", 0.75),
        ("Probability: 0.75", 0.75),
        ("Probability is 80%", 0.80),
        ("Probability of 0.42", 0.42),
        ("After thinking: 25%", 0.25),
        ("The answer is 50 percent.", 0.50),
        ("My estimate is 0.33", 0.33),
    ])
    def test_extracts_supported_formats(self, text: str, expected: float):
        result = ChainOfThoughtQA.extract_probability_from_text(text)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_uses_last_explicit_match(self):
        # The model may revise its estimate mid-CoT; we trust the
        # final "Probability: X" line.
        text = "Probability: 30%. Wait, on reflection, Probability: 65%."
        assert ChainOfThoughtQA.extract_probability_from_text(text) == pytest.approx(0.65)

    def test_returns_none_when_no_signal(self):
        assert ChainOfThoughtQA.extract_probability_from_text("the answer is unclear") is None

    def test_rejects_out_of_range_probability(self):
        # A bare 150% (or similar) should not be treated as a probability
        # via the explicit "Probability:" pattern; the function falls
        # through and either finds a valid percent later or returns None.
        assert ChainOfThoughtQA.extract_probability_from_text("Probability: 150%") is None


# ----------------------------------------------------------------------
# get_answer_from_model_output — fallback contract
# ----------------------------------------------------------------------

class TestGetAnswerFromModelOutput:
    def test_returns_extracted_value(self, cot_qa: ChainOfThoughtQA):
        result = cot_qa.get_answer_from_model_output("Probability: 80%")
        assert result == pytest.approx(0.80)

    def test_returns_default_on_extraction_failure(self, cot_qa: ChainOfThoughtQA):
        # Silent fallback to 0.5 keeps the benchmark from crashing on a
        # single bad generation; the failure is logged at ERROR.
        result = cot_qa.get_answer_from_model_output("nonsense with no probability")
        assert result == 0.5


# ----------------------------------------------------------------------
# Benchmark._validate_config — CoT ↔ chat-template asymmetry.
# The CoT path applies the tokenizer's chat template internally inside
# `generate_text_batch`; combining it with `use_chat_template=True` would
# double-wrap the prompt. This must be rejected upfront.
# ----------------------------------------------------------------------

class TestValidateConfigCoTChatInteraction:
    def test_rejects_chat_template_with_cot(self):
        config = BenchmarkConfig(use_chat_template=True, cot_prompting=True)
        with pytest.raises(ValueError, match="chat_template.*cot|cot.*chat_template"):
            Benchmark._validate_config(config)

    def test_rejects_chat_template_with_thinking(self):
        config = BenchmarkConfig(use_chat_template=True, enable_thinking=True)
        with pytest.raises(ValueError, match="chat_template.*cot|cot.*chat_template"):
            Benchmark._validate_config(config)

    def test_accepts_cot_without_chat_template(self):
        # The supported configuration: CoT applies its own chat template
        # internally (or falls back to raw prompt for base models).
        Benchmark._validate_config(
            BenchmarkConfig(use_chat_template=False, cot_prompting=True)
        )

    def test_accepts_chat_template_without_cot(self):
        Benchmark._validate_config(
            BenchmarkConfig(use_chat_template=True, cot_prompting=False)
        )
