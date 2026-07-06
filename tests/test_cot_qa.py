"""Unit tests for `ChainOfThoughtQA` — guards the regex extraction (the most
fragile part of the CoT path) and the LSP-compliant `get_question_prompt`
signature added during the rebase onto main's chat-template work.
"""
from __future__ import annotations

import pytest

from folktexts.benchmark import Benchmark, BenchmarkConfig
from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import (
    ChainOfThoughtQA,
    Choice,
    DirectNumericQA,
    MultipleChoiceQA,
)
from folktexts.task import TaskMetadata


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


# ----------------------------------------------------------------------
# Benchmark._configure_task_question — singleton state reset.
#
# `TaskMetadata.get_task` is a class-level cache, so the same task object
# is reused across benchmark runs. _configure_task_question MUST clear any
# Q&A state written by a prior run — without this, switching from a
# CoT/thinking cell to a plain chat_mcq cell silently dispatches
# ChainOfThoughtQA inference (max_new_tokens=8000) for what should be a
# 1-token MC prediction. Caught while running scripts/cot_e2e_sweep.py:
# gpt-oss-20b chat_mcq came right after a Qwen3-14B cot_thinking cell, the
# leaked CoT question made the run emit 38 000 chars of gibberish per row
# at ~50 s/batch.
# ----------------------------------------------------------------------

@pytest.fixture
def fresh_task() -> TaskMetadata:
    """A real `TaskMetadata` with both MC and numeric Q&A interfaces wired up,
    so `_configure_task_question` can be exercised end-to-end without
    monkeypatching the singleton cache.

    Uses a unique name per test run via the test id to avoid colliding with
    `TaskMetadata._tasks`'s cross-test cache."""
    mc_qa = MultipleChoiceQA(
        column="TARGET",
        text="Is the value high?",
        choices=(
            Choice("low", data_value=0, numeric_value=0.0),
            Choice("high", data_value=1, numeric_value=1.0),
        ),
    )
    num_qa = DirectNumericQA(
        column="TARGET",
        text="What is the value?",
    )
    name = f"_test_state_leak_{id(mc_qa)}"
    task = TaskMetadata(
        name=name,
        features=["x"],
        target="TARGET",
        cols_to_text={
            "x": ColumnToText("x", short_description="x"),
            "TARGET": ColumnToText("TARGET", short_description="target"),
        },
        multiple_choice_qa=mc_qa,
        direct_numeric_qa=num_qa,
    )
    yield task
    # Cleanup the class-level cache so subsequent tests don't inherit state.
    TaskMetadata._tasks.pop(name, None)


class TestConfigureTaskQuestionStateReset:
    def test_chat_mcq_after_cot_thinking_resets_to_mc(self, fresh_task):
        # Bug repro: configure CoT+thinking first, then chat_mcq. Without the
        # else-branch reset in _configure_task_question, task.question would
        # still return the leaked ChainOfThoughtQA.
        Benchmark._configure_task_question(
            fresh_task,
            BenchmarkConfig(cot_prompting=True, enable_thinking=True),
        )
        assert isinstance(fresh_task.question, ChainOfThoughtQA)
        assert fresh_task._use_cot_qa is True

        Benchmark._configure_task_question(
            fresh_task,
            BenchmarkConfig(use_chat_template=True),  # chat_mcq: no CoT/numeric/thinking
        )
        assert isinstance(fresh_task.question, MultipleChoiceQA)
        assert fresh_task._use_cot_qa is False
        assert fresh_task._use_numeric_qa is False

    def test_chat_mcq_after_numeric_resets_to_mc(self, fresh_task):
        # Symmetric case: chat_numeric -> chat_mcq must also clear state.
        Benchmark._configure_task_question(
            fresh_task,
            BenchmarkConfig(numeric_risk_prompting=True),
        )
        assert isinstance(fresh_task.question, DirectNumericQA)
        assert fresh_task._use_numeric_qa is True

        Benchmark._configure_task_question(
            fresh_task,
            BenchmarkConfig(use_chat_template=True),
        )
        assert isinstance(fresh_task.question, MultipleChoiceQA)
        assert fresh_task._use_cot_qa is False
        assert fresh_task._use_numeric_qa is False

    def test_chat_mcq_after_cot_resets_to_mc(self, fresh_task):
        # CoT without thinking — same leak pattern.
        Benchmark._configure_task_question(
            fresh_task, BenchmarkConfig(cot_prompting=True),
        )
        assert isinstance(fresh_task.question, ChainOfThoughtQA)

        Benchmark._configure_task_question(
            fresh_task, BenchmarkConfig(),  # all flags default-False = chat_mcq
        )
        assert isinstance(fresh_task.question, MultipleChoiceQA)

    def test_cot_after_numeric_overrides(self, fresh_task):
        # CoT branch already calls set_question(), which clears both flags.
        # This is just a regression guard — make sure that path still works.
        Benchmark._configure_task_question(
            fresh_task, BenchmarkConfig(numeric_risk_prompting=True),
        )
        Benchmark._configure_task_question(
            fresh_task, BenchmarkConfig(cot_prompting=True),
        )
        assert isinstance(fresh_task.question, ChainOfThoughtQA)
        assert fresh_task._use_numeric_qa is False
        assert fresh_task._use_cot_qa is True

    def test_numeric_after_cot_thinking_overrides(self, fresh_task):
        # Numeric branch sets `task.use_numeric_qa = True`, whose setter
        # also clears `_use_cot_qa`. Regression guard for that setter contract.
        Benchmark._configure_task_question(
            fresh_task,
            BenchmarkConfig(cot_prompting=True, enable_thinking=True),
        )
        Benchmark._configure_task_question(
            fresh_task, BenchmarkConfig(numeric_risk_prompting=True),
        )
        assert isinstance(fresh_task.question, DirectNumericQA)
        assert fresh_task._use_cot_qa is False
        assert fresh_task._use_numeric_qa is True


class TestDefaultTemperature:
    """Greedy for plain CoT; 1.0 in thinking mode (greedy is discouraged for
    thinking models). MC/numeric keep the temperature-free ClassVar default."""

    def test_plain_cot_is_greedy(self, cot_qa):
        assert cot_qa.enable_thinking is False
        assert cot_qa.default_temperature == 0.0

    def test_thinking_mode_samples(self):
        q = ChainOfThoughtQA(column="PINCP", text="dummy", enable_thinking=True)
        assert q.default_temperature == 1.0

    def test_token_probability_modes_stay_greedy(self):
        assert DirectNumericQA(column="PINCP", text="dummy").default_temperature == 0.0
        mcq = MultipleChoiceQA(
            column="PINCP", text="dummy",
            choices=(Choice("Yes", 1), Choice("No", 0)),
        )
        assert mcq.default_temperature == 0.0
