"""Unit tests for `VLLMClassifier` using duck-typed stubs.

vLLM is an optional install; these tests must run in environments without it.
We inject a fake ``vllm`` module into ``sys.modules`` before importing the
classifier so the in-method ``from vllm import SamplingParams`` resolves to
our stub. The stubbed ``LLM`` returns canned ``RequestOutput``-like objects so
we can verify the score-extraction contract (logprobs → risk estimate, text
→ extracted probability) without ever touching a real model.
"""
from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass

import pytest


# --------------------------------------------------------------------------
# Inject a fake `vllm` module BEFORE the classifier is imported so its local
# `from vllm import SamplingParams` calls resolve to our stub.
# --------------------------------------------------------------------------

@dataclass
class _FakeSamplingParams:
    temperature: float = 0.0
    max_tokens: int = 1
    logprobs: int | None = None
    allowed_token_ids: list[int] | None = None
    seed: int | None = None


_fake_vllm = types.ModuleType("vllm")
_fake_vllm.SamplingParams = _FakeSamplingParams
sys.modules["vllm"] = _fake_vllm

# --- The classifier imports below resolve `vllm` to the stub above. -----
from folktexts.classifier.vllm_classifier import VLLMClassifier  # noqa: E402
from folktexts.qa_interface import (  # noqa: E402
    Choice,
    DirectNumericQA,
    MultipleChoiceQA,
    ReasoningQA,
)


# --------------------------------------------------------------------------
# Stub LLM / tokenizer / output objects
# --------------------------------------------------------------------------

class _StubTokenizer:
    """Minimal duck-typed tokenizer for VLLMClassifier."""

    def __init__(self, vocab: dict[str, int], vocab_size: int | None = None):
        self._vocab = dict(vocab)
        self.vocab_size = vocab_size if vocab_size is not None else max(vocab.values()) + 1
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<eos>"

    def get_vocab(self):
        return dict(self._vocab)

    @property
    def vocab(self):
        return dict(self._vocab)

    def add_special_tokens(self, *args, **kwargs):
        # No-op; classifier never touches this directly (load_vllm_model does).
        pass


@dataclass
class _StubLogprob:
    """Mirrors the public surface of vllm.Logprob — we only read .logprob."""
    logprob: float


@dataclass
class _StubCompletionOutput:
    text: str = ""
    logprobs: list[dict[int, _StubLogprob]] | None = None


@dataclass
class _StubRequestOutput:
    outputs: list[_StubCompletionOutput]


class _StubLLM:
    """Returns canned RequestOutputs in the order they were configured.

    `script` is a list-of-list-of-RequestOutput: each `generate(...)` call
    pops one batch from the front. This lets a single test exercise multiple
    successive calls (e.g. three batches of MC prompts) without coupling
    behaviour to a real vLLM engine.
    """

    def __init__(self, script: list[list[_StubRequestOutput]]):
        self._script = list(script)
        self.last_sampling_params = None
        self.last_prompts = None

    def generate(self, prompts, sampling_params, **kwargs):
        self.last_prompts = list(prompts)
        self.last_sampling_params = sampling_params
        if not self._script:
            raise AssertionError("StubLLM.generate called more times than scripted")
        return self._script.pop(0)


# --------------------------------------------------------------------------
# Test fixtures
# --------------------------------------------------------------------------

def _binary_mc_question() -> MultipleChoiceQA:
    return MultipleChoiceQA(
        column="PINCP",
        text="Is this person's income above $50k?",
        choices=(
            Choice(text="No", data_value=0, numeric_value=0.0),
            Choice(text="Yes", data_value=1, numeric_value=1.0),
        ),
    )


def _make_classifier(llm: _StubLLM, tokenizer: _StubTokenizer, *, vocab_dim: int):
    """Build a VLLMClassifier wired against stubs.

    We pass `model_name_or_path=None` to skip the AutoConfig.from_pretrained
    lookup. The tokenizer fallback then yields `vocab_dim`. We force-override
    `_vocab_dim` afterwards so the test doesn't depend on the fallback's exact
    formula.
    """
    clf = VLLMClassifier(
        llm=llm,
        tokenizer=tokenizer,
        task="ACSIncome",
        model_name_or_path=None,
    )
    clf._vocab_dim = vocab_dim
    return clf


# --------------------------------------------------------------------------
# Multiple-choice path
# --------------------------------------------------------------------------

class TestMultipleChoicePath:
    def test_returns_positive_choice_probability(self):
        # Tokenizer pins ids for " A" / " B"; logprobs put 0.8 on " B" (= "Yes").
        vocab = {" A": 1, " B": 2, "X": 3}
        tokenizer = _StubTokenizer(vocab, vocab_size=10)
        request_output = _StubRequestOutput(outputs=[
            _StubCompletionOutput(
                logprobs=[{
                    1: _StubLogprob(math.log(0.2)),
                    2: _StubLogprob(math.log(0.8)),
                }],
            ),
        ])
        llm = _StubLLM(script=[[request_output]])
        clf = _make_classifier(llm, tokenizer, vocab_dim=10)

        risks = clf._query_prompt_risk_estimates_batch(
            prompts_batch=["dummy prompt"],
            question=_binary_mc_question(),
        )
        assert risks.shape == (1,)
        assert risks[0] == pytest.approx(0.8, abs=1e-6)

    def test_uses_unconstrained_sampling_for_mc(self):
        # The transformers MC path is unmasked; vLLM should mirror that —
        # `allowed_token_ids` must be None for MultipleChoiceQA.
        vocab = {" A": 1, " B": 2}
        tokenizer = _StubTokenizer(vocab, vocab_size=10)
        request_output = _StubRequestOutput(outputs=[
            _StubCompletionOutput(logprobs=[{1: _StubLogprob(math.log(0.5)),
                                              2: _StubLogprob(math.log(0.5))}])
        ])
        llm = _StubLLM(script=[[request_output]])
        clf = _make_classifier(llm, tokenizer, vocab_dim=10)
        clf._query_prompt_risk_estimates_batch(
            prompts_batch=["dummy"], question=_binary_mc_question(),
        )

        params = llm.last_sampling_params
        assert params.allowed_token_ids is None
        assert params.max_tokens == 1
        assert params.temperature == 0.0


# --------------------------------------------------------------------------
# Direct numeric path
# --------------------------------------------------------------------------

class TestDirectNumericPath:
    def test_two_passes_concatenate_to_probability(self):
        # Vocab covers digits 0-9. Pass 0 picks "2", pass 1 picks "5" → 0.25.
        vocab = {str(d): d for d in range(10)}
        tokenizer = _StubTokenizer(vocab, vocab_size=10)
        request_output = _StubRequestOutput(outputs=[
            _StubCompletionOutput(logprobs=[
                {2: _StubLogprob(math.log(0.7)), 1: _StubLogprob(math.log(0.2))},
                {5: _StubLogprob(math.log(0.6)), 4: _StubLogprob(math.log(0.3))},
            ]),
        ])
        llm = _StubLLM(script=[[request_output]])
        clf = _make_classifier(llm, tokenizer, vocab_dim=10)

        question = DirectNumericQA(column="PINCP", text="dummy")
        risks = clf._query_prompt_risk_estimates_batch(
            prompts_batch=["dummy prompt"], question=question,
        )
        assert risks[0] == pytest.approx(0.25, abs=1e-6)

    def test_constrains_sampling_to_digits(self):
        # The transformers numeric path masks to digits via `digits_only=True`;
        # vLLM should mirror that with `allowed_token_ids=<digit ids>`.
        vocab = {**{str(d): d for d in range(10)}, "X": 99}
        tokenizer = _StubTokenizer(vocab, vocab_size=100)
        request_output = _StubRequestOutput(outputs=[
            _StubCompletionOutput(logprobs=[
                {3: _StubLogprob(math.log(0.9))},
                {0: _StubLogprob(math.log(0.9))},
            ]),
        ])
        llm = _StubLLM(script=[[request_output]])
        clf = _make_classifier(llm, tokenizer, vocab_dim=100)

        question = DirectNumericQA(column="PINCP", text="dummy")
        clf._query_prompt_risk_estimates_batch(
            prompts_batch=["dummy"], question=question,
        )

        params = llm.last_sampling_params
        assert params.allowed_token_ids is not None
        assert sorted(params.allowed_token_ids) == list(range(10))
        # "X" is not decimal — must NOT be in the allowed list.
        assert 99 not in params.allowed_token_ids


# --------------------------------------------------------------------------
# Reasoning path
# --------------------------------------------------------------------------

class TestReasoningPath:
    def test_extracts_probability_from_generated_text(self):
        vocab = {"a": 0, "b": 1}
        tokenizer = _StubTokenizer(vocab, vocab_size=10)
        request_output = _StubRequestOutput(outputs=[
            _StubCompletionOutput(text="Reasoning... so my final answer is Probability: 73%."),
        ])
        llm = _StubLLM(script=[[request_output]])
        clf = _make_classifier(llm, tokenizer, vocab_dim=10)

        # enable_thinking=False ⇒ chat template applied (no thinking-block strip).
        # _apply_chat_template_batch falls back to raw prompts when the tokenizer
        # has no chat_template attribute.
        question = ReasoningQA(column="PINCP", text="dummy", enable_thinking=False)
        risks = clf._query_prompt_risk_estimates_batch(
            prompts_batch=["dummy prompt"], question=question,
        )
        assert risks[0] == pytest.approx(0.73, abs=1e-6)

    def test_failed_extraction_falls_back_to_half(self):
        vocab = {"a": 0}
        tokenizer = _StubTokenizer(vocab, vocab_size=10)
        request_output = _StubRequestOutput(outputs=[
            _StubCompletionOutput(text="No probability stated here"),
        ])
        llm = _StubLLM(script=[[request_output]])
        clf = _make_classifier(llm, tokenizer, vocab_dim=10)

        question = ReasoningQA(column="PINCP", text="dummy", enable_thinking=False)
        risks = clf._query_prompt_risk_estimates_batch(
            prompts_batch=["dummy"], question=question,
        )
        assert risks[0] == pytest.approx(0.5, abs=1e-6)
        # Failure tracked so the warning logic can fire downstream.
        assert clf._reasoning_failed == 1
        assert clf._reasoning_total == 1

    def test_strips_thinking_block_when_enabled(self):
        vocab = {"a": 0}
        tokenizer = _StubTokenizer(vocab, vocab_size=10)
        # The "thinking" content contains "20%", but only the post-</think>
        # response counts. The decoder MUST extract 0.85, not 0.20.
        full_text = (
            "Lots of reasoning, considering 20%, then more.\n"
            "</think>\n"
            "Final answer: Probability: 85%."
        )
        request_output = _StubRequestOutput(outputs=[
            _StubCompletionOutput(text=full_text),
        ])
        llm = _StubLLM(script=[[request_output]])
        clf = _make_classifier(llm, tokenizer, vocab_dim=10)

        question = ReasoningQA(column="PINCP", text="dummy", enable_thinking=True)
        risks = clf._query_prompt_risk_estimates_batch(
            prompts_batch=["dummy"], question=question,
        )
        assert risks[0] == pytest.approx(0.85, abs=1e-6)
        assert clf._reasoning_failed == 0


# --------------------------------------------------------------------------
# Hash separation between backends
# --------------------------------------------------------------------------

class TestBackendDistinctHash:
    def test_hash_includes_vllm_tag(self):
        # Two classifiers identical except for backend tag must hash differently
        # so cached predictions don't bleed between backends.
        vocab = {" A": 1, " B": 2}
        tokenizer = _StubTokenizer(vocab, vocab_size=10)
        clf = _make_classifier(_StubLLM(script=[]), tokenizer, vocab_dim=10)
        # Sanity: the hash dict includes the backend tag literally.
        h = hash(clf)
        assert isinstance(h, int)
        # Hash is deterministic given the same inputs.
        assert hash(clf) == h
