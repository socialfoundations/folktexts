"""Regression tests for ``WebAPILLMClassifier._query_webapi_batch``.

These exercise the system-prompt handling without a live web API: the
classifier is built with ``__new__`` (bypassing the litellm / API-key setup in
``__init__``) and the network call (``text_completion_api``) is replaced by a
recorder that captures the ``messages`` payload.

The key regression: for non-CoT questions, ``system_prompt`` must always be
bound. Before the fix it was assigned only inside
``if self.prompt_config.system_prompt is not None``, so a config with the
system role explicitly disabled (``system_prompt=None``) raised
``NameError: name 'system_prompt' is not defined``.
"""
from __future__ import annotations

import pytest

from folktexts.classifier import WebAPILLMClassifier
from folktexts.prompting import PromptConfig
from folktexts.qa_interface import ChainOfThoughtQA


@pytest.fixture(scope="module")
def mcq_task():
    from folktexts.acs import ACSTaskMetadata
    return ACSTaskMetadata.get_task("ACSIncome", use_numeric_qa=False)


def _make_classifier(prompt_config: PromptConfig):
    """Build a WebAPILLMClassifier without touching litellm / the network."""
    clf = WebAPILLMClassifier.__new__(WebAPILLMClassifier)
    clf._model_name = "test-model"
    clf._seed = 42
    clf._prompt_config = prompt_config
    clf._total_cost = 0  # consumed by __del__
    clf.max_api_rpm = 10**9  # make the inter-call sleep negligible
    clf.supported_params = {
        "temperature", "max_tokens", "stream", "seed", "logprobs", "top_logprobs",
    }

    calls: list[list[dict]] = []

    def _fake_completion(*, model, messages, **kwargs):
        calls.append(messages)
        return {"choices": [{"message": {"content": "Probability: 50%"}}]}

    clf.text_completion_api = _fake_completion
    return clf, calls


def _system_contents(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "system"]


# --- Non-CoT (MCQ / Numeric) -------------------------------------------------

def test_mcq_with_disabled_system_prompt_does_not_raise(mcq_task):
    """Regression: system_prompt=None previously raised NameError on MCQ."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task, system_prompt=None)
    assert cfg.system_prompt is None  # precondition: role disabled

    clf, calls = _make_classifier(cfg)
    clf._query_webapi_batch(["some prompt"], question=mcq_task.multiple_choice_qa)

    assert len(calls) == 1
    # System role is omitted entirely when disabled (no content=None turn).
    assert _system_contents(calls[0]) == []
    assert any(m["role"] == "user" for m in calls[0])


def test_numeric_with_disabled_system_prompt_does_not_raise(mcq_task):
    cfg = PromptConfig.from_dict(
        pv={}, task=mcq_task, question=mcq_task.direct_numeric_qa, system_prompt=None,
    )
    assert cfg.system_prompt is None

    clf, calls = _make_classifier(cfg)
    clf._query_webapi_batch(["p"], question=mcq_task.direct_numeric_qa)

    assert _system_contents(calls[0]) == []


def test_mcq_default_system_prompt_is_sent(mcq_task):
    """Default config carries the MCQ system prompt → emitted as the system turn."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)  # PROMPT_DEFAULT
    assert cfg.system_prompt is not None

    clf, calls = _make_classifier(cfg)
    clf._query_webapi_batch(["p"], question=mcq_task.multiple_choice_qa)

    sys_contents = _system_contents(calls[0])
    assert sys_contents == [cfg.system_prompt()]
    assert sys_contents[0]  # non-empty


def test_custom_system_prompt_overrides_default(mcq_task):
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task, system_prompt="CUSTOM SYS")
    clf, calls = _make_classifier(cfg)
    clf._query_webapi_batch(["p"], question=mcq_task.multiple_choice_qa)
    assert _system_contents(calls[0]) == ["CUSTOM SYS"]


# --- CoT ---------------------------------------------------------------------

def _cot_question(mcq_task) -> ChainOfThoughtQA:
    base = mcq_task.direct_numeric_qa
    return ChainOfThoughtQA(column=base.column, text=base.text, enable_thinking=False)


def test_cot_uses_default_instruction_when_no_system_prompt(mcq_task):
    """CoT default config has system_prompt=None → falls back to CoT instruction."""
    cfg = PromptConfig.from_dict(
        pv={}, task=mcq_task, question=_cot_question(mcq_task),
    )
    assert cfg.system_prompt is None  # CoT default ClassVar is None

    clf, calls = _make_classifier(cfg)
    clf._query_webapi_batch(["p"], question=_cot_question(mcq_task))

    sys_contents = _system_contents(calls[0])
    assert len(sys_contents) == 1
    assert "step-by-step" in sys_contents[0].lower()
    assert "probability" in sys_contents[0].lower()


def test_cot_threads_custom_system_prompt(mcq_task):
    """Regression: --system-prompt was ignored on the web_api CoT path."""
    cfg = PromptConfig.from_dict(
        pv={}, task=mcq_task, question=_cot_question(mcq_task),
        system_prompt="MY COT SYSTEM PROMPT",
    )
    assert cfg.system_prompt is not None

    clf, calls = _make_classifier(cfg)
    clf._query_webapi_batch(["p"], question=_cot_question(mcq_task))

    assert _system_contents(calls[0]) == ["MY COT SYSTEM PROMPT"]
