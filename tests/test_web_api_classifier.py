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


def _make_classifier(prompt_config: PromptConfig, *, supported_params: set | None = None):
    """Build a WebAPILLMClassifier without touching litellm / the network."""
    clf = WebAPILLMClassifier.__new__(WebAPILLMClassifier)
    clf._model_name = "test-model"
    clf._seed = 42
    clf._prompt_config = prompt_config
    clf._temperature = None  # no override → use each question's default_temperature
    clf._total_cost = 0  # consumed by __del__
    clf.max_api_rpm = 10**9  # make the inter-call sleep negligible
    clf.supported_params = supported_params if supported_params is not None else {
        "temperature", "max_tokens", "stream", "seed", "logprobs", "top_logprobs",
        "reasoning_effort",
    }
    clf._warned_unsupported_params = set()
    clf._family_quirks = {  # default quirks (matches _DEFAULT_FAMILY_QUIRKS)
        "top_logprobs_max": 20,
        "max_tokens_overhead": 0,
        "force_reasoning_none": False,
    }

    calls: list[list[dict]] = []

    def _fake_completion(*, model, messages, **kwargs):
        calls.append(messages)
        clf.last_call_params = kwargs  # capture the resolved api_call_params
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


# --- Temperature contract ----------------------------------------------------

def test_mcq_uses_temperature_zero(mcq_task):
    """MCQ (token-probability) defaults to temperature 0 for determinism."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)
    clf, _ = _make_classifier(cfg)
    clf._query_webapi_batch(["p"], question=mcq_task.multiple_choice_qa)
    assert clf.last_call_params["temperature"] == 0.0


def test_numeric_uses_temperature_zero(mcq_task):
    """Direct-numeric (token-probability) defaults to temperature 0."""
    cfg = PromptConfig.from_dict(
        pv={}, task=mcq_task, question=mcq_task.direct_numeric_qa,
    )
    clf, _ = _make_classifier(cfg)
    clf._query_webapi_batch(["p"], question=mcq_task.direct_numeric_qa)
    assert clf.last_call_params["temperature"] == 0.0


def test_cot_without_thinking_defaults_to_greedy(mcq_task):
    """Plain CoT defaults to greedy — sampling collapses small models' output
    format (regex fallbacks) and previous CoT results were produced greedy."""
    cfg = PromptConfig.from_dict(
        pv={}, task=mcq_task, question=_cot_question(mcq_task),
    )
    clf, _ = _make_classifier(cfg)
    clf._query_webapi_batch(["p"], question=_cot_question(mcq_task))
    assert clf.last_call_params["temperature"] == 0.0


def test_cot_with_thinking_defaults_to_one(mcq_task):
    """Thinking-mode CoT defaults to sampling (greedy is discouraged there)."""
    base = mcq_task.direct_numeric_qa
    q = ChainOfThoughtQA(column=base.column, text=base.text, enable_thinking=True)
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task, question=q)
    clf, _ = _make_classifier(cfg)
    clf._query_webapi_batch(["p"], question=q)
    assert clf.last_call_params["temperature"] == 1.0


def test_explicit_temperature_override_applies_to_cot_only(mcq_task):
    """An explicit temperature overrides the CoT default; MC/numeric always
    read untempered token probabilities and stay at 0."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)
    clf, _ = _make_classifier(cfg)
    clf._temperature = 0.25  # explicit override

    clf._query_webapi_batch(["p"], question=_cot_question(mcq_task))
    assert clf.last_call_params["temperature"] == 0.25

    clf._query_webapi_batch(["p"], question=mcq_task.multiple_choice_qa)
    assert clf.last_call_params["temperature"] == 0


# --- Unsupported-parameter filtering -----------------------------------------

def test_unsupported_temperature_is_filtered_with_warning(mcq_task, caplog):
    """Models that reject `temperature` (e.g. o1/o3) must have it dropped, not raise."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)
    # Model supports everything EXCEPT temperature.
    clf, _ = _make_classifier(
        cfg,
        supported_params={"max_tokens", "stream", "seed", "logprobs", "top_logprobs"},
    )

    with caplog.at_level("WARNING"):
        clf._query_webapi_batch(["p"], question=mcq_task.multiple_choice_qa)

    # temperature was dropped from the outgoing request instead of raising.
    assert "temperature" not in clf.last_call_params
    # The drop is visible in the logs and names the offending parameter.
    assert any(
        "temperature" in rec.getMessage() and rec.levelname == "WARNING"
        for rec in caplog.records
    )


def test_unsupported_param_filtering_applies_to_cot(mcq_task, caplog):
    """Filtering must also cover the CoT path (previously exempt from the check)."""
    cfg = PromptConfig.from_dict(
        pv={}, task=mcq_task, question=_cot_question(mcq_task),
    )
    clf, _ = _make_classifier(
        cfg,
        supported_params={"max_tokens", "stream"},  # no temperature, no seed
    )

    with caplog.at_level("WARNING"):
        clf._query_webapi_batch(["p"], question=_cot_question(mcq_task))

    assert "temperature" not in clf.last_call_params
    assert "seed" not in clf.last_call_params
    assert "max_tokens" in clf.last_call_params  # supported params survive


def test_all_supported_params_pass_through_unchanged(mcq_task, caplog):
    """No warning and no dropped keys when every param is supported."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)
    clf, _ = _make_classifier(cfg)  # default supported_params covers all

    with caplog.at_level("WARNING"):
        clf._query_webapi_batch(["p"], question=mcq_task.multiple_choice_qa)

    assert "temperature" in clf.last_call_params
    assert not any(
        "does not support API parameter" in rec.getMessage()
        for rec in caplog.records
    )


def test_missing_logprobs_support_fails_fast_for_mcq(mcq_task):
    """`logprobs` is required to decode MCQ/numeric; dropping it must raise,
    not send a doomed request that only fails deep in response decoding."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)
    clf, calls = _make_classifier(
        cfg, supported_params={"temperature", "max_tokens", "stream", "seed"},
    )

    with pytest.raises(RuntimeError, match="logprobs"):
        clf._query_webapi_batch(["p"], question=mcq_task.multiple_choice_qa)
    assert calls == []  # no API call was made


# --- gpt-5 family quirks -----------------------------------------------------

_GPT5_QUIRKS = {"top_logprobs_max": 5, "max_tokens_overhead": 3, "force_reasoning_none": True}
_DEFAULT_QUIRKS = {"top_logprobs_max": 20, "max_tokens_overhead": 0, "force_reasoning_none": False}


@pytest.mark.parametrize("model_name, expected", [
    ("gpt-5.4-mini", _GPT5_QUIRKS),
    ("gpt-5.4-nano", _GPT5_QUIRKS),
    ("openai/gpt-5-turbo", _GPT5_QUIRKS),
    ("gpt-4o-mini", _DEFAULT_QUIRKS),
    ("claude-sonnet-5", _DEFAULT_QUIRKS),
    ("some-future-model", _DEFAULT_QUIRKS),
])
def test_resolve_family_quirks(model_name, expected):
    from folktexts.classifier.web_api_classifier import _resolve_family_quirks
    assert _resolve_family_quirks(model_name) == expected


def test_mcq_sends_family_specific_top_logprobs_cap(mcq_task):
    """gpt-5 models must receive top_logprobs=5 (any higher → OpenAI 400)."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)
    clf, _ = _make_classifier(cfg)
    clf._family_quirks = _GPT5_QUIRKS

    clf._query_webapi_batch(["p"], question=mcq_task.multiple_choice_qa)
    assert clf.last_call_params["top_logprobs"] == 5


def test_mcq_max_tokens_absorbs_overhead_for_gpt5(mcq_task):
    """gpt-5 has a ~3-token hidden preamble that consumes max_completion_tokens;
    a single-token MCQ needs `max_tokens = 1 + overhead = 4`."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)
    clf, _ = _make_classifier(cfg)
    clf._family_quirks = _GPT5_QUIRKS

    clf._query_webapi_batch(["p"], question=mcq_task.multiple_choice_qa)
    assert clf.last_call_params["max_tokens"] == 4


def test_numeric_max_tokens_absorbs_overhead_for_gpt5(mcq_task):
    """Numeric needs `num_forward_passes + 2 + overhead` visible tokens for gpt-5;
    without the overhead the model outputs only '0.' and decodes to 0.0."""
    cfg = PromptConfig.from_dict(
        pv={}, task=mcq_task, question=mcq_task.direct_numeric_qa,
    )
    clf, _ = _make_classifier(cfg)
    clf._family_quirks = _GPT5_QUIRKS
    expected = mcq_task.direct_numeric_qa.num_forward_passes + 2 + 3

    clf._query_webapi_batch(["p"], question=mcq_task.direct_numeric_qa)
    assert clf.last_call_params["max_tokens"] == expected


def test_mcq_default_max_tokens_unchanged_for_non_gpt5(mcq_task):
    """Non-gpt-5 models keep max_tokens=1 for single-token MCQ (no overhead)."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)
    clf, _ = _make_classifier(cfg)  # default quirks
    clf._query_webapi_batch(["p"], question=mcq_task.multiple_choice_qa)
    assert clf.last_call_params["max_tokens"] == 1


def test_mcq_injects_reasoning_effort_none_for_gpt5(mcq_task):
    """gpt-5 needs reasoning_effort='none' or logprobs requests are refused."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)
    clf, _ = _make_classifier(cfg)
    clf._family_quirks = _GPT5_QUIRKS

    clf._query_webapi_batch(["p"], question=mcq_task.multiple_choice_qa)
    assert clf.last_call_params.get("reasoning_effort") == "none"


def test_numeric_injects_reasoning_effort_none_for_gpt5(mcq_task):
    cfg = PromptConfig.from_dict(
        pv={}, task=mcq_task, question=mcq_task.direct_numeric_qa,
    )
    clf, _ = _make_classifier(cfg)
    clf._family_quirks = _GPT5_QUIRKS

    clf._query_webapi_batch(["p"], question=mcq_task.direct_numeric_qa)
    assert clf.last_call_params.get("reasoning_effort") == "none"


def test_cot_does_not_inject_reasoning_effort_even_for_gpt5(mcq_task):
    """CoT reads free-form text — reasoning budget should stay at the model default."""
    cfg = PromptConfig.from_dict(
        pv={}, task=mcq_task, question=_cot_question(mcq_task),
    )
    clf, _ = _make_classifier(cfg)
    clf._family_quirks = _GPT5_QUIRKS

    clf._query_webapi_batch(["p"], question=_cot_question(mcq_task))
    assert "reasoning_effort" not in clf.last_call_params


def test_mcq_omits_reasoning_effort_for_non_gpt5(mcq_task):
    """Non-gpt-5 models must not receive reasoning_effort (gpt-4o-mini rejects it)."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)
    clf, _ = _make_classifier(cfg)  # default quirks
    clf._query_webapi_batch(["p"], question=mcq_task.multiple_choice_qa)
    assert "reasoning_effort" not in clf.last_call_params


def test_unsupported_param_warning_fires_once_across_batches(mcq_task, caplog):
    """The drop-warning must not spam the logs once per batch."""
    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)
    clf, _ = _make_classifier(
        cfg,
        supported_params={"max_tokens", "stream", "seed", "logprobs", "top_logprobs"},
    )

    with caplog.at_level("WARNING"):
        clf._query_webapi_batch(["p1"], question=mcq_task.multiple_choice_qa)
        clf._query_webapi_batch(["p2"], question=mcq_task.multiple_choice_qa)

    warnings = [
        rec for rec in caplog.records
        if "does not support API parameter" in rec.getMessage()
    ]
    assert len(warnings) == 1
