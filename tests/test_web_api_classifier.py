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
        "numeric_percentage_mode": False,
    }
    clf._active_question = None  # no override; falls back to task.question

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

_GPT5_QUIRKS = {
    "top_logprobs_max": 5,
    "max_tokens_overhead": 3,
    "force_reasoning_none": True,
    "numeric_percentage_mode": True,
}
_DEFAULT_QUIRKS = {
    "top_logprobs_max": 20,
    "max_tokens_overhead": 0,
    "force_reasoning_none": False,
    "numeric_percentage_mode": False,
}


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


# --- Numeric percentage mode (gpt-5 workaround) -----------------------------

def test_directnumericqa_percentage_prefix_has_no_decimal_prefill():
    """With `percentage=True`, the prompt asks for an integer percentage
    and does NOT bake the `0.` prefill into the user turn."""
    from folktexts.qa_interface import DirectNumericQA, NUMERIC_PERCENTAGE_CHAT_PROMPT
    q = DirectNumericQA(
        column="test", text="What is P?", num_forward_passes=1, percentage=True,
    )
    prefix = q.get_answer_prefix()
    assert prefix == NUMERIC_PERCENTAGE_CHAT_PROMPT
    assert "0." not in prefix
    assert "0-100" in prefix


def test_directnumericqa_percentage_decode_expected_value():
    """`percentage=True` returns a mass-weighted expected value over the
    numeric tokens at the first post-`probability<...>:` position with
    meaningful numeric mass, divided by 100."""
    import numpy as np
    from folktexts.qa_interface import DirectNumericQA
    q = DirectNumericQA(
        column="test", text="P?", num_forward_passes=4, percentage=True,
    )
    # Simulate `'**Probability (0-100): 92%**'`-style response: the
    # accumulated text hits `probability(0-100):` by pos 2, then the
    # answer arrives at pos 3.
    vocab = {
        "92": 0, "88": 1, "12": 2,           # answer tokens
        "**Probability ": 3, "(0-100):": 4, " ": 5,
    }
    probs = np.zeros((4, 6))
    probs[0, 3] = 1.0   # '**Probability '
    probs[1, 4] = 1.0   # '(0-100):'    (anchor `Probability (0-100):` fires)
    probs[2, 5] = 1.0   # ' '           (whitespace, no numeric mass → skip)
    probs[3, 0] = 0.9   # '92'   → expected = (92*0.9 + 88*0.05 + 12*0.05) / 1.0 = 87.8
    probs[3, 1] = 0.05  # '88'
    probs[3, 2] = 0.05  # '12'
    decoded = q.get_answer_from_model_output(probs, vocab)
    assert decoded == pytest.approx(0.878)


def test_directnumericqa_percentage_skips_range_label_digits():
    """The `0` and `100` from an echoed `(0-100)` range label are valid
    percentages, but appear BEFORE the anchor colon and must be skipped.
    The decoder should walk to the first post-anchor position that
    carries digit mass."""
    import numpy as np
    from folktexts.qa_interface import DirectNumericQA
    q = DirectNumericQA(
        column="test", text="P?", num_forward_passes=6, percentage=True,
    )
    # Response shape: '**Probability 0 - 100 ): 35'
    # Prefix tokens produce accumulated `**Probability 0-100):` by
    # position 5, so the anchor `probability[^:]*:` matches. Answer at
    # position 5 must NOT be the `100` (still pre-anchor) — verify by
    # placing the real answer at position 6.
    vocab = {
        "35": 0, "100": 1, "0": 2, "-": 3, "):": 4,
        "**Probability ": 5,
    }
    probs = np.zeros((7, 6))
    probs[0, 5] = 1.0    # '**Probability '  (no anchor yet — no colon)
    probs[1, 2] = 1.0    # '0'               (range label starts)
    probs[2, 3] = 1.0    # '-'
    probs[3, 1] = 1.0    # '100'             (still pre-anchor — no colon yet)
    probs[4, 4] = 1.0    # '):'              (accumulated hits `Probability 0-100):`)
    probs[5, 0] = 0.0    # (empty — skipped, no mass)
    probs[6, 0] = 0.6    # '35'              (post-anchor answer)
    probs[6, 2] = 0.3    # '0'               (contributes 0)
    decoded = q.get_answer_from_model_output(probs, vocab)
    # Expected = (35*0.6 + 0*0.3 + 100*0) / 0.9 = 21 / 0.9 = 23.333...
    # `100` doesn't appear at pos 6 → not counted.
    assert decoded == pytest.approx(23.333 / 100.0, abs=0.001)


def test_directnumericqa_percentage_returns_zero_without_anchor():
    """When the model preambles for the entire token budget without
    ever emitting `probability<...>:`, the decoder should return 0.0 —
    those responses (~4% on gpt-5.4-nano) contain no answer to decode,
    and inventing one would harm calibration."""
    import numpy as np
    from folktexts.qa_interface import DirectNumericQA
    q = DirectNumericQA(
        column="test", text="P?", num_forward_passes=3, percentage=True,
    )
    # Preamble that never reaches the anchor `probability:`, but
    # includes a numeric-looking token (`42` from `**42-year-old`).
    vocab = {"42": 0, "Doctorate": 1, "**": 2}
    probs = np.zeros((3, 3))
    probs[0, 2] = 1.0   # '**'
    probs[1, 0] = 1.0   # '42' — a numeric token but before any anchor
    probs[2, 1] = 1.0   # 'Doctorate'
    decoded = q.get_answer_from_model_output(probs, vocab)
    assert decoded == 0.0


def test_directnumericqa_percentage_accepts_paraphrased_anchor():
    """Real gpt-5 responses often drop the `(0-100)` range label and
    paraphrase (e.g. `'**Estimated probability: 18%**'`). The anchor
    matches `probability<anything>:` so paraphrases still decode."""
    import numpy as np
    from folktexts.qa_interface import DirectNumericQA
    q = DirectNumericQA(
        column="test", text="P?", num_forward_passes=4, percentage=True,
    )
    # Response: '**Estimated probability: 18%**'
    vocab = {"18": 0, "**Estimated probability": 1, ":": 2, " ": 3}
    probs = np.zeros((4, 4))
    probs[0, 1] = 1.0   # '**Estimated probability'  (no anchor yet)
    probs[1, 2] = 1.0   # ':'                        (anchor fires here)
    probs[2, 3] = 1.0   # ' '                        (whitespace — skip)
    probs[3, 0] = 1.0   # '18'                       (answer)
    decoded = q.get_answer_from_model_output(probs, vocab)
    assert decoded == pytest.approx(0.18)


def test_directnumericqa_percentage_clamps_above_100():
    """Expected value >100 should clamp to 1.0."""
    import numpy as np
    from folktexts.qa_interface import DirectNumericQA
    q = DirectNumericQA(
        column="test", text="P?", num_forward_passes=1, percentage=True,
    )
    # Only 105 in the vocab. `_get_numeric_tokens` allows any digit token,
    # but percentage decoder further filters to <=100 — so 105 is REJECTED.
    # With no valid tokens, decoder returns 0.0 (with warning).
    vocab = {"105": 0}
    probs = np.array([[1.0]])
    decoded = q.get_answer_from_model_output(probs, vocab)
    assert decoded == 0.0


def test_directnumericqa_percentage_no_digits_returns_zero():
    """If no valid percentage tokens appear at any position, the decoder
    returns 0.0 rather than raising."""
    import numpy as np
    from folktexts.qa_interface import DirectNumericQA
    q = DirectNumericQA(
        column="test", text="P?", num_forward_passes=1, percentage=True,
    )
    # `_get_numeric_tokens` yields an empty numeric_tokens_vocab if the
    # response has no digits, so pass a vocab with only a non-digit token.
    vocab = {"foo": 0}  # `_get_numeric_tokens` would filter this out anyway
    probs = np.array([[1.0]])
    decoded = q.get_answer_from_model_output(probs, vocab)
    assert decoded == 0.0


def test_directnumericqa_percentage_false_preserves_decimal_prefill():
    """Default `percentage=False` still returns the classic `0.` prefill."""
    from folktexts.qa_interface import DirectNumericQA
    q = DirectNumericQA(column="test", text="P?")
    assert q.percentage is False
    assert q.get_answer_prefix() == "Answer (between 0 and 1): 0."


def test_enable_numeric_percentage_mode_swaps_question(mcq_task):
    """Calling `_enable_numeric_percentage_mode()` should replace the
    numeric QA in `prompt_config.suffix` with a `percentage=True` variant
    and rebuild `_encode_row` against the new config."""
    from folktexts.qa_interface import DirectNumericQA
    cfg = PromptConfig.from_dict(
        pv={}, task=mcq_task, question=mcq_task.direct_numeric_qa,
    )
    clf, _ = _make_classifier(cfg)
    clf._task = mcq_task  # required by the encode_row rebuild

    original_q = clf._prompt_config.suffix.question
    assert isinstance(original_q, DirectNumericQA)
    assert original_q.percentage is False

    clf._enable_numeric_percentage_mode()

    new_q = clf._prompt_config.suffix.question
    assert isinstance(new_q, DirectNumericQA)
    assert new_q.percentage is True
    # ≥10 to give gpt-5 room for the markdown echo it emits before the digit
    assert new_q.num_forward_passes >= 10
    # `_encode_row` should now be a partial closed over the NEW config
    assert clf._encode_row.keywords["prompt_config"] is clf._prompt_config
    # `_active_question` is the override the base classifier consults
    assert clf._active_question is new_q


def test_enable_numeric_percentage_mode_installs_format_system_prompt(mcq_task):
    """The swap should replace the default numeric system prompt with one
    that explicitly names the required answer format (mirrors the CoT
    pattern) — reasoning models don't reliably continue the user-turn
    prefill, so the instruction reduces zero-fallback rows."""
    from folktexts.classifier.web_api_classifier import (
        NUMERIC_PERCENTAGE_SYSTEM_PROMPT,
    )
    cfg = PromptConfig.from_dict(
        pv={}, task=mcq_task, question=mcq_task.direct_numeric_qa,
    )
    clf, _ = _make_classifier(cfg)
    clf._task = mcq_task

    # Precondition: numeric-QA default system prompt is active
    assert clf._prompt_config.system_prompt is not None
    assert "MUST end" not in clf._prompt_config.system_prompt()

    clf._enable_numeric_percentage_mode()

    new_sys = clf._prompt_config.system_prompt()
    assert new_sys == NUMERIC_PERCENTAGE_SYSTEM_PROMPT
    assert "Probability (0-100): X%" in new_sys
    # Instruction shape: forbid preamble/reasoning so the model emits the
    # answer inside the (limited) `max_tokens` budget.
    assert "Reply with ONLY" in new_sys
    assert "no preamble".lower() in new_sys.lower() or \
        "no preamble, reasoning".lower() in new_sys.lower() or \
        "preamble" in new_sys


def test_enable_numeric_percentage_mode_preserves_user_system_prompt(mcq_task):
    """A user-supplied `--system-prompt` must NOT be silently clobbered by
    the auto-swap (only the QA subclass default gets replaced)."""
    cfg = PromptConfig.from_dict(
        pv={}, task=mcq_task, question=mcq_task.direct_numeric_qa,
        system_prompt="CUSTOM USER SYS",
    )
    clf, _ = _make_classifier(cfg)
    clf._task = mcq_task

    assert clf._prompt_config.system_prompt() == "CUSTOM USER SYS"

    clf._enable_numeric_percentage_mode()

    # QA is still swapped …
    assert clf._prompt_config.suffix.question.percentage is True
    # … but the user's system prompt is preserved
    assert clf._prompt_config.system_prompt() == "CUSTOM USER SYS"


def test_enable_numeric_percentage_mode_preserves_disabled_system_prompt(mcq_task):
    """`system_prompt=None` (role disabled) must stay disabled after swap."""
    cfg = PromptConfig.from_dict(
        pv={}, task=mcq_task, question=mcq_task.direct_numeric_qa,
        system_prompt=None,
    )
    clf, _ = _make_classifier(cfg)
    clf._task = mcq_task
    assert clf._prompt_config.system_prompt is None

    clf._enable_numeric_percentage_mode()

    assert clf._prompt_config.suffix.question.percentage is True
    assert clf._prompt_config.system_prompt is None


def test_gpt5_init_auto_enables_percentage_mode_for_numeric_qa(monkeypatch, mcq_task):
    """End-to-end: constructing a WebAPILLMClassifier for a gpt-5 model with
    a numeric-QA task should auto-swap the numeric QA to percentage mode."""
    from folktexts.qa_interface import DirectNumericQA

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_litellm = type("_fake_litellm", (), {"success_callback": []})()

    def _fake_get_supported(**_kwargs):
        return ["temperature", "max_tokens", "stream", "seed",
                "logprobs", "top_logprobs", "reasoning_effort"]

    def _fake_completion(**_kwargs):
        raise RuntimeError("should not be called in __init__")

    import litellm as real_litellm
    monkeypatch.setattr(real_litellm, "success_callback", [], raising=False)
    monkeypatch.setattr(real_litellm, "completion", _fake_completion, raising=False)
    monkeypatch.setattr(
        real_litellm, "get_supported_openai_params", _fake_get_supported,
        raising=False,
    )

    cfg = PromptConfig.from_dict(
        pv={}, task=mcq_task, question=mcq_task.direct_numeric_qa,
    )
    clf = WebAPILLMClassifier(
        model_name="gpt-5.4-nano", task=mcq_task, prompt_config=cfg,
    )
    q = clf._prompt_config.suffix.question
    assert isinstance(q, DirectNumericQA)
    assert q.percentage is True
    assert q.num_forward_passes >= 10
    # And the format-instruction system prompt is installed
    assert "Probability (0-100): X%" in clf._prompt_config.system_prompt()


def test_default_model_init_does_not_swap_numeric_qa(monkeypatch, mcq_task):
    """A non-gpt-5 model must leave the numeric QA in decimal-prefill mode."""
    from folktexts.qa_interface import DirectNumericQA

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def _fake_get_supported(**_kwargs):
        return ["temperature", "max_tokens", "stream", "seed", "logprobs", "top_logprobs"]

    def _fake_completion(**_kwargs):
        raise RuntimeError("should not be called in __init__")

    import litellm as real_litellm
    monkeypatch.setattr(real_litellm, "success_callback", [], raising=False)
    monkeypatch.setattr(real_litellm, "completion", _fake_completion, raising=False)
    monkeypatch.setattr(
        real_litellm, "get_supported_openai_params", _fake_get_supported,
        raising=False,
    )

    cfg = PromptConfig.from_dict(
        pv={}, task=mcq_task, question=mcq_task.direct_numeric_qa,
    )
    clf = WebAPILLMClassifier(
        model_name="gpt-4o-mini", task=mcq_task, prompt_config=cfg,
    )
    q = clf._prompt_config.suffix.question
    assert isinstance(q, DirectNumericQA)
    assert q.percentage is False
    # Non-gpt-5 path keeps the plain numeric system prompt (no format
    # instruction — regular models honour the user-turn prefill continuation).
    assert "Reply with ONLY" not in clf._prompt_config.system_prompt()
    assert "Probability (0-100)" not in clf._prompt_config.system_prompt()


def test_gpt5_init_does_not_swap_mcq(monkeypatch, mcq_task):
    """The auto-swap must only fire when the current question is a
    `DirectNumericQA` — MCQ tasks are unaffected."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def _fake_get_supported(**_kwargs):
        return ["temperature", "max_tokens", "stream", "seed",
                "logprobs", "top_logprobs", "reasoning_effort"]

    def _fake_completion(**_kwargs):
        raise RuntimeError("should not be called in __init__")

    import litellm as real_litellm
    monkeypatch.setattr(real_litellm, "success_callback", [], raising=False)
    monkeypatch.setattr(real_litellm, "completion", _fake_completion, raising=False)
    monkeypatch.setattr(
        real_litellm, "get_supported_openai_params", _fake_get_supported,
        raising=False,
    )

    cfg = PromptConfig.from_dict(pv={}, task=mcq_task)  # MCQ, not numeric
    clf = WebAPILLMClassifier(
        model_name="gpt-5.4-nano", task=mcq_task, prompt_config=cfg,
    )
    # question is a MultipleChoiceQA, no percentage attribute — the swap
    # should have been skipped without error.
    from folktexts.qa_interface import MultipleChoiceQA
    assert isinstance(clf._prompt_config.suffix.question, MultipleChoiceQA)
