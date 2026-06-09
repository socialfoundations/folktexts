"""Tests for chat-template prompting (folktexts.prompting).

These tests exercise the layer added in PR #27: `apply_chat_template`,
`encode_row_prompt_chat`, `resolve_chat_defaults`, and the
`tokenizer_supports_system_prompt` probe.

Tokenizer-dependent tests use a local copy of
`meta-llama/Llama-3.2-3B-Instruct`. They are skipped if the snapshot is not
present locally (we never download from the Hub during tests).
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from transformers import AutoTokenizer

from folktexts.prompting import (
    PromptConfig,
    VaryFormat,
    VarySuffix,
    apply_chat_template,
    encode_row_prompt,
    encode_row_prompt_chat,
    resolve_chat_defaults,
    tokenizer_supports_system_prompt,
)
from folktexts.qa_interface import (
    ANTHROPIC_CHAT_PROMPT,
    NUMERIC_CHAT_PROMPT,
    NUMERIC_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    ChainOfThoughtQA,
    Choice,
    DirectNumericQA,
    MultipleChoiceQA,
)

# Local snapshot dir for a chat-tuned tokenizer. Override via env var if needed.
LOCAL_CHAT_TOKENIZER_PATH = Path(
    os.environ.get(
        "FOLKTEXTS_TEST_CHAT_TOKENIZER",
        Path.home() / "huggingface-models" / "meta-llama--Llama-3.2-3B-Instruct",
    )
)

# Jinja template that mimics Gemma's behavior of rejecting the system role.
GEMMA_LIKE_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ raise_exception('System role not supported') }}"
    "{% endif %}"
    "{% for m in messages %}"
    "<|{{ m['role'] }}|>{{ m['content'] }}<|end|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)

# Template that drops the assistant prefill entirely; used to verify the
# rfind-miss guard in `apply_chat_template`.
PREFILL_DROPPING_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] != 'assistant' %}"
    "<|{{ m['role'] }}|>{{ m['content'] }}<|end|>"
    "{% endif %}"
    "{% endfor %}"
)

# Minimal "what you pass is what you get" template — emits exactly the roles
# and content provided, with no auto-injected system metadata. Lets us check
# whether a role was actually included without fighting model-specific
# template quirks (Llama auto-injects a "Cutting Knowledge Date" system block
# even when no system message is supplied, which makes role-omission asserts
# brittle on the real chat template).
MINIMAL_TEMPLATE = (
    "{% for m in messages %}"
    "<|{{ m['role'] }}|>{{ m['content'] }}<|end|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def chat_tokenizer() -> AutoTokenizer:
    """Real chat-tuned tokenizer (Llama-3.2-3B-Instruct), loaded from disk."""
    if not LOCAL_CHAT_TOKENIZER_PATH.exists():
        pytest.skip(
            f"Local chat tokenizer not found at {LOCAL_CHAT_TOKENIZER_PATH}; "
            "set FOLKTEXTS_TEST_CHAT_TOKENIZER to a local snapshot dir."
        )
    return AutoTokenizer.from_pretrained(
        str(LOCAL_CHAT_TOKENIZER_PATH), local_files_only=True
    )


@pytest.fixture
def gemma_like_tokenizer(chat_tokenizer) -> AutoTokenizer:
    """A tokenizer whose chat template raises on system messages.

    Built by re-loading the chat tokenizer and swapping in a Gemma-style
    Jinja template, so we don't need a real Gemma snapshot on disk.
    """
    tok = AutoTokenizer.from_pretrained(
        str(LOCAL_CHAT_TOKENIZER_PATH), local_files_only=True
    )
    tok.chat_template = GEMMA_LIKE_TEMPLATE
    return tok


@pytest.fixture
def prefill_dropping_tokenizer(chat_tokenizer) -> AutoTokenizer:
    """A tokenizer whose template silently omits the assistant prefill."""
    tok = AutoTokenizer.from_pretrained(
        str(LOCAL_CHAT_TOKENIZER_PATH), local_files_only=True
    )
    tok.chat_template = PREFILL_DROPPING_TEMPLATE
    return tok


@pytest.fixture
def minimal_tokenizer(chat_tokenizer) -> AutoTokenizer:
    """A tokenizer whose chat template emits exactly the provided roles/content."""
    tok = AutoTokenizer.from_pretrained(
        str(LOCAL_CHAT_TOKENIZER_PATH), local_files_only=True
    )
    tok.chat_template = MINIMAL_TEMPLATE
    return tok


@pytest.fixture
def fake_task() -> MagicMock:
    """Minimal TaskMetadata stand-in for prompting tests.

    `encode_row_prompt` only touches `task.get_row_description(row)` and
    `task.question.get_question_prompt()`, so a MagicMock is sufficient.
    """
    age_col = MagicMock()
    age_col.short_description = "Age"
    age_col.__getitem__ = lambda self, k: f"{k} years old"
    occp_col = MagicMock()
    occp_col.short_description = "Occupation"
    occp_col.__getitem__ = lambda self, k: str(k)

    task = MagicMock()
    task.name = "ACSFakeTask"
    task.features = ["AGE", "OCCP"]
    task.cols_to_text = {"AGE": age_col, "OCCP": occp_col}
    task.question.get_question_prompt.return_value = (
        "Question: What is the income bracket?\nA. <50k\nB. >=50k\nAnswer:"
    )
    task.question.get_answer_prefix.return_value = "Answer:"
    task.question.default_system_prompt = SYSTEM_PROMPT
    task.question.default_chat_prompt = ANTHROPIC_CHAT_PROMPT
    return task


@pytest.fixture
def sample_row() -> pd.Series:
    return pd.Series({"AGE": 35, "OCCP": "Engineer"})


# ----------------------------------------------------------------------
# resolve_chat_defaults — pure unit tests
# ----------------------------------------------------------------------


class TestResolveChatDefaults:
    def test_mc_mode_uses_mc_defaults(self):
        mc_q = MultipleChoiceQA(
            column="X",
            text="Q?",
            choices=(Choice("No", 0), Choice("Yes", 1)),
        )
        sys_p, chat_p = resolve_chat_defaults(question=mc_q)
        assert sys_p == SYSTEM_PROMPT
        assert chat_p == ANTHROPIC_CHAT_PROMPT

    def test_numeric_mode_uses_numeric_defaults(self):
        numeric_q = DirectNumericQA(column="X", text="Q?")
        sys_p, chat_p = resolve_chat_defaults(question=numeric_q)
        assert sys_p == NUMERIC_SYSTEM_PROMPT
        assert chat_p == NUMERIC_CHAT_PROMPT

    def test_explicit_values_are_returned_unchanged(self):
        numeric_q = DirectNumericQA(column="X", text="Q?")
        sys_p, chat_p = resolve_chat_defaults(
            question=numeric_q,
            system_prompt="custom system",
            chat_prompt="custom prefill",
        )
        assert sys_p == "custom system"
        assert chat_p == "custom prefill"

    def test_partial_override_only_replaces_provided_field(self):
        mc_q = MultipleChoiceQA(
            column="X",
            text="Q?",
            choices=(Choice("No", 0), Choice("Yes", 1)),
        )
        sys_p, chat_p = resolve_chat_defaults(
            question=mc_q,
            system_prompt="custom system",
        )
        assert sys_p == "custom system"
        assert chat_p == ANTHROPIC_CHAT_PROMPT

    def test_cot_mode_returns_none_none(self):
        cot_q = ChainOfThoughtQA(column="X", text="Q?")
        sys_p, chat_p = resolve_chat_defaults(question=cot_q)
        assert sys_p is None
        assert chat_p is None


# ----------------------------------------------------------------------
# tokenizer_supports_system_prompt
# ----------------------------------------------------------------------


class TestTokenizerSupportsSystemPrompt:
    def test_chat_tokenizer_supports_system(self, chat_tokenizer):
        assert tokenizer_supports_system_prompt(chat_tokenizer) is True

    def test_gemma_like_tokenizer_rejects_system(self, gemma_like_tokenizer):
        assert tokenizer_supports_system_prompt(gemma_like_tokenizer) is False


# ----------------------------------------------------------------------
# apply_chat_template
# ----------------------------------------------------------------------


class TestApplyChatTemplate:
    def test_full_conversation_assembles_all_three_roles(self, chat_tokenizer):
        out = apply_chat_template(
            chat_tokenizer,
            user_prompt="USER_CONTENT_MARKER",
            system_prompt="SYSTEM_CONTENT_MARKER",
            chat_prompt="PREFILL_MARKER",
        )
        assert "SYSTEM_CONTENT_MARKER" in out
        assert "USER_CONTENT_MARKER" in out
        assert out.endswith("PREFILL_MARKER"), (
            "After trimming, the prompt must end exactly with the assistant "
            "prefill so LLMClassifier reads probabilities at the right token."
        )

    def test_system_omitted_when_none(self, minimal_tokenizer):
        # Use the minimal template so we can directly assert that no system
        # role tag is emitted — Llama's real template auto-injects default
        # system metadata, which would mask whether our code added one.
        out = apply_chat_template(
            minimal_tokenizer,
            user_prompt="USER_CONTENT_MARKER",
            system_prompt=None,
            chat_prompt="PREFILL_MARKER",
        )
        assert "<|system|>" not in out
        assert "<|user|>USER_CONTENT_MARKER<|end|>" in out
        assert out.endswith("PREFILL_MARKER")

    def test_system_included_when_provided(self, minimal_tokenizer):
        out = apply_chat_template(
            minimal_tokenizer,
            user_prompt="USER_CONTENT_MARKER",
            system_prompt="SYSTEM_CONTENT_MARKER",
            chat_prompt="PREFILL_MARKER",
        )
        assert "<|system|>SYSTEM_CONTENT_MARKER<|end|>" in out
        assert out.endswith("PREFILL_MARKER")

    def test_no_chat_prompt_uses_generation_prompt(self, chat_tokenizer):
        out = apply_chat_template(
            chat_tokenizer,
            user_prompt="USER_CONTENT_MARKER",
            system_prompt="SYSTEM_CONTENT_MARKER",
            chat_prompt=None,
        )
        # Without a prefill, Llama's template emits a trailing assistant header.
        assert "USER_CONTENT_MARKER" in out
        assert "SYSTEM_CONTENT_MARKER" in out
        assert "<|start_header_id|>assistant<|end_header_id|>" in out

    def test_raises_when_prefill_missing_from_output(self, prefill_dropping_tokenizer):
        # Regression test for the silent-truncation bug: if the template strips
        # or transforms the prefill, rfind returns -1 and the old code would
        # have sliced the output to garbage. We now raise a clear error.
        with pytest.raises(ValueError, match="prefill not found"):
            apply_chat_template(
                prefill_dropping_tokenizer,
                user_prompt="USER",
                system_prompt="SYS",
                chat_prompt="THIS_PREFILL_GETS_DROPPED",
            )

    def test_trims_trailing_template_tokens_after_prefill(self, chat_tokenizer):
        # Llama's template appends an `<|eot_id|>` after assistant content.
        # Verify the trim removes it, so the prompt ends with the prefill.
        prefill = "If had to select one of the options, my answer would be"
        out = apply_chat_template(
            chat_tokenizer,
            user_prompt="USER_CONTENT_MARKER",
            system_prompt="SYS",
            chat_prompt=prefill,
        )
        assert out.endswith(prefill)
        assert (
            "<|eot_id|>"
            not in out.split("USER_CONTENT_MARKER", 1)[1].split(prefill, 1)[1]
        )


# ----------------------------------------------------------------------
# encode_row_prompt_chat
# ----------------------------------------------------------------------


class TestEncodeRowPromptChat:
    def test_default_args_apply_mc_defaults(
        self, chat_tokenizer, fake_task, sample_row
    ):
        out = encode_row_prompt_chat(
            row=sample_row,
            task=fake_task,
            tokenizer=chat_tokenizer,
        )
        assert SYSTEM_PROMPT.strip() in out
        assert out.endswith(ANTHROPIC_CHAT_PROMPT)

    def test_numeric_question_applies_numeric_defaults(
        self, chat_tokenizer, fake_task, sample_row
    ):
        # Numeric defaults are driven by question type, not a separate flag.
        # This mirrors make_benchmark, which calls _configure_task_question
        # (setting task.question to DirectNumericQA) before building the
        # prompt_config, so the two are always in sync.
        numeric_qa = DirectNumericQA(column="PINCP", text="What is the income bracket?")
        out = encode_row_prompt_chat(
            row=sample_row,
            task=fake_task,
            tokenizer=chat_tokenizer,
            question=numeric_qa,
        )
        assert NUMERIC_SYSTEM_PROMPT.strip() in out
        assert out.endswith(NUMERIC_CHAT_PROMPT)
        # Cross-check that the MC defaults are NOT injected for a numeric question.
        assert SYSTEM_PROMPT.strip() not in out

    def test_explicit_system_prompt_none_does_not_reinject_default(
        self, minimal_tokenizer, fake_task, sample_row
    ):
        # Regression test for the Gemma-fallback bug: `Benchmark.make_benchmark`
        # explicitly passes `system_prompt=None` for tokenizers that reject the
        # system role. The pre-fix `encode_row_prompt_chat` re-ran the resolver
        # and silently reinstated SYSTEM_PROMPT, which would then crash the
        # chat template on Gemma-style tokenizers. With the sentinel-based
        # default, `None` must mean "explicitly disabled". We use the minimal
        # template so the assertion is unambiguous (no auto-injected system).
        out = encode_row_prompt_chat(
            row=sample_row,
            task=fake_task,
            tokenizer=minimal_tokenizer,
            system_prompt=None,
            chat_prompt=ANTHROPIC_CHAT_PROMPT,
        )
        assert "<|system|>" not in out
        assert SYSTEM_PROMPT.strip() not in out
        assert out.endswith(ANTHROPIC_CHAT_PROMPT)

    def test_explicit_system_prompt_none_with_gemma_like_tokenizer(
        self, gemma_like_tokenizer, fake_task, sample_row
    ):
        # End-to-end regression: replicate the Benchmark code path where
        # `tokenizer_supports_system_prompt` returns False and the caller
        # passes `system_prompt=None`. Must not raise.
        assert tokenizer_supports_system_prompt(gemma_like_tokenizer) is False
        out = encode_row_prompt_chat(
            row=sample_row,
            task=fake_task,
            tokenizer=gemma_like_tokenizer,
            system_prompt=None,
            chat_prompt=ANTHROPIC_CHAT_PROMPT,
        )
        # The Gemma-like template wraps content in <|user|>...<|end|> tags.
        assert "<|user|>" in out
        assert "<|system|>" not in out
        assert out.endswith(ANTHROPIC_CHAT_PROMPT)

    def test_explicit_chat_prompt_overrides_default(
        self, chat_tokenizer, fake_task, sample_row
    ):
        custom_prefill = "My final answer is"
        out = encode_row_prompt_chat(
            row=sample_row,
            task=fake_task,
            tokenizer=chat_tokenizer,
            chat_prompt=custom_prefill,
        )
        assert out.endswith(custom_prefill)
        assert ANTHROPIC_CHAT_PROMPT not in out

    def test_user_content_includes_row_description_and_question(
        self, chat_tokenizer, fake_task, sample_row
    ):
        out = encode_row_prompt_chat(
            row=sample_row,
            task=fake_task,
            tokenizer=chat_tokenizer,
        )
        assert "35 years old" in out
        assert "Engineer" in out
        assert "What is the income bracket?" in out

    def test_custom_system_prompt_appears_in_output(
        self, minimal_tokenizer, acs_income_task, acs_row
    ):
        custom_system = "You are a test assistant."
        out = encode_row_prompt_chat(
            acs_row,
            task=acs_income_task,
            tokenizer=minimal_tokenizer,
            system_prompt=custom_system,
            chat_prompt=ANTHROPIC_CHAT_PROMPT,
        )
        assert custom_system in out

    def test_prompt_variation_changes_output(
        self, minimal_tokenizer, acs_income_task, acs_row
    ):
        base_config = PromptConfig.default(task=acs_income_task)
        prompt_bullet = encode_row_prompt_chat(
            acs_row,
            task=acs_income_task,
            tokenizer=minimal_tokenizer,
            prompt_config=dataclasses.replace(base_config, format=VaryFormat("bullet")),
        )
        prompt_comma = encode_row_prompt_chat(
            acs_row,
            task=acs_income_task,
            tokenizer=minimal_tokenizer,
            prompt_config=dataclasses.replace(base_config, format=VaryFormat("comma")),
        )
        assert prompt_bullet != prompt_comma


# ----------------------------------------------------------------------
# Numeric-prefill duplication regression tests
# ----------------------------------------------------------------------
# Background: when zero-shot scoring, `DirectNumericQA.get_question_prompt()`
# bakes `"Answer (between 0 and 1): 0."` into the question text — the
# zero-shot path scores the digits the model emits right after that prefill.
# The chat-template path appends `NUMERIC_CHAT_PROMPT` (byte-identical to
# that suffix) as the assistant turn; if both paths emitted the prefill, it
# would render twice and silently degrade scoring (Mistral 7B IT chat-numeric
# AUC collapsed from 0.815 to 0.578 before the fix). The fix is structural:
# `build_chat` forces `with_answer_prefill=False` on the `VarySuffix` via
# `dataclasses.replace` before calling `encode_row_prompt`, so the user message
# stops short of the prefill; the `chat_prompt` assistant turn is the only place
# it appears. These tests pin that invariant.


class TestEncodeRowPromptChatNumericPrefill:
    def _numeric_question(self) -> DirectNumericQA:
        return DirectNumericQA(
            column="PINCP",
            text="What is this person's estimated yearly income?",
        )

    def test_chat_numeric_renders_prefill_exactly_once(
        self, chat_tokenizer, fake_task, sample_row
    ):
        out = encode_row_prompt_chat(
            row=sample_row,
            task=fake_task,
            tokenizer=chat_tokenizer,
            question=self._numeric_question(),
        )
        assert out.count(NUMERIC_CHAT_PROMPT) == 1, (
            "User content must omit the answer prefill (chat path passes "
            "with_answer_prefill=False); only the assistant turn should "
            "contribute NUMERIC_CHAT_PROMPT to the rendered prompt."
        )

    def test_chat_numeric_tail_invariant_preserved(
        self, chat_tokenizer, fake_task, sample_row
    ):
        out = encode_row_prompt_chat(
            row=sample_row,
            task=fake_task,
            tokenizer=chat_tokenizer,
            question=self._numeric_question(),
        )
        assert out.endswith(NUMERIC_CHAT_PROMPT), (
            "LLMClassifier reads probabilities from the last token of the "
            "rendered prompt; the tail-equals-prefill invariant must hold "
            "(prefill comes from the assistant turn, not the user message)."
        )

    def test_chat_mc_path_unaffected(self, chat_tokenizer, fake_task, sample_row):
        out = encode_row_prompt_chat(
            row=sample_row,
            task=fake_task,
            tokenizer=chat_tokenizer,
        )
        assert out.count(ANTHROPIC_CHAT_PROMPT) == 1
        assert "Answer (between 0 and 1): 0." not in out, (
            "MC path must not introduce the numeric prefill anywhere — the "
            "assistant prefill is ANTHROPIC_CHAT_PROMPT and the user turn "
            "carries an MC-style question."
        )

    def test_zero_shot_numeric_prefill_unchanged(self, fake_task, sample_row):
        # The zero-shot path is what the paper used and what already
        # reproduces; `encode_row_prompt` defaults to `with_answer_prefill=True`,
        # so the zero-shot prompt must include the prefill exactly once at the
        # tail (last-token scoring relies on it).
        out = encode_row_prompt(
            row=sample_row,
            task=fake_task,
            question=self._numeric_question(),
        )
        prefill = "Answer (between 0 and 1): 0."
        assert out.endswith(prefill)
        assert out.count(prefill) == 1

    def test_chat_numeric_user_turn_omits_prefill(
        self, chat_tokenizer, fake_task, sample_row
    ):
        # Pin the structural property directly: the bare user content (before
        # chat-template wrapping) must not contain the numeric prefill at all.
        base_config = PromptConfig.from_dict({}, task=fake_task)
        config_no_prefill = dataclasses.replace(
            base_config,
            suffix=dataclasses.replace(base_config.suffix, with_answer_prefill=False),
        )
        user_content = encode_row_prompt(
            row=sample_row,
            task=fake_task,
            question=self._numeric_question(),
            prompt_config=config_no_prefill,
        )
        assert "Answer (between 0 and 1): 0." not in user_content
        assert user_content.rstrip().endswith(
            "What is this person's estimated yearly income?"
        )


# ----------------------------------------------------------------------
# encode_row_prompt — `with_answer_prefill` plumbing
# ----------------------------------------------------------------------
# `with_answer_prefill` lives on `VarySuffix` (part of `PromptConfig`).
# Setting it to False tells the QAInterface to omit the answer prefix from
# the question text — required by the chat path, where the prefill is the
# assistant turn. These tests pin that the flag round-trips through
# `PromptConfig` → `VarySuffix` → `question.get_question_prompt` correctly
# for both QA types.


class TestEncodeRowPromptThreadsAnswerPrefillFlag:
    def test_kwarg_forwarded_to_question_get_question_prompt(
        self, fake_task, sample_row
    ):
        """Plumbing: with_answer_prefill on VarySuffix reaches question.get_question_prompt."""
        mock_q = MagicMock()
        mock_q.get_question_prompt.return_value = "Q?"
        base_config = PromptConfig.from_dict({}, task=fake_task)
        config = dataclasses.replace(
            base_config,
            suffix=VarySuffix(question=mock_q, with_answer_prefill=False),
        )
        encode_row_prompt(row=sample_row, task=fake_task, prompt_config=config)
        mock_q.get_question_prompt.assert_called_once_with(with_answer_prefill=False)

    def test_default_includes_answer_prefill(self, fake_task, sample_row):
        """Default config (with_answer_prefill=True) bakes prefill into output."""
        q = DirectNumericQA(column="x", text="Numeric Q?")
        config = PromptConfig.from_dict({}, task=fake_task, question=q)
        out = encode_row_prompt(row=sample_row, task=fake_task, prompt_config=config)
        assert out.rstrip().endswith("Answer (between 0 and 1): 0.")

    def test_with_answer_prefill_false_omits_prefill(self, fake_task, sample_row):
        """VarySuffix(with_answer_prefill=False) omits the prefill from output."""
        q = DirectNumericQA(column="x", text="Numeric Q?")
        base_config = PromptConfig.from_dict({}, task=fake_task, question=q)
        config = dataclasses.replace(
            base_config,
            suffix=dataclasses.replace(base_config.suffix, with_answer_prefill=False),
        )
        out = encode_row_prompt(row=sample_row, task=fake_task, prompt_config=config)
        assert "Answer (between 0 and 1): 0." not in out
        assert out.rstrip().endswith("Numeric Q?")

    def test_numeric_round_trip_drops_prefill(self, fake_task, sample_row):
        q = DirectNumericQA(column="x", text="Numeric Q?")
        base_config = PromptConfig.from_dict({}, task=fake_task, question=q)
        no_prefill_config = dataclasses.replace(
            base_config,
            suffix=dataclasses.replace(base_config.suffix, with_answer_prefill=False),
        )
        with_prefill = encode_row_prompt(
            row=sample_row,
            task=fake_task,
            prompt_config=base_config,
        )
        without_prefill = encode_row_prompt(
            row=sample_row,
            task=fake_task,
            prompt_config=no_prefill_config,
        )
        assert with_prefill.endswith("Answer (between 0 and 1): 0.")
        assert "Answer (between 0 and 1)" not in without_prefill
        # Without the prefill, the rendered prompt must end at the question
        # text (which is the chat-path expectation).
        assert without_prefill.rstrip().endswith("Numeric Q?")

    def test_mc_round_trip_drops_answer_prefix(self, fake_task, sample_row):
        q = MultipleChoiceQA(
            column="x",
            text="MC Q?",
            choices=(
                Choice(text="No", data_value=0, numeric_value=0.0),
                Choice(text="Yes", data_value=1, numeric_value=1.0),
            ),
        )
        base_config = PromptConfig.from_dict({}, task=fake_task, question=q)
        no_prefill_config = dataclasses.replace(
            base_config,
            suffix=dataclasses.replace(base_config.suffix, with_answer_prefill=False),
        )
        with_prefill = encode_row_prompt(
            row=sample_row,
            task=fake_task,
            prompt_config=base_config,
        )
        without_prefill = encode_row_prompt(
            row=sample_row,
            task=fake_task,
            prompt_config=no_prefill_config,
        )
        assert with_prefill.rstrip().endswith("Answer:")
        # Choices remain in both forms — they're question content, not prefill.
        assert "A. No." in with_prefill and "A. No." in without_prefill
        assert "B. Yes." in with_prefill and "B. Yes." in without_prefill
        # The `Answer:` trailer is the only thing the False flag should drop.
        assert not without_prefill.rstrip().endswith("Answer:")
