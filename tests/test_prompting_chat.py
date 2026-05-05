"""Tests for chat-template prompting (folktexts.prompting).

These tests exercise the layer added in PR #27: `apply_chat_template`,
`encode_row_prompt_chat`, `resolve_chat_defaults`, and the
`tokenizer_supports_system_prompt` probe.

Tokenizer-dependent tests use a local copy of
`meta-llama/Llama-3.2-3B-Instruct`. They are skipped if the snapshot is not
present locally (we never download from the Hub during tests).
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from transformers import AutoTokenizer

from folktexts.prompting import (
    ANTHROPIC_CHAT_PROMPT,
    NUMERIC_CHAT_PROMPT,
    NUMERIC_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    apply_chat_template,
    encode_row_prompt_chat,
    resolve_chat_defaults,
    tokenizer_supports_system_prompt,
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
    task = MagicMock()
    task.get_row_description.return_value = "Age: 35\nOccupation: Engineer"
    task.question.get_question_prompt.return_value = (
        "Question: What is the income bracket?\nA. <50k\nB. >=50k\nAnswer:"
    )
    return task


@pytest.fixture
def sample_row() -> pd.Series:
    return pd.Series({"AGE": 35, "OCCP": "Engineer"})


# ----------------------------------------------------------------------
# resolve_chat_defaults — pure unit tests
# ----------------------------------------------------------------------

class TestResolveChatDefaults:
    def test_mc_mode_uses_mc_defaults(self):
        sys_p, chat_p = resolve_chat_defaults(numeric=False)
        assert sys_p == SYSTEM_PROMPT
        assert chat_p == ANTHROPIC_CHAT_PROMPT

    def test_numeric_mode_uses_numeric_defaults(self):
        sys_p, chat_p = resolve_chat_defaults(numeric=True)
        assert sys_p == NUMERIC_SYSTEM_PROMPT
        assert chat_p == NUMERIC_CHAT_PROMPT

    def test_explicit_values_are_returned_unchanged(self):
        sys_p, chat_p = resolve_chat_defaults(
            numeric=True,
            system_prompt="custom system",
            chat_prompt="custom prefill",
        )
        assert sys_p == "custom system"
        assert chat_p == "custom prefill"

    def test_partial_override_only_replaces_provided_field(self):
        sys_p, chat_p = resolve_chat_defaults(
            numeric=False,
            system_prompt="custom system",
        )
        assert sys_p == "custom system"
        assert chat_p == ANTHROPIC_CHAT_PROMPT


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

    def test_raises_when_prefill_missing_from_output(
        self, prefill_dropping_tokenizer
    ):
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
        assert "<|eot_id|>" not in out.split("USER_CONTENT_MARKER", 1)[1].split(
            prefill, 1
        )[1]


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

    def test_numeric_kwarg_applies_numeric_defaults(
        self, chat_tokenizer, fake_task, sample_row
    ):
        out = encode_row_prompt_chat(
            row=sample_row,
            task=fake_task,
            tokenizer=chat_tokenizer,
            numeric=True,
        )
        assert NUMERIC_SYSTEM_PROMPT.strip() in out
        assert out.endswith(NUMERIC_CHAT_PROMPT)
        # Cross-check that the MC defaults are NOT injected when numeric=True.
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
        assert (
            tokenizer_supports_system_prompt(gemma_like_tokenizer) is False
        )
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
        assert "Age: 35" in out
        assert "Occupation: Engineer" in out
        assert "What is the income bracket?" in out
