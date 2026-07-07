"""Regression test for the vocab-mismatch bug in llm_utils.

`tokenizer.vocab` (the dict), `tokenizer.vocab_size` (the base property), and
`model.config.vocab_size` (the logits dim) are not interchangeable across
families:

  - Gemma-3-1b-it: `len(vocab) == vocab_size + 1`, logits dim == `vocab_size`.
  - Llama-3.2:    `len(vocab) == vocab_size + 256`, logits dim == `len(vocab)`.

The allowed-tokens mask must be sized against `model.config.vocab_size` (the
actual logits axis); anything else crashes on one family or the other. This
test pins both directions of the mismatch without needing real checkpoints.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from folktexts.llm_utils import query_model_batch_multiple_passes


class _Tokenizer:
    """Configurable tokenizer mock — `len(vocab)` and `vocab_size` may differ."""

    def __init__(self, vocab_size: int, n_extra: int):
        self.vocab_size = vocab_size
        # Base IDs 0..vocab_size-1 are decimal-named so digits_only catches them.
        self.vocab = {str(i): i for i in range(vocab_size)}
        # Extra added/special tokens beyond vocab_size (Gemma-style when
        # logits_dim == vocab_size; or in-range when logits_dim == len(vocab)).
        for k in range(n_extra):
            self.vocab[f"<extra{k}>"] = vocab_size + k
        self.pad_token_id = 0
        self.padding_side = "right"
        self.truncation_side = "right"

    def encode(self, text, return_tensors=None):
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        return ids if return_tensors == "pt" else ids.tolist()

    def __call__(self, texts, return_tensors=None, padding=None,
                 truncation=None, max_length=None, add_special_tokens=None):
        n = len(texts)
        input_ids = torch.tensor([[1, 2, 3]] * n, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return type("Enc", (), {"input_ids": input_ids, "attention_mask": attention_mask})()

    def decode(self, ids):
        return str(int(ids[0]))


class _Model(nn.Module):
    """Returns logits of shape (batch, seq, logits_dim) — the only thing we mask."""

    def __init__(self, logits_dim: int):
        super().__init__()
        self._param = nn.Parameter(torch.zeros(1))
        self.config = type("Cfg", (), {"vocab_size": logits_dim})()
        self._logits_dim = logits_dim

    def forward(self, input_ids, attention_mask):
        batch, seq = input_ids.shape
        return type("Out", (), {"logits": torch.zeros(batch, seq, self._logits_dim)})()


# (label, tokenizer.vocab_size, n_extra_tokens, model.config.vocab_size)
MISMATCH_CASES = [
    ("gemma_style", 10, 1, 10),    # logits_dim == vocab_size,    len(vocab) > logits_dim
    ("llama_style", 10, 5, 15),    # logits_dim == len(vocab),    vocab_size  < logits_dim
    ("matched",     10, 0, 10),    # all three equal — sanity check
]


@pytest.mark.parametrize("label,vocab_size,n_extra,logits_dim", MISMATCH_CASES)
@pytest.mark.parametrize("digits_only", [False, True])
def test_vocab_mismatch_does_not_crash(label, vocab_size, n_extra, logits_dim, digits_only):
    tokenizer = _Tokenizer(vocab_size=vocab_size, n_extra=n_extra)
    model = _Model(logits_dim=logits_dim)

    out = query_model_batch_multiple_passes(
        text_inputs=["hello", "world"],
        model=model,
        tokenizer=tokenizer,
        context_size=128,
        n_passes=2,
        digits_only=digits_only,
    )
    assert out.shape == (2, 2, logits_dim)
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# generate_text_batch temperature / sampling contract
# ---------------------------------------------------------------------------


class _RecordingModel(nn.Module):
    """Captures the kwargs passed to `.generate` and echoes back canned tokens."""

    def __init__(self):
        super().__init__()
        self._param = nn.Parameter(torch.zeros(1))
        self.generate_kwargs: dict | None = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        batch, _ = input_ids.shape
        generated = torch.full((batch, 2), 7, dtype=torch.long)
        return torch.cat([input_ids, generated], dim=1)


class _GenTokenizer:
    """Minimal tokenizer for `generate_text_batch` (no chat template → raw prompts)."""

    def __init__(self):
        self.padding_side = "right"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.chat_template = None

    def __call__(self, texts, return_tensors=None, padding=None,
                 truncation=None, max_length=None):
        n = len(texts)
        input_ids = torch.ones((n, 3), dtype=torch.long)
        attention_mask = torch.ones((n, 3), dtype=torch.long)
        return type("Enc", (), {"input_ids": input_ids, "attention_mask": attention_mask})()

    def decode(self, tokens, skip_special_tokens=False):
        return "Probability: 50%"


def test_generate_text_batch_is_greedy_by_default():
    from folktexts.llm_utils import generate_text_batch

    model = _RecordingModel()
    generate_text_batch(["hi"], model=model, tokenizer=_GenTokenizer(), max_new_tokens=2)

    assert model.generate_kwargs["do_sample"] is False
    assert "temperature" not in model.generate_kwargs


def test_generate_text_batch_samples_when_temperature_positive():
    from folktexts.llm_utils import generate_text_batch

    model = _RecordingModel()
    generate_text_batch(
        ["hi"], model=model, tokenizer=_GenTokenizer(),
        max_new_tokens=2, temperature=1.0, seed=42,
    )

    assert model.generate_kwargs["do_sample"] is True
    assert model.generate_kwargs["temperature"] == 1.0
