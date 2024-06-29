"""Test functions in folktexts.llm_utils
"""

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from folktexts.llm_utils import load_model_tokenizer


def test_load_model_tokenizer(causal_lm_name_or_path: str):
    model, tokenizer = load_model_tokenizer(causal_lm_name_or_path)

    assert isinstance(model, PreTrainedModel), \
        f"Expected model type `PreTrainedModel`, got {type(model)}."

    assert isinstance(tokenizer, PreTrainedTokenizerBase), \
        f"Expected tokenizer type `PreTrainedTokenizer`, got {type(tokenizer)}."
