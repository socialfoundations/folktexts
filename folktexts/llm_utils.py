"""Common functions to use with transformer LLMs."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Will warn if the sum of digit probabilities is below this threshold
PROB_WARN_THR = 0.5


def query_model_batch(
    text_inputs: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context_size: int
) -> np.array:
    """Queries the model with a batch of text inputs.

    Parameters
    ----------
    text_inputs : list[str]
        The inputs to the model as a list of strings.
    model : AutoModelForCausalLM
        The model to query.
    tokenizer : AutoTokenizer
        The tokenizer used to encode the text inputs.
    context_size : int
        The maximum context size to consider for each input (in tokens).

    Returns
    -------
    last_token_probs : np.array
        Model's last token *linear* probabilities for each input as an
        np.array of shape (batch_size, vocab_size).
    """
    model_device = next(model.parameters()).device

    # Tokenize
    token_inputs = [tokenizer.encode(text, return_tensors="pt").flatten()[-context_size:] for text in text_inputs]
    idx_last_token = [tok_seq.shape[0] - 1 for tok_seq in token_inputs]

    # Pad
    tensor_inputs = torch.nn.utils.rnn.pad_sequence(
        token_inputs,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    ).to(model_device)

    # Mask padded context
    attention_mask = tensor_inputs.ne(tokenizer.pad_token_id)

    # Query: run one forward pass, i.e., generate the next token
    with torch.no_grad():
        logits = model(input_ids=tensor_inputs, attention_mask=attention_mask).logits

    # Probabilities corresponding to the last token after the prompt
    last_token_logits = logits[torch.arange(len(idx_last_token)), idx_last_token]
    last_token_probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
    return last_token_probs.to(dtype=torch.float16).cpu().numpy()


def query_model_batch_multiple_passes(
    text_inputs: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context_size: int,
    n_passes: int,
    digits_only: bool = False,
) -> np.array:
    """Queries an LM for multiple forward passes.

    Greedy token search over multiple forward passes: Each forward pass takes
    the highest likelihood token from the previous pass.

    NOTE: could use model.generate in the future!

    Parameters
    ----------
    text_inputs : list[str]
        The batch inputs to the model as a list of strings.
    model : AutoModelForCausalLM
        The model to query.
    tokenizer : AutoTokenizer
        The tokenizer used to encode the text inputs.
    context_size : int
        The maximum context size to consider for each input (in tokens).
    n_passes : int, optional
        The number of forward passes to run.
    digits_only : bool, optional
        Whether to only sample for digit tokens.

    Returns
    -------
    last_token_probs : np.array
        Last token *linear* probabilities for each forward pass, for each text
        in the input batch. The output has shape (batch_size, n_passes, vocab_size).
    """
    # If `digits_only`, get token IDs for digit tokens
    allowed_tokens_filter = np.ones(len(tokenizer.vocab), dtype=bool)
    if digits_only:
        allowed_token_ids = np.array([
            tok_id
            for token, tok_id in tokenizer.vocab.items() if token.isdecimal()
        ])

        allowed_tokens_filter = np.zeros(len(tokenizer.vocab), dtype=bool)
        allowed_tokens_filter[allowed_token_ids] = True

    # Current text batch
    current_batch = text_inputs

    # For each forward pass, add one token to each text in the batch
    last_token_probs = []

    for iter in range(n_passes):
        # Query the model with the current batch
        current_probs = query_model_batch(current_batch, model, tokenizer, context_size)

        # Filter out probabilities for tokens that are not allowed
        current_probs[:, ~allowed_tokens_filter] = 0

        # Sanity check digit probabilities
        if iter == 0 and digits_only:
            total_digit_probs = np.sum(current_probs, axis=-1)
            if any(probs < PROB_WARN_THR for probs in total_digit_probs):
                logging.error(f"Digit probabilities are too low: {total_digit_probs}")

        # Add the highest likelihood token to each text in the batch
        next_tokens = [tokenizer.decode([np.argmax(probs)]) for probs in current_probs]
        current_batch = [text + next_token for text, next_token in zip(current_batch, next_tokens)]

        # Store the probabilities of the last token for each text in the batch
        last_token_probs.append(current_probs)

    # Cast output to np.array with correct shape
    last_token_probs_array = np.array(last_token_probs)
    last_token_probs_array = np.moveaxis(last_token_probs_array, 0, 1)
    assert last_token_probs_array.shape == (len(text_inputs), n_passes, len(tokenizer.vocab))
    return last_token_probs_array


def add_pad_token(tokenizer):
    """Add a pad token to the model and tokenizer if it doesn't already exist.

    Here we're using the end-of-sentence token as the pad token. Both the model
    weights and tokenizer vocabulary are untouched.

    Another possible way would be to add a new token `[PAD]` to the tokenizer
    and update the tokenizer vocabulary and model weight embeddings accordingly.
    The embedding for the new pad token would be the average of all other
    embeddings.
    """
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})


def is_bf16_compatible() -> bool:
    """Checks if the current environment is bfloat16 compatible."""
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def load_model_tokenizer(model_name_or_path: str | Path, **kwargs) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model and tokenizer from the given local path (or using the model name).

    Parameters
    ----------
    model_name_or_path : str | Path
        Model name or local path to the model folder.
    kwargs : dict
        Additional keyword arguments to pass to the model `from_pretrained` call.

    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]
        The loaded model and tokenizer, respectively.
    """
    logging.info(f"Loading model '{model_name_or_path}'")

    # Load tokenizer from disk
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Set default keyword arguments for loading the pretrained model
    model_kwargs = dict(
        torch_dtype=torch.bfloat16 if is_bf16_compatible() else torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    model_kwargs.update(kwargs)

    # Load model from disk
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )

    # Add pad token to the tokenizer if it doesn't already exist
    add_pad_token(tokenizer)

    # Move model to the correct device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    logging.info(f"Moving model to device: {device}")
    if model.device.type != device:
        model.to(device)

    return model, tokenizer


def get_model_folder_path(model_name: str, root_dir="/tmp") -> str:
    """Returns the folder where the model is saved."""
    folder_name = model_name.replace("/", "--")
    return (Path(root_dir) / folder_name).resolve().as_posix()


def get_model_size_B(model_name: str, default: int = None) -> int:
    """Get the model size from the model name, in Billions of parameters.
    """
    regex = re.search(r"((?P<times>\d+)[xX])?(?P<size>\d+)[bB]", model_name)
    if regex:
        return int(regex.group("size")) * int(regex.group("times") or 1)

    if default is not None:
        return default

    logging.warning(f"Could not infer model size from name '{model_name}'.")
    return default
