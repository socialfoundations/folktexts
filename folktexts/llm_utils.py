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


def _apply_chat_template_batch(
    inputs: list[str],
    *,
    tokenizer: AutoTokenizer,
    enable_thinking: bool | None,
) -> list[str]:
    """Apply tokenizer chat template to a list of prompts, if requested.

    Parameters
    ----------
    inputs : list[str]
        Raw (non-chat) prompts.
    tokenizer : AutoTokenizer
        Tokenizer used to apply the chat template.
    enable_thinking : bool | None
        If None, no chat template is applied. If True/False, chat template is
        applied and (if supported) the `enable_thinking` kwarg is forwarded.

    Returns
    -------
    formatted_inputs : list[str]
        Prompts formatted using the tokenizer's chat template when requested.
    """
    if enable_thinking is None:
        return inputs

    # Base models (e.g. raw Llama-3-8B, GPT-2) have no chat template; falling
    # through to apply_chat_template would raise ValueError mid-batch. Use the
    # raw prompts instead — the reasoning prompt itself is already self-contained.
    if getattr(tokenizer, "chat_template", None) is None:
        if enable_thinking:
            logging.warning(
                "Tokenizer has no chat_template; cannot honor enable_thinking=True. "
                "Falling back to raw prompts (base model)."
            )
        else:
            logging.info("Tokenizer has no chat_template; using raw prompts (base model).")
        return inputs

    processed: list[str] = []
    for text in inputs:
        messages = [{"role": "user", "content": text}]
        try:
            processed.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            )
        except TypeError:
            if enable_thinking:
                logging.warning(
                    "Tokenizer does not support 'enable_thinking' parameter. "
                    "Falling back to standard chat template."
                )
            processed.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

    logging.debug(f"Applied chat template (enable_thinking={enable_thinking})")
    return processed


def _postprocess_generated_text(
    full_text: str,
    *,
    enable_thinking: bool | None,
    i: int,
    n: int,
) -> str:
    """Return the content to use for downstream extraction.

    In thinking mode, the model may emit a `<think> ... </think>` block. We log
    the thinking content (debug) but return only the response content after
    `</think>` for extraction.

    Parameters
    ----------
    full_text : str
        Full decoded generation (new tokens only).
    enable_thinking : bool | None
        Whether thinking mode is enabled.
    i : int
        Index of this generation in the batch (0-based).
    n : int
        Total number of generations in the batch.

    Returns
    -------
    response_text : str
        Text to be used for probability extraction.
    """
    if enable_thinking is not True:
        logging.debug(f"=== Generated output {i+1}/{n} ===")
        logging.debug(f"Content ({len(full_text)} chars):\n{full_text[:500]}...")
        return full_text.strip()

    think_end_marker = "</think>"
    if think_end_marker not in full_text:
        logging.warning(
            f"</think> marker not found in output (thinking mode was enabled). "
            f"Using full generated text ({len(full_text)} chars)."
        )
        return full_text.strip()

    parts = full_text.split(think_end_marker, 1)
    thinking_content = parts[0].strip()
    response_content = parts[1].strip() if len(parts) > 1 else ""

    logging.debug(f"=== Generated output {i+1}/{n} ===")
    logging.debug(f"Thinking content ({len(thinking_content)} chars) [IGNORED for extraction]:")
    logging.debug(f"{thinking_content[:500]}..." if len(thinking_content) > 500 else thinking_content)
    logging.debug(f"Response content ({len(response_content)} chars) [USED for extraction]:")
    logging.debug(response_content)

    if response_content:
        return response_content

    logging.warning(
        "Response content after </think> is empty. "
        "Model may not have generated a proper response. "
        "Probability extraction will likely fail."
    )
    return ""


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
    # Mask is sized to the model's logits dim, not the tokenizer's vocab dict:
    # neither `len(tokenizer.vocab)` nor `tokenizer.vocab_size` is reliable
    # (Gemma-3 has `len(vocab) == vocab_size + 1`; Llama-3.2 has
    # `len(vocab) == vocab_size + 256`). Only `model.config.vocab_size` matches
    # the actual logits axis we're masking.
    vocab_dim = model.config.vocab_size
    allowed_tokens_filter = np.ones(vocab_dim, dtype=bool)
    if digits_only:
        allowed_token_ids = np.array([
            tok_id
            for token, tok_id in tokenizer.vocab.items()
            if token.isdecimal() and tok_id < vocab_dim
        ])

        allowed_tokens_filter = np.zeros(vocab_dim, dtype=bool)
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
    assert last_token_probs_array.shape == (len(text_inputs), n_passes, vocab_dim)
    return last_token_probs_array


def generate_text_batch(
    text_inputs: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 1024,
    context_size: int = None,
    enable_thinking: bool = None,
) -> list[str]:
    """Generate text completions for a batch of prompts.

    Uses the model's generate() method for autoregressive text generation,
    suitable for reasoning-based Q&A where the model needs to produce
    free-form text before outputting a probability estimate. Generation is
    greedy (do_sample=False) so runs are reproducible — matches the web-API
    path's temperature=0 contract.

    Parameters
    ----------
    text_inputs : list[str]
        The input prompts as a list of strings.
    model : AutoModelForCausalLM
        The model to use for generation.
    tokenizer : AutoTokenizer
        The tokenizer used to encode/decode text.
    max_new_tokens : int, optional
        Maximum number of new tokens to generate, by default 1024.
    context_size : int, optional
        The maximum context size for input tokens. If None, no truncation
        is applied to inputs.
    enable_thinking : bool, optional
        Controls chat template application and thinking mode:
        - None: Do not apply chat template (use raw prompts, for base models)
        - False: Apply chat template WITHOUT thinking mode (for instruction-tuned models)
        - True: Apply chat template WITH thinking mode, and extract response
          content after </think> marker (for thinking models like Qwen3)

    Returns
    -------
    generated_texts : list[str]
        The generated text completions for each input prompt. Only the
        newly generated tokens are returned (not the input prompt).
    """
    model_device = next(model.parameters()).device

    # Save original padding side and set to left for generation
    # (decoder-only models require left-padding for correct generation)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        formatted_inputs = _apply_chat_template_batch(
            text_inputs,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
        )

        tokenized = tokenizer(
            formatted_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True if context_size else False,
            max_length=context_size,
        )

        tensor_inputs = tokenized.input_ids.to(model_device)
        attention_mask = tokenized.attention_mask.to(model_device)
        input_seq_length = tensor_inputs.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=tensor_inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        generated_texts: list[str] = []
        for i, output in enumerate(outputs):
            generated_tokens = output[input_seq_length:]
            full_generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(
                _postprocess_generated_text(
                    full_generated_text,
                    enable_thinking=enable_thinking,
                    i=i,
                    n=len(outputs),
                )
            )

        return generated_texts

    finally:
        tokenizer.padding_side = original_padding_side


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
