"""Common functions to use with transformer LLMs."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from jinja2 import TemplateError
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from folktexts.qa_interface import DirectNumericQA, MultipleChoiceQA

# Will warn if the sum of digit probabilities is below this threshold
PROB_WARN_THR = 0.5


def _apply_chat_template_batch(
    inputs: list[str],
    *,
    tokenizer: AutoTokenizer,
    enable_thinking: bool | None,
    system_prompt: str | None = None,
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
    system_prompt : str | None, optional
        System prompt to prepend as a system role message. Ignored when
        ``enable_thinking`` is None (no chat template applied).

    Returns
    -------
    formatted_inputs : list[str]
        Prompts formatted using the tokenizer's chat template when requested.
    """
    if enable_thinking is None:
        return inputs

    # Base models (e.g. raw Llama-3-8B, GPT-2) have no chat template; falling
    # through to apply_chat_template would raise ValueError mid-batch. Use the
    # raw prompts instead — the CoT prompt itself is already self-contained.
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
        # Two features may be unsupported by a given tokenizer; track each
        # independently and strip on the first exception that names it.
        use_system = system_prompt is not None
        use_thinking = enable_thinking  # set to None if TypeError is raised

        while True:
            msgs = (
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
                if use_system
                else [{"role": "user", "content": text}]
            )
            kw = {} if use_thinking is None else {"enable_thinking": use_thinking}
            try:
                processed.append(
                    tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True, **kw
                    )
                )
                break
            except TypeError:
                # `enable_thinking` kwarg not accepted — strip it and retry.
                if use_thinking is not None:
                    if use_thinking:
                        logging.warning(
                            "Tokenizer does not support 'enable_thinking'; "
                            "falling back to standard chat template."
                        )
                    use_thinking = None
                else:
                    raise
            except (TemplateError, ValueError):
                # System role rejected (e.g. Gemma) — drop it and retry.
                if use_system:
                    logging.warning(
                        "Tokenizer does not support system role; dropping system "
                        "prompt for CoT generation."
                    )
                    use_system = False
                else:
                    raise

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

    # Batched tokenization. `truncation_side="left"` reproduces the previous
    # per-row `[-context_size:]` semantics (keep the tail, drop the head);
    # `padding_side="right"` matches the previous `pad_sequence` layout so the
    # last-real-token index still comes from the pre-pad length. Restore both
    # attributes even if the tokenizer call raises.
    old_pad_side = tokenizer.padding_side
    old_trunc_side = tokenizer.truncation_side
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    try:
        tokenized = tokenizer(
            text_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=context_size,
            add_special_tokens=True,
        )
    finally:
        tokenizer.padding_side = old_pad_side
        tokenizer.truncation_side = old_trunc_side

    # Compute the last-real-token index on CPU before moving tensors to the
    # model device, so the downstream `logits[torch.arange(...), idx]` gather
    # uses plain Python ints and doesn't mix CPU/CUDA index tensors.
    idx_last_token = (tokenized.attention_mask.sum(dim=1) - 1).tolist()
    tensor_inputs = tokenized.input_ids.to(model_device)
    attention_mask = tokenized.attention_mask.to(model_device)

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
    # the actual logits axis we're masking. Multimodal Gemma-3 puts vocab_size
    # under `config.text_config` instead of the top-level config.
    vocab_dim = getattr(model.config, "vocab_size", None)
    if vocab_dim is None:
        vocab_dim = getattr(getattr(model.config, "text_config", None), "vocab_size", None)
    if vocab_dim is None:
        raise AttributeError(
            f"Could not resolve vocab_size from {type(model.config).__name__} "
            "(checked top-level and text_config)."
        )
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


def decode_topk_logprobs_to_risk_estimate(
    per_pass_topk: list[dict[int, float]],
    *,
    tokenizer_vocab: dict[str, int],
    vocab_dim: int,
    question: "MultipleChoiceQA | DirectNumericQA",
) -> float:
    """Convert top-K log-probabilities into a single risk-estimate float.

    Parameters
    ----------
    per_pass_topk : list[dict[int, float]]
        One dict per generated token position, mapping token_id -> log-prob. The
        token_ids must match the values in `tokenizer_vocab`. Tokens absent from
        the top-K are assumed to have probability ~0.
    tokenizer_vocab : dict[str, int]
        Token string -> token_id map used by the QA decoder for prefix-variant
        lookup (MultipleChoiceQA) or digit/decimal lookup (DirectNumericQA).
    vocab_dim : int
        Size of the linear-probability array's vocab axis. For local backends
        this is `model.config.vocab_size` (the logits axis); for the synthetic
        WebAPI path it is the size of the synthesised vocab.
    question : MultipleChoiceQA | DirectNumericQA
        The QA interface used to interpret the probabilities.

    Returns
    -------
    risk_estimate : float
        Risk score in [0, 1] from `question.get_answer_from_model_output`.

    Notes
    -----
    Both the WebAPI backend (top_logprobs=20 from OpenAI-style responses) and
    the vLLM backend (top-K logprobs from `SamplingParams(logprobs=K)`) call
    this helper. The transformers backend reads the full softmax directly and
    bypasses this path; see `query_model_batch_multiple_passes`.
    """
    n_passes = len(per_pass_topk)
    probs = np.zeros((n_passes, vocab_dim), dtype=np.float64)
    for i, pass_dict in enumerate(per_pass_topk):
        for tok_id, logprob in pass_dict.items():
            if 0 <= tok_id < vocab_dim:
                probs[i, tok_id] = float(np.exp(logprob))

    # Drop tokenizer-vocab entries that point past the array. MultipleChoiceQA's
    # decoder does an unchecked `last_token_probs[choice_token_id]` lookup; on
    # tokenizers where added tokens sit beyond `model.config.vocab_size`
    # (Llama-3.2, Gemma-3) those ids would IndexError. DirectNumericQA already
    # filters the same way internally — this keeps both modes consistent.
    in_range_vocab = {
        tok: tok_id
        for tok, tok_id in tokenizer_vocab.items()
        if 0 <= tok_id < vocab_dim
    }

    return question.get_answer_from_model_output(
        probs,
        tokenizer_vocab=in_range_vocab,
    )


def generate_text_batch(
    text_inputs: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 1024,
    context_size: int = None,
    enable_thinking: bool = None,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    seed: int | None = None,
) -> list[str]:
    """Generate text completions for a batch of prompts.

    Uses the model's generate() method for autoregressive text generation,
    suitable for chain-of-thought Q&A where the model needs to produce
    free-form text before outputting a probability estimate. Generation is
    greedy when temperature <= 0 (the default); otherwise it samples at the
    given temperature and can be seeded via ``seed`` for reproducibility.

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
    system_prompt : str | None, optional
        System prompt to inject as a system role message when applying the
        chat template. Ignored when ``enable_thinking`` is None.
    temperature : float, optional
        Sampling temperature. Values <= 0 use greedy decoding; values > 0
        enable sampling at the given temperature. Defaults to 0.0.
    seed : int | None, optional
        Random seed to set immediately before generation when sampling is
        enabled. Ignored for greedy generation.

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
            system_prompt=system_prompt,
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

        do_sample = temperature is not None and temperature > 0.0
        generate_kwargs = dict(
            input_ids=tensor_inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
        )
        if do_sample:
            generate_kwargs["temperature"] = temperature

        with torch.no_grad():
            if do_sample and seed is not None:
                torch.manual_seed(seed)
            outputs = model.generate(**generate_kwargs)

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


def load_vllm_model(
    model_name_or_path: str | Path,
    *,
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.85,
    max_model_len: int | None = None,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = True,
    seed: int = 42,
    max_logprobs: int = 50,
    **kwargs,
):
    """Load a vLLM `LLM` engine and its tokenizer.

    Mirrors `load_model_tokenizer` for the vLLM backend. vLLM allocates the KV
    cache statically at startup based on `gpu_memory_utilization` and
    `max_model_len`; tune these per-GPU. `vllm` is an optional install — if it
    is not importable, this function raises a pointed error.

    Parameters
    ----------
    model_name_or_path : str | Path
        Model name or local path to the model folder. Pre-cached snapshots
        under `/fast/groups/sf/huggingface-models/` work without download.
    dtype : str, optional
        Compute dtype: ``"auto"`` (default; vLLM picks bf16/fp16 from the
        config), ``"bfloat16"``, ``"float16"``, or ``"float32"``.
    gpu_memory_utilization : float, optional
        Fraction of GPU VRAM vLLM may use for weights + KV cache. Default 0.85
        (vLLM's own default is 0.9, which is aggressive on shared cluster
        nodes). vLLM fails fast at startup if this isn't enough — bump down
        if you hit OOM at LLM().
    max_model_len : int, optional
        Maximum number of tokens (input + output) per request. If ``None``,
        vLLM reads it from the model config — which on some Llama checkpoints
        is 131072 and will allocate enormous KV cache. Pass an explicit value
        sized as ``context_size + max_new_tokens + buffer`` for the workload.
    tensor_parallel_size : int, optional
        Number of GPUs to shard the model across; default 1. Set higher when
        the cluster job grants multiple GPUs and the model fits with
        tensor-parallel sharding.
    trust_remote_code : bool, optional
        Forwarded to vLLM (mirrors `load_model_tokenizer`).
    seed : int, optional
        Random seed for vLLM. Doesn't affect greedy (`temperature=0`) decoding
        — used by multiple-choice / numeric QA — but governs reproducibility of
        sampled paths such as chain-of-thought (`temperature=1` by default).
    max_logprobs : int, optional
        Engine-level cap on top-K logprobs SamplingParams may request.
        Default 50 — must be ≥ ``VLLMClassifier._TOPK_LOGPROBS`` or the engine
        rejects the request at predict time (`VLLMValidationError: Requested
        sample logprobs of K, which is greater than max allowed`).
    **kwargs
        Additional keyword arguments forwarded verbatim to ``vllm.LLM(...)``.

    Returns
    -------
    tuple[vllm.LLM, AutoTokenizer]
        Loaded engine and its tokenizer. The tokenizer has had `add_pad_token`
        applied so it matches the transformers path's tokenizer state.
    """
    try:
        from vllm import LLM
    except ImportError as exc:  # pragma: no cover - exercised in user-facing CLI
        raise ImportError(
            "vLLM is not installed. Install the optional extra with "
            "`pip install 'folktexts[vllm]'`, or run with "
            "`--inference-backend transformers` to use the HuggingFace path."
        ) from exc

    # vLLM is extremely chatty during model loading; quieten it unless the
    # caller has explicitly opted into verbose logs.
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    # `processed_logprobs` returns top-K logprobs computed AFTER `allowed_token_ids`
    # masking. The default `raw_logprobs` would return top-K from the unmasked
    # distribution — which on `DirectNumericQA` causes the decoder to see non-digit
    # tokens (e.g., '.', '\n') as high-probability "numeric tokens" and pick them
    # over the only-allowed digit, collapsing Llama-3 base numeric output to 0.5
    # (answer text "5." → regex "5" → 0.5). MC has no `allowed_token_ids` so this
    # defaults to the raw distribution either way; numeric is the only mode this
    # affects.
    kwargs.setdefault("logprobs_mode", "processed_logprobs")
    logging.info(f"Loading vLLM model '{model_name_or_path}'")
    llm = LLM(
        model=str(model_name_or_path),
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
        seed=seed,
        max_logprobs=max_logprobs,
        **kwargs,
    )
    tokenizer = llm.get_tokenizer()
    add_pad_token(tokenizer)
    return llm, tokenizer


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
