"""Utils for querying an LLM model.
"""
import math
import logging
from pathlib import Path
from typing import Callable
from functools import partial

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .datasets import Dataset
from .prompting import encode_row_prompt as default_encode_row_prompt
from .decoding import get_answer_to_question, get_risk_estimate_from_answers


DEFAULT_CONTEXT_SIZE = 500
DEFAULT_BATCH_SIZE = 8

SCORE_COL_NAME = "risk_score"
LABEL_COL_NAME = "label"


def compute_task_risk_estimates(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    encode_row: Callable[[pd.Series], str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    context_size: int = DEFAULT_CONTEXT_SIZE,
):
    """Computes risk estimates for each row in the dataset.

    Parameters
    ----------
    model : AutoModelForCausalLM
        The LLM to use for risk estimation.
    tokenizer : AutoTokenizer
        The LLM's tokenizer.
    dataset : Dataset
        The dataset to compute risk estimates for.
    encode_row_prompt: Callable[[pd.Series], str], optional
        Function to encode a row into a natural language prompt.

    Returns
    -------
    results : dict[str, np.ndarray]
        The risk estimates for each data type in the dataset (usually "train",
        "val", "test").
    """
    # Default encoding function
    if encode_row is None:
        encode_row = partial(default_encode_row_prompt, dataset=dataset, randomize=False)

    data_types = {
        "train": dataset.get_train()[0],
        "test": dataset.get_test()[0],
    }
    if dataset.get_val() is not None:
        data_types["val"] = dataset.get_val()[0]

    results = {
        data_type: compute_risk_estimates_for_dataframe(
            model,
            tokenizer,
            df,
            dataset=dataset,
            encode_row=encode_row,
            batch_size=batch_size,
            context_size=context_size,
        )
        for data_type, df in data_types.items()
    }

    return results


def compute_risk_estimates_for_dataframe(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
    dataset: Dataset,
    encode_row: Callable[[pd.Series], str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    predictions_save_path: str | Path = None,
) -> np.ndarray:
    """Helper to compute risk estimates for a specific dataframe."""

    # Check if predictions were already computed and saved to disk
    if predictions_save_path is not None and Path(predictions_save_path).exists():
        predictions_df = pd.read_csv(predictions_save_path, index_col=0)

        # Check if index matches our current dataframe
        if predictions_df.index.equals(df.index):
            logging.warning(f"Re-using predictions saved at {predictions_save_path.as_posix()}.")
            return predictions_df[SCORE_COL_NAME].values
        else:
            logging.error("Saved predictions do not match the current dataframe. Recomputing...")

    # Initialize risk scores and other constants
    fill_value = -1
    risk_scores = np.empty(len(df))
    risk_scores.fill(fill_value)    # fill with -1's

    num_batches = math.ceil(len(df) / batch_size)

    # Compute risk estimates per batch
    for batch_idx in tqdm(range(num_batches), desc="Computing risk estimates"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(df))
        batch_data = df.iloc[start_idx:end_idx]

        # Encode batch data into natural text prompts
        data_texts_batch = [
            encode_row(row)
            for _, row in batch_data.iterrows()
        ]

        # Query model
        last_token_probs_batch = query_model_batch(
            data_texts_batch,
            model,
            tokenizer,
            context_size=context_size,
        )

        # Decode model output -> map output to choices
        answers_batch = [
            get_answer_to_question(
                dataset.question,
                ltp,
                tokenizer)
            for ltp in last_token_probs_batch
        ]

        # Compute risk estimate
        risk_estimates_batch = [
            get_risk_estimate_from_answers(answers)
            for answers in answers_batch
        ]

        risk_scores[start_idx:end_idx] = risk_estimates_batch

    # Check that all risk scores were computed
    assert not np.isclose(risk_scores, fill_value).any()

    # Save predictions to disk (if path provided)
    if predictions_save_path is not None:
        predictions_df = pd.DataFrame(risk_scores, index=df.index, columns=[SCORE_COL_NAME])
        predictions_df[LABEL_COL_NAME] = dataset.get_target_data().loc[df.index]
        predictions_df.to_csv(predictions_save_path, index=True, mode="w")

    return risk_scores


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
    np.array
        Model's last token probabilities for each input as a np.array of shape
        (len(text_inputs), vocab_size).
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
