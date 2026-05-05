"""Module for using huggingface transformers models as classifiers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from folktexts.llm_utils import generate_text_batch, query_model_batch_multiple_passes
from folktexts.qa_interface import DirectNumericQA, MultipleChoiceQA, ReasoningQA
from folktexts.task import TaskMetadata

from .._utils import hash_dict
from .base import LLMClassifier


class TransformersLLMClassifier(LLMClassifier):
    """Use a huggingface transformers model to produce risk scores."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        task: TaskMetadata | str,
        custom_prompt_prefix: str = None,
        encode_row: Callable[[pd.Series], str] = None,
        threshold: float = 0.5,
        correct_order_bias: bool = True,
        seed: int = 42,
        **inference_kwargs,
    ):
        """Creates an LLMClassifier based on a huggingface transformers model.

        Parameters
        ----------
        model : AutoModelForCausalLM
            The torch language model to use for inference.
        tokenizer : AutoTokenizer
            The tokenizer used to train the model.
        task : TaskMetadata | str
            The task metadata object or name of an already created task.
        custom_prompt_prefix : str, optional
            A custom prompt prefix to supply to the model before the encoded
            row data, by default None.
        encode_row : Callable[[pd.Series], str], optional
            The function used to encode tabular rows into natural text. If not
            provided, will use the default encoding function for the task.
        threshold : float, optional
            The classification threshold to use when outputting binary
            predictions, by default 0.5. Must be between 0 and 1. Will be
            re-calibrated if `fit` is called.
        correct_order_bias : bool, optional
            Whether to correct ordering bias in multiple-choice Q&A questions,
            by default True.
        seed : int, optional
            The random seed - used for reproducibility.
        **inference_kwargs
            Additional keyword arguments to be used at inference time. Options
            include `context_size` and `batch_size`.
        """
        # Transformers objects for the model and tokenizer
        self._model = model
        self._tokenizer = tokenizer

        # Fetch name for transformers model
        model_name = Path(self._model.name_or_path).name

        super().__init__(
            model_name=model_name,
            task=task,
            custom_prompt_prefix=custom_prompt_prefix,
            encode_row=encode_row,
            correct_order_bias=correct_order_bias,
            threshold=threshold,
            seed=seed,
            **inference_kwargs,
        )

        # Logging controls (used mainly for ReasoningQA generation debugging).
        # By default, log only the first N prompt/generation pairs; users can
        # enable logging all generations via env var or CLI wrapper.
        self._log_generations_all = os.getenv("FOLKTEXTS_LOG_GENERATIONS", "0").strip() in {"1", "true", "True"}
        try:
            self._log_generations_first_n = int(os.getenv("FOLKTEXTS_LOG_GENERATIONS_FIRST_N", "3"))
        except ValueError:
            self._log_generations_first_n = 3
        self._logged_generations_count = 0

    def _should_log_generation(self) -> bool:
        """Return True if we should log the next prompt/generation pair."""
        if self._log_generations_all:
            return True
        return self._logged_generations_count < max(self._log_generations_first_n, 0)

    def __hash__(self) -> int:
        """Generate a unique hash for the LLMClassifier object."""

        # All parameters that affect the model's behavior
        hash_params = dict(
            super_hash=super().__hash__(),
            model_size=self._model.num_parameters(),
            tokenizer_vocab_size=self._tokenizer.vocab_size,
        )

        return int(hash_dict(hash_params), 16)

    @property
    def model(self) -> AutoModelForCausalLM:
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    def _query_prompt_risk_estimates_batch(
        self,
        prompts_batch: list[str],
        *,
        question: MultipleChoiceQA | DirectNumericQA | ReasoningQA,
        context_size: int = None,
    ) -> np.ndarray:
        """Query model with a batch of prompts and return risk estimates.

        Parameters
        ----------
        prompts_batch : list[str]
            A batch of string prompts to query the model with.
        question : MultipleChoiceQA | DirectNumericQA | ReasoningQA
            The question (`QAInterface`) object to use for querying the model.
        context_size : int, optional
            The maximum context size to consider for each input (in tokens).

        Returns
        -------
        risk_estimates : np.ndarray
            The risk estimates for each prompt in the batch.
        """
        # Handle ReasoningQA with text generation
        if isinstance(question, ReasoningQA):
            # Pass enable_thinking to generate_text_batch:
            # - True: enable thinking mode (uses chat template with enable_thinking=True)
            # - False: explicitly disable thinking mode (uses chat template with enable_thinking=False)
            # Always apply chat template for ReasoningQA to properly format the prompt
            generated_texts = generate_text_batch(
                text_inputs=prompts_batch,
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=question.max_new_tokens,
                context_size=context_size or self.inference_kwargs["context_size"],
                enable_thinking=question.enable_thinking,
            )

            # Extract probability from generated text and log each sample
            risk_estimates_batch = []
            for idx, (prompt, generated_text) in enumerate(zip(prompts_batch, generated_texts)):
                risk_estimate = question.get_answer_from_model_output(generated_text)
                risk_estimates_batch.append(risk_estimate)

                if self._should_log_generation():
                    # Log prompt, generated answer, and extracted risk score at INFO level
                    logging.info(
                        "\n"
                        + "=" * 60
                        + "\n"
                        + f"[ReasoningQA Sample {self._logged_generations_count + 1}]"
                        + "\n"
                        + "=" * 60
                        + "\n"
                        + "PROMPT:\n"
                        + prompt
                        + "\n"
                        + "-" * 60
                        + "\n"
                        + "GENERATED ANSWER:\n"
                        + generated_text
                        + "\n"
                        + "-" * 60
                        + "\n"
                        + f"EXTRACTED RISK SCORE: {risk_estimate:.6f}\n"
                        + "=" * 60
                    )
                    self._logged_generations_count += 1

            return risk_estimates_batch

        # TODO: Add support for any unicode character used as a prefix to " A".

        # Query model using token probabilities for DirectNumericQA and MultipleChoiceQA
        last_token_probs_batch = query_model_batch_multiple_passes(
            text_inputs=prompts_batch,
            model=self.model,
            tokenizer=self.tokenizer,
            context_size=context_size or self.inference_kwargs["context_size"],
            n_passes=question.num_forward_passes,
            digits_only=True if isinstance(question, DirectNumericQA) else False,
        )

        # Decode model output
        risk_estimates_batch = [
            question.get_answer_from_model_output(
                ltp,
                tokenizer_vocab=self._tokenizer.vocab,
            )
            for ltp in last_token_probs_batch
        ]

        return risk_estimates_batch
