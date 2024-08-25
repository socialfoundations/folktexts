"""Module for using huggingface transformers models as classifiers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from folktexts.evaluation import compute_best_threshold
from folktexts.llm_utils import query_model_batch_multiple_passes
from folktexts.qa_interface import DirectNumericQA, MultipleChoiceQA
from folktexts.task import TaskMetadata

from .base import LLMClassifier
from .._utils import hash_dict


class TransformersLLMClassifier(LLMClassifier):
    """Use a huggingface transformers model to produce risk scores."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        task: TaskMetadata | str,
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
            encode_row=encode_row,
            correct_order_bias=correct_order_bias,
            threshold=threshold,
            seed=seed,
            **inference_kwargs,
        )

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

    def fit(self, X, y, *, false_pos_cost=1.0, false_neg_cost=1.0, **kwargs):
        """Uses the provided data sample to fit the prediction threshold."""

        # Compute risk estimates for the data
        y_pred_scores = self._get_positive_class_scores(
            self.predict_proba(X, **kwargs)
        )

        # Compute the best threshold for the given data
        self.threshold = compute_best_threshold(
            y, y_pred_scores,
            false_pos_cost=false_pos_cost,
            false_neg_cost=false_neg_cost,
        )

        # Update sklearn is_fitted status
        self._is_fitted = True
        return self

    def _query_prompt_risk_estimates_batch(
        self,
        prompts_batch: list[str],
        *,
        question: MultipleChoiceQA | DirectNumericQA,
        context_size: int = None,
    ) -> np.ndarray:
        """Query model with a batch of prompts and return risk estimates.

        Parameters
        ----------
        prompts_batch : list[str]
            A batch of string prompts to query the model with.
        question : MultipleChoiceQA | DirectNumericQA
            The question (`QAInterface`) object to use for querying the model.
        context_size : int, optional
            The maximum context size to consider for each input (in tokens).

        Returns
        -------
        risk_estimates : np.ndarray
            The risk estimates for each prompt in the batch.
        """

        # Query model
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
