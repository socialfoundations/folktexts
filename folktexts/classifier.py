from __future__ import annotations

import logging
import math
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from folktexts.dataset import Dataset
from folktexts.evaluation import compute_best_threshold
from folktexts.llm_utils import query_model_batch
from folktexts.prompting import encode_row_prompt as default_encode_row_prompt
from folktexts.qa_interface import DirectNumericQA, MultipleChoiceQA
from folktexts.task import TaskMetadata

from ._utils import hash_dict, hash_function

DEFAULT_CONTEXT_SIZE = 500
DEFAULT_BATCH_SIZE = 16
DEFAULT_CORRECT_ORDER_BIAS = True

SCORE_COL_NAME = "risk_score"
LABEL_COL_NAME = "label"


class LLMClassifier(BaseEstimator, ClassifierMixin):
    """An interface to use a transformer-based LLM as a binary classifier."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        task: TaskMetadata | str,
        encode_row: Callable[[pd.Series], str] = None,
        correct_order_bias: bool = DEFAULT_CORRECT_ORDER_BIAS,
        threshold: float = 0.5,
        **inference_kwargs,
    ):
        """Creates an LLMClassifier object.

        Parameters
        ----------
        model : AutoModelForCausalLM
            The torch language model to use for inference.
        tokenizer : AutoTokenizer
            The tokenizer used to train the model.
        task : TaskMetadata | str
            The task metadata object or name of an already created task. This
            will be used to encode tabular rows into natural text prompts.
        encode_row : Callable[[pd.Series], str], optional
            The function used to encode tabular rows into natural text. If not
            provided, will use the default encoding function for the task.
        threshold : float, optional
            The classification threshold to use when outputting binary
            predictions, by default 0.5. Must be between 0 and 1. Will be
            re-calibrated if `fit` is called.
        **inference_kwargs
            Additional keyword arguments to pass to `self.predict_proba(...)`
            during inference. By default, will use `context_size=500` and
            `batch_size=16`.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._task = TaskMetadata.get_task(task) if isinstance(task, str) else task
        self._encode_row = encode_row or partial(
            default_encode_row_prompt,
            task=self.task,
        )
        self._threshold = threshold

        self.correct_order_bias = correct_order_bias

        # Default inference kwargs
        self._inference_kwargs = inference_kwargs
        self._inference_kwargs.setdefault("context_size", DEFAULT_CONTEXT_SIZE)
        self._inference_kwargs.setdefault("batch_size", DEFAULT_BATCH_SIZE)

    @property
    def model(self) -> AutoModelForCausalLM:
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def task(self) -> TaskMetadata:
        return self._task

    @property
    def encode_row(self) -> Callable[[pd.Series], str]:
        return self._encode_row

    @property
    def model_name(self) -> str:
        return Path(self.model.name_or_path).name

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        if not 0 <= value <= 1:
            logging.error(f"Threshold must be between 0 and 1; got {value}.")

        # Clip threshold to valid range
        self._threshold = np.clip(value, 0, 1)
        logging.info(f"Set threshold to {self._threshold}.")

    def __hash__(self) -> int:
        """Generate a unique hash for the LLMClassifier object."""

        # All parameters that affect the model's behavior
        hash_params = dict(
            model_name=self.model_name,
            model_size=self.model.num_parameters(),
            tokenizer_vocab_size=self.tokenizer.vocab_size,
            task_hash=hash(self.task),
            correct_order_bias=self.correct_order_bias,
            threshold=self.threshold,
            encode_row_hash=hash_function(self.encode_row),
        )

        return int(hash_dict(hash_params), 16)

    def fit(self, X, y, *, false_pos_cost=1.0, false_neg_cost=1.0, **kwargs):
        """Uses the provided data sample to fit the prediction threshold."""

        # Compute risk estimates for the data
        y_pred_scores = self.predict_proba(X, **kwargs)
        if len(y_pred_scores.shape) > 1:
            y_pred_scores = y_pred_scores[:, -1]

        # Compute the best threshold for the given data
        self.threshold = compute_best_threshold(
            y, y_pred_scores,
            false_pos_cost=false_pos_cost,
            false_neg_cost=false_neg_cost,
        )

        # Update sklearn is_fitted status
        self._is_fitted = True
        return self

    def __sklearn_is_fitted__(self):
        """Check fitted status and return a Boolean value."""
        return hasattr(self, "_is_fitted") and self._is_fitted

    @staticmethod
    def _get_positive_class_scores(risk_scores: np.ndarray) -> np.ndarray:
        """Helper function to get positive class scores from risk scores."""
        if len(risk_scores.shape) > 1:
            return risk_scores[:, -1]
        else:
            return risk_scores

    def predict(
        self,
        data: pd.DataFrame | Dataset,
        batch_size: int = None,
        context_size: int = None,
        predictions_save_path: str | Path = None,
        labels: pd.Series | np.ndarray = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Returns binary predictions for the given data."""
        risk_scores = self.predict_proba(
            data,
            batch_size=batch_size,
            context_size=context_size,
            predictions_save_path=predictions_save_path,
            labels=labels,
        )
        if isinstance(risk_scores, dict):
            return {
                data_type: (self._get_positive_class_scores(data_scores) >= self.threshold).astype(int)
                for data_type, data_scores in risk_scores.items()
            }
        else:
            return (self._get_positive_class_scores(risk_scores) >= self.threshold).astype(int)

    def _load_predictions_from_disk(
        self,
        predictions_save_path: str | Path,
        data: pd.DataFrame | Dataset,
    ) -> np.ndarray | dict[str, np.ndarray] | None:
        """Attempts to load pre-computed predictions from disk."""
        predictions_save_path = Path(predictions_save_path)

        # If DF, try to load predictions as a CSV file
        if isinstance(data, pd.DataFrame):
            predictions_save_path = predictions_save_path.with_suffix(".csv")
            predictions_df = pd.read_csv(predictions_save_path, index_col=0)

            # Check if index matches our current dataframe
            if predictions_df.index.equals(data.index):
                return predictions_df[SCORE_COL_NAME].values
            else:
                logging.error("Saved predictions do not match the current dataframe.")

        # If Dataset, try to load predictions as a pickled dict
        elif isinstance(data, Dataset):
            from ._io import load_pickle
            predictions_save_path = predictions_save_path.with_suffix(".pkl")
            predictions_dict = load_pickle(predictions_save_path)
            if not isinstance(predictions_dict, dict):
                logging.error("Loaded predictions are not in the expected dictionary format.")
                return None

            # Check if the predictions' indices match the current dataset
            if all(
                preds_array.index.equals(data.get_data_split(data_type)[0].index)
                for data_type, preds_array in predictions_dict.items()
            ):
                return {
                    data_type: predictions_dict[data_type][SCORE_COL_NAME].values
                    for data_type in predictions_dict.keys()
                }
            else:
                logging.error("Saved predictions do not match the current dataset splits.")

        else:
            logging.error(f"Cannot load predictions from disk for data type {type(data)}.")
            return None

    def predict_proba(
        self,
        data: pd.DataFrame | Dataset,
        batch_size: int = None,
        context_size: int = None,
        predictions_save_path: str | Path = None,
        labels: pd.Series | np.ndarray = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Returns probability estimates for the given data.

        Parameters
        ----------
        data : pd.DataFrame | Dataset
            The data to compute risk estimates for. Can be a pandas DataFrame or
            a Dataset object. If a Dataset object is provided, will compute risk
            scores for all available data splits.
        batch_size : int, optional
            The batch size to use when running inference.
        context_size : int, optional
            The context size to use when running inference (in number of tokens).
        predictions_save_path : str | Path, optional
            If provided, will save the computed risk scores to this path in
            disk. If the path exists, will attempt to load pre-computed
            predictions from this path.
        labels : pd.Series | np.ndarray, optional
            The labels corresponding to the provided data. Not required to
            compute predictions. Will only be used to save alongside predictions
            to disk.

        Returns
        -------
        risk_scores : np.ndarray | dict[str, np.ndarray]
            The risk scores for the given data, or a dictionary of data split
            name to risk scores if a Dataset object was provided.
        """
        # Validate arguments
        if labels is not None and predictions_save_path is None:
            logging.error(
                "** Ignoring `labels` argument as `predictions_save_path` was not provided. **"
                "The `labels` argument is only used in conjunction with "
                "`predictions_save_path` to save alongside predictions to disk. ")

        # Check if `predictions_save_path` exists and load predictions if possible
        if predictions_save_path is not None and Path(predictions_save_path).exists():
            result = self._load_predictions_from_disk(predictions_save_path, data=data)
            if result is not None:
                logging.info(f"Loaded predictions from {predictions_save_path}.")
                return result
            else:
                logging.error(
                    f"Failed to load predictions from {predictions_save_path}. "
                    f"Re-computing predictions and overwriting local file..."
                )

        # Compute risk estimates
        # (if local save path was not provided or does not match current data)
        if isinstance(data, pd.DataFrame):
            risk_scores = self._compute_risk_estimates_for_dataframe(
                df=data,
                batch_size=batch_size,
                context_size=context_size,
            )

            # Save to disk if `predictions_save_path` is provided
            if predictions_save_path is not None:
                predictions_save_path = Path(predictions_save_path).with_suffix(".csv")

                predictions_df = pd.DataFrame(risk_scores, index=data.index, columns=[SCORE_COL_NAME])
                predictions_df[LABEL_COL_NAME] = labels
                predictions_df.to_csv(predictions_save_path, index=True, mode="w")

            return risk_scores

        elif isinstance(data, Dataset):
            # TODO: save predictions in a standardized file format when given a Dataset
            # > we have the dataset name, splits, and seed, so we can safely save predictions for future use
            scores_dict = self._compute_risk_estimates_for_dataset(
                dataset=data,
                batch_size=batch_size,
                context_size=context_size,
            )

            # Save to disk if `predictions_save_path` is provided
            if predictions_save_path is not None:
                predictions_save_path = Path(predictions_save_path).with_suffix(".pkl")

                from ._io import save_pickle
                logging.warning(    # TODO
                    f"Saving dataset predictions to {predictions_save_path.as_posix()}. "
                    f"TODO: remove pickling functionality and save everything as csv files "
                    f"to re-use file-handling code from `_compute_risk_estimates_for_dataframe`."
                )
                save_pickle(scores_dict, predictions_save_path)

            return scores_dict

        else:
            raise ValueError(
                f"`data` must be a pandas DataFrame or a Dataset object; "
                f"received {type(data)} instead."
            )

    def _compute_risk_estimates_for_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: int = None,
        context_size: int = None,
    ) -> np.ndarray:
        """Compute risk estimates for a specific dataframe (internal helper function).

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to compute risk estimates for.
        batch_size : int, optional
            The batch size to use when running inference.
        context_size : int, optional
            The context size to use when running inference (in number of tokens).

        Returns
        -------
        risk_scores : np.ndarray
            The risk estimates for each row in the dataframe.
        """

        # Initialize risk scores and other constants
        fill_value = -1
        risk_scores = np.empty(len(df))
        risk_scores.fill(fill_value)    # fill with -1's

        batch_size = batch_size or self._inference_kwargs["batch_size"]
        context_size = context_size or self._inference_kwargs["context_size"]
        num_batches = math.ceil(len(df) / batch_size)

        # Get questions to ask
        q = self.task.question
        questions = [q]
        if self.correct_order_bias:
            if isinstance(q, DirectNumericQA):
                logging.info("No need to correct ordering bias for DirectNumericQA prompting.")
            elif isinstance(q, MultipleChoiceQA):
                questions = list(MultipleChoiceQA.create_answer_keys_permutations(q))
            else:
                logging.error(f"Unknown question type '{type(q)}'; cannot correct ordering bias.")

        # Compute risk estimates per batch
        for batch_idx in tqdm(range(num_batches), desc="Computing risk estimates"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_data = df.iloc[start_idx:end_idx]

            batch_risk_scores = np.empty((len(batch_data), len(questions)))
            for idx, q in enumerate(questions):

                # Encode batch data into natural text prompts
                data_texts_batch = [
                    self.encode_row(row, question=q)
                    for _, row in batch_data.iterrows()
                ]

                # Query model
                last_token_probs_batch = query_model_batch(
                    data_texts_batch,
                    self.model,
                    self.tokenizer,
                    context_size=context_size,
                )

                # Decode model output
                risk_estimates_batch = [
                    q.get_answer_from_model_output(
                        ltp,
                        tokenizer=self.tokenizer,
                    )
                    for ltp in last_token_probs_batch
                ]

                # Store risk estimates for current question
                batch_risk_scores[:, idx] = risk_estimates_batch

            risk_scores[start_idx:end_idx] = batch_risk_scores.mean(axis=1)

        # Check that all risk scores were computed
        assert not np.isclose(risk_scores, fill_value).any()
        return risk_scores

    def _compute_risk_estimates_for_dataset(
        self,
        dataset: Dataset,
        batch_size: int = None,
        context_size: int = None,
    ):
        """Computes risk estimates for each row in the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to compute risk estimates for.
        batch_size : int, optional
            The batch size to use when running inference.
        context_size : int, optional
            The context size to use when running inference (in number of tokens).

        Returns
        -------
        results : dict[str, np.ndarray]
            The risk estimates for each data type in the dataset (usually "train",
            "val", "test").
        """
        data_types = {
            "train": dataset.get_train()[0],
            "test": dataset.get_test()[0],
        }
        if dataset.get_val() is not None:
            data_types["val"] = dataset.get_val()[0]

        results = {
            data_type: self._compute_risk_estimates_for_dataframe(
                df=df,
                batch_size=batch_size,
                context_size=context_size,
            )
            for data_type, df in data_types.items()
        }

        return results
