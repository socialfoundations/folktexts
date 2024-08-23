from __future__ import annotations

import re
import time
import math
import logging
from functools import partial
from pathlib import Path
from typing import Callable
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from folktexts.dataset import Dataset
from folktexts.evaluation import compute_best_threshold
from folktexts.llm_utils import query_model_batch_multiple_passes
from folktexts.prompting import encode_row_prompt as default_encode_row_prompt
from folktexts.qa_interface import DirectNumericQA, MultipleChoiceQA
from folktexts.task import TaskMetadata

from ._utils import hash_dict, hash_function

DEFAULT_CONTEXT_SIZE = 500
DEFAULT_BATCH_SIZE = 16

SCORE_COL_NAME = "risk_score"
LABEL_COL_NAME = "label"


class LLMClassifier(BaseEstimator, ClassifierMixin, ABC):
    """An interface to produce risk scores and class predictions with an LLM."""

    DEFAULT_INFERENCE_KWARGS = {
        "context_size": DEFAULT_CONTEXT_SIZE,
        "batch_size": DEFAULT_BATCH_SIZE,
    }

    def __init__(
        self,
        model_name: str,
        task: TaskMetadata | str,
        encode_row: Callable[[pd.Series], str] = None,
        threshold: float = 0.5,
        correct_order_bias: bool = True,
        seed: int = 42,
        **inference_kwargs,
    ):
        """Creates an LLMClassifier object.

        Parameters
        ----------
        model_name : str
            The model name or ID.
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

        # Set classifier metadata
        self._model_name = model_name

        self._task = TaskMetadata.get_task(task) if isinstance(task, str) else task
        self._encode_row = encode_row or partial(
            default_encode_row_prompt,
            task=self.task,
        )

        self._threshold = threshold
        self._correct_order_bias = correct_order_bias
        self._seed = seed

        # Default inference kwargs
        self._inference_kwargs = self.DEFAULT_INFERENCE_KWARGS.copy()
        self._inference_kwargs.update(inference_kwargs)

        # Fixed sklearn parameters
        self.classes_ = np.array([0, 1])
        self._is_fitted = False

    def __hash__(self) -> int:
        """Generate a unique hash for this object."""

        # All parameters that affect the model's behavior
        hash_params = dict(
            model_name=self.model_name,
            task_hash=hash(self.task),
            correct_order_bias=self.correct_order_bias,
            threshold=self.threshold,
            encode_row_hash=hash_function(self.encode_row),
        )

        return int(hash_dict(hash_params), 16)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def task(self) -> TaskMetadata:
        return self._task

    @property
    def encode_row(self) -> Callable[[pd.Series], str]:
        return self._encode_row

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        if not 0 <= value <= 1:
            logging.error(f"Threshold must be between 0 and 1; got {value}.")

        # Clip threshold to valid range
        self._threshold = np.clip(value, 0, 1)
        logging.warning(f"Setting {self.model_name} threshold to {self._threshold}.")

    @property
    def correct_order_bias(self) -> bool:
        return self._correct_order_bias

    @correct_order_bias.setter
    def correct_order_bias(self, value: bool):
        self._correct_order_bias = value
        logging.warning(f"Setting {self.model_name} correct_order_bias to {value}.")

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def inference_kwargs(self) -> dict:
        return self._inference_kwargs

    def set_inference_kwargs(self, **kwargs):
        """Set inference kwargs for the model."""
        self._inference_kwargs.update(kwargs)

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

    @staticmethod
    def _make_predictions_multiclass(pos_class_scores: np.ndarray) -> np.ndarray:
        """Converts positive class scores to multiclass scores."""
        return np.column_stack([1 - pos_class_scores, pos_class_scores])

    def _load_predictions_from_disk(
        self,
        predictions_save_path: str | Path,
        data: pd.DataFrame,
    ) -> np.ndarray | None:
        """Attempts to load pre-computed predictions from disk."""

        # Load predictions from disk
        predictions_save_path = Path(predictions_save_path).with_suffix(".csv")
        predictions_df = pd.read_csv(predictions_save_path, index_col=0)

        # Check if index matches our current dataframe
        if predictions_df.index.equals(data.index):
            return predictions_df[SCORE_COL_NAME].values
        else:
            logging.error("Saved predictions do not match the current dataframe.")
            return None

    def predict(
        self,
        data: pd.DataFrame,
        predictions_save_path: str | Path = None,
        labels: pd.Series | np.ndarray = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Returns binary predictions for the given data."""
        risk_scores = self.predict_proba(
            data,
            predictions_save_path=predictions_save_path,
            labels=labels,
        )
        return (self._get_positive_class_scores(risk_scores) >= self.threshold).astype(int)

    def predict_proba(
        self,
        data: pd.DataFrame,
        predictions_save_path: str | Path = None,
        labels: pd.Series | np.ndarray = None,
    ) -> np.ndarray:
        """Returns probability estimates for the given data.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to compute risk estimates for.
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
        risk_scores : np.ndarray
            The risk scores for the given data.
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
                return self._make_predictions_multiclass(result)
            else:
                logging.error(
                    f"Failed to load predictions from {predictions_save_path}. "
                    f"Re-computing predictions and overwriting local file..."
                )

        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                f"`data` must be a pd.DataFrame, received {type(data)} instead.")

        # Compute risk estimates
        risk_scores = self.compute_risk_estimates_for_dataframe(df=data)

        # Save to disk if `predictions_save_path` is provided
        if predictions_save_path is not None:
            predictions_save_path = Path(predictions_save_path).with_suffix(".csv")

            predictions_df = pd.DataFrame(risk_scores, index=data.index, columns=[SCORE_COL_NAME])
            predictions_df[LABEL_COL_NAME] = labels
            predictions_df.to_csv(predictions_save_path, index=True, mode="w")

        return self._make_predictions_multiclass(risk_scores)

    @abstractmethod
    def _query_prompt_risk_estimates_batch(
        self,
        prompts_batch: list[str],
        *,
        question: MultipleChoiceQA | DirectNumericQA,
        context_size: int = None,
    ) -> np.ndarray:
        """Query model with a batch of prompts and return risk estimates."""
        raise NotImplementedError("Calling an abstract method :: Use one of the subclasses of LLMClassifier.")

    def compute_risk_estimates_for_dataframe(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """Compute risk estimates for a specific dataframe (internal helper function).

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to compute risk estimates for.

        Returns
        -------
        risk_scores : np.ndarray
            The risk estimates for each row in the dataframe.
        """

        # Initialize risk scores and other constants
        fill_value = -1
        risk_scores = np.empty(len(df))
        risk_scores.fill(fill_value)    # fill with -1's

        batch_size = self._inference_kwargs["batch_size"]
        context_size = self._inference_kwargs["context_size"]
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
            for q_idx, q in enumerate(questions):

                # Encode batch data into natural text prompts
                data_texts_batch = [
                    self.encode_row(row, question=q)
                    for _, row in batch_data.iterrows()
                ]

                # Query the model with the batch of data
                risk_estimates_batch = self._query_prompt_risk_estimates_batch(
                    prompts_batch=data_texts_batch,
                    question=q,
                    context_size=context_size,
                )

                # Store risk estimates for current question
                batch_risk_scores[:, q_idx] = risk_estimates_batch

            risk_scores[start_idx: end_idx] = batch_risk_scores.mean(axis=1)

        # Check that all risk scores were computed
        assert not np.isclose(risk_scores, fill_value).any()
        return risk_scores

    def compute_risk_estimates_for_dataset(
        self,
        dataset: Dataset,
    ) -> dict[str, np.ndarray]:
        """Computes risk estimates for each row in the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to compute risk estimates for.

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
            data_type: self.compute_risk_estimates_for_dataframe(
                df=df,
            )
            for data_type, df in data_types.items()
        }

        return results


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


class WebAPILLMClassifier(LLMClassifier):
    """Use an LLM through a web API to produce risk scores."""

    def __init__(
        self,
        model_name: str,
        task: TaskMetadata | str,
        encode_row: Callable[[pd.Series], str] = None,
        threshold: float = 0.5,
        correct_order_bias: bool = True,
        max_requests_per_minute: int = 1000,
        seed: int = 42,
        **inference_kwargs,
    ):
        """Creates an LLMClassifier object that uses a web API for inference.

        Parameters
        ----------
        model_name : str
            The model ID to be resolved by `litellm`.
        task : TaskMetadata | str
            The task metadata object or name of an already created task.
        correct_order_bias : bool, optional
            Whether to correct ordering bias in multiple-choice Q&A, by default
            True.
        threshold : float, optional
            The threshold used to binarize risk scores produced by the model, by
            default 0.5.
        """
        super().__init__(
            model_name=model_name,
            task=task,
            encode_row=encode_row,
            threshold=threshold,
            correct_order_bias=correct_order_bias,
            seed=seed,
            **inference_kwargs,
        )

        self.max_requests_per_minute = max_requests_per_minute
        if self.max_requests_per_minute > 5000:
            raise ValueError(f"Maximum RPM must be less than 5K, got {self.max_requests_per_minute}")

        # Check extra dependencies
        assert self.check_webAPI_deps(), "Web API dependencies are not installed."

        # Set-up litellm API client
        self._total_cost = 0

        import litellm
        litellm.success_callback = [self.track_cost_callback]

        from litellm import completion
        self.text_completion_api = completion

        # Get supported parameters
        from litellm import get_supported_openai_params
        self.supported_params = set(get_supported_openai_params(model=self.model_name))

    @staticmethod
    def check_webAPI_deps() -> bool:
        """Check if litellm dependencies are available."""
        try:
            import litellm      # noqa: F401
        except ImportError:
            logging.critical(
                "Please install extra API dependencies with "
                "`pip install 'folktexts[apis]'` "
                "to use the WebAPILLMClassifier."
            )
            return False
        return True

    def _query_webapi_batch(
        self,
        prompts_batch: list[str],
        *,
        question: MultipleChoiceQA | DirectNumericQA,
        context_size: int = None,
    ) -> list[dict]:
        """Query the web API with a batch of prompts and returns the json response.

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
        responses_batch : list[dict]
            The returned JSON responses for each prompt in the batch.
        """
        # Adapt number of forward passes
        # > Single token answers should require only one forward pass
        if question.num_forward_passes == 1:
            num_forward_passes = 1

        # NOTE: Models often generate "0." instead of directly outputting the fractional part
        # > Therefore: for multi-token answers, extra forward passes may be required
        else:
            num_forward_passes = question.num_forward_passes + 2    # +2 tokens for "0."

        api_call_params = dict(
            temperature=1,
            max_tokens=num_forward_passes,
            stream=False,
            seed=self.seed,
            logprobs=True,
            top_logprobs=20,
        )

        if set(api_call_params.keys()) - self.supported_params:
            raise RuntimeError(
                f"Unsupported API parameters for model '{self.model_name}': "
                f"{set(api_call_params.keys()) - self.supported_params}"
            )

        # Get system prompt depending on Q&A type
        if isinstance(question, DirectNumericQA):
            # system_prompt = "Please respond with number."
            system_prompt = "Please respond with number representing the estimated probability."
        elif isinstance(question, MultipleChoiceQA):
            system_prompt = "Please respond with a single letter."
        else:
            raise ValueError(f"Unknown question type '{type(question)}'.")

        # Query model for each prompt in the batch
        responses_batch = []
        for prompt in prompts_batch:

            # Construct prompt messages object
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            # Query the model API
            response = self.text_completion_api(
                model=self.model_name,
                messages=messages,
                **api_call_params,
            )
            responses_batch.append(response)

            # Sleep for short period to avoid rate-limiting (max 5K RPM for OpenAI API)
            time.sleep(60 / self.max_requests_per_minute)

        return responses_batch

    def _decode_risk_estimate_from_api_response(
        self,
        response: dict,
        question: MultipleChoiceQA | DirectNumericQA,
    ) -> float:
        """Decode model output from API response to get risk estimate.

        Parameters
        ----------
        response : dict
            The response from the API call.
        question : MultipleChoiceQA | DirectNumericQA
            The question (`QAInterface`) object to use for querying the model.

        Returns
        -------
        risk_estimate : float
            The risk estimate for the API query.
        """
        # Get response message
        response_message: str = response.choices[0].message.content

        # Get top token choices for each forward pass
        token_choices_all_passes = response.choices[0].logprobs["content"]

        # Construct dictionary of token to linear token probability for each forward pass
        token_probs_all_passes = [
            {
                token_metadata["token"]: np.exp(token_metadata["logprob"])
                for token_metadata in top_token_logprobs["top_logprobs"]
            }
            for top_token_logprobs in token_choices_all_passes
        ]

        # Decode model output into risk estimates
        # 1. Construct vocabulary dict for this response
        vocab_tokens = {tok for forward_pass in token_probs_all_passes for tok in forward_pass}

        token_to_id = {tok: i for i, tok in enumerate(vocab_tokens)}
        id_to_token = {i: tok for i, tok in enumerate(vocab_tokens)}

        # 2. Parse `token_probs_all_passes` into an array of shape (num_passes, vocab_size) 
        token_probs_array = np.array([
            [
                forward_pass.get(id_to_token[i], 0)
                for i in range(len(vocab_tokens))
            ]
            for forward_pass in token_probs_all_passes
        ])
        # NOTE: token_probs.shape = (num_passes, vocab_size)

        # Get risk estimate
        risk_estimate = question.get_answer_from_model_output(
            token_probs_array,
            tokenizer_vocab=token_to_id,
        )

        # Sanity check numeric answers based on global model response:
        if isinstance(question, DirectNumericQA):
            try:
                numeric_response = re.match(r"[-+]?\d*\.\d+|\d+", response_message).group()
                risk_estimate_full_text = float(numeric_response)
            except Exception as e:
                logging.error(
                    f"Failed to extract numeric response from '{response_message}': {e}; "
                    f"Falling back on standard risk estimate of {risk_estimate}."
                )

            if not np.isclose(risk_estimate, risk_estimate_full_text, atol=1e-2):
                logging.info(
                    f"Numeric answer mismatch: {risk_estimate} != {risk_estimate_full_text} "
                    f"from response '{response_message}'."
                )

            # Using full text answer as it more tightly relates to the ChatGPT web answer
            risk_estimate = risk_estimate_full_text

        return risk_estimate

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

        Raises
        ------
        RuntimeError
            Raised when web API call is unsuccessful.
        """

        # Query model through web API
        api_responses_batch = self._query_webapi_batch(
            prompts_batch=prompts_batch,
            question=question,
            context_size=context_size,
        )

        # Parse API responses and decode model output
        risk_estimates_batch = [
            self._decode_risk_estimate_from_api_response(response, question)
            for response in api_responses_batch
        ]

        return risk_estimates_batch

    def track_cost_callback(
        self,
        kwargs,
        completion_response,
        start_time,
        end_time,
    ):
        """Callback function to cost of API calls."""
        try:
            response_cost = kwargs.get("response_cost", 0)
            self._total_cost += response_cost

        except Exception as e:
            logging.error(f"Failed to track cost of API calls: {e}")
