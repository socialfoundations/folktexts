"""Module for using a language model through a web API for risk classification.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Callable

import numpy as np
import pandas as pd

from folktexts.qa_interface import DirectNumericQA, MultipleChoiceQA
from folktexts.task import TaskMetadata

from .base import LLMClassifier


class WebAPILLMClassifier(LLMClassifier):
    """Use an LLM through a web API to produce risk scores."""

    def __init__(
        self,
        model_name: str,
        task: TaskMetadata | str,
        custom_prompt_prefix: str = None,
        encode_row: Callable[[pd.Series], str] = None,
        threshold: float = 0.5,
        correct_order_bias: bool = True,
        max_api_rpm: int = 5000,    # NOTE: OpenAI Tier 1 limit is only 500 RPM !
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
        max_api_rpm : int, optional
            The maximum number of requests per minute allowed for the API.
        seed : int, optional
            The random seed - used for reproducibility.
        **inference_kwargs
            Additional keyword arguments to be used at inference time. Options
            include `context_size` and `batch_size`.
        """
        super().__init__(
            model_name=model_name,
            task=task,
            custom_prompt_prefix=custom_prompt_prefix,
            encode_row=encode_row,
            threshold=threshold,
            correct_order_bias=correct_order_bias,
            seed=seed,
            **inference_kwargs,
        )

        # Initialize total cost of API calls
        self._total_cost = 0

        # Set maximum requests per minute
        self.max_api_rpm = max_api_rpm
        if "MAX_API_RPM" in os.environ:
            self.max_api_rpm = int(os.getenv("MAX_API_RPM"))
            logging.warning(
                f"MAX_API_RPM environment variable is set. "
                f"Overriding previous value of {max_api_rpm} with {self.max_api_rpm}."
            )

        # Check extra dependencies
        assert self.check_webAPI_deps(), "Web API dependencies are not installed."

        # Check OpenAI API key was passed
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OpenAI API key not found in environment variables")

        # Set-up litellm API client
        import litellm
        litellm.success_callback = [self.track_cost_callback]

        from litellm import completion
        self.text_completion_api = completion

        # Get supported parameters
        from litellm import get_supported_openai_params
        supported_params = get_supported_openai_params(model=self.model_name)
        if supported_params is None:
            raise RuntimeError(f"Failed to get supported parameters for model '{self.model_name}'.")
        self.supported_params = set(supported_params)

        # Set litellm logger level to WARNING
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    @staticmethod
    def check_webAPI_deps() -> bool:
        """Check if litellm dependencies are available."""
        try:
            import litellm  # noqa: F401
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

        TODO! Retry on non-successful API calls (e.g., RPM exceeded).

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
            # Add extra tokens for textual prefix, e.g., "The probability is: ..."
            num_forward_passes = question.num_forward_passes + 2

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
            system_prompt = "Your response must start with a number representing the estimated probability."
            # system_prompt = (
            #     "You are a highly specialized assistant that always responds with a single number. "
            #     "For every input, you must analyze the request and respond with only the relevant single number, "
            #     "without any additional text, explanation, or symbols."
            # )
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
            # TODO: Retry on non-successful API calls (e.g., RPM exceeded).
            response = self.text_completion_api(
                model=self.model_name,
                messages=messages,
                **api_call_params,
            )
            responses_batch.append(response)

            # Sleep for short period to avoid rate-limiting (max 5K RPM for OpenAI API)
            time.sleep(60 / self.max_api_rpm)

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

                if not np.isclose(risk_estimate, risk_estimate_full_text, atol=1e-2):
                    logging.info(
                        f"Numeric answer mismatch: {risk_estimate} != {risk_estimate_full_text} "
                        f"from response '{response_message}'."
                    )

                    # Using full text answer as it more tightly relates to the ChatGPT web answer
                    risk_estimate = risk_estimate_full_text

                    if risk_estimate > 1:
                        logging.info(
                            f"Got risk estimate > 1: {risk_estimate}. Using "
                            f"output as a percentage: {risk_estimate / 100.0} instead."
                        )
                        risk_estimate = risk_estimate / 100.0

            except Exception:
                logging.error(
                    f"Failed to extract numeric response from message='{response_message}';\n"
                    f"Falling back on standard risk estimate of {risk_estimate}.")

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

    def __del__(self):
        """Destructor to report total cost of API calls."""
        msg = f"Total cost of API calls: ${self._total_cost:.2f}"
        print(msg)
        logging.info(msg)
