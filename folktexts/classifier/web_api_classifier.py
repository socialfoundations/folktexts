"""Module for using a language model through a web API for risk classification."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Callable

import numpy as np
import pandas as pd

from folktexts.llm_utils import decode_topk_logprobs_to_risk_estimate
from folktexts.qa_interface import ChainOfThoughtQA, DirectNumericQA, MultipleChoiceQA
from folktexts.task import TaskMetadata

from .base import LLMClassifier


class WebAPILLMClassifier(LLMClassifier):
    """Use an LLM through a web API to produce risk scores."""

    def __init__(
        self,
        model_name: str,
        task: TaskMetadata | str,
        encode_row: Callable[[pd.Series], str] = None,
        threshold: float = 0.5,
        correct_order_bias: bool = True,
        max_api_rpm: int = 5000,  # NOTE: OpenAI Tier 1 limit is only 500 RPM !
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
            logging.info(
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
            raise RuntimeError(
                f"Failed to get supported parameters for model '{self.model_name}'."
            )
        self.supported_params = set(supported_params)
        self._warned_unsupported_params: set[str] = set()

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

    def _filter_supported_params(self, params: dict) -> dict:
        """Drop params the model's API doesn't support, warning about each drop.

        Web APIs (notably OpenAI reasoning models such as o1/o3) reject
        parameters like `temperature`; filtering keeps the request valid
        instead of raising, while the warning keeps the drop visible.
        """
        unsupported = [k for k in params if k not in self.supported_params]
        if set(unsupported) - self._warned_unsupported_params:  # warn once per param
            self._warned_unsupported_params.update(unsupported)
            logging.warning(
                f"Model '{self.model_name}' does not support API "
                f"parameter(s) {sorted(unsupported)}; dropping them from the "
                f"request (this may reduce determinism/reproducibility). "
                f"Supported params: {sorted(self.supported_params)}."
            )
        return {k: v for k, v in params.items() if k in self.supported_params}

    def _query_webapi_batch(
        self,
        prompts_batch: list[str],
        *,
        question: MultipleChoiceQA | DirectNumericQA | ChainOfThoughtQA,
        context_size: int = None,
    ) -> list[dict]:
        """Query the web API with a batch of prompts and returns the json response.

        TODO! Retry on non-successful API calls (e.g., RPM exceeded).

        Parameters
        ----------
        prompts_batch : list[str]
            A batch of string prompts to query the model with.
        question : MultipleChoiceQA | DirectNumericQA | ChainOfThoughtQA
            The question (`QAInterface`) object to use for querying the model.
        context_size : int, optional
            The maximum context size to consider for each input (in tokens).

        Returns
        -------
        responses_batch : list[dict]
            The returned JSON responses for each prompt in the batch.
        """
        # Handle ChainOfThoughtQA with longer text generation
        if isinstance(question, ChainOfThoughtQA):
            api_call_params = dict(
                temperature=self._resolve_temperature(question),
                max_tokens=question.max_new_tokens,
                stream=False,
                seed=self.seed,
            )
            # Use the user-supplied system prompt (via PromptConfig / --system-prompt)
            # when set; otherwise fall back to the default CoT instruction.
            if self.prompt_config.system_prompt is not None:
                system_prompt = self.prompt_config.system_prompt()
            else:
                system_prompt = (
                    "You are a helpful assistant. Reason step-by-step about the question "
                    "and provide your final probability estimate. Your response MUST end "
                    "with 'Probability: X%' where X is a number between 0 and 100."
                )
        # Adapt number of forward passes for token-probability based methods
        elif question.num_forward_passes == 1:
            # Single token answers should require only one forward pass
            num_forward_passes = 1
            api_call_params = dict(
                temperature=self._resolve_temperature(question),
                max_tokens=num_forward_passes,
                stream=False,
                seed=self.seed,
                logprobs=True,
                top_logprobs=20,
            )
        else:
            # NOTE: Models often generate "0." instead of directly outputting the fractional part
            # > Therefore: for multi-token answers, extra forward passes may be required
            # Add extra tokens for textual prefix, e.g., "The probability is: ..."
            num_forward_passes = question.num_forward_passes + 2
            api_call_params = dict(
                temperature=self._resolve_temperature(question),
                max_tokens=num_forward_passes,
                stream=False,
                seed=self.seed,
                logprobs=True,
                top_logprobs=20,
            )

        api_call_params = self._filter_supported_params(api_call_params)

        # `logprobs` are load-bearing for token-probability decoding: dropping
        # them would only fail later, deep inside response decoding. Fail fast
        # instead (e.g. OpenAI o1/o3 don't support logprobs).
        if not isinstance(question, ChainOfThoughtQA) and "logprobs" not in api_call_params:
            raise RuntimeError(
                f"Model '{self.model_name}' does not support `logprobs`, which "
                f"are required to decode multiple-choice/numeric risk estimates. "
                f"Use chain-of-thought prompting (--cot-prompting) instead."
            )

        # Get system prompt depending on Q&A type (if not already set for ChainOfThoughtQA)
        if not isinstance(question, ChainOfThoughtQA):
            # Use the system prompt carried by PromptConfig (always the QA subclass
            # default unless the caller explicitly cleared it). `None` disables the
            # system role entirely. Bind unconditionally so it is always defined.
            system_prompt = (
                self.prompt_config.system_prompt()
                if self.prompt_config.system_prompt is not None
                else None
            )
            logging.debug(f"System prompt: {system_prompt}")

        # Query model for each prompt in the batch
        responses_batch = []
        for prompt in prompts_batch:
            # Construct prompt messages object (omit the system role when disabled)
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

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
        question: MultipleChoiceQA | DirectNumericQA | ChainOfThoughtQA,
    ) -> float:
        """Decode model output from API response to get risk estimate.

        Parameters
        ----------
        response : dict
            The response from the API call.
        question : MultipleChoiceQA | DirectNumericQA | ChainOfThoughtQA
            The question (`QAInterface`) object to use for querying the model.

        Returns
        -------
        risk_estimate : float
            The risk estimate for the API query.
        """
        # Get response message
        response_message: str = response.choices[0].message.content

        # Handle ChainOfThoughtQA by extracting probability from generated text
        if isinstance(question, ChainOfThoughtQA):
            risk_estimate = question.get_answer_from_model_output(response_message)
            logging.debug(
                f"ChainOfThoughtQA extracted probability: {risk_estimate:.2%}"
            )
            return risk_estimate

        # Get top-K logprobs per forward pass (keyed by decoded token string).
        # OpenAI-style API returns string keys; we synthesise an integer ID per
        # unique string so we can share the same scatter/decode helper as the
        # vLLM backend (which provides real token IDs directly).
        token_choices_all_passes = response.choices[0].logprobs.content

        token_logprobs_per_pass = [
            {
                token_metadata.token: token_metadata.logprob
                for token_metadata in top_token_logprobs.top_logprobs
            }
            for top_token_logprobs in token_choices_all_passes
        ]

        all_tokens = sorted({tok for d in token_logprobs_per_pass for tok in d})
        synthetic_vocab = {tok: idx for idx, tok in enumerate(all_tokens)}

        per_pass_topk = [
            {synthetic_vocab[tok]: lp for tok, lp in pass_logprobs.items()}
            for pass_logprobs in token_logprobs_per_pass
        ]

        risk_estimate = decode_topk_logprobs_to_risk_estimate(
            per_pass_topk,
            tokenizer_vocab=synthetic_vocab,
            vocab_dim=len(synthetic_vocab),
            question=question,
        )

        # Sanity check numeric answers based on global model response:
        if isinstance(question, DirectNumericQA):
            try:
                numeric_response = re.match(
                    r"[-+]?\d*\.\d+|\d+", response_message
                ).group()
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
                logging.info(
                    f"Failed to extract numeric response from message='{response_message}';\n"
                    f"Falling back on standard risk estimate of {risk_estimate}."
                )

        return risk_estimate

    def _query_prompt_risk_estimates_batch(
        self,
        prompts_batch: list[str],
        *,
        question: MultipleChoiceQA | DirectNumericQA | ChainOfThoughtQA,
        context_size: int = None,
    ) -> np.ndarray:
        """Query model with a batch of prompts and return risk estimates.

        Parameters
        ----------
        prompts_batch : list[str]
            A batch of string prompts to query the model with.
        question : MultipleChoiceQA | DirectNumericQA | ChainOfThoughtQA
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
