"""Module for using vLLM as the local LLM-inference backend.

Mirrors `TransformersLLMClassifier` for the score-extraction contract — the
same QA decoders are reused, only the model-call inner loop changes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoTokenizer

from folktexts.llm_utils import (
    _apply_chat_template_batch,
    _postprocess_generated_text,
    decode_topk_logprobs_to_risk_estimate,
)
from folktexts.qa_interface import ChainOfThoughtQA, DirectNumericQA, MultipleChoiceQA
from folktexts.task import TaskMetadata

from .._utils import hash_dict
from .base import LLMClassifier

if TYPE_CHECKING:
    import vllm

# Top-K logprobs to request from vLLM. WebAPI uses 20; we bump to 50 here
# because zero-shot prompts on instruction-tuned base models can push A/B
# answer letters into the long tail behind prose continuations
# ("I", "The", "Based"…). 50 covers those without materially changing
# results on models where A/B are clearly top-2.
_TOPK_LOGPROBS = 50


class VLLMClassifier(LLMClassifier):
    """Use a vLLM `LLM` engine to produce risk scores."""

    def __init__(
        self,
        llm: "vllm.LLM",
        tokenizer: AutoTokenizer,
        task: TaskMetadata | str,
        *,
        model_name_or_path: str | Path | None = None,
        encode_row: Callable[[pd.Series], str] = None,
        threshold: float = 0.5,
        correct_order_bias: bool = True,
        seed: int = 42,
        **inference_kwargs,
    ):
        """Creates an LLMClassifier backed by vLLM.

        Parameters
        ----------
        llm : vllm.LLM
            A loaded vLLM engine. See `folktexts.llm_utils.load_vllm_model`.
        tokenizer : AutoTokenizer
            The HuggingFace tokenizer for the model. Usually obtained via
            `llm.get_tokenizer()`; passed in explicitly so observability hooks
            (chat-template helpers, vocab lookups) work without reaching into
            vLLM internals.
        task : TaskMetadata | str
            The task metadata object or name of an already created task.
        model_name_or_path : str | Path, optional
            The model path / name used to load the engine. Used for the
            display name and as a stable hash input. Defaults to the engine's
            internal `model_config.model` if available.
        encode_row, threshold, correct_order_bias, seed,
        **inference_kwargs
            Forwarded to `LLMClassifier`. See base-class docs.
        """
        self._llm = llm
        self._tokenizer = tokenizer

        # Resolve a stable name + vocab_dim without poking vLLM internals where
        # possible. AutoConfig is the canonical source of truth for vocab_size
        # (the logits axis) — same value `model.config.vocab_size` would give
        # on the transformers path.
        resolved_path = self._resolve_model_path(model_name_or_path)
        model_name = Path(resolved_path).name if resolved_path else "vllm-model"
        self._model_name_or_path = resolved_path
        self._vocab_dim = self._resolve_vocab_dim(resolved_path, tokenizer)

        super().__init__(
            model_name=model_name,
            task=task,
            encode_row=encode_row,
            correct_order_bias=correct_order_bias,
            threshold=threshold,
            seed=seed,
            **inference_kwargs,
        )

        # Observability — mirrors transformers_classifier.py:88-125 so the
        # ChainOfThoughtQA failure-rate warning fires identically across backends.
        self._log_generations_all = os.getenv(
            "FOLKTEXTS_LOG_GENERATIONS", "0"
        ).strip() in {"1", "true", "True"}
        try:
            self._log_generations_first_n = int(
                os.getenv("FOLKTEXTS_LOG_GENERATIONS_FIRST_N", "3")
            )
        except ValueError:
            self._log_generations_first_n = 3
        self._logged_generations_count = 0

        self._cot_total = 0
        self._cot_failed = 0

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _resolve_model_path(self, model_name_or_path: str | Path | None) -> str | None:
        """Best-effort lookup of the model path used to load the engine."""
        if model_name_or_path is not None:
            return str(model_name_or_path)
        # vLLM has shuffled this attribute across versions; try a few paths
        # before giving up.
        for getter in (
            lambda llm: llm.llm_engine.model_config.model,
            lambda llm: llm.llm_engine.get_model_config().model,
            lambda llm: llm.llm_engine.vllm_config.model_config.model,
        ):
            try:
                return str(getter(self._llm))
            except AttributeError:
                continue
        return None

    def _resolve_vocab_dim(
        self,
        model_name_or_path: str | None,
        tokenizer: AutoTokenizer,
    ) -> int:
        """Return the model's logits-axis vocab size.

        Always prefer `AutoConfig.vocab_size` over `len(tokenizer.get_vocab())`
        (the latter diverges across families — Gemma-3 has it == vocab_size+1,
        Llama-3.2 has it == vocab_size+256). See CLAUDE.md "Gotchas".
        """
        if model_name_or_path is not None:
            try:
                config = AutoConfig.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                )
                # Multimodal Gemma-3 and similar wrap the language-model config
                # under `text_config`; check both top-level and nested.
                vs = getattr(config, "vocab_size", None)
                if vs is None:
                    vs = getattr(
                        getattr(config, "text_config", None), "vocab_size", None
                    )
                if vs is not None:
                    return int(vs)
                logging.warning(
                    f"AutoConfig {type(config).__name__} for {model_name_or_path} "
                    "exposes no vocab_size at top level or text_config; falling back."
                )
            except Exception as exc:
                logging.warning(
                    f"AutoConfig.from_pretrained failed for {model_name_or_path}: "
                    f"{exc!r}; falling back to tokenizer-derived vocab size."
                )
        # Fallback: best of the two tokenizer-derived numbers; warn loudly,
        # since this is the path that the vocab-mismatch bug used to trip on.
        fallback = max(
            getattr(tokenizer, "vocab_size", 0),
            len(tokenizer.get_vocab()) if hasattr(tokenizer, "get_vocab") else 0,
        )
        logging.warning(
            f"Falling back to tokenizer-derived vocab_dim={fallback}; "
            f"this can mis-size the logits mask on Gemma-3 / Llama-3.2 "
            f"families. Pass `model_name_or_path` explicitly to use AutoConfig."
        )
        return int(fallback)

    # ------------------------------------------------------------------
    # Properties / hashing
    # ------------------------------------------------------------------

    @property
    def llm(self):
        return self._llm

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    def __hash__(self) -> int:
        """Hash distinct from `TransformersLLMClassifier.__hash__`.

        Including the backend tag prevents result-file paths
        (`results.bench-{hash}.json`) from colliding when the same model is
        run under both backends — predictions can differ on the order of 1e-3
        due to attention-kernel differences, so re-using a transformers CSV
        for a vLLM run would silently mix them.
        """
        hash_params = dict(
            super_hash=super().__hash__(),
            backend="vllm",
            vocab_dim=self._vocab_dim,
        )
        return int(hash_dict(hash_params), 16)

    # ------------------------------------------------------------------
    # ChainOfThoughtQA failure-rate observability — mirrors transformers backend
    # ------------------------------------------------------------------

    _COT_FAILURE_WARN_THRESHOLD = 0.25
    _COT_FAILURE_WARN_MIN_SAMPLES = 20

    def _should_log_generation(self) -> bool:
        if self._log_generations_all:
            return True
        return self._logged_generations_count < max(self._log_generations_first_n, 0)

    def _maybe_warn_cot_failure_rate(self) -> None:
        if self._cot_total < self._COT_FAILURE_WARN_MIN_SAMPLES:
            return
        if (
            self._cot_total % 200 != 0
            and self._cot_total != self._COT_FAILURE_WARN_MIN_SAMPLES
        ):
            return
        rate = self._cot_failed / self._cot_total
        if rate >= self._COT_FAILURE_WARN_THRESHOLD:
            logging.warning(
                f"ChainOfThoughtQA: probability extraction failed for "
                f"{self._cot_failed}/{self._cot_total} samples "
                f"({rate:.1%}); these fall back to 0.5 and will collapse AUC. "
                f"Inspect generations with FOLKTEXTS_LOG_GENERATIONS_FIRST_N."
            )

    # ------------------------------------------------------------------
    # Inference dispatch
    # ------------------------------------------------------------------

    def _query_prompt_risk_estimates_batch(
        self,
        prompts_batch: list[str],
        *,
        question: MultipleChoiceQA | DirectNumericQA | ChainOfThoughtQA,
        context_size: int = None,
    ) -> np.ndarray:
        """Query vLLM with a batch of prompts and return risk estimates."""
        if isinstance(question, ChainOfThoughtQA):
            return self._risk_estimates_cot(prompts_batch, question, context_size)

        if isinstance(question, DirectNumericQA):
            return self._risk_estimates_numeric(prompts_batch, question, context_size)

        return self._risk_estimates_multiple_choice(
            prompts_batch, question, context_size
        )

    # ------------------------------------------------------------------
    # ChainOfThoughtQA path: text generation + regex extraction
    # ------------------------------------------------------------------

    def _risk_estimates_cot(
        self,
        prompts_batch: list[str],
        question: ChainOfThoughtQA,
        context_size: int | None,
    ) -> np.ndarray:
        from vllm import (
            SamplingParams,  # local import — keeps module importable without vllm
        )

        # Apply chat template (or fall back to raw prompts for base models) —
        # exact same path the transformers backend takes. `enable_thinking` is
        # threaded through.
        formatted_inputs = _apply_chat_template_batch(
            prompts_batch,
            tokenizer=self._tokenizer,
            enable_thinking=question.enable_thinking,
            system_prompt=(
                self.prompt_config.system_prompt()
                if self.prompt_config.system_prompt is not None
                else None
            ),
        )

        sampling_params = SamplingParams(
            temperature=self._resolve_temperature(question),
            max_tokens=question.max_new_tokens,
            seed=self.seed,
        )
        outputs = self._llm.generate(formatted_inputs, sampling_params)

        risk_estimates_batch: list[float] = []
        for idx, (prompt, request_output) in enumerate(zip(prompts_batch, outputs)):
            generated_text = request_output.outputs[0].text
            response_text = _postprocess_generated_text(
                generated_text,
                enable_thinking=question.enable_thinking,
                i=idx,
                n=len(outputs),
            )

            extracted = question.extract_probability_from_text(response_text)
            self._cot_total += 1
            if extracted is None:
                self._cot_failed += 1
            risk_estimate = 0.5 if extracted is None else extracted
            risk_estimates_batch.append(risk_estimate)
            self._maybe_warn_cot_failure_rate()

            if self._should_log_generation():
                logging.info(
                    "\n"
                    + "=" * 60
                    + "\n"
                    + f"[ChainOfThoughtQA Sample {self._logged_generations_count + 1}]"
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

        return np.asarray(risk_estimates_batch, dtype=float)

    # ------------------------------------------------------------------
    # DirectNumericQA path: greedy + digit-only constraint
    # ------------------------------------------------------------------

    def _risk_estimates_numeric(
        self,
        prompts_batch: list[str],
        question: DirectNumericQA,
        context_size: int | None,
    ) -> np.ndarray:
        from vllm import SamplingParams

        digit_token_ids = sorted(
            {
                tok_id
                for token, tok_id in self._tokenizer.get_vocab().items()
                if token.isdecimal() and 0 <= tok_id < self._vocab_dim
            }
        )
        if not digit_token_ids:
            raise RuntimeError(
                "No digit tokens found in tokenizer vocabulary; cannot run "
                "DirectNumericQA on this model."
            )

        # Always 0.0 — numeric QA reads the next-token distribution, it doesn't
        # sample. With `logprobs_mode="processed_logprobs"` any temperature > 0
        # would rescale the returned logprobs (vLLM divides logits by the
        # temperature before computing them) and silently change risk scores
        # relative to the other backends, which read untempered probabilities.
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=question.num_forward_passes,
            logprobs=_TOPK_LOGPROBS,
            allowed_token_ids=digit_token_ids,
            seed=self.seed,
        )
        outputs = self._llm.generate(prompts_batch, sampling_params)

        risk_estimates_batch: list[float] = []
        for request_output in outputs:
            per_pass_topk = self._extract_per_pass_topk(request_output)
            risk_estimate = decode_topk_logprobs_to_risk_estimate(
                per_pass_topk,
                tokenizer_vocab=self._tokenizer.get_vocab(),
                vocab_dim=self._vocab_dim,
                question=question,
            )
            risk_estimates_batch.append(risk_estimate)

        return np.asarray(risk_estimates_batch, dtype=float)

    # ------------------------------------------------------------------
    # MultipleChoiceQA path: unconstrained next-token logprobs
    # ------------------------------------------------------------------

    def _risk_estimates_multiple_choice(
        self,
        prompts_batch: list[str],
        question: MultipleChoiceQA,
        context_size: int | None,
    ) -> np.ndarray:
        # Match the transformers contract: MC reads the unconstrained next-token
        # softmax (no `allowed_token_ids` mask). The QA decoder's prefix-variant
        # logic + answer-token renormalisation handles candidates not in top-K
        # the same way it handles low-mass tokens on the transformers path.
        from vllm import SamplingParams

        # Always 0.0 — see the equivalent comment in `_risk_estimates_numeric`.
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=_TOPK_LOGPROBS,
            seed=self.seed,
        )
        outputs = self._llm.generate(prompts_batch, sampling_params)

        risk_estimates_batch: list[float] = []
        for request_output in outputs:
            per_pass_topk = self._extract_per_pass_topk(request_output)
            risk_estimate = decode_topk_logprobs_to_risk_estimate(
                per_pass_topk,
                tokenizer_vocab=self._tokenizer.get_vocab(),
                vocab_dim=self._vocab_dim,
                question=question,
            )
            risk_estimates_batch.append(risk_estimate)

        return np.asarray(risk_estimates_batch, dtype=float)

    # ------------------------------------------------------------------
    # vLLM output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_per_pass_topk(request_output) -> list[dict[int, float]]:
        """Convert vLLM's per-position logprobs into ``[{token_id: logprob}, ...]``.

        ``request_output.outputs[0].logprobs`` is a list (one entry per
        generated token) of dicts ``{token_id: Logprob(logprob, ...)}``. We
        only need the ``logprob`` field; the rest is metadata.
        """
        completion = request_output.outputs[0]
        position_logprobs = completion.logprobs or []
        per_pass: list[dict[int, float]] = []
        for pos in position_logprobs:
            per_pass.append(
                {
                    int(tok_id): float(getattr(lp, "logprob", lp))
                    for tok_id, lp in pos.items()
                }
            )
        return per_pass
