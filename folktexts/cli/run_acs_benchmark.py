#!/usr/bin/env python3
"""Runs the LLM calibration benchmark from the command line."""

from __future__ import annotations

import json
import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from folktexts._utils import ParseDict
from folktexts.prompting import DEFAULT_PROMPT_STYLE, PROMPT_DEFAULT

DEFAULT_ACS_TASK = "ACSIncome"

DEFAULT_BATCH_SIZE = 16
DEFAULT_CONTEXT_SIZE = 600
DEFAULT_SEED = 42

DEFAULT_INFERENCE_BACKEND = "vllm"
DEFAULT_GPU_MEM_UTIL = 0.85
DEFAULT_VLLM_DTYPE = "auto"
DEFAULT_TENSOR_PARALLEL_SIZE = 1


def setup_arg_parser() -> ArgumentParser:

    # Init parser
    parser = ArgumentParser(
        description="Benchmark risk scores produced by a language model on ACS data."
    )

    # Define a custom argument type for a list of strings
    def list_of_strings(arg):
        return arg.split(",")

    # List of command-line arguments, with type and helper string
    cli_args: list[Any] = [
        ("--model", str, "[str] Model name or path to model saved on disk"),
        (
            "--results-dir",
            str,
            "[str] Directory under which this experiment's results will be saved",
        ),
        ("--data-dir", str, "[str] Root folder to find datasets on"),
        (
            "--task",
            str,
            "[str] Name of the ACS task to run the experiment on",
            False,
            DEFAULT_ACS_TASK,
        ),
        (
            "--few-shot",
            int,
            "[int] Use few-shot prompting with the given number of shots",
            False,
        ),
        (
            "--batch-size",
            int,
            "[int] The batch size to use for inference",
            False,
            DEFAULT_BATCH_SIZE,
        ),
        (
            "--context-size",
            int,
            "[int] The maximum context size when prompting the LLM",
            False,
            DEFAULT_CONTEXT_SIZE,
        ),
        (
            "--fit-threshold",
            int,
            "[int] Whether to fit the prediction threshold, and on how many samples",
            False,
        ),
        (
            "--subsampling",
            float,
            "[float] Which fraction of the dataset to use (if omitted will use all data)",
            False,
        ),
        (
            "--seed",
            int,
            "[int] Random seed -- to set for reproducibility",
            False,
            DEFAULT_SEED,
        ),
    ]

    for arg in cli_args:
        parser.add_argument(
            arg[0],
            type=arg[1],
            help=arg[2],
            required=(arg[3] if len(arg) > 3 else True),  # NOTE: required by default
            default=(arg[4] if len(arg) > 4 else None),  # default value if provided
        )

    # Add special arguments (e.g., boolean flags or multiple-choice args)
    parser.add_argument(
        "--use-web-api-model",
        help="[bool] Whether use a model hosted on a web API (instead of a local model)",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--inference-backend",
        type=str,
        choices=["transformers", "vllm"],
        default=DEFAULT_INFERENCE_BACKEND,
        help=(
            "[str] Local inference backend to use; default is 'vllm'. "
            "Pass 'transformers' to fall back to the HuggingFace path. "
            "Ignored when --use-web-api-model is set."
        ),
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEM_UTIL,
        help="[float] vLLM gpu_memory_utilization (default 0.85). Lower if vLLM OOMs at startup.",
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "[int] vLLM max_model_len (input + output tokens). If unset, derived "
            "from --context-size + ChainOfThoughtQA.max_new_tokens for the prompting "
            "mode (currently 8000 for CoT/thinking, 1 otherwise)."
        ),
    )

    parser.add_argument(
        "--vllm-dtype",
        type=str,
        default=DEFAULT_VLLM_DTYPE,
        help="[str] vLLM compute dtype (auto/bfloat16/float16/float32).",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help=(
            "[int] vLLM tensor_parallel_size. If unset, auto-detected from "
            "CUDA_VISIBLE_DEVICES (1 if unset)."
        ),
    )

    parser.add_argument(
        "--dont-correct-order-bias",
        help="[bool] Whether to avoid correcting ordering bias, by default will correct it",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--numeric-risk-prompting",
        help="[bool] Whether to prompt for numeric risk-estimates instead of multiple-choice Q&A",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--cot-prompting",
        help=(
            "[bool] Whether to use chain-of-thought (CoT) prompting where the "
            "model reasons step-by-step before outputting a probability"
        ),
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--enable-thinking",
        help=(
            "[bool] Whether to enable thinking mode for tokenizers that support "
            "it (e.g., Qwen3). Only applies with --cot-prompting"
        ),
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--reuse-few-shot-examples",
        help="[bool] Whether to reuse the same samples for few-shot prompting (or sample new ones every time)",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--compose-few-shot-examples",
        help=(
            "[str|list] How to select samples in few-shot prompting: random, balanced or list of speicified "
            "class counts. Defaults to random."
        ),
        default="random",
    )

    parser.add_argument(
        "--example-order",
        help=(
            "[str] Comma-separated permutation of few-shot example indices, e.g. '2,0,1'. "
            "Only used when --few-shot is set."
        ),
    )

    parser.add_argument(
        "--variation",
        help="[dict] Dictionary specifying variations of data point serialization.",
        nargs="*",
        action=ParseDict,
        required=False,
        default={},
    )

    parser.add_argument(
        "--use-chat-template",
        help="[bool] Whether to format prompts using the tokenizer's chat template (for instruct/chat models)",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--chat-prompt",
        type=str,
        help="[str] Custom assistant prefill text to use with chat templates",
        required=False,
        default=PROMPT_DEFAULT,
    )

    parser.add_argument(
        "--system-prompt",
        type=str,
        help="[str] Custom system prompt text to use with chat templates",
        required=False,
        default=PROMPT_DEFAULT,
    )

    # Optionally, receive a list of features to use (subset of original list)
    parser.add_argument(
        "--use-feature-subset",
        type=list_of_strings,
        help="[str] Optional subset of features to use for prediction, comma separated",
        required=False,
    )

    parser.add_argument(
        "--use-population-filter",
        type=list_of_strings,
        help=(
            "[str] Optional population filter for this benchmark; must follow "
            "the format 'column_name=value' to filter the dataset by a specific value."
        ),
        required=False,
    )

    parser.add_argument(
        "--max-api-rpm",
        type=int,
        help="[int] Maximum number of API requests per minute (if using a web-hosted model)",
        required=False,
    )

    parser.add_argument(
        "--logger-level",
        type=str,
        help="[str] The logging level to use for the experiment",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        required=False,
        default="WARNING",
    )

    return parser


def _loggable_args(args) -> dict:
    """Return the parsed args as a JSON-serializable dict for logging.

    ``--chat-prompt``/``--system-prompt`` default to the ``PROMPT_DEFAULT`` sentinel
    (``object()``), which ``json.dumps`` cannot serialize. Resolve it to ``"default"``
    for logging only (mirrors ``BenchmarkConfig.__hash__``/``save_to_disk``); the sentinel
    itself still flows unchanged into ``BenchmarkConfig``.
    """
    return {k: ("default" if v is PROMPT_DEFAULT else v) for k, v in vars(args).items()}


def main():
    """Prepare and launch the LLM-as-classifier experiment using ACS data."""

    # Setup parser and process cmd-line args
    parser = setup_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(level=args.logger_level)
    pretty_args_str = json.dumps(_loggable_args(args), indent=4, sort_keys=True)
    logging.info(f"Current python executable: '{sys.executable}'")
    logging.info(f"Received the following cmd-line args: {pretty_args_str}")

    # Parse prompt variation dict
    prompt_variation_dict = DEFAULT_PROMPT_STYLE
    if args.variation != {}:
        # update with args.variation
        prompt_variation_dict = {**prompt_variation_dict, **args.variation}

    # Parse population filter if provided
    population_filter_dict = None
    if args.use_population_filter:
        from folktexts.cli._utils import cmd_line_args_to_kwargs

        population_filter_dict = cmd_line_args_to_kwargs(args.use_population_filter)

    # Load model and tokenizer
    backend = None  # webapi when --use-web-api-model; otherwise the local choice
    # Web-hosted LLM
    if args.use_web_api_model:
        model = args.model
        tokenizer = None
        backend = "webapi"

    # Local LLM via vLLM (default)
    elif args.inference_backend == "vllm":
        from folktexts.llm_utils import load_vllm_model

        tensor_parallel_size = args.tensor_parallel_size
        if tensor_parallel_size is None:
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            tensor_parallel_size = max(
                1, len([d for d in cuda_visible.split(",") if d.strip()])
            )
        max_model_len = args.max_model_len
        if max_model_len is None:
            # CoT runs need much more output budget than baseline. Pull from
            # ChainOfThoughtQA.max_new_tokens so this stays in sync if the CoT
            # budget is bumped (Qwen3-Thinking needs ≥ 8k to close </think.
            from folktexts.qa_interface import ChainOfThoughtQA

            cot_max_new_tokens = (
                ChainOfThoughtQA.max_new_tokens
                if (args.cot_prompting or args.enable_thinking)
                else 1
            )
            max_model_len = args.context_size + cot_max_new_tokens + 256

        model, tokenizer = load_vllm_model(
            args.model,
            dtype=args.vllm_dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            seed=args.seed,
        )
        backend = "vllm"

    # Local LLM via HuggingFace transformers (opt-in fallback)
    else:
        from folktexts.llm_utils import load_model_tokenizer

        model, tokenizer = load_model_tokenizer(args.model)
        backend = "transformers"

    # Build FewShotConfig if few-shot prompting is requested
    from folktexts.prompting import FewShotConfig

    few_shot_config = None
    if args.few_shot:
        few_shot_config = FewShotConfig(
            n_shots=args.few_shot,
            compose=args.compose_few_shot_examples,
            reuse_examples=args.reuse_few_shot_examples,
            example_order=args.example_order,
        )

    # Fill ACS Benchmark config
    from folktexts.benchmark import BenchmarkConfig

    config = BenchmarkConfig(
        few_shot_config=few_shot_config,
        prompt_variation=prompt_variation_dict,
        numeric_risk_prompting=args.numeric_risk_prompting,
        cot_prompting=args.cot_prompting,
        enable_thinking=args.enable_thinking,
        use_chat_template=args.use_chat_template,
        chat_prompt=args.chat_prompt,
        system_prompt=args.system_prompt,
        batch_size=args.batch_size,
        context_size=args.context_size,
        correct_order_bias=not args.dont_correct_order_bias,
        feature_subset=args.use_feature_subset or None,
        population_filter=population_filter_dict,
        seed=args.seed,
    )

    # Create ACS Benchmark object
    from folktexts.benchmark import Benchmark

    bench = Benchmark.make_acs_benchmark(
        task_name=args.task,
        model=model,
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        config=config,
        subsampling=args.subsampling,
        max_api_rpm=args.max_api_rpm,
        backend=backend,
        model_name_or_path=args.model if backend == "vllm" else None,
    )

    # Set-up results directory
    from folktexts.cli._utils import get_or_create_results_dir

    results_dir = get_or_create_results_dir(
        model_name=Path(args.model).name,
        task_name=args.task,
        results_root_dir=args.results_dir,
    )
    logging.info(f"Saving results to {results_dir.as_posix()}")

    # Run benchmark
    bench.run(results_root_dir=results_dir, fit_threshold=args.fit_threshold)
    bench.save_results()

    # Save results
    import pprint

    pprint.pprint(bench.results, indent=4, sort_dicts=True)

    # Finish
    from folktexts._utils import get_current_timestamp

    print(f"\nFinished experiment successfully at {get_current_timestamp()}\n")


if __name__ == "__main__":
    main()
