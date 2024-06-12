#!/usr/bin/env python3
"""Runs the LLM calibration benchmark from the command line.
"""
import json
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

DEFAULT_BATCH_SIZE = 30
DEFAULT_CONTEXT_SIZE = 500
DEFAULT_SEED = 42


def setup_arg_parser() -> ArgumentParser:

    # Init parser
    parser = ArgumentParser(description="Run an LLM as a classifier experiment.")

    # List of command-line arguments, with type and helper string
    cli_args = [
        ("--model",         str, "[str] Model name or path to model saved on disk"),
        ("--task-name",     str, "[str] Name of the ACS task to run the experiment on"),
        ("--results-dir",   str, "[str] Directory under which this experiment's results will be saved"),
        ("--data-dir",      str, "[str] Root folder to find datasets on"),
        ("--few-shot",      int, "[int] Use few-shot prompting with the given number of shots", False),
        ("--batch-size",    int, "[int] The batch size to use for inference", False, DEFAULT_BATCH_SIZE),
        ("--context-size",  int, "[int] The maximum context size when prompting the LLM", False, DEFAULT_CONTEXT_SIZE),
        ("--fit-threshold", int, "[int] Whether to fit the prediction threshold, and on how many samples", False),
        ("--subsampling",   float, "[float] Which fraction of the dataset to use (if omitted will use all data)", False),
        ("--seed",          int, "[int] Random seed -- to set for reproducibility", False, DEFAULT_SEED),
    ]

    for arg in cli_args:
        parser.add_argument(
            arg[0],
            type=arg[1],
            help=arg[2],
            required=(arg[3] if len(arg) > 3 else True),    # NOTE: required by default
            default=(arg[4] if len(arg) > 4 else None),     # default value if provided
        )

    # Add special arguments (e.g., boolean flags or multiple-choice args)
    parser.add_argument(
        "--dont-correct-order-bias",
        help="[bool] Whether to avoid correcting ordering bias, by default will correct it",
        action="store_false",
        default=True,
    )

    parser.add_argument(
        "--chat-prompt",
        help="[bool] Whether to use chat-based prompting (for instruct models)",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--direct-risk-prompting",
        help="[bool] Whether to directly prompt for risk-estimates instead of multiple-choice Q&A",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--reuse-few-shot-examples",
        help="[bool] Whether to reuse the same samples for few-shot prompting (or sample new ones every time)",
        action="store_true",
        default=False,
    )

    # Optionally, receive a list of features to use (subset of original list)
    parser.add_argument(
        "--use-feature-subset",
        type=str,
        nargs="*",
        help="[str] Optional subset of features to use for prediction",
        required=False,
    )

    parser.add_argument(
        "--use-population-filter",
        type=str,
        nargs="*",
        help=(
            "[str] Optional population filter for this benchmark; must follow "
            "the format 'column_name=value' to filter the dataset by a specific value."
        ),
        required=False,
    )

    parser.add_argument(
        "--logger-level",
        type=str,
        help="[str] The logging level to use for the experiment",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        required=False,
    )

    return parser


def main():
    """Prepare and launch the LLM-as-classifier experiment using ACS data."""

    # Setup parser and process cmd-line args
    parser = setup_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(level=args.logger_level or "INFO")
    pretty_args_str = json.dumps(vars(args), indent=4, sort_keys=True)
    logging.info(f"Current python executable: '{sys.executable}'")
    logging.info(f"Received the following cmd-line args: {pretty_args_str}")

    # Parse population filter if provided
    population_filter_dict = None
    if args.use_population_filter:
        from ._utils import cmd_line_args_to_kwargs
        population_filter_dict = cmd_line_args_to_kwargs(args.use_population_filter)

    # Load model and tokenizer
    from folktexts.llm_utils import load_model_tokenizer
    model, tokenizer = load_model_tokenizer(args.model)

    # Fill ACS Benchmark config
    from folktexts.benchmark import BenchmarkConfig
    config = BenchmarkConfig(
        few_shot=args.few_shot,
        chat_prompt=args.chat_prompt,
        direct_risk_prompting=args.direct_risk_prompting,
        reuse_few_shot_examples=args.reuse_few_shot_examples,
        batch_size=args.batch_size,
        context_size=args.context_size,
        correct_order_bias=not args.dont_correct_order_bias,
        feature_subset=args.use_feature_subset or None,
        population_filter=population_filter_dict,
        seed=args.seed,
    )

    # Create ACS Benchmark object
    from folktexts.benchmark import CalibrationBenchmark
    bench = CalibrationBenchmark.make_acs_benchmark(
        model=model,
        tokenizer=tokenizer,
        task_name=args.task_name,
        data_dir=args.data_dir,
        config=config,
        subsampling=args.subsampling,
    )

    # Create results directory if needed
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=False, exist_ok=True)

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
