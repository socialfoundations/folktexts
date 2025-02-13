#!/usr/bin/env python3
"""Runs the LLM calibration benchmark from the command line.
"""
import json
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

DEFAULT_ACS_TASK = "ACSIncome"

DEFAULT_BATCH_SIZE = 16
DEFAULT_CONTEXT_SIZE = 600
DEFAULT_SEED = 42


def setup_arg_parser() -> ArgumentParser:

    # Init parser
    parser = ArgumentParser(description="Benchmark risk scores produced by a language model on ACS data.")

    # Define a custom argument type for a list of strings
    def list_of_strings(arg):
        return arg.split(',')

    # List of command-line arguments, with type and helper string
    cli_args = [
        ("--model",         str, "[str] Model name or path to model saved on disk"),
        ("--results-dir",   str, "[str] Directory under which this experiment's results will be saved"),
        ("--data-dir",      str, "[str] Root folder to find datasets on"),
        ("--task",          str, "[str] Name of the ACS task to run the experiment on", False, DEFAULT_ACS_TASK),
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
        "--use-web-api-model",
        help="[bool] Whether use a model hosted on a web API (instead of a local model)",
        action="store_true",
        default=False,
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
        "--reuse-few-shot-examples",
        help="[bool] Whether to reuse the same samples for few-shot prompting (or sample new ones every time)",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--balance-few-shot-examples",
        help="[bool] Whether to sample evenly from all classes in few-shot prompting",
        action="store_true",
        default=False,
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


def main():
    """Prepare and launch the LLM-as-classifier experiment using ACS data."""

    # Setup parser and process cmd-line args
    parser = setup_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(level=args.logger_level)
    pretty_args_str = json.dumps(vars(args), indent=4, sort_keys=True)
    logging.info(f"Current python executable: '{sys.executable}'")
    logging.info(f"Received the following cmd-line args: {pretty_args_str}")

    # Parse population filter if provided
    population_filter_dict = None
    if args.use_population_filter:
        from folktexts.cli._utils import cmd_line_args_to_kwargs
        population_filter_dict = cmd_line_args_to_kwargs(args.use_population_filter)

    # Load model and tokenizer
    # > Web-hosted LLM
    if args.use_web_api_model:
        model = args.model
        tokenizer = None

    # > Local LLM
    else:
        from folktexts.llm_utils import load_model_tokenizer
        model, tokenizer = load_model_tokenizer(args.model)

    # Fill ACS Benchmark config
    from folktexts.benchmark import BenchmarkConfig
    config = BenchmarkConfig(
        few_shot=args.few_shot,
        numeric_risk_prompting=args.numeric_risk_prompting,
        reuse_few_shot_examples=args.reuse_few_shot_examples,
        balance_few_shot_examples=args.balance_few_shot_examples,
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
