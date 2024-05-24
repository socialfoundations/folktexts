#!/usr/bin/env python3
"""Runs the "LLM as a classifier" experiment.
"""
import sys
import logging
from pathlib import Path
from functools import partial
from argparse import ArgumentParser

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from folktexts.datasets import Dataset


def setup_arg_parser() -> ArgumentParser:

    # Init parser
    parser = ArgumentParser(description="Run an LLM as a classifier experiment.")

    # List of command-line arguments, with type and helper string
    cli_args = [
        ("--model",         str, "[str] Model name or path to model saved on disk"),
        ("--acs-task-name", str, "[str] Name of the ACS task to run the experiment on"),
        ("--results-dir",   str, "[str] Directory under which this experiment's results will be saved"),
        ("--data-dir",      str, "[str] Root folder to find datasets on"),
        ("--sens-attr",     str, "[str] Column with sensitive attribute", False),                           # TODO
        ("--few-shot",      int, "[int] Use few-shot prompting with the given number of shots", False),     # TODO
        ("--fine-tuning",   float, "[float] Fine-tune model on the given fraction of the data", False),     # TODO
        ("--batch-size",    int, "[int] The batch size to use for inference", False, 8),
        ("--context-size",  int, "[int] The maximum context size when prompting the LLM", False, 2000),
        ("--subsampling",   float, "[float] Which fraction of the dataset to use (if omitted will use 100%)", False),
        ("--seed",          int, "[int] Random seed to use for the experiment", False, 42),
        # ("--hash",          str, "[str] Random seed to use for the experiment", False),
    ]

    for arg in cli_args:
        parser.add_argument(
            arg[0],
            type=arg[1],
            help=arg[2],
            required=(arg[3] if len(arg) > 3 else True),    # NOTE: required by default
            default=(arg[4] if len(arg) > 4 else None),     # default value if provided
        )

    # Add special arguments
    parser.add_argument(
        "--chat-prompt",
        help="[bool] Whether to use chat-based prompting (for instruct models)",
        action="store_true",
        default=False,
    )

    parser.add_argument(    # TODO: implement
        "--direct-risk-prompting",
        help="[bool] Whether to directly prompt for risk-estimates instead of multiple-choice Q&A",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--logger-level",
        type=str,
        help="[str] The logging level to use for the experiment",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        required=False,
    )

    return parser


def run_experiment(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        results_dir: Path,
        few_shot: int | bool = False,
        chat_prompt: bool = False,
        direct_risk_prompting: bool = False,
        batch_size: int = 8,
        context_size: int = 2000,
        seed: int = 42,
        all_args: dict = None,
    ) -> dict:

    # Get test data
    X_test, y_test = dataset.get_test()
    print(f"Test data features shape: {X_test.shape}")

    # Create random number generator
    rng = np.random.default_rng(seed)

    # Get prompting function
    if chat_prompt:
        from folktexts.prompting import encode_row_prompt_chat
        encode_row_function = partial(encode_row_prompt_chat, dataset=dataset, tokenizer=tokenizer)
    else:
        from folktexts.prompting import encode_row_prompt
        encode_row_function = partial(encode_row_prompt, dataset=dataset)

    if few_shot:
        from folktexts.prompting import encode_row_prompt_few_shot

        def _encode_row_few_shot(row) -> str:
            return encode_row_prompt_few_shot(
                row,
                dataset=dataset,
                n_shots=few_shot,
                seed=rng.integers(1e3),
                # NOTE ^ this ensures different examples are used for each row
            )

        encode_row_function = _encode_row_few_shot

    if direct_risk_prompting:
        raise NotImplementedError("Direct risk prompting not implemented yet.")

    # Check subsampling fraction
    subsampling = all_args.get("subsampling", None)
    subsampling_str = f"subsampling-{subsampling:.2f}" if subsampling else "full"

    # Get LLM risk-estimate predictions for each row in the test set
    from folktexts.querying import compute_risk_estimates_for_dataframe
    y_test_scores = compute_risk_estimates_for_dataframe(
        model=model,
        tokenizer=tokenizer,
        df=X_test,
        dataset=dataset,
        encode_row=encode_row_function,
        batch_size=batch_size,
        context_size=context_size,
        predictions_save_path=(
            results_dir
            / f"test_predictions.seed-{seed}.{subsampling_str}.csv"
        ),
    )

    # Evaluate test risk scores
    from folktexts.evaluation import evaluate_predictions
    results = evaluate_predictions(
        y_test.to_numpy(),
        y_test_scores,
        threshold="best",
        imgs_dir=results_dir,
        model_name=Path(model.name_or_path).name,
    )

    # Save results to disk
    from folktexts.cli._utils import save_json
    from folktexts.cli._constants import RESULTS_JSON_FILE_NAME
    save_json(results, path=results_dir / RESULTS_JSON_FILE_NAME)

    # Log main results
    msg = (
        f"\n** Test results **\n"
        f"Model balanced accuracy:  {results['balanced_accuracy']:.1%};\n"
        f"Model accuracy:           {results['accuracy']:.1%};\n"
        f"Model ROC AUC :           {results['roc_auc']:.1%};\n"
    )
    logging.info(msg)
    logging.info(f"Saved experiment results to '{results_dir.as_posix()}'")

    return results


if __name__ == '__main__':
    """Prepare and launch a single experiment trial."""

    # Setup parser and process cmd-line args
    parser = setup_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(level=args.logger_level or "INFO")
    logging.info(f"Received the following cmd-line args: {args.__dict__}")
    logging.info(f"Current python executable: '{sys.executable}'\n")

    # Create results directory if needed
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment's cmd-line args to save-dir
    from folktexts.cli._constants import ARGS_JSON_FILE_NAME
    from folktexts.cli._utils import save_json
    save_json(vars(args), path=results_dir / ARGS_JSON_FILE_NAME)

    # Load ACS dataset
    from folktexts.acs import ACSDataset
    logging.info(f"Loading ACS dataset '{args.acs_task_name}'")
    dataset = ACSDataset(
        task_name=args.acs_task_name,
        cache_dir=args.data_dir,
        seed=args.seed,
        subsampling=args.subsampling,
    )
    # TODO: eventually add option to use other datasets

    # Load model and tokenizer
    logging.info(f"Loading model '{args.model}'")
    from folktexts.llm_utils import load_model_tokenizer
    model, tokenizer = load_model_tokenizer(args.model)

    # Run experiment
    run_experiment(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        results_dir=results_dir,
        few_shot=args.few_shot or False,
        chat_prompt=args.chat_prompt,
        direct_risk_prompting=args.direct_risk_prompting,
        batch_size=args.batch_size,
        context_size=args.context_size,
        seed=args.seed,
        all_args=vars(args),
    )

    from folktexts.cli._utils import get_current_timestamp
    print(f"\nFinished experiment successfully at {get_current_timestamp()}\n")
