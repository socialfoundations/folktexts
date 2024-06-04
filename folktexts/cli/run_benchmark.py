#!/usr/bin/env python3
"""Runs the LLM calibration benchmark from the command line.
"""
import logging
import sys
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

from folktexts._io import save_json
from folktexts.classifier import LLMClassifier
from folktexts.dataset import Dataset
from folktexts.evaluation import evaluate_predictions
from folktexts.prompting import (
    encode_row_prompt,
    encode_row_prompt_chat,
    encode_row_prompt_few_shot,
)
from folktexts.task import TaskMetadata

DEFAULT_BATCH_SIZE = 16
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
        # ("--sens-attr",     str, "[str] Column with sensitive attribute", False),
        ("--few-shot",      int, "[int] Use few-shot prompting with the given number of shots", False),
        ("--batch-size",    int, "[int] The batch size to use for inference", False, DEFAULT_BATCH_SIZE),
        ("--context-size",  int, "[int] The maximum context size when prompting the LLM", False, DEFAULT_CONTEXT_SIZE),
        ("--subsampling",   float, "[float] Which fraction of the dataset to use (if omitted will use all data)", False),
        ("--seed",          int, "[int] Random seed to use for the experiment", False, DEFAULT_SEED),
        # ("--hash",          str, "[str] Random seed to use for the experiment", False),
        ("--fit-threshold", int, "[int] Whether to fit the prediction threshold, and on how many samples", False),
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

    parser.add_argument(
        "--logger-level",
        type=str,
        help="[str] The logging level to use for the experiment",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        required=False,
    )

    return parser


def run_llm_risk_scores_evaluation(     # TODO! RUN AND CHECK IF THIS WORKS!
    llm_clf: LLMClassifier,
    dataset: Dataset,
    results_dir: Path,
    seed: int = DEFAULT_SEED,
) -> dict:

    # Get test data
    X_test, y_test = dataset.get_test()
    print(f"Test data features shape: {X_test.shape}")

    # Get sensitive attribute data if available
    s_test = None
    if llm_clf.task.sensitive_attribute is not None:
        s_test = dataset.get_sensitive_attribute_data().loc[y_test.index]

    # Get LLM risk-estimate predictions for each row in the test set
    y_test_scores = llm_clf.predict_proba(
        data=X_test,
        predictions_save_path=(
            results_dir
            / f"{dataset.get_name()}.test_predictions.csv"
        ),
        labels=y_test,  # used only to save alongside predictions in disk
    )

    # Evaluate test risk scores
    results = evaluate_predictions(
        y_true=y_test.to_numpy(),
        y_pred_scores=y_test_scores,
        sensitive_attribute=s_test,
        threshold=llm_clf.threshold,
        imgs_dir=results_dir,
        model_name=Path(llm_clf.model.name_or_path).name,
    )

    # Save results to disk
    from folktexts.cli._constants import RESULTS_JSON_FILE_NAME
    save_json(results, path=results_dir / RESULTS_JSON_FILE_NAME)

    # Log main results
    msg = (
        f"\n** Test results **\n"
        f"Model balanced accuracy:  {results['balanced_accuracy']:.1%};\n"
        f"Model accuracy:           {results['accuracy']:.1%};\n"
        f"Model ROC AUC :           {results['roc_auc']:.1%};\n"
    )
    print(msg)
    print(f"Saved experiment results to '{results_dir.as_posix()}'")

    return results


def run_llm_calibration_evaluation() -> dict:
    # TODO
    raise NotImplementedError("Calibration evaluation is not yet implemented.")


if __name__ == '__main__':
    """Prepare and launch the LLM-as-classifier experiment using ACS data."""

    # Setup parser and process cmd-line args
    parser = setup_arg_parser()
    args = parser.parse_args()

    # NOTE: the following code block parses extra non-argparse kwargs
    # from folktexts.cli._utils import cmd_line_args_to_kwargs
    # args, extra_kwargs = parser.parse_known_args()
    # extra_kwargs = cmd_line_args_to_kwargs(extra_kwargs)
    # if extra_kwargs:
    #     logging.warning(f"Received the following extra kwargs: {extra_kwargs}")

    logging.getLogger().setLevel(level=args.logger_level or "INFO")
    logging.info(f"Received the following cmd-line args: {args.__dict__}")
    logging.info(f"Current python executable: '{sys.executable}'\n")

    # Create results directory if needed
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nStarting experiment with `results_dir='{results_dir.as_posix()}'`.")

    # Save experiment's cmd-line args to save-dir
    from folktexts.cli._constants import ARGS_JSON_FILE_NAME
    save_json(vars(args), path=results_dir / ARGS_JSON_FILE_NAME)

    # ### ### ### ### ### ### ### ### #
    # Construct LLMClassifier object  #
    # ### ### ### ### ### ### ### ### #

    # Load ACS Task object
    task = TaskMetadata.get_task(args.task_name)

    # Load ACS dataset
    print(f"\nLoading ACS dataset for task {args.task_name}...")
    from folktexts.acs import ACSDataset
    dataset = ACSDataset(
        task_name=args.task_name,
        cache_dir=args.data_dir,
        seed=args.seed,
        subsampling=args.subsampling,
    )

    # Load model and tokenizer
    from folktexts.llm_utils import load_model_tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_tokenizer(args.model)

    # Get prompting function
    if args.chat_prompt:
        encode_row_function = partial(encode_row_prompt_chat, task=task, tokenizer=tokenizer)
    else:
        encode_row_function = partial(encode_row_prompt, task=task)

    if args.few_shot:
        encode_row_function = partial(
            encode_row_prompt_few_shot,
            task=task,
            n_shots=args.few_shot,
            dataset=dataset,
            reuse_examples=args.reuse_few_shot_examples,
        )

    # Load the QA interface to be used for risk-score prompting
    from folktexts.acs.acs_questions import acs_multiple_choice_qa_map, acs_numeric_qa_map
    if args.direct_risk_prompting:
        question = acs_numeric_qa_map[task.target]
    else:
        question = acs_multiple_choice_qa_map[task.target]

    # Set the task's target question
    task.cols_to_text[task.target]._question = question

    # Construct the LLMClassifier object
    llm_clf = LLMClassifier(
        model=model,
        tokenizer=tokenizer,
        task=task,
        encode_row=encode_row_function,
        batch_size=args.batch_size,
        context_size=args.context_size,
    )

    # Run benchmark
    run_llm_risk_scores_evaluation(     # TODO: extract this functionality to a benchmark module
        llm_clf=llm_clf,
        dataset=dataset,
        results_dir=results_dir,
        seed=args.seed,
    )

    from folktexts.cli._utils import get_current_timestamp
    print(f"\nFinished experiment successfully at {get_current_timestamp()}\n")
