#!/usr/bin/env python
import logging
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from sklearn.inspection import permutation_importance

from folktexts._io import save_json, save_pickle
from folktexts.classifier import LLMClassifier
from folktexts.dataset import Dataset
from folktexts.llm_utils import get_model_folder_path, load_model_tokenizer

# Local paths
# DEFAULT_ROOT_DIR = Path("/fast/groups/sf")            # CLUSTER dir
DEFAULT_ROOT_DIR = Path("~").expanduser().resolve()     # LOCAL dir

DEFAULT_MODELS_DIR = DEFAULT_ROOT_DIR / "huggingface-models"
DEFAULT_DATA_DIR = DEFAULT_ROOT_DIR / "data"
DEFAULT_RESULTS_DIR = Path(".")

DEFAULT_TASK_NAME = "ACSIncome"

DEFAULT_CONTEXT_SIZE = 600
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEED = 42

DEFAULT_SUBSAMPLING = 0.1           # NOTE: by default, uses 10% of the dataset
DEFAULT_PERMUTATION_REPEATS = 5


def setup_arg_parser() -> ArgumentParser:

    # Init parser
    parser = ArgumentParser(description="Evaluate LLM feature importance on a given ACS task.")

    # List of command-line arguments, with type and helper string
    cli_args = [
        ("--model",
         str,
         "[str] Model name or path to model saved on disk"),
        ("--task",
         str,
         "[str] Name of the ACS task to run the experiment on",
         False,
         DEFAULT_TASK_NAME,
         ),
        ("--results-dir",
         str,
         "[str] Directory under which this experiment's results will be saved",
         False,
         DEFAULT_RESULTS_DIR,
         ),
        ("--data-dir",
         str,
         "[str] Root folder to find datasets on",
         False,
         DEFAULT_DATA_DIR,
         ),
        ("--models-dir",
         str,
         "[str] Root folder to find huggingface models on",
         False,
         DEFAULT_MODELS_DIR,
         ),
        ("--scorer",
         str,
         "[str] Name of the scorer to use for evaluation",
         False,
         "roc_auc",
         ),
        ("--batch-size",
         int,
         "[int] The batch size to use for inference",
         False,
         DEFAULT_BATCH_SIZE,
         ),
        ("--context-size",
         int,
         "[int] The maximum context size when prompting the LLM",
         False,
         DEFAULT_CONTEXT_SIZE,
         ),
        ("--subsampling",
         float,
         "[float] Which fraction of the dataset to use (if omitted will use all data)",
         False,
         DEFAULT_SUBSAMPLING,
         ),
        ("--fit-threshold",
         int,
         "[int] Whether to fit the prediction threshold, and on how many samples",
         False,
         ),
        ("--seed",
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
            required=(arg[3] if len(arg) > 3 else True),    # NOTE: required by default
            default=(arg[4] if len(arg) > 4 else None),     # default value if provided
        )
    return parser


def parse_feature_importance(results: dict, columns: list[str]) -> dict:
    """Parse the results dictionary of sklearn's permutation_importance."""
    parsed_r = defaultdict(dict)
    for idx, col in enumerate(columns):
        parsed_r[col]["imp_mean"] = results.importances_mean[idx]
        parsed_r[col]["imp_std"] = results.importances_std[idx]

    return parsed_r


def compute_feature_importance(
    llm_clf: LLMClassifier,
    dataset: Dataset,
    scorer: str,
    results_dir: Path,
    fit_threshold=None,
    seed=DEFAULT_SEED,
) -> dict:

    # Get train and test data
    X_test, y_test = dataset.get_test()
    logging.info(f"{X_test.shape=}")

    permutation_kwargs = dict(
        X=X_test, y=y_test,
        scoring=scorer,
        n_repeats=DEFAULT_PERMUTATION_REPEATS,
        random_state=seed,
    )

    # Optionally, fit the LLM classifier's threshold on a few data samples.
    if fit_threshold and isinstance(fit_threshold, int):
        X_train_sample, y_train_sample = dataset.sample_n_train_examples(n=fit_threshold)
        llm_clf.fit(X_train_sample, y_train_sample)

    # LLM feature importance
    r = permutation_importance(llm_clf, **permutation_kwargs)
    llm_imp_file_path = results_dir / f"feature-importance.{llm_clf.task.name}.{llm_clf.model_name}.pkl"
    save_pickle(obj=r, path=llm_imp_file_path)
    save_json(
        parse_feature_importance(results=r, columns=X_test.columns),
        path=llm_imp_file_path.with_suffix(".json"))

    print("LLM feature importance:")
    for i in r.importances_mean.argsort()[::-1]:
        print(
            f"{X_test.columns[i]:<8}"
            f"{r.importances_mean[i]:.3f}"
            f" +/- {r.importances_std[i]:.3f}")

    print(X_test.columns.tolist())


def main():
    # Parse arguments from command line
    args = setup_arg_parser().parse_args()

    # Set logging level
    logging.getLogger().setLevel(logging.INFO)

    # Resolve model folder path
    model_folder_path = Path(get_model_folder_path(model_name=args.model, root_dir=args.models_dir))
    if not model_folder_path.exists() or not model_folder_path.is_dir():
        model_folder_path = Path(args.model).resolve()

    # Load model and tokenizer
    logging.info(f"Loading model from {model_folder_path.as_posix()}")
    model, tokenizer = load_model_tokenizer(model_folder_path)

    # Create results directory if needed
    # Set-up results directory
    from folktexts.cli._utils import get_or_create_results_dir
    results_dir = get_or_create_results_dir(
        model_name=Path(args.model).name,
        task_name=args.task,
        results_root_dir=args.results_dir,
    )
    logging.info(f"Saving results to {results_dir.as_posix()}")

    # Load Task and Dataset
    from folktexts.acs import ACSTaskMetadata
    task = ACSTaskMetadata.get_task(args.task)

    from folktexts.acs import ACSDataset
    dataset = ACSDataset.make_from_task(
        task=task,
        cache_dir=args.data_dir,
        seed=args.seed,
    )

    # Optionally, subsample dataset
    if args.subsampling:
        dataset.subsample(args.subsampling)      # subsample in-place
        logging.info(f"{dataset.subsampling=}")

    # Construct LLM Classifier
    from folktexts.classifier import TransformersLLMClassifier
    llm_clf = TransformersLLMClassifier(
        model=model,
        tokenizer=tokenizer,
        task=task,
        batch_size=args.batch_size,
        context_size=args.context_size,
    )

    # Compute feature importance
    compute_feature_importance(
        llm_clf,
        dataset=dataset,
        scorer=args.scorer,
        results_dir=results_dir,
        fit_threshold=args.fit_threshold,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
