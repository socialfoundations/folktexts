#!/usr/bin/env python
import logging
from pathlib import Path
from argparse import ArgumentParser

from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance

from folktexts.llm_utils import load_model_tokenizer, get_model_folder_path
from folktexts._io import save_pickle


# Local paths
ROOT_DIR = Path("/fast/groups/sf")            # CLUSTER dir
# ROOT_DIR = Path("~").expanduser().resolve()     # LOCAL dir

MODELS_DIR = ROOT_DIR / "huggingface-models"
DATA_DIR = ROOT_DIR / "data"
RESULTS_ROOT_DIR = ROOT_DIR / "folktexts-results"

# MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_NAME = "google/gemma-2b"    # NOTE: this is among the smallest models

TASK_NAME = "ACSIncome"

DEFAULT_CONTEXT_SIZE = 500
DEFAULT_BATCH_SIZE = 30
DEFAULT_SEED = 42


def setup_arg_parser() -> ArgumentParser:

    # Init parser
    parser = ArgumentParser(description="Evaluate LLM feature importance on a given ACS task.")

    # List of command-line arguments, with type and helper string
    cli_args = [
        ("--model",         str, "[str] Model name or path to model saved on disk"),
        ("--task-name",     str, "[str] Name of the ACS task to run the experiment on"),
        ("--results-dir",   str, "[str] Directory under which this experiment's results will be saved"),
        ("--data-dir",      str, "[str] Root folder to find datasets on"),
        ("--batch-size",    int, "[int] The batch size to use for inference", False, DEFAULT_BATCH_SIZE),
        ("--context-size",  int, "[int] The maximum context size when prompting the LLM", False, DEFAULT_CONTEXT_SIZE),
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
    return parser


def compute_feature_importance(llm_clf, dataset):

    # # Optionally, fit the LLM classifier's threshold on a few data samples.
    # llm_clf.fit(*dataset[:1000])

    # Get train and test data
    X_train, y_train = dataset.get_train()
    X_test, y_test = dataset.get_test()

    permutation_kwargs = dict(
        X=X_test, y=y_test,
        scoring="roc_auc",
        n_repeats=5,
        random_state=SEED,
    )

    # Baseline: GBM feature importance
    gbm_clf = LGBMClassifier()
    gbm_clf.fit(X_train, y_train)

    r = permutation_importance(gbm_clf, **permutation_kwargs)

    save_pickle(obj=r, path=f"permutation-importance.{TASK_NAME}.GBM.pkl")

    # Print results:
    print("GBM feature importance:")
    for i in r.importances_mean.argsort()[::-1]:
        # if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(
            f"{X_test.columns[i]:<8}"
            f"{r.importances_mean[i]:.3f}"
            f" +/- {r.importances_std[i]:.3f}")

    # LLM feature importance
    r = permutation_importance(llm_clf, **permutation_kwargs)
    save_pickle(obj=r, path=f"permutation-importance.{TASK_NAME}.{llm_clf.model_name}.pkl")

    print("LLM feature importance:")
    for i in r.importances_mean.argsort()[::-1]:
        # if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(
            f"{X_test.columns[i]:<8}"
            f"{r.importances_mean[i]:.3f}"
            f" +/- {r.importances_std[i]:.3f}")

    print(X_test.columns.tolist())


def main():
    # Parse arguments from command line
    args = setup_arg_parser().parse_args()      # TODO: use args to set up the experiment

    # Set logging level
    logging.getLogger().setLevel(logging.INFO)

    # Load model and tokenizer
    model_folder_path = get_model_folder_path(model_name=MODEL_NAME, root_dir=MODELS_DIR)
    model, tokenizer = load_model_tokenizer(model_folder_path)

    results_dir = RESULTS_ROOT_DIR / Path(model_folder_path).name
    results_dir.mkdir(exist_ok=True, parents=True)
    results_dir

    # Load Task and Dataset
    from folktexts.acs import ACSTaskMetadata
    task = ACSTaskMetadata.get_task(TASK_NAME)

    from folktexts.acs import ACSDataset
    dataset = ACSDataset.make_from_task(task=task, cache_dir=DATA_DIR)

    # Optionally, subsample dataset # TODO: use command line argument
    # dataset.subsample(0.1)
    # print(f"{dataset.subsampling=}")

    # Construct LLM Classifier
    from folktexts.classifier import LLMClassifier
    llm_clf = LLMClassifier(
        model=model,
        tokenizer=tokenizer,
        task=task,
        batch_size=32,
    )

    # Compute feature importance
    compute_feature_importance(llm_clf, tokenizer, dataset)


if __name__ == "__main__":
    main()
