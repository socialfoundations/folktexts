#!/usr/bin/env python3
"""Launch htcondor jobs for all ACS benchmark experiments.
"""
import argparse
import logging
import math
from pathlib import Path
from pprint import pprint

from folktexts._io import load_json, save_json
from folktexts.llm_utils import get_model_folder_path, get_model_size_B

from .experiments import Experiment, launch_experiment_job

# All ACS prediction tasks
ACS_TASKS = (
    "ACSIncome",
    "ACSEmployment",
    "ACSMobility",
    "ACSTravelTime",
    "ACSPublicCoverage",
)

################
# Useful paths #
################
ROOT_DIR = Path("/fast/groups/sf")
# ROOT_DIR = Path("~").expanduser().resolve()               # on local machine

# ACS data directory
ACS_DATA_DIR = ROOT_DIR / "data"

# Models save directory
MODELS_DIR = ROOT_DIR / "huggingface-models"


##################
# Global configs #
##################
BATCH_SIZE = 20
CONTEXT_SIZE = 600

JOB_CPUS = 4
JOB_MEMORY_GB = 60
JOB_BID = 250

# LLMs to evaluate
LLM_MODELS = [
    # ** Small models **
    "google/gemma-2b",
    "google/gemma-1.1-2b-it",

    # # ** Medium models **
    "google/gemma-7b",
    "google/gemma-1.1-7b-it",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",

    # # ** Large models **
    "01-ai/Yi-34B",
    "01-ai/Yi-34B-Chat",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "mistralai/Mixtral-8x22B-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
]


# Function that defines common settings among all LLM-as-clf experiments
def make_llm_clf_experiment(
    executable_path: str,
    model_name: str,
    task: str,
    results_dir: str,
    **kwargs,
) -> Experiment:
    """Create an experiment object to run.
    """
    # Get model size
    model_size_B = get_model_size_B(model_name)

    # Get model path
    model_path = get_model_folder_path(model_name, root_dir=MODELS_DIR)
    assert Path(model_path).exists(), f"Model path '{model_path}' does not exist."

    # Split experiment and job kwargs
    job_kwargs = {key: val for key, val in kwargs.items() if key.startswith("job_")}
    experiment_kwargs = {key: val for key, val in kwargs.items() if key not in job_kwargs}

    # Set default job kwargs
    job_kwargs.setdefault("job_cpus", JOB_CPUS)
    job_kwargs.setdefault("job_gpus", math.ceil(model_size_B / 40))     # One GPU per 40B parameters
    job_kwargs.setdefault("job_memory_gb", JOB_MEMORY_GB)
    job_kwargs.setdefault("job_gpu_memory_gb", 35 if model_size_B < 5 else 60)
    job_kwargs.setdefault("job_bid", JOB_BID)

    # Set default experiment kwargs
    n_shots = int(experiment_kwargs.get("few_shot", 1))
    experiment_kwargs.setdefault("batch_size", math.ceil(BATCH_SIZE / n_shots))
    experiment_kwargs.setdefault("context_size", CONTEXT_SIZE * n_shots)
    experiment_kwargs.setdefault("data_dir", ACS_DATA_DIR.as_posix())
    # experiment_kwargs.setdefault("fit_threshold", FIT_THRESHOLD)

    # Define experiment
    exp = Experiment(
        executable_path=executable_path,
        kwargs=dict(
            model=model_path,
            task=task,
            results_dir=results_dir,
            **experiment_kwargs,
        ),
        **job_kwargs,
    )

    # Create LLM results directory
    save_json(
        obj=exp.to_dict(),
        path=Path(results_dir) / f"experiment.{exp.hash()}.json",
        overwrite=True,
    )

    return exp


def setup_arg_parser() -> argparse.ArgumentParser:
    # Init parser
    parser = argparse.ArgumentParser(description="Launch experiments to evaluate LLMs as classifiers.")

    parser.add_argument(
        "--executable-path",
        type=str,
        help="[string] Path to the executable script to run.",
        required=True,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        help="[string] Directory under which results will be saved.",
        required=True,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="[string] Model name on huggingface hub - can provide multiple!",
        required=False,
        action="append",
    )

    parser.add_argument(
        "--task",
        type=str,
        help="[string] ACS task name to run experiments on - can provide multiple!",
        required=False,
        action="append",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct folder structure and print experiments without launching them.",
        default=False,
    )

    parser.add_argument(
        "--experiment-json",
        type=str,
        help="[string] Path to an experiment JSON file to load. Will override all other args.",
        required=False,
    )

    return parser


def main():
    # Parse command-line arguments
    parser = setup_arg_parser()
    args, extra_kwargs = parser.parse_known_args()

    # Parse extra kwargs
    from ._utils import cmd_line_args_to_kwargs
    extra_kwargs = cmd_line_args_to_kwargs(extra_kwargs)

    # Prepare command-line arguments
    models = args.model or LLM_MODELS
    tasks = args.task or ACS_TASKS
    executable_path = Path(args.executable_path).resolve()
    if not executable_path.exists() or not executable_path.is_file():
        raise FileNotFoundError(f"Executable script not found at '{executable_path}'.")

    # Set-up results directory
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # Load experiment from JSON file if provided
    if args.experiment_json:
        print(f"Launching job for experiment at '{args.experiment_json}'...")
        exp = Experiment(**load_json(args.experiment_json))
        all_experiments = [exp]

    # Otherwise, run all experiments planned
    else:
        all_experiments = [
            make_llm_clf_experiment(
                executable_path=executable_path.as_posix(),
                model_name=model,
                task=task,
                results_dir=args.results_dir,
                **extra_kwargs,
            )
            for model in models
            for task in tasks
        ]

    # Log total number of experiments
    print(f"Launching {len(all_experiments)} experiment(s)...\n")
    for i, exp in enumerate(all_experiments):
        cluster_id = launch_experiment_job(exp).cluster() if not args.dry_run else None
        print(f"{i:2}. cluster-id={cluster_id}")
        pprint(exp.to_dict(), indent=4)


if __name__ == "__main__":
    main()
