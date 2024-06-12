"""A benchmark class for measuring and evaluating LLM calibration.
"""
from __future__ import annotations

import dataclasses
import logging
from functools import partial
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._io import load_json, save_json
from ._utils import hash_dict, is_valid_number
from .acs.acs_dataset import ACSDataset
from .acs.acs_questions import acs_multiple_choice_qa_map, acs_numeric_qa_map
from .acs.acs_tasks import ACSTaskMetadata
from .classifier import LLMClassifier
from .dataset import Dataset
from .evaluation import evaluate_predictions
from .plotting import render_evaluation_plots, render_fairness_plots
from .prompting import encode_row_prompt, encode_row_prompt_chat, encode_row_prompt_few_shot
from .task import TaskMetadata

DEFAULT_SEED = 42
DEFAULT_FIT_THRESHOLD_N = 100


@dataclasses.dataclass(frozen=True, eq=True)
class BenchmarkConfig:
    """A dataclass to hold the configuration for a calibration benchmark."""

    chat_prompt: bool = False
    direct_risk_prompting: bool = False
    few_shot: int | None = None
    reuse_few_shot_examples: bool = False
    batch_size: int | None = None
    context_size: int | None = None
    correct_order_bias: bool = True
    feature_subset: list[str] | None = None
    population_filter: dict | None = None
    seed: int = DEFAULT_SEED

    @classmethod
    def default_config(cls, **changes):
        """Returns the default configuration with optional changes."""
        return cls(**changes)

    @classmethod
    def load_from_disk(cls, path: str | Path):
        """Load the configuration from disk."""
        return cls(**load_json(path))

    def save_to_disk(self, path: str | Path):
        """Save the configuration to disk."""
        save_json(dataclasses.asdict(self), path)

    def __hash__(self) -> int:
        """Generates a unique hash for the configuration."""
        cfg = dataclasses.asdict(self)
        cfg["feature_subset"] = tuple(cfg["feature_subset"]) if cfg["feature_subset"] else None
        cfg["population_filter_hash"] = (
            hash_dict(cfg["population_filter"])
            if cfg["population_filter"] else None
        )
        return int(hash_dict(cfg), 16)


class CalibrationBenchmark:
    """A benchmark class for measuring and evaluating LLM calibration."""

    DEFAULT_BENCHMARK_METRIC = "ece"

    """
    Standardized configurations for the ACS data to use for benchmarking.
    """
    ACS_DATASET_CONFIGS = {
        # ACS survey configs
        "survey_year": "2018",
        "horizon": "1-Year",
        "survey": "person",

        # Data split configs
        "test_size": 0.1,
        "val_size": 0.1,
        "subsampling": None,

        # Fixed random seed
        "seed": 42,
    }

    def __init__(
        self,
        llm_clf: LLMClassifier,
        dataset: Dataset | str,
        results_dir: str | Path,
        config: BenchmarkConfig,
    ):
        self.llm_clf = llm_clf
        self.dataset = dataset
        self.config = config

        # Create sub-folder under the given root folder
        subfolder_name = f"{self.llm_clf.model_name}_bench-{hash(self)}"
        self.results_dir = Path(results_dir).resolve() / subfolder_name
        self.results_dir.mkdir(exist_ok=True, parents=False)

        # Create sub-folder for images
        self.imgs_dir = self.results_dir / "imgs"
        self.imgs_dir.mkdir(exist_ok=True, parents=False)

        self._y_test_scores: np.ndarray = None
        self._results: dict = None
        self._plots: dict = None

        # Log initialization
        msg = (
            f"\n** Benchmark initialization **\n"
            f"Model: {self.llm_clf.model_name};\n"
            f"Task: {self.llm_clf.task.name};\n"
            f"Results dir: {self.results_dir.as_posix()};\n"
            f"Hash: {hash(self)};\n"
        )
        logging.info(msg)

    @property
    def configs_dict(self) -> dict:
        cnf = dataclasses.asdict(self.config)

        # Add info on model, task, and dataset
        cnf["model_name"] = self.llm_clf.model_name
        cnf["model_hash"] = hash(self.llm_clf)
        cnf["task_name"] = self.llm_clf.task.name
        cnf["task_hash"] = hash(self.llm_clf.task)
        cnf["dataset_name"] = self.dataset.get_name()
        cnf["dataset_hash"] = hash(self.dataset)

        return cnf

    @property
    def results(self):
        # Add benchmark configs to the results
        results = self._results.copy()
        results["config"] = self.configs_dict
        return results

    @property
    def task(self):
        return self.llm_clf.task

    def __hash__(self) -> int:
        hash_params = dict(
            llm_clf_hash=hash(self.llm_clf),
            dataset_hash=hash(self.dataset),
            config_hash=hash(self.config),
        )

        return int(hash_dict(hash_params), 16)

    def _get_predictions_save_path(self, data_split: str) -> Path:
        """Standardized path to file containing predictions for the given data split."""
        assert data_split in ("train", "val", "test")
        return self.results_dir / f"{self.dataset.get_name()}.{data_split}_predictions.csv"

    def run(self, fit_threshold: int | bool = False) -> float:
        """Run the calibration benchmark experiment."""

        # Get test data
        X_test, y_test = self.dataset.get_test()
        logging.info(f"Test data features shape: {X_test.shape}")

        # Get sensitive attribute data if available
        s_test = None
        if self.task.sensitive_attribute is not None:
            s_test = self.dataset.get_sensitive_attribute_data().loc[y_test.index]

        # Get LLM risk-estimate predictions for each row in the test set
        test_predictions_save_path = self._get_predictions_save_path("test")
        self._y_test_scores = self.llm_clf.predict_proba(
            data=X_test,
            predictions_save_path=test_predictions_save_path,
            labels=y_test,  # used only to save alongside predictions in disk
        )

        # If requested, fit the threshold on a small portion of the train set
        if fit_threshold:
            if fit_threshold is True:
                fit_threshold = DEFAULT_FIT_THRESHOLD_N
            elif not is_valid_number(fit_threshold) or fit_threshold <= 0:
                raise ValueError(f"Invalid fit_threshold={fit_threshold}")

            logging.info(f"Fitting threshold on {fit_threshold} train samples")
            X_train, y_train = self.dataset.sample_n_train_examples(fit_threshold)
            self.llm_clf.fit(X_train, y_train)

        # Evaluate test risk scores
        self._results = evaluate_predictions(
            y_true=y_test.to_numpy(),
            y_pred_scores=self._y_test_scores,
            sensitive_attribute=s_test,
            threshold=self.llm_clf.threshold,
            model_name=self.llm_clf.model_name,
        )

        # Save predictions save path
        self._results["predictions_path"] = test_predictions_save_path.as_posix()

        # Log main results
        msg = (
            f"\n** Test results **\n"
            f"Model balanced accuracy:  {self.results['balanced_accuracy']:.1%};\n"
            f"Model accuracy:           {self.results['accuracy']:.1%};\n"
            f"Model ROC AUC :           {self.results['roc_auc']:.1%};\n"
        )
        logging.info(msg)

        return self._results[self.DEFAULT_BENCHMARK_METRIC]

    def plot_results(self, imgs_dir: str | Path = None):
        """Render and save evaluation plots."""

        imgs_dir = Path(imgs_dir) if imgs_dir else self.imgs_dir
        _, y_test = self.dataset.get_test()

        plots_paths = render_evaluation_plots(
            y_true=y_test.to_numpy(),
            y_pred_scores=self._y_test_scores,
            eval_results=self.results,
            imgs_dir=imgs_dir,
            model_name=self.llm_clf.model_name,
        )

        # Plot fairness plots if sensitive attribute is provided
        if self.task.sensitive_attribute is not None:
            s_test = self.dataset.get_sensitive_attribute_data().loc[y_test.index]

            plots_paths.update(render_fairness_plots(
                y_true=y_test.to_numpy(),
                y_pred_scores=self._y_test_scores,
                sensitive_attribute=s_test,
                imgs_dir=imgs_dir,
                eval_results=self.results,
                model_name=self.llm_clf.model_name,
                group_value_map=self.task.sensitive_attribute_value_map(),
            ))

        self._results["plots"] = plots_paths

        return plots_paths

    def save_results(self, results_dir: str | Path = None):
        """Saves results to disk."""
        if self.results is None:
            raise ValueError("No results to save. Run the benchmark first.")

        # Get path to results file
        results_dir = Path(results_dir) if results_dir else self.results_dir
        results_file_name = f"results.bench-{hash(self)}.json"

        # Save results to disk
        save_json(self.results, path=results_dir / results_file_name)
        logging.info(f"Saved experiment results to '{results_dir.as_posix()}'")

    @classmethod
    def make_acs_benchmark(
        cls,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        task_name: str,
        results_dir: str | Path,
        data_dir: str | Path = None,
        config: BenchmarkConfig = BenchmarkConfig.default_config(),
        **kwargs,
    ) -> CalibrationBenchmark:
        """Create a standardized calibration benchmark on ACS data.

        Parameters
        ----------
        model : AutoModelForCausalLM
            The torch/transformers language model to use.
        tokenizer : AutoTokenizer
            The tokenizer used to train the model.
        task_name : str
            The name of the ACS task to use.
        results_dir : str | Path
            Path to the directory to save results in.
        data_dir : str | Path, optional
            Path to the directory to load data from and save data in.
        config : BenchmarkConfig, optional
            Extra benchmark configurations, by default will use
            `BenchmarkConfig.default_config()`.
        **kwargs
            Additional arguments passed to `ACSDataset`. By default will use a
            set of standardized dataset configurations for reproducibility.

        Returns
        -------
        bench : CalibrationBenchmark
            The ACS calibration benchmark object.
        """

        # Handle non-standard ACS arguments
        acs_dataset_configs = cls.ACS_DATASET_CONFIGS.copy()
        for arg in acs_dataset_configs:
            if arg in kwargs and kwargs[arg] != cls.ACS_DATASET_CONFIGS[arg]:
                logging.warning(
                    f"Received non-standard ACS argument '{arg}' (using "
                    f"{arg}={kwargs[arg]} instead of default {arg}={cls.ACS_DATASET_CONFIGS[arg]}). "
                    f"This may affect reproducibility.")
                acs_dataset_configs[arg] = kwargs.pop(arg)

        # Log unused kwargs
        if kwargs:
            logging.warning(f"Unused key-word arguments: {kwargs}")

        # Fetch ACS task and dataset
        acs_task = ACSTaskMetadata.get_task(task_name)
        acs_dataset = ACSDataset(
            task=acs_task,
            cache_dir=data_dir,
            **acs_dataset_configs)

        return cls.make_benchmark(
            model=model,
            tokenizer=tokenizer,
            task=acs_task,
            dataset=acs_dataset,
            results_dir=results_dir,
            config=config,
        )

    @classmethod
    def make_benchmark(
        cls,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        task: TaskMetadata | str,
        dataset: Dataset,
        results_dir: str | Path,
        config: BenchmarkConfig = BenchmarkConfig.default_config(),
    ) -> CalibrationBenchmark:
        """Create a calibration benchmark from a given configuration.

        Parameters
        ----------
        model : AutoModelForCausalLM
            The torch/transformers language model to use.
        tokenizer : AutoTokenizer
            The tokenizer used to train the model.
        task : TaskMetadata | str
            The task metadata object or name of the task to use.
        dataset : Dataset
            The dataset to use for the benchmark.
        results_dir : str | Path
            Path to the directory to save results in.
        config : BenchmarkConfig, optional
            Extra benchmark configurations, by default will use
            `BenchmarkConfig.default_config()`.

        Returns
        -------
        bench : CalibrationBenchmark
            The calibration benchmark object.
        """
        # Handle TaskMetadata object
        if isinstance(task, str):
            task = TaskMetadata.get_task(task)

        if config.feature_subset is not None and len(config.feature_subset) > 0:
            task = task.create_task_with_feature_subset(config.feature_subset)
            dataset.task = task

        # Check dataset is compatible with task
        if dataset.task is not task and dataset.task.name != task.name:
            raise ValueError(
                f"Dataset task '{dataset.task.name}' does not match the "
                f"provided task '{task.name}'.")

        if config.population_filter is not None:
            dataset = dataset.filter(config.population_filter)

        # Get prompting function
        if config.chat_prompt:
            logging.warning(f"Untested feature: chat_prompt={config.chat_prompt}")  # TODO!
            encode_row_function = partial(encode_row_prompt_chat, task=task, tokenizer=tokenizer)
        else:
            encode_row_function = partial(encode_row_prompt, task=task)

        if config.few_shot:
            assert not config.chat_prompt, "Few-shot prompting is not currently compatible with chat prompting."
            encode_row_function = partial(
                encode_row_prompt_few_shot,
                task=task,
                n_shots=config.few_shot,
                dataset=dataset,
                reuse_examples=config.reuse_few_shot_examples,
            )

        # Load the QA interface to be used for risk-score prompting
        if config.direct_risk_prompting:
            logging.warning(f"Untested feature: direct_risk_prompting={config.direct_risk_prompting}")  # TODO!
            question = acs_numeric_qa_map[task.target]
        else:
            question = acs_multiple_choice_qa_map[task.target]

        # Set the task's target question
        task.cols_to_text[task.target]._question = question

        # Construct the LLMClassifier object
        llm_inference_kwargs = {"correct_order_bias": config.correct_order_bias}
        if config.batch_size:
            llm_inference_kwargs["batch_size"] = config.batch_size
        if config.context_size:
            llm_inference_kwargs["context_size"] = config.context_size

        llm_clf = LLMClassifier(
            model=model,
            tokenizer=tokenizer,
            task=task,
            encode_row=encode_row_function,
            **llm_inference_kwargs,
        )

        return cls(
            llm_clf=llm_clf,
            dataset=dataset,
            results_dir=results_dir,
            config=config,
        )
