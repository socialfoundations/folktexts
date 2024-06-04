"""A benchmark class for measuring and evaluating LLM calibration.
"""
from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._io import save_json
from ._utils import hash_dict, is_valid_number
from .acs.acs_dataset import ACSDataset
from .acs.acs_questions import acs_multiple_choice_qa_map, acs_numeric_qa_map
from .acs.acs_tasks import ACSTaskMetadata
from .classifier import LLMClassifier
from .dataset import Dataset
from .evaluation import evaluate_predictions
from .plotting import render_evaluation_plots, render_fairness_plots
from .prompting import encode_row_prompt, encode_row_prompt_chat, encode_row_prompt_few_shot

DEFAULT_SEED = 42

RESULTS_JSON_FILE_NAME = "results.json"
ARGS_JSON_FILE_NAME = "cmd-line-args.json"


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
        seed: int = DEFAULT_SEED,
    ):
        self.llm_clf = llm_clf
        self.dataset = dataset
        self.seed = seed

        # Create sub-folder under the given root folder
        subfolder_name = f"{self.llm_clf.model_name}_bench-{hash(self)}"
        self.results_dir = Path(results_dir).resolve() / subfolder_name
        self.results_dir.mkdir(exist_ok=True, parents=False)

        # Create sub-folder for images
        self.imgs_dir = self.results_dir / "imgs"
        self.imgs_dir.mkdir(exist_ok=True, parents=False)

        self._rng = np.random.default_rng(self.seed)
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
    def results(self):
        return self._results

    @property
    def task(self):
        return self.llm_clf.task

    def __hash__(self) -> int:
        hash_params = dict(
            llm_clf_hash=hash(self.llm_clf),
            dataset_hash=hash(self.dataset),
            seed=self.seed,
        )

        return int(hash_dict(hash_params), 16)

    def _get_predictions_save_path(self, data_split: str) -> Path:
        """Standardized path to file containing predictions for the given data split."""
        assert data_split in ("train", "val", "test")
        return self.results_dir / f"{self.dataset.get_name()}.{data_split}_predictions.csv"

    def run(self, fit_threshold: int | False = False) -> float:
        """Run the calibration benchmark experiment."""

        # Get test data
        X_test, y_test = self.dataset.get_test()
        logging.info(f"Test data features shape: {X_test.shape}")

        # Get sensitive attribute data if available
        s_test = None
        if self.task.sensitive_attribute is not None:
            s_test = self.dataset.get_sensitive_attribute_data().loc[y_test.index]

        # Get LLM risk-estimate predictions for each row in the test set
        self._y_test_scores = self.llm_clf.predict_proba(
            data=X_test,
            predictions_save_path=self._get_predictions_save_path("test"),
            labels=y_test,  # used only to save alongside predictions in disk
        )

        # If requested, fit the threshold on a small portion of the train set
        if fit_threshold:
            if not is_valid_number(fit_threshold):
                raise ValueError(f"Invalid fit_threshold={fit_threshold}")

            X_train, y_train = self.dataset.sample_n_train_examples(fit_threshold)
            self.llm_clf.fit(X_train, y_train)
            logging.info(
                f"Fitted threshold on {len(y_train)} train examples; "
                f"threshold={self.llm_clf.threshold:.3f};"
            )

        # Evaluate test risk scores
        self._results = evaluate_predictions(
            y_true=y_test.to_numpy(),
            y_pred_scores=self._y_test_scores,
            sensitive_attribute=s_test,
            threshold=self.llm_clf.threshold,
            model_name=self.llm_clf.model_name,
        )

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

        results_dir = Path(results_dir) if results_dir else self.results_dir
        results_file_name = f"{RESULTS_JSON_FILE_NAME.with_suffix('')}.{hash_dict(self.results)}.json"
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
        **kwargs,
    ) -> CalibrationBenchmark:
        """Create a standardized calibration benchmark on ACS data."""

        # Handle non-standard ACS arguments
        acs_dataset_configs = cls.ACS_DATASET_CONFIGS.copy()
        for arg in acs_dataset_configs:
            if arg in kwargs:
                logging.warning(
                    f"Received non-standard ACS argument '{arg}' (using "
                    f"{arg}={kwargs[arg]} instead of default {arg}={cls.ACS_DATASET_CONFIGS[arg]}). "
                    f"This may affect reproducibility.")
                acs_dataset_configs[arg] = kwargs.pop(arg)

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
            seed=acs_dataset_configs["seed"],
            **kwargs,
        )

    @classmethod
    def make_benchmark(
        cls,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        task: ACSTaskMetadata | str,
        dataset: Dataset,
        results_dir: str | Path,
        chat_prompt: bool = False,
        few_shot: int | None = None,
        reuse_few_shot_examples: bool = False,
        direct_risk_prompting: bool = False,
        batch_size: int = None,
        context_size: int = None,
        correct_order_bias: bool = True,
        seed: int = DEFAULT_SEED,
    ) -> CalibrationBenchmark:
        """Create a calibration benchmark from a given configuration."""

        # Handle TaskMetadata object
        if isinstance(task, str):
            task = ACSTaskMetadata.get_task(task)

        # Check dataset is compatible with task
        if dataset.task is not task and dataset.task.name != task.name:
            raise ValueError(
                f"Dataset task '{dataset.task.name}' does not match the "
                f"provided task '{task.name}'.")

        # Get prompting function
        if chat_prompt:
            encode_row_function = partial(encode_row_prompt_chat, task=task, tokenizer=tokenizer)
        else:
            encode_row_function = partial(encode_row_prompt, task=task)

        if few_shot:
            assert not chat_prompt, "Few-shot prompting is not currently compatible with chat prompting."
            encode_row_function = partial(
                encode_row_prompt_few_shot,
                task=task,
                n_shots=few_shot,
                dataset=dataset,
                reuse_examples=reuse_few_shot_examples,
            )

        # Load the QA interface to be used for risk-score prompting
        if direct_risk_prompting:
            question = acs_numeric_qa_map[task.target]
        else:
            question = acs_multiple_choice_qa_map[task.target]

        # Set the task's target question
        task.cols_to_text[task.target]._question = question

        # Construct the LLMClassifier object
        llm_inference_kwargs = {"correct_order_bias": correct_order_bias}
        if batch_size:
            llm_inference_kwargs["batch_size"] = batch_size
        if context_size:
            llm_inference_kwargs["context_size"] = context_size

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
            seed=seed,
        )
