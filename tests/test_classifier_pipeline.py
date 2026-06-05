"""Integration tests for the Benchmark and TransformersLLMClassifier pipeline.

Uses the tiny-random-gpt2 model and the 10-row ACS fixture for fast runs.
The tiny model produces meaningless logits; tests assert shape/range correctness
and structural properties (e.g. that order-bias correction generates distinct
prompts per permutation).
"""

from __future__ import annotations

import dataclasses
import json
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest

from folktexts.benchmark import Benchmark, BenchmarkConfig
from folktexts.classifier import TransformersLLMClassifier
from folktexts.prompting import (
    FewShotConfig,
    PromptConfig,
    encode_row_prompt_few_shot,
)


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer(causal_lm_name_or_path):
    from folktexts.llm_utils import load_model_tokenizer

    return load_model_tokenizer(causal_lm_name_or_path)


@pytest.fixture(scope="module")
def clf(tiny_model_and_tokenizer, acs_income_task):
    model, tokenizer = tiny_model_and_tokenizer
    return TransformersLLMClassifier(
        model=model,
        tokenizer=tokenizer,
        task=acs_income_task,
        correct_order_bias=True,
        batch_size=3,
        context_size=256,
    )


@pytest.fixture(scope="module")
def clf_no_bias(tiny_model_and_tokenizer, acs_income_task):
    model, tokenizer = tiny_model_and_tokenizer
    return TransformersLLMClassifier(
        model=model,
        tokenizer=tokenizer,
        task=acs_income_task,
        correct_order_bias=False,
        batch_size=3,
        context_size=256,
    )


class TestPredictProba:
    def test_output_shape(self, clf, acs_income_dataset):
        X_test, _ = acs_income_dataset.get_test()
        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)

    def test_output_range(self, clf, acs_income_dataset):
        X_test, _ = acs_income_dataset.get_test()
        proba = clf.predict_proba(X_test)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_rows_sum_to_one(self, clf, acs_income_dataset):
        X_test, _ = acs_income_dataset.get_test()
        proba = clf.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_few_shot(
        self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset
    ):
        model, tokenizer = tiny_model_and_tokenizer
        X_test, _ = acs_income_dataset.get_test()
        prompt_config = PromptConfig.from_dict({}, task=acs_income_task)
        encode_row_fn = partial(
            encode_row_prompt_few_shot,
            task=acs_income_task,
            dataset=acs_income_dataset,
            prompt_config=prompt_config,
            few_shot_config=FewShotConfig(n_shots=2, reuse_examples=True),
        )
        clf_fs = TransformersLLMClassifier(
            model=model,
            tokenizer=tokenizer,
            task=acs_income_task,
            encode_row=encode_row_fn,
            correct_order_bias=True,
            batch_size=3,
            context_size=512,
        )
        proba = clf_fs.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predictions_saved_to_csv(self, clf, acs_income_dataset, tmp_path):
        """predict_proba with predictions_save_path writes a CSV with scores and labels."""
        import pandas as pd

        X_test, y_test = acs_income_dataset.get_test()
        save_path = tmp_path / "preds.csv"
        clf.predict_proba(X_test, predictions_save_path=save_path, labels=y_test)
        assert save_path.exists()
        df = pd.read_csv(save_path, index_col=0)
        assert "risk_score" in df.columns
        assert "label" in df.columns
        assert len(df) == len(X_test)

    def test_predictions_loaded_from_disk(self, clf, acs_income_dataset, tmp_path):
        """predict_proba returns cached scores without re-running inference when the file exists."""
        X_test, y_test = acs_income_dataset.get_test()
        save_path = tmp_path / "preds_cached.csv"
        proba1 = clf.predict_proba(
            X_test, predictions_save_path=save_path, labels=y_test
        )

        with patch.object(
            clf,
            "_query_prompt_risk_estimates_batch",
            side_effect=AssertionError("inference should not run when cache exists"),
        ):
            proba2 = clf.predict_proba(
                X_test, predictions_save_path=save_path, labels=y_test
            )

        np.testing.assert_array_equal(proba1, proba2)


class TestClassifierOrderBias:
    def test_distinct_prompts_per_permutation(self, clf, acs_income_dataset):
        """Each MCQ permutation must produce a distinct prompt — guards against
        the bug where `question` was silently dropped when `prompt_config` was set."""
        X_test, _ = acs_income_dataset.get_test()

        captured: list[tuple] = []  # (question, prompt)
        original_encode_row = clf.encode_row

        def capturing_encode_row(row, **kwargs):
            prompt = original_encode_row(row, **kwargs)
            captured.append((kwargs.get("question"), prompt))
            return prompt

        clf._encode_row = capturing_encode_row
        try:
            clf.predict_proba(X_test)
        finally:
            clf._encode_row = original_encode_row

        # With 2-choice MCQ there are 2 permutations → 2 captures per row
        n_rows = len(X_test)
        assert len(captured) == n_rows * 2, (
            f"Expected {n_rows * 2} encode_row calls (2 permutations × {n_rows} rows), got {len(captured)}"
        )

        # For each row, the two prompts (one per permutation) must differ.
        # Loop order is batch-outer, question-inner, rows-innermost:
        # captured[0..n_rows-1] = all rows under permutation 0,
        # captured[n_rows..2*n_rows-1] = all rows under permutation 1.
        for i in range(n_rows):
            q0, p0 = captured[i]
            q1, p1 = captured[n_rows + i]
            assert q0 != q1, (
                f"Row {i}: question objects are identical across permutations"
            )
            assert p0 != p1, (
                f"Row {i}: prompts are identical across permutations — question override was dropped"
            )

    def test_no_bias_correction_single_prompt_per_row(
        self, clf_no_bias, acs_income_dataset
    ):
        X_test, _ = acs_income_dataset.get_test()

        captured: list[str] = []
        original_encode_row = clf_no_bias.encode_row

        def capturing_encode_row(row, **kwargs):
            prompt = original_encode_row(row, **kwargs)
            captured.append(prompt)
            return prompt

        clf_no_bias._encode_row = capturing_encode_row
        try:
            clf_no_bias.predict_proba(X_test)
        finally:
            clf_no_bias._encode_row = original_encode_row

        assert len(captured) == len(X_test)


class TestBenchmarkConfig:
    def test_default_is_hashable(self):
        cfg = BenchmarkConfig()
        assert isinstance(hash(cfg), int)

    def test_distinct_configs_have_different_hashes(self):
        assert hash(BenchmarkConfig()) != hash(BenchmarkConfig(seed=99))

    def test_hash_with_few_shot_config(self):
        cfg = BenchmarkConfig(few_shot_config=FewShotConfig(n_shots=2))
        assert isinstance(hash(cfg), int)

    def test_few_shot_hash_is_deterministic_across_processes(self):
        """B4 regression: __hash__ hashed few_shot_config with Python's salted builtin
        hash(), so `results.bench-{hash}.json` got a different name every process. The
        few-shot hash must be stable across PYTHONHASHSEED values."""
        import os
        import subprocess
        import sys

        code = (
            "from folktexts.benchmark import BenchmarkConfig;"
            "from folktexts.prompting import FewShotConfig;"
            "print(hash(BenchmarkConfig(few_shot_config=FewShotConfig(n_shots=2))))"
        )

        def _hash_with_seed(seed: int) -> str:
            return subprocess.check_output(
                [sys.executable, "-c", code],
                env={**os.environ, "PYTHONHASHSEED": str(seed)},
            ).strip()

        hashes = {_hash_with_seed(seed) for seed in (1, 2)}
        assert len(hashes) == 1, f"few-shot config hash is not deterministic: {hashes}"

    def test_hash_with_feature_subset(self):
        cfg = BenchmarkConfig(feature_subset=["AGEP", "WKHP"])
        assert isinstance(hash(cfg), int)

    def test_hash_with_prompt_variation(self):
        cfg = BenchmarkConfig(prompt_variation={"format": "bullet"})
        assert isinstance(hash(cfg), int)

    def test_hash_differs_with_few_shot_config(self):
        cfg_zero_shot = BenchmarkConfig()
        cfg_few_shot = BenchmarkConfig(few_shot_config=FewShotConfig(n_shots=2))
        assert hash(cfg_zero_shot) != hash(cfg_few_shot)

    def test_save_load_roundtrip(self, tmp_path):
        cfg = BenchmarkConfig(seed=7, batch_size=4)
        path = tmp_path / "config.json"
        cfg.save_to_disk(path)
        loaded = BenchmarkConfig.load_from_disk(path)
        assert cfg == loaded

    def test_save_load_with_few_shot_config(self, tmp_path):
        """FewShotConfig nested object survives a save/load roundtrip."""
        few_shot = FewShotConfig(n_shots=3, compose="balanced", reuse_examples=True)
        cfg = BenchmarkConfig(few_shot_config=few_shot)
        path = tmp_path / "config_fs.json"
        cfg.save_to_disk(path)
        loaded = BenchmarkConfig.load_from_disk(path)
        assert loaded.few_shot_config == few_shot

    def test_save_load_null_few_shot_config(self, tmp_path):
        """None few_shot_config round-trips correctly (not reconstructed as FewShotConfig)."""
        cfg = BenchmarkConfig()
        path = tmp_path / "config_null_fs.json"
        cfg.save_to_disk(path)
        loaded = BenchmarkConfig.load_from_disk(path)
        assert loaded.few_shot_config is None

    def test_update_ignores_unknown_keys(self):
        cfg = BenchmarkConfig()
        updated = cfg.update(nonexistent_key="value")
        assert updated == cfg

    def test_update_returns_new_object(self):
        cfg = BenchmarkConfig()
        updated = cfg.update(seed=1)
        assert updated is not cfg

    def test_prompt_variation_preserved(self):
        pv = {"format": "bullet", "connector": ":"}
        cfg = BenchmarkConfig(prompt_variation=pv)
        assert cfg.prompt_variation == pv

    def test_no_chat_prompt_warning_on_default_prompts(self, caplog):
        """Spin-off of B1: the chat-only warning used `is not None`, but the defaults
        are the PROMPT_DEFAULT sentinel (not None), so it fired on every default run."""
        import logging

        with caplog.at_level(logging.WARNING):
            Benchmark._validate_config(BenchmarkConfig(use_chat_template=False))
        assert "will be ignored" not in caplog.text

    def test_chat_prompt_warning_when_user_set_without_chat_template(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            Benchmark._validate_config(
                BenchmarkConfig(use_chat_template=False, system_prompt="custom")
            )
        assert "will be ignored" in caplog.text


class TestBenchmarkRun:
    """End-to-end tests for Benchmark.make_benchmark() + Benchmark.run().

    Each test constructs its own Benchmark to avoid shared mutable state
    (make_benchmark can mutate task fields like use_text_output_for_qa).
    """

    def _make_bench(self, model, tokenizer, task, dataset, **config_overrides):
        config_params = {"batch_size": 3, "context_size": 256}
        config_params.update(config_overrides)
        config = BenchmarkConfig(**config_params)
        return Benchmark.make_benchmark(
            task=task,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            config=config,
        )

    def test_run_produces_results(
        self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path
    ):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(model, tokenizer, acs_income_task, acs_income_dataset)
        bench.run(results_root_dir=tmp_path)
        assert bench.results is not None
        assert isinstance(bench.results, dict)

    def test_results_has_expected_keys(
        self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path
    ):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(model, tokenizer, acs_income_task, acs_income_dataset)
        bench.run(results_root_dir=tmp_path)
        for key in ("accuracy", "roc_auc", "model_name", "threshold_fitted_on"):
            assert key in bench.results, f"Missing key '{key}' in results"

    def test_test_scores_shape_and_range(
        self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path
    ):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(model, tokenizer, acs_income_task, acs_income_dataset)
        bench.run(results_root_dir=tmp_path)
        X_test, _ = acs_income_dataset.get_test()
        assert bench._y_test_scores is not None
        assert len(bench._y_test_scores) == len(X_test)
        assert np.all(bench._y_test_scores >= 0) and np.all(bench._y_test_scores <= 1)

    def test_save_results_writes_json(
        self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path
    ):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(model, tokenizer, acs_income_task, acs_income_dataset)
        bench.run(results_root_dir=tmp_path)
        bench.save_results()
        result_files = list(bench.results_dir.glob("results.bench-*.json"))
        assert len(result_files) == 1
        with open(result_files[0]) as f:
            saved = json.load(f)
        assert "accuracy" in saved

    def test_save_results_content_matches_results_dict(
        self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path
    ):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(model, tokenizer, acs_income_task, acs_income_dataset)
        bench.run(results_root_dir=tmp_path)
        bench.save_results()
        result_file = next(bench.results_dir.glob("results.bench-*.json"))
        with open(result_file) as f:
            saved = json.load(f)
        assert saved["accuracy"] == pytest.approx(bench.results["accuracy"])

    def test_run_with_few_shot_config(
        self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path
    ):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(
            model,
            tokenizer,
            acs_income_task,
            acs_income_dataset,
            context_size=512,
            few_shot_config=FewShotConfig(n_shots=2, reuse_examples=True),
        )
        bench.run(results_root_dir=tmp_path)
        assert bench.results is not None
        X_test, _ = acs_income_dataset.get_test()
        assert len(bench._y_test_scores) == len(X_test)

    def test_run_with_balanced_few_shot(
        self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path
    ):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(
            model,
            tokenizer,
            acs_income_task,
            acs_income_dataset,
            context_size=512,
            few_shot_config=FewShotConfig(
                n_shots=2, compose="balanced", reuse_examples=True
            ),
        )
        bench.run(results_root_dir=tmp_path)
        assert bench.results is not None

    def test_run_with_per_class_few_shot(
        self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path
    ):
        """B4 (related): per-class `compose` is normalized to a tuple by FewShotConfig,
        but Dataset.sample_n_train_examples used to accept only list/str -> the documented
        per-class few-shot feature crashed end-to-end. A tuple compose must run."""
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(
            model,
            tokenizer,
            acs_income_task,
            acs_income_dataset,
            context_size=512,
            few_shot_config=FewShotConfig(
                n_shots=2, compose=[1, 1], reuse_examples=True
            ),
        )
        bench.run(results_root_dir=tmp_path)
        assert bench.results is not None

    def test_run_with_prompt_variation(
        self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path
    ):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(
            model,
            tokenizer,
            acs_income_task,
            acs_income_dataset,
            prompt_variation={"format": "bullet", "connector": "is"},
        )
        bench.run(results_root_dir=tmp_path)
        assert bench.results is not None

    def test_run_no_order_bias(
        self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path
    ):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(
            model,
            tokenizer,
            acs_income_task,
            acs_income_dataset,
            correct_order_bias=False,
        )
        bench.run(results_root_dir=tmp_path)
        assert bench.results is not None

    def test_fit_threshold(
        self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset, tmp_path
    ):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(model, tokenizer, acs_income_task, acs_income_dataset)
        bench.run(results_root_dir=tmp_path, fit_threshold=3)
        assert bench.llm_clf._threshold_fitted_on == 3
        assert bench.results["threshold_fitted_on"] == 3

    def test_benchmark_hash_is_stable(
        self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset
    ):
        model, tokenizer = tiny_model_and_tokenizer
        bench = self._make_bench(model, tokenizer, acs_income_task, acs_income_dataset)
        assert hash(bench) == hash(bench)

    def test_benchmark_config_round_trips_via_hash(
        self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset
    ):
        """Two Benchmark objects built from identical configs share the same hash."""
        model, tokenizer = tiny_model_and_tokenizer
        bench1 = self._make_bench(
            model, tokenizer, acs_income_task, acs_income_dataset, seed=7
        )
        bench2 = self._make_bench(
            model, tokenizer, acs_income_task, acs_income_dataset, seed=7
        )
        assert hash(bench1) == hash(bench2)

    def test_different_configs_differ_in_hash(
        self, tiny_model_and_tokenizer, acs_income_task, acs_income_dataset
    ):
        model, tokenizer = tiny_model_and_tokenizer
        bench1 = self._make_bench(
            model, tokenizer, acs_income_task, acs_income_dataset, seed=1
        )
        bench2 = self._make_bench(
            model, tokenizer, acs_income_task, acs_income_dataset, seed=2
        )
        assert hash(bench1) != hash(bench2)
