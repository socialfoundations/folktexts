"""General Dataset functionality for text-based datasets.

TODO
----
- Re-assess if the Dataset needs permanent access to the task;
    - The task is already in the LLMClassifier;
    - Maybe the Dataset should simply receive the `task` object whenever a
    method needs it.
"""
from __future__ import annotations

import logging
from abc import ABC

import numpy as np
import pandas as pd

from ._utils import hash_dict, is_valid_number
from .task import TaskMetadata

DEFAULT_TEST_SIZE = 0.1
DEFAULT_VAL_SIZE = None
DEFAULT_SEED = 42


class Dataset(ABC):
    def __init__(
        self,
        data: pd.DataFrame,
        task: TaskMetadata,
        test_size: float = DEFAULT_TEST_SIZE,
        val_size: float = DEFAULT_VAL_SIZE,
        subsampling: float = None,
        seed: int = DEFAULT_SEED,
    ):
        """Construct a Dataset object.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset's data in pandas DataFrame format.
        task : TaskMetadata
            The metadata for the prediction task.
        test_size : float, optional
            The size of the test set, as a fraction of the total dataset size,
            by default 0.1.
        val_size : float, optional
            The size of the validation set, as a fraction of the total dataset
            size, by default 0.1.
        subsampling : float, optional
            Whether to use sub-sampling, and which fraction of the data to keep.
            By default will not use sub-sampling (`subsampling=None`).
        seed : int, optional
            The random state seed, by default 42.
        """
        self._data = data
        self._task = task
        if not isinstance(self._task, TaskMetadata):
            raise ValueError(
                f"Invalid `task` type: {type(self._task)}. "
                f"Expected `TaskMetadata`.")

        self._test_size = test_size
        self._val_size = val_size or 0
        self._train_size = 1 - self._test_size - self._val_size
        assert self._train_size > 0

        self._seed = seed
        self._rng = np.random.default_rng(self._seed)

        # Make train/test/val split
        self._train_indices, self._test_indices, self._val_indices = (
            self._make_train_test_val_split(
                self._data, self.test_size, self.val_size, self._rng)
        )

        # Subsample the train/test/val data (if requested)
        self._subsampling = None
        if subsampling is not None:
            self._subsample_inplace(subsampling)

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def task(self) -> TaskMetadata:
        return self._task

    @task.setter
    def task(self, new_task: TaskMetadata):
        # Check if task columns are in the data
        if not all(col in self.data.columns for col in (new_task.features + [new_task.get_target()])):
            raise ValueError(
                f"Task columns not found in dataset: "
                f"features={new_task.features}, target={new_task.get_target()}")

        self._task = new_task

    @property
    def train_size(self) -> float:
        return self._train_size

    @property
    def test_size(self) -> float:
        return self._test_size

    @property
    def val_size(self) -> float:
        return self._val_size

    @property
    def subsampling(self) -> float:
        return getattr(self, "_subsampling", None)

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def name(self) -> str:
        """A unique name for this dataset."""
        subsampling_str = f"subsampled-{self.subsampling:.3}" if self.subsampling else "full"
        seed_str = f"seed-{self._seed}"
        hash_str = f"hash-{hash(self)}"
        return f"{self.task.name}_{subsampling_str}_{seed_str}_{hash_str}"

    @staticmethod
    def _make_train_test_val_split(
        data,
        test_size: float,
        val_size: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Permute indices
        indices = rng.permutation(len(data))

        # Split train/test
        train_size = 1 - test_size - val_size
        train_indices = indices[: int(len(indices) * train_size)]
        test_indices = indices[
            len(train_indices):
            int(len(indices) * (train_size + test_size))]

        # Split val if requested
        if val_size is not None and val_size > 0:
            val_indices = indices[
                len(train_indices) + len(test_indices):]
        else:
            val_indices = None

        return (
            train_indices,
            test_indices,
            val_indices,
        )

    def _subsample_inplace(self, subsampling: float) -> "Dataset":
        """Subsample the dataset in-place."""

        # Check argument is valid
        if not is_valid_number(subsampling) or not (0 < subsampling <= 1):
            raise ValueError(f"`subsampling={subsampling}` must be in the range (0, 1]")

        # Update train/test/val indices
        new_train_size = int(len(self._train_indices) * subsampling + 0.5)
        new_test_size = int(len(self._test_indices) * subsampling + 0.5)

        self._train_indices = self._train_indices[: new_train_size]
        self._test_indices = self._test_indices[: new_test_size]
        if self._val_indices is not None:
            new_val_size = int(len(self._val_indices) * subsampling + 0.5)
            self._val_indices = self._val_indices[: new_val_size]

        # Update subsampling factor
        self._subsampling = (getattr(self, "_subsampling", None) or 1) * subsampling

        # Log new dataset size
        msg = (
            f"Subsampled dataset to {self.subsampling:.1%} of the original size. "
            f"Train size: {len(self._train_indices)}, "
            f"Test size: {len(self._test_indices)}, "
            f"Val size: {len(self._val_indices) if self._val_indices is not None else 0};"
        )
        logging.info(msg)

        return self

    def subsample(self, subsampling: float):
        """Subsamples this dataset in-place."""
        return self._subsample_inplace(subsampling)

    def _filter_inplace(
        self,
        population_feature_values: dict,
    ) -> "Dataset":
        """Subset the dataset in-place: keep only samples with the given feature values."""
        # Check argument is of valid type
        if not isinstance(population_feature_values, dict):
            raise ValueError(
                f"Invalid `population_feature_values` type: "
                f"{type(population_feature_values)}.")

        # Check argument keys are valid columns
        if not all(key in self.data.columns for key in population_feature_values.keys()):
            raise ValueError(
                f"Invalid `population_feature_values` keys; columns don't exist "
                f"in the dataset: {list(population_feature_values.keys())}.")

        # Create boolean filter based on the given feature values
        population_filter = pd.Series(True, index=self.data.index)
        for key, value in population_feature_values.items():
            population_filter &= (self.data[key] == value)

        # Update train/test/val indices
        train_pop_filter = population_filter.iloc[self._train_indices]
        test_pop_filter = population_filter.iloc[self._test_indices]
        val_pop_filter = population_filter.iloc[self._val_indices] if self._val_indices is not None else None

        self._train_indices = self._train_indices[train_pop_filter]
        self._test_indices = self._test_indices[test_pop_filter]
        self._val_indices = self._val_indices[val_pop_filter] if self._val_indices is not None else None

        return self

    def filter(self, population_feature_values: dict):
        """Filter dataset rows in-place."""
        self._filter_inplace(population_feature_values)

    def get_features_data(self) -> pd.DataFrame:
        return self.data[self.task.features]

    def get_target_data(self) -> pd.Series:
        return self.data[self.task.get_target()]

    def get_sensitive_attribute_data(self) -> pd.Series:
        if self.task.sensitive_attribute is not None:
            return self.data[self.task.sensitive_attribute]
        return None

    def get_data_split(self, split: str) -> tuple[pd.DataFrame, pd.Series]:
        if split == "train":
            return self.get_train()
        elif split == "test":
            return self.get_test()
        elif split == "val":
            return self.get_val()
        else:
            raise ValueError(f"Invalid split '{split}'")

    def get_train(self):
        train_data = self.data.iloc[self._train_indices]
        return train_data[self.task.features], train_data[self.task.get_target()]

    def sample_n_train_examples(
        self,
        n: int,
        reuse_examples: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Return a set of samples from the training set.

        Parameters
        ----------
        n : int
            The number of example rows to return.
        reuse_examples : bool, optional
            Whether to reuse the same examples for consistency. By default will
            sample new examples each time (`reuse_examples=False`).

        Returns
        -------
        X, y : tuple[pd.DataFrame, pd.Series]
            The features and target data for the sampled examples.
        """
        # TODO: make sure examples are class-balanced?
        if reuse_examples:
            example_indices = self._train_indices[:n]
        else:
            example_indices = self._rng.choice(self._train_indices, size=n, replace=False)

        return (
            self.data.iloc[example_indices][self.task.features],
            self.data.iloc[example_indices][self.task.get_target()],
        )

    def get_test(self):
        test_data = self.data.iloc[self._test_indices]
        return test_data[self.task.features], test_data[self.task.get_target()]

    def get_val(self):
        if self._val_indices is None:
            return None
        val_data = self.data.iloc[self._val_indices]
        return val_data[self.task.features], val_data[self.task.get_target()]

    def __getitem__(self, i) -> tuple[pd.DataFrame, pd.Series]:
        """Returns the i-th training sample."""
        curr_indices = self._train_indices[i]
        curr_data = self.data.iloc[curr_indices]
        return curr_data[self.task.features], curr_data[self.task.get_target()]

    def __iter__(self):
        """Iterates over the training data."""
        for i in range(len(self._train_indices)):
            yield self[i]

    def __len__(self) -> int:
        return len(self.data)

    def __hash__(self) -> int:
        hashable_params = {
            "data_shape": self.data.shape,
            "task": hash(self.task),
            "train_size": len(self._train_indices),
            "test_size": len(self._test_indices),
            "val_size": len(self._val_indices) if self._val_indices is not None else 0,
            "subsampling": self.subsampling or 1,
            "seed": self.seed,
        }

        return int(hash_dict(hashable_params), 16)
