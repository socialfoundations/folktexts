"""General Dataset functionality for text-based datasets."""
from __future__ import annotations

import logging
from abc import ABC

import numpy as np
import pandas as pd

from ._commons import is_valid_number
from .task import TaskMetadata

DEFAULT_TEST_SIZE = 0.1
DEFAULT_VAL_SIZE = None


class Dataset(ABC):
    def __init__(
        self,
        data: pd.DataFrame,
        task: TaskMetadata,
        test_size: float = DEFAULT_TEST_SIZE,
        val_size: float = DEFAULT_VAL_SIZE,
        subsampling: float = None,
        seed: int = 42,
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

        self._test_size = test_size
        self._val_size = val_size or 0
        self._train_size = 1 - self._test_size - self._val_size
        self._subsampling = subsampling
        assert self._train_size > 0

        self._seed = seed
        self._rng = np.random.default_rng(self._seed)

        # Make train/test/val split
        indices = self._rng.permutation(len(self._data))
        self._train_indices = indices[: int(len(indices) * self._train_size)]
        self._test_indices = indices[
            len(self._train_indices):
            int(len(indices) * (self._train_size + self._test_size))]

        if val_size is not None and val_size > 0:
            self._val_indices = indices[
                len(self._train_indices) + len(self._test_indices):]
        else:
            self._val_indices = None

        # Subsample the train/test/val data (if requested)
        if self._subsampling is not None:
            self.subsample(self._subsampling)

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def task(self) -> TaskMetadata:
        return self._task

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
        return self._subsampling

    @property
    def seed(self) -> int:
        return self._seed

    def subsample(self, subsampling: float) -> "Dataset":
        """Subsample the dataset in-place."""
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
        self._subsampling = (self._subsampling or 1) * subsampling

        # Log new dataset size
        msg = (
            f"Subsampled dataset to {self.subsampling:.3f} of the original size. "
            f"Train size: {len(self._train_indices)}, "
            f"Test size: {len(self._test_indices)}, "
            f"Val size: {len(self._val_indices) if self._val_indices is not None else 0};"
        )
        logging.info(msg)

        return self

    def get_name(self) -> str:
        subsampling_str = f"subsampled-{self.subsampling:.3}" if self.subsampling else "full"
        seed_str = f"seed-{self._seed}"
        return f"{self.task.name}_{subsampling_str}_{seed_str}"

    def get_features_data(self) -> pd.DataFrame:
        return self.data[self.task.features]

    def get_target_data(self) -> pd.Series:
        return self.data[self.task.target]

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
        return train_data[self.task.features], train_data[self.task.target]

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
            self.data.iloc[example_indices][self.task.target],
        )

    def get_test(self):
        test_data = self.data.iloc[self._test_indices]
        return test_data[self.task.features], test_data[self.task.target]

    def get_val(self):
        if self._val_indices is None:
            return None
        val_data = self.data.iloc[self._val_indices]
        return val_data[self.task.features], val_data[self.task.target]
