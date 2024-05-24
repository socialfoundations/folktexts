"""General Dataset functionality for text-based datasets."""

from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, ClassVar

import numpy as np
import pandas as pd

from .questions import Question


@dataclass
class TaskMetadata(ABC):
    """A base class to hold information on a prediction task."""
    name: str
    description: str
    features: list[str]
    target: str

    # The name of the column used as the sensitive attribute data (if provided)
    sensitive_attribute: str = None

    # The threshold used to binarize the target column (if provided)
    target_threshold: float = None

    # Class-level task storage
    _tasks: ClassVar[dict[str, "TaskMetadata"]] = field(default={}, init=False, repr=False)

    def __post_init__(self):
        # Add this task to the class-level dictionary
        self._tasks[self.name] = self

    @classmethod
    def get_task(cls, name: str) -> "TaskMetadata":
        if name not in cls._tasks:
            raise ValueError(f"Task '{name}' has not been created yet.")
        return cls._tasks[name]


class ColumnToText:
    """Maps a single column's values to natural text."""

    def __init__(
            self,
            column: str,
            short_description: str,
            value_map: dict[object, str] | Callable = None,
            connector_verb: str = "is",
            question: Question = None,
            missing_value_fill: str = "N/A",
        ):
        """Constructs a `ColumnToText` object.

        Parameters
        ----------
        column : str
            The column's name.
        short_description : str
            A short description of the column to be used before different
            values. For example, short_description="yearly income" will result
            in "The yearly income is [...]".
        value_map : dict[int | str, str] | Callable, optional
            A map between column values and their textual meaning. If not
            provided will use the str() function to convert the values to text.
        connector_verb : str, optional
            Which verb to use when connecting the column's description to its
            value; by default "is".
        missing_value_fill : str, optional
            The value to use when the column's value is not found in the
            `value_map`, by default "N/A".
        """
        self.column = column
        self.short_description = short_description
        self.value_map = value_map
        self.connector_verb = connector_verb
        self.question = question
        self.missing_value_fill = missing_value_fill

        assert (self.value_map is None) or (self.question is None), \
            f"Must provide at most one of (`value_map`, `question`) for col='{self.column}'."

        # If `questions` was provided, build `value_map`
        if self.question is not None:
            self.value_map = self.question.get_value_to_text_map()

        # Else, if `value_map` was provided, build `question`
        elif self.value_map is not None and isinstance(self.value_map, dict):
            assert self.question is None, "Sanity check failed."

            self.question = Question.make_question_from_value_map(
                self.value_map, self.short_description,
            )

    def __getitem__(self, key):
        if self.value_map is None:
            return str(key)

        if isinstance(self.value_map, dict):
            return self.value_map.get(key, self.missing_value_fill)
        else:
            return self.value_map(key)

    def get_text(self, value: object) -> str:
        return f"The {self.short_description} {self.connector_verb} {self[value]}."


# TODO: make the test and val sets optional; we're doing zero-shot so there's no need to split the data...
class Dataset(ABC):
    def __init__(
            self,
            name: str,
            data: pd.DataFrame,
            task_metadata: TaskMetadata,
            cols_to_text: dict[str, ColumnToText],
            test_size: float = 0.1,
            val_size: float = 0.1,
            subsampling: float = None,
            seed: int = 42,
        ):
        """Construct a Dataset object.

        Parameters
        ----------
        name : str
            The dataset's name.
        data : pd.DataFrame
            The dataset's data in pandas DataFrame format.
        task_metadata : TaskMetadata
            The metadata for the prediction task.
        cols_to_text : dict[str, ColumnToText]
            A mapping between column names and their textual descriptions.
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
        self.name = name
        self.data = data        # NOTE: subsample only after train/test/val split
        self.task = task_metadata
        self.cols_to_text = cols_to_text

        self.test_size = test_size
        self.val_size = val_size
        self.train_size = 1 - test_size - val_size
        self.subsampling = subsampling
        assert self.train_size > 0

        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Get question for this dataset
        self._question = self.cols_to_text[self.task.target].question
        assert self._question is not None, \
            f"No question provided for the target column '{self.task.target}'."

        # Make train/test/val split
        indices = self._rng.permutation(len(self.data))
        self._train_indices = indices[: int(len(indices) * self.train_size)]
        self._test_indices = indices[
            len(self._train_indices):
            int(len(indices) * (self.train_size + self.test_size))]

        if val_size is not None and val_size > 0:
            self._val_indices = indices[
                len(self._train_indices) + len(self._test_indices):]
        else:
            self._val_indices = None

        # Subsample the train/test/val data (if requested)
        if subsampling is not None:
            assert 0 < subsampling <= 1, \
                f"`subsampling={subsampling}` must be in the range (0, 1]"

            self._train_indices = self._train_indices[: int(len(self._train_indices) * subsampling)]
            self._test_indices = self._test_indices[: int(len(self._test_indices) * subsampling)]
            if self._val_indices is not None:
                self._val_indices = self._val_indices[: int(len(self._val_indices) * subsampling)]

    @property
    def question(self):
        return self._question

    def get_features_data(self) -> pd.DataFrame:
        return self.data[self.task.features]

    def get_target_data(self) -> pd.Series:
        return self.data[self.task.target]

    def get_train(self):
        train_data = self.data.iloc[self._train_indices]
        return train_data[self.task.features], train_data[self.task.target]

    def sample_n_train_examples(self, n: int) -> tuple[pd.DataFrame, pd.Series]:
        """Return a balanced set of samples from the training set."""
        raise NotImplementedError("TODO: implement this method.")
        # example_indices = self._rng.choice(self._train_indices, size=n, replace=False)
        # return self.data.iloc[example_indices], y_train.loc[n_samples.index]

    def get_test(self):
        test_data = self.data.iloc[self._test_indices]
        return test_data[self.task.features], test_data[self.task.target]

    def get_val(self):
        if self._val_indices is None:
            return None
        val_data = self.data.iloc[self._val_indices]
        return val_data[self.task.features], val_data[self.task.target]

    def get_row_description(self, row: pd.Series = None, *, row_idx: int = None):
        assert (row is not None) ^ (row_idx is not None), "Either `row` or `row_idx` must be provided."
        if row is None:
            row = self.data.iloc[row_idx]

        return (
            "\n".join(
                "- " + self.cols_to_text[col].get_text(val)
                for col, val in row.items()
            )
        )
