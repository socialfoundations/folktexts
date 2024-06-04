"""Definition of a generic TaskMetadata class.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Callable, ClassVar

import pandas as pd

from ._utils import hash_dict
from .col_to_text import ColumnToText


@dataclass
class TaskMetadata:
    """A base class to hold information on a prediction task.

    Attributes
    ----------
    name : str
        The name of the task.
    description : str
        A description of the task, including the population to which the task
        pertains to.
    features : list[str]
        The names of the features used in the task.
    target : str
        The name of the target column.
    cols_to_text : dict[str, ColumnToText]
        A mapping between column names and their textual descriptions.
    sensitive_attribute : str, optional
        The name of the column used as the sensitive attribute data (if provided).
    target_threshold : float, optional
        The threshold used to binarize the target column (if provided).
    """
    name: str
    description: str
    features: list[str]
    target: str
    cols_to_text: dict[str, ColumnToText]
    sensitive_attribute: str = None
    target_threshold: float | int = None

    # Class-level task storage
    _tasks: ClassVar[dict[str, "TaskMetadata"]] = field(default={}, init=False, repr=False)

    def __post_init__(self):
        # Check if this task had already been created
        if self.name in TaskMetadata._tasks:
            logging.error(f"A task with `name='{self.name}'` already exists. Overwriting...")

        # Add this task to the class-level dictionary
        TaskMetadata._tasks[self.name] = self

    def __hash__(self) -> int:
        hashable_params = asdict(self)
        hashable_params.pop("cols_to_text")
        hashable_params["question_hash"] = hash(self.question)
        return int(hash_dict(hashable_params), 16)

    @classmethod
    def get_task(cls, name: str) -> "TaskMetadata":
        if name not in cls._tasks:
            raise ValueError(f"Task '{name}' has not been created yet.")
        return cls._tasks[name]

    @property
    def question(self):
        if self.cols_to_text[self.target]._question is None:
            raise ValueError(f"No question provided for the target column '{self.target}'.")
        return self.cols_to_text[self.target].question

    def get_row_description(self, row: pd.Series) -> str:
        """Encode a description of a given data row in textual form."""
        # TODO: add different encodings other than bullet point?
        return (
            "\n".join(
                "- " + self.cols_to_text[col].get_text(val)
                for col, val in row.items()
            )
        )

    def sensitive_attribute_value_map(self) -> Callable:
        """Returns a mapping between sensitive attribute values and their descriptions."""
        if self.sensitive_attribute is None:
            logging.warning("No sensitive attribute provided for this task.")
            return {}
        return self.cols_to_text[self.sensitive_attribute].value_map

    def create_task_with_feature_subset(self):
        """Creates a new task with a subset of the original features."""
        raise NotImplementedError # TODO: create tasks that use subset of these features?
