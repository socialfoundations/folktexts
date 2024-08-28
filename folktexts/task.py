"""Definition of a generic TaskMetadata class.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Iterable

import pandas as pd

from ._utils import hash_dict
from .col_to_text import ColumnToText
from .qa_interface import QAInterface, MultipleChoiceQA, DirectNumericQA
from .threshold import Threshold


@dataclass
class TaskMetadata:
    """A base class to hold information on a prediction task."""

    name: str
    """The name of the task."""

    description: str
    """A description of the task, including the population to which the task pertains to."""

    features: list[str]
    """The names of the features used in the task."""

    target: str
    """The name of the target column."""

    cols_to_text: dict[str, ColumnToText]
    """A mapping between column names and their textual descriptions."""

    sensitive_attribute: str = None
    """The name of the column used as the sensitive attribute data (if provided)."""

    target_threshold: Threshold = None
    """The threshold used to binarize the target column (if provided)."""

    population_description: str = None
    """A description of the population to which the task pertains to."""

    multiple_choice_qa: MultipleChoiceQA = None
    """The multiple-choice question and answer interface for this task."""

    direct_numeric_qa: DirectNumericQA = None
    """The direct numeric question and answer interface for this task."""

    _use_numeric_qa: bool = False
    """Whether to use numeric Q&A instead of multiple-choice Q&A prompts. Default is False."""

    # Class-level task storage
    _tasks: ClassVar[dict[str, "TaskMetadata"]] = field(default={}, init=False, repr=False)

    def __post_init__(self):
        # Check if this task had already been created
        if self.name in TaskMetadata._tasks:
            logging.error(f"A task with `name='{self.name}'` already exists. Overwriting...")

        # Add this task to the class-level dictionary
        TaskMetadata._tasks[self.name] = self

        # If no question is explicitly provided, use the question from the target column
        if self.multiple_choice_qa is None and self.direct_numeric_qa is None:
            logging.warning(
                f"No question was explicitly provided for task '{self.name}'. "
                f"Inferring from target column's default question ({self.get_target()}).")

            if self.cols_to_text[self.get_target()]._question is not None:
                question = self.cols_to_text[self.get_target()]._question
                self.set_question(question)

        # Make sure Q&A related attributes are consistent
        if (
            self._use_numeric_qa is True and self.direct_numeric_qa is None
            or self._use_numeric_qa is False and self.multiple_choice_qa is None
        ):
            raise ValueError("Inconsistent Q&A attributes provided.")

    def __hash__(self) -> int:
        hashable_params = dataclasses.asdict(self)
        hashable_params.pop("cols_to_text")
        hashable_params["question_hash"] = hash(self.question)
        return int(hash_dict(hashable_params), 16)

    def get_target(self) -> str:
        """Resolves the name of the target column depending on `self.target_threshold`."""
        if self.target_threshold is None:
            return self.target
        else:
            return self.target_threshold.apply_to_column_name(self.target)

    def set_question(self, question: QAInterface):
        """Sets the Q&A interface for this task."""
        logging.info(f"Setting question for task '{self.name}' to '{question.text}'.")

        if isinstance(question, MultipleChoiceQA):
            self.multiple_choice_qa = question
            self._use_numeric_qa = False
        elif isinstance(question, DirectNumericQA):
            self.direct_numeric_qa = question
            self._use_numeric_qa = True
        else:
            raise ValueError(f"Invalid question type: {type(question).__name__}")

    @property
    def use_numeric_qa(self) -> bool:
        """Getter for whether to use numeric Q&A instead of multiple-choice Q&A prompts."""
        return self._use_numeric_qa

    @use_numeric_qa.setter
    def use_numeric_qa(self, use_numeric_qa: bool):
        """Setter for whether to use numeric Q&A instead of multiple-choice Q&A prompts."""
        logging.info(
            f"Changing Q&A mode for task '{self.name}' to "
            f"{'numeric' if use_numeric_qa else 'multiple-choice'}.")
        self._use_numeric_qa = use_numeric_qa

    @classmethod
    def get_task(cls, name: str, use_numeric_qa: bool = False) -> TaskMetadata:
        """Fetches a previously created task by its name.

        Parameters
        ----------
        name : str
            The name of the task to fetch.
        use_numeric_qa : bool, optional
            Whether to set the retrieved task to use verbalized numeric Q&A
            instead of the default multiple-choice Q&A prompts. Default is False.

        Returns
        -------
        task : TaskMetadata
            The task object with the given name.

        Raises
        ------
        ValueError
            Raised if the task with the given name has not been created yet.
        """
        if name not in cls._tasks:
            raise ValueError(f"Task '{name}' has not been created yet.")

        # Retrieve the task object
        task = cls._tasks[name]

        # Set Q&A interface type
        task.use_numeric_qa = use_numeric_qa

        return task

    @property
    def question(self) -> QAInterface:
        """Getter for the Q&A interface for this task."""

        # Resolve direct numeric Q&A vs multiple-choice Q&A
        if self._use_numeric_qa:
            q = self.direct_numeric_qa
        else:
            q = self.multiple_choice_qa

        if q is None:
            raise ValueError(f"Invalid Q&A interface configured for task {self.name}.")
        return q

    def get_row_description(self, row: pd.Series) -> str:
        """Encode a description of a given data row in textual form."""
        row = row[self.features]
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

    def create_task_with_feature_subset(self, feature_subset: Iterable[str]):
        """Creates a new task with a subset of the original features."""
        # Convert iterable to list
        feature_subset = list(feature_subset)

        # Check if features are a subset of the original features
        if not set(feature_subset).issubset(self.features):
            raise ValueError(
                f"`feature_subset` must be a subset of the original features; "
                f"following features are not in the original set: "
                f"{set(feature_subset) - set(self.features)}"
            )

        # Return new TaskMetadata object
        return dataclasses.replace(
            self,
            name=f"{self.name}_" + "_".join(sorted(feature_subset)),
            features=feature_subset,
        )
