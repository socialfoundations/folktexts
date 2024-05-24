import logging
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd

import folktables
from folktables import ACSDataSource, BasicProblem
from folktables.load_acs import state_list

from . import acs_columns
from ..datasets import ColumnToText, Dataset, TaskMetadata


DEFAULT_ACS_DATA_DIR = Path("~/data/folktables").expanduser().resolve()

# A dict containing all ColumnToText objects from acs_columns.py
acs_columns_map = {
    col_mapper.column: col_mapper
    for col_mapper in acs_columns.__dict__.values()
    if isinstance(col_mapper, ColumnToText)
}


@dataclass
class ACSFolktablesTask(TaskMetadata):
    """A class to hold information on an ACS prediction task."""
    task_object: BasicProblem = None

    @classmethod
    def make_folktables_task(cls, name: str, description: str, target_threshold: float) -> "ACSFolktablesTask":
        # Check if this task has already been created
        if name in TaskMetadata._tasks:
            raise ValueError(f"Task '{name}' has already been created.")

        try:
            task_object = getattr(folktables, name)
        except AttributeError:
            raise ValueError(f"Could not find task '{name}' in folktables package.")

        acs_task = ACSFolktablesTask(
            name=name,
            description=description,
            features=task_object.features,
            target=task_object.target,
            target_threshold=target_threshold,
            task_object=task_object,
        )

        return acs_task


# Instantiate folktables tasks
acs_income_task = ACSFolktablesTask.make_folktables_task(
    name="ACSIncome",
    description="predict whether an individual's income is above $50,000",
    target_threshold=50000,
)

acs_public_coverage_task = ACSFolktablesTask.make_folktables_task(
    name="ACSPublicCoverage",
    description="predict whether an individual is covered by public health insurance",
    target_threshold=0.5,   # target column is already binary
)

acs_mobility_task = ACSFolktablesTask.make_folktables_task(
    name="ACSMobility",
    description="predict whether an individual had the same residential address one year ago",
    target_threshold=0.5,
)

acs_employment_task = ACSFolktablesTask.make_folktables_task(
    name="ACSEmployment",
    description="predict whether an individual is employed",
    target_threshold=0.5,
)

acs_travel_time_task = ACSFolktablesTask.make_folktables_task(
    name="ACSTravelTime",
    description="predict whether an individual has a commute to work that is longer than 20 minutes",
    target_threshold=20,
)


class ACSDataset(Dataset):
    """Wrapper for ACS folktables datasets."""

    def __init__(
            self,
            task_name: str,
            cache_dir: str | Path = DEFAULT_ACS_DATA_DIR,
            **kwargs,
        ):

        # Load ACS data source
        data_source = ACSDataSource(
            survey_year='2018', horizon='1-Year', survey='person',
            root_dir=str(cache_dir),
        )

        # Get ACS data in a pandas DF
        data = data_source.get_data(states=state_list, download=True)

        # Get information on this ACS/folktables task
        acs_task = TaskMetadata.get_task(task_name)

        # Keep only rows used in this task
        data = acs_task.task_object._preprocess(data)

        # Threshold the target column
        new_target_col = f"{acs_task.target}_binary"
        if acs_task.target_threshold is not None:
            data[new_target_col] = (data[acs_task.target] >= acs_task.target_threshold).astype(int)
            acs_task.target = new_target_col

        super().__init__(
            name=task_name,
            data=data,
            task_metadata=acs_task,
            cols_to_text=acs_columns_map,
            **kwargs,
        )
