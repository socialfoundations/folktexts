"""A collection of ACS prediction tasks based on the folktables package.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import folktables
from folktables import BasicProblem

from .._utils import hash_dict
from ..col_to_text import ColumnToText as _ColumnToText
from ..task import TaskMetadata
from . import acs_columns

# Map of ACS column names to ColumnToText objects
_acs_columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in acs_columns.__dict__.values()
    if isinstance(col_mapper, _ColumnToText)
}


@dataclass
class ACSTaskMetadata(TaskMetadata):
    """A class to hold information on an ACS prediction task."""

    # The ACS task object from the folktables package
    folktables_obj: BasicProblem = None

    @classmethod
    def make_folktables_task(
        cls,
        name: str,
        description: str,
        target_threshold: float | int = None,
    ) -> "ACSTaskMetadata":

        # Get the task object from the folktables package
        try:
            folktables_task = getattr(folktables, name)
        except AttributeError:
            raise ValueError(f"Could not find task '{name}' in folktables package.")

        acs_task = ACSTaskMetadata(
            name=name,
            description=description,
            features=folktables_task.features,
            target=folktables_task.target,
            cols_to_text=_acs_columns_map,
            # sensitive_attribute=folktables_task.group,    # NOTE: by default do not run fairness evaluation
            target_threshold=target_threshold,
            folktables_obj=folktables_task,
        )

        return acs_task

    def __hash__(self) -> int:
        hashable_params = asdict(self)
        hashable_params.pop("cols_to_text")
        hashable_params.pop("folktables_obj")
        return int(hash_dict(hashable_params), 16)


# Instantiate folktables tasks
acs_income_task = ACSTaskMetadata.make_folktables_task(
    name="ACSIncome",
    description="predict whether an individual's income is above $50,000",
    target_threshold=50000,
)

acs_public_coverage_task = ACSTaskMetadata.make_folktables_task(
    name="ACSPublicCoverage",
    description="predict whether an individual is covered by public health insurance",
)

acs_mobility_task = ACSTaskMetadata.make_folktables_task(
    name="ACSMobility",
    description="predict whether an individual had the same residential address one year ago",
)

acs_employment_task = ACSTaskMetadata.make_folktables_task(
    name="ACSEmployment",
    description="predict whether an individual is employed",
)

acs_travel_time_task = ACSTaskMetadata.make_folktables_task(
    name="ACSTravelTime",
    description="predict whether an individual has a commute to work that is longer than 20 minutes",
    target_threshold=20,
)
