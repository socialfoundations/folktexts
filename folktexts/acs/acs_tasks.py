"""A collection of ACS prediction tasks based on the folktables package.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import folktables
from folktables import BasicProblem

from .._utils import hash_dict
from ..col_to_text import ColumnToText as _ColumnToText
from ..qa_interface import DirectNumericQA, MultipleChoiceQA
from ..task import TaskMetadata
from ..threshold import Threshold
from . import acs_columns, acs_questions
from .acs_thresholds import (
    acs_employment_threshold,
    acs_health_insurance_threshold,
    acs_income_threshold,
    acs_mobility_threshold,
    acs_poverty_ratio_threshold,
    acs_public_coverage_threshold,
    acs_travel_time_threshold,
)

# Map of ACS column names to ColumnToText objects
acs_columns_map: dict[str, object] = {
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
    def make_task(
        cls,
        name: str,
        features: list[str],
        target: str = None,
        sensitive_attribute: str = None,
        target_threshold: Threshold = None,
        multiple_choice_qa: MultipleChoiceQA = None,
        direct_numeric_qa: DirectNumericQA = None,
        description: str = None,
        folktables_obj: BasicProblem = None,
    ) -> ACSTaskMetadata:
        """Create an ACS task object from the given parameters."""
        # Resolve target column name
        target_col_name = (
            target_threshold.apply_to_column_name(target)
            if target_threshold is not None else target)

        # Get default Q&A interfaces for this task's target column
        if multiple_choice_qa is None:
            multiple_choice_qa = acs_questions.acs_multiple_choice_qa_map.get(target_col_name)
        if direct_numeric_qa is None:
            direct_numeric_qa = acs_questions.acs_numeric_qa_map.get(target_col_name)

        return cls(
            name=name,
            features=features,
            target=target,
            cols_to_text=acs_columns_map,
            sensitive_attribute=sensitive_attribute,
            target_threshold=target_threshold,
            multiple_choice_qa=multiple_choice_qa,
            direct_numeric_qa=direct_numeric_qa,
            description=description,
            folktables_obj=folktables_obj,
        )

    @classmethod
    def make_folktables_task(
        cls,
        name: str,
        target_threshold: Threshold = None,
        description: str = None,
    ) -> ACSTaskMetadata:

        # Get the task object from the folktables package
        try:
            folktables_task = getattr(folktables, name)
        except AttributeError:
            raise ValueError(f"Could not find task '{name}' in folktables package.")

        acs_task = cls.make_task(
            name=name,
            features=folktables_task.features,
            target=folktables_task.target,
            sensitive_attribute=folktables_task.group,
            target_threshold=target_threshold,
            description=description,
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
    target_threshold=acs_income_threshold,
    description="predict whether an individual's income is above $50,000",
)

acs_public_coverage_task = ACSTaskMetadata.make_folktables_task(
    name="ACSPublicCoverage",
    target_threshold=acs_public_coverage_threshold,
    description="predict whether an individual is covered by public health insurance",
)

acs_mobility_task = ACSTaskMetadata.make_folktables_task(
    name="ACSMobility",
    target_threshold=acs_mobility_threshold,
    description="predict whether an individual had the same residential address one year ago",
)

acs_employment_task = ACSTaskMetadata.make_folktables_task(
    name="ACSEmployment",
    target_threshold=acs_employment_threshold,
    description="predict whether an individual is employed",
)

acs_travel_time_task = ACSTaskMetadata.make_folktables_task(
    name="ACSTravelTime",
    target_threshold=acs_travel_time_threshold,
    description="predict whether an individual has a commute to work that is longer than 20 minutes",
)

acs_income_poverty_ratio_task = ACSTaskMetadata.make_folktables_task(
    name="ACSIncomePovertyRatio",
    target_threshold=acs_poverty_ratio_threshold,
    description="predict whether an individual's income-to-poverty ratio is below 2.5",
)

# Dummy/test ACS task to predict health insurance coverage using all other available features
acs_full_task = ACSTaskMetadata.make_task(
    name="ACSHealthInsurance-test",
    features=sorted(list({
        *acs_income_task.features,
        *acs_public_coverage_task.features,
        *acs_mobility_task.features,
        *acs_employment_task.features,
        *acs_travel_time_task.features,
    })),
    target="HINS2",
    target_threshold=acs_health_insurance_threshold,
    description=(
        "predict whether an individual has purchased health insurance directly "
        "from an insurance company (as opposed to being insured through an "
        "employer, Medicare, Medicaid, or any other source)"
    ),
)
