"""Module to hold ACS column mappings from values to natural text.
"""
from functools import partial
from pathlib import Path

from ..col_to_text import ColumnToText
from ..qa_interface import Choice, DirectNumericQA, MultipleChoiceQA
from .._utils import get_thresholded_column_name
from ._utils import parse_pums_code

# Path to ACS codebook files
ACS_OCCP_FILE = Path(__file__).parent / "data" / "OCCP-codes-acs.txt"
ACS_POBP_FILE = Path(__file__).parent / "data" / "POBP-codes-acs.txt"


# Describing ACS columns and corresponding questions
acs_age = ColumnToText(
    "AGEP",
    short_description="age",
    value_map=lambda x: f"{int(x)} years old",
)

acs_class_of_worker = ColumnToText(
    "COW",
    short_description="current employment status",
    question=MultipleChoiceQA(
        column="COW",
        text=(
            "Which one of the following best describes this person's employment "
            "last week or the most recent employment in the past 5 years?",
        ),
        choices=[
            Choice("working for a for-profit private company or organization", 1),
            Choice("working for a non-profit organization", 2),
            Choice("working for the local government", 3),
            Choice("working for the state government", 4),
            Choice("working for the federal government", 5),
            Choice("owner of non-incorporated business, professional practice, or farm", 6),
            Choice("owner of incorporated business, professional practice, or farm", 7),
            Choice("working without pay in a for-profit family business or farm", 8),
            Choice("unemployed and last worked 5 years ago or earlier or never worked", 9),
        ]
    ),
)

acs_schooling = ColumnToText(
    "SCHL",
    short_description="highest grade completed",
    question=MultipleChoiceQA(
        column="SCHL",
        text="What is this person's highest grade or level of school completed?",
        choices=[
            Choice("N/A - no schooling completed", 1),
            Choice("nursery school / preschool", 2),
            Choice("kindergarten", 3),
            Choice("1st grade only", 4),
            Choice("2nd grade", 5),
            Choice("3rd grade", 6),
            Choice("4th grade", 7),
            Choice("5th grade", 8),
            Choice("6th grade", 9),
            Choice("7th grade", 10),
            Choice("8th grade", 11),
            Choice("9th grade", 12),
            Choice("10th grade", 13),
            Choice("11th grade", 14),
            Choice("12th grade, no diploma", 15),
            Choice("regular high school diploma", 16),
            Choice("GED or alternative credential", 17),
            Choice("some college, less than 1 year", 18),
            Choice("some college, 1 or more years, no degree", 19),
            Choice("Associate's degree", 20),
            Choice("Bachelor's degree", 21),
            Choice("Master's degree", 22),
            Choice("Professional degree beyond a bachelor's degree", 23),
            Choice("Doctorate degree", 24),
        ],
    ),
)

acs_marital_status = ColumnToText(
    "MAR",
    short_description="marital status",
    value_map={
        1: "married",
        2: "widowed",
        3: "divorced",
        4: "separated",
        5: "never married",
    },
)

acs_occupation = ColumnToText(
    "OCCP",
    short_description="occupation",
    value_map=partial(
        parse_pums_code,
        file=ACS_OCCP_FILE,
        postprocess=lambda x: x[4:].lower().strip(),
    ),
)

acs_place_of_birth = ColumnToText(
    "POBP",
    short_description="place of birth",
    value_map=partial(parse_pums_code, file=ACS_POBP_FILE),
)

acs_relationship = ColumnToText(
    "RELP",
    short_description="relationship to the reference person in the household",
    value_map={
        0: "the 'reference person' itself",
        1: "husband/wife",
        2: "biological son or daughter",
        3: "adopted son or daughter",
        4: "stepson or stepdaughter",
        5: "brother or sister",
        6: "father or mother",
        7: "grandchild",
        8: "parent-in-law",
        9: "son-in-law or daughter-in-law",
        10: "other relative",
        11: "roomer or boarder",
        12: "housemate or roommate",
        13: "unmarried partner",
        14: "foster child",
        15: "other non-relative",
        16: "institutionalized group quarters population",
        17: "non-institutionalized group quarters population",
    },
)

acs_work_hours = ColumnToText(
    "WKHP",
    short_description="usual number of hours worked per week",
    missing_value_fill="N/A (less than 16 years old, or did not work during the past 12 months)",
    value_map=lambda x: f"{int(x)} hours",
)

acs_sex = ColumnToText(
    "SEX",
    short_description="sex",
    value_map={
        1: "Male",
        2: "Female",
    },
)

acs_race = ColumnToText(
    "RAC1P",
    short_description="race",
    value_map={
        1: "White",
        2: "Black or African American",
        3: "American Indian",
        4: "Alaska Native",
        5: (
            "American Indian and Alaska Native tribes specified, or American "
            "Indian or Alaska Native, not specified and no other races"),
        6: "Asian",
        7: "Native Hawaiian and Other Pacific Islander",
        # 8: "Some other race alone",
        8: "Some other race alone (non-White)",
        9: "Two or more races",
    },
)

acs_income = ColumnToText(
    "PINCP",
    short_description="yearly income",
    missing_value_fill="N/A (less than 15 years old)",
    value_map=lambda x: f"${int(x):,}",
)

acs_income_binary_qa = MultipleChoiceQA(
    column=get_thresholded_column_name("PINCP", 50_000),
    text="What is this person's estimated yearly income?",
    choices=[
        Choice("Below $50,000", 0),
        Choice("Above $50,000", 1),
    ],
)

acs_income_numeric_qa = DirectNumericQA(
    column=get_thresholded_column_name("PINCP", 50_000),
    text=(
        "What is the probability that this person's estimated yearly income is "
        "above $50,000 ?"
    ),
    answer_probability=True,
    num_forward_passes=2,
)

acs_income = ColumnToText(
    name=get_thresholded_column_name("PINCP", 50_000),
    short_description="yearly income",
    missing_value_fill="N/A (less than 15 years old)",
    question=acs_income_binary_qa,
)

acs_income_brackets = ColumnToText(
    "PINCP_brackets",
    short_description="yearly income",
    missing_value_fill="N/A (less than 15 years old)",
    question=MultipleChoiceQA(
        column="PINCP_brackets",
        text="What is this person's estimated yearly income?",
        choices=[
            Choice("Less than $25,000", data_value="(0.0, 25000.0]", numeric_value=12_500),
            Choice("Between $25,000 and $50,000", data_value="(25000.0, 50000.0]", numeric_value=37_500),
            Choice("Between $50,000 and $100,000", data_value="(50000.0, 100000.0]", numeric_value=75_000),
            Choice("Above $100,000", data_value="(100000.0, inf]", numeric_value=150_000),
        ],
    ),
)

acs_pubcov = ColumnToText(
    "PUBCOV",
    short_description="public health insurance coverage",
    question=MultipleChoiceQA(
        column="PUBCOV",
        text="Does this person have public health insurance coverage?",
        choices=[
            Choice("No, individual is not covered by public health insurance", 0),
            Choice("Yes, individual is covered by public health insurance", 1),
        ],
    ),
)
