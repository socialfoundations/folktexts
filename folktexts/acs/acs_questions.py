"""A collection of instantiated ACS column objects and ACS tasks."""
from __future__ import annotations

from folktexts.qa_interface import DirectNumericQA as _DirectNumericQA
from folktexts.qa_interface import MultipleChoiceQA as _MultipleChoiceQA
from folktexts.col_to_text import ColumnToText

from . import acs_columns
from .acs_tasks import _acs_columns_map

# Map of numeric ACS questions
acs_numeric_qa_map: dict[str, object] = {
    question.column: question
    for question in acs_columns.__dict__.values()
    if isinstance(question, _DirectNumericQA)
}

# Map of multiple-choice ACS questions
acs_multiple_choice_qa_map: dict[str, object] = {
    question.column: question
    for question in acs_columns.__dict__.values()
    if isinstance(question, _MultipleChoiceQA)
}

# ... include all multiple-choice questions defined in the column descriptions
acs_multiple_choice_qa_map.update({
    col_to_text.name: col_to_text.question
    for col_to_text in _acs_columns_map.values()
    if (
        isinstance(col_to_text, ColumnToText)
        and col_to_text._question is not None
    )
})
