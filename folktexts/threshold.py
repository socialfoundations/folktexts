"""Helper function for defining binarization thresholds.
"""
from __future__ import annotations

import dataclasses
import operator
from typing import ClassVar

import pandas as pd


@dataclasses.dataclass(frozen=True, eq=True)
class Threshold:
    """A class to represent a threshold value and its comparison operator.

    Attributes
    ----------
    value : float | int
        The threshold value to compare against.
    op : str
        The comparison operator to use. One of '>', '<', '>=', '<=', '=='.
    """
    value: float | int
    op: str

    valid_ops: ClassVar[dict] = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
    }

    def __post_init__(self):
        if self.op not in self.valid_ops.keys():
            raise ValueError(f"Invalid comparison operator '{self.op}'.")

    def __str__(self):
        return f"{self.op}{self.value}"

    def apply_to_column_data(self, data: float | int | pd.Series) -> int | pd.Series:
        """Applies the threshold operation to a pandas Series or scalar value."""
        if isinstance(data, pd.Series):
            return self.valid_ops[self.op](data, self.value).astype(int)
        elif isinstance(data, (float, int)):
            return int(self.valid_ops[self.op](data, self.value))
        else:
            raise TypeError(f"Invalid data type '{type(data)}'.")

    def apply_to_column_name(self, column_name: str) -> str:
        """Standardizes naming of thresholded columns."""
        return column_name + str(self)
