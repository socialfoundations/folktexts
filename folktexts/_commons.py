"""Common set of utility functions and constants used across the project."""
import logging
import operator
from functools import reduce

import numpy as np


def is_valid_number(num) -> bool:
    """Check if the given number is a valid numerical value and not NaN."""
    return isinstance(num, (float, int, np.number)) and not np.isnan(num)


def safe_division(a: float, b: float, *, worst_result: float):
    """Try to divide the given arguments and return `worst_result` if unsuccessful."""
    if b == 0 or not is_valid_number(a) or not is_valid_number(b):
        logging.warning(f"Error in the following division: {a} / {b}")
        return worst_result
    else:
        return a / b


def join_dictionaries(*dicts) -> dict:
    """Joins a sequence of dictionaries into a single dictionary."""
    return reduce(operator.or_, dicts)
