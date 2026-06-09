"""Common set of utility functions and constants used across the project."""

from __future__ import annotations

import hashlib
import json
import logging
import operator
from argparse import Action
from contextlib import contextmanager
from datetime import datetime
from functools import partial, reduce
from pathlib import Path
from typing import Any

import numpy as np


def is_valid_number(num) -> bool:
    """Check if the given number is a valid numerical value and not NaN."""
    return isinstance(num, (float, int, np.number)) and not np.isnan(num)


def safe_division(a: float, b: float, *, worst_result: float):
    """Try to divide the given arguments and return `worst_result` if unsuccessful."""
    if b == 0 or not is_valid_number(a) or not is_valid_number(b):
        logging.debug(
            f"Using `worst_result={worst_result}` in place of the following "
            f"division: {a} / {b}"
        )
        return worst_result
    else:
        return a / b


def join_dictionaries(*dicts) -> dict:
    """Joins a sequence of dictionaries into a single dictionary."""
    return reduce(operator.or_, dicts)


def get_current_timestamp() -> str:
    """Return a timestamp representing the current time up to the second."""
    return datetime.now().strftime("%Y.%m.%d-%H.%M.%S")


def get_current_date() -> str:
    """Return a timestamp representing the current time up to the second."""
    return datetime.now().strftime("%Y-%m-%d")


def hash_dict(d: dict, length: int = 8) -> str:
    """Hashes a dictionary using SHAKE-256 and returns the hexdigest.

    Parameters
    ----------
    d : dict
        The dictionary to hash.
    length : int, optional
        The length of the hexdigest in number of text characters, by default 8.

    Returns
    -------
    hexdigest : str
        A string representing the hash in hexadecimal format.
    """
    d_enc = json.dumps(d, sort_keys=True).encode()
    return hashlib.shake_256(d_enc).hexdigest(length // 2)


def hash_function(func, length: int = 8) -> str:
    """Hashes a function using SHAKE-256 and returns the hexdigest.

    Parameters
    ----------
    func : Callable
        The function to hash. Can be a partial function.
    length : int, optional
        The length of the hexdigest in number of text characters, by default 8.

    Returns
    -------
    hexdigest : str
        A string representing the hash in hexadecimal format.
    """
    if not callable(func):
        raise ValueError("The input must be a callable function.")
    elif isinstance(func, partial):
        return hash_function(func.func)

    func_str = func.__code__.co_code
    return hashlib.shake_256(func_str).hexdigest(length // 2)


def standardize_path(path: str | Path) -> str:
    """Represents a posix path as a standardized string."""
    return Path(path).expanduser().resolve().as_posix()


@contextmanager
def suppress_logging(new_level):
    """Suppresses all logs of a given level within a context block."""
    logger = logging.getLogger()
    previous_level = logger.level
    logger.setLevel(new_level)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


def is_float(element: Any) -> bool:
    # If you expect None to be passed:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def is_int(element: Any) -> bool:
    # If you expect None to be passed:
    if element is None:
        return False
    try:
        int(element)
        return True
    except ValueError:
        return False


def is_str(element: Any) -> bool:
    return isinstance(element, str)


def is_bool(element: Any) -> bool:
    return isinstance(element, str) and element.strip().lower() in ("true", "false")


def _parse_cli_scalar(value: str):
    """Coerce a CLI string to int/float/bool when unambiguous, else keep it a str."""
    if is_int(value):
        return int(value)
    if is_float(value):
        return float(value)
    if value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return value


class ParseDict(Action):
    """argparse Action that parses 'key1=val1;key2=val2' strings into a dict."""

    def __call__(self, parser, namespace, values, option_string=None):
        # nargs="*" passes a list; accept a bare string defensively.
        if isinstance(values, str):
            values = [values]
        logging.debug(f"ParseDict received values: {values}")
        # Accept both '--variation a=b;c=d' and '--variation a=b c=d'; no args -> {}.
        pairs = [pair for value in (values or []) for pair in value.split(";") if pair]
        result: dict = {}
        for pair in pairs:
            key, sep, value = pair.partition("=")
            assert sep and value != "", (
                f"Could not parse '{pair}'; expected 'key=value'."
            )
            result[key] = _parse_cli_scalar(value)
        setattr(namespace, self.dest, result)
