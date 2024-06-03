import json
import hashlib
from pathlib import Path
from datetime import datetime


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


def standardize_path(path: str | Path) -> str:
    """Represents a posix path as a standardized string."""
    return Path(path).expanduser().resolve().as_posix()


def cmd_line_args_to_kwargs(cmdline_args: list) -> dict:
    """Converts a list of command-line arguments to a dictionary of keyword arguments."""
    def _handle_str_value(val: str) -> str | bool:
        # Try bool
        if val.lower() in ("true", "false"):
            return val.lower() == "true"

        # Try int
        try:
            return int(val)
        except ValueError:
            pass

        # Try float
        try:
            return float(val)
        except ValueError:
            pass

        # Otherwise, assume it's a string
        return val

    kwargs_dict = {}
    for arg in cmdline_args:
        parsed_arg = arg.lstrip("-").replace("-", "_")
        if "=" in parsed_arg:
            key, val = parsed_arg.split("=")
            kwargs_dict[key] = _handle_str_value(val)
        else:
            kwargs_dict[parsed_arg] = True

    return kwargs_dict
