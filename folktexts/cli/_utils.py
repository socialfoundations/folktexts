import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Collection

import pickle
import cloudpickle


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


def save_json(obj: Collection, path: str | Path, overwrite: bool = True):
    """Saves a JSON-serializable object to disk."""
    logging.info(f"Saving JSON file to '{str(path)}'")
    with open(path, "w" if overwrite else "x") as f_out:
        json.dump(obj, f_out, indent=4, sort_keys=True)


def load_json(path: str | Path) -> object:
    """Loads a JSON file from disk and returns the deserialized object."""
    with open(path, "r") as f_in:
        return json.load(f_in)


def save_pickle(obj: object, path: str | Path, overwrite: bool = True) -> bool:
    """Saves the given object as a pickle with the given file path.

    Parameters
    ----------
    obj : object
        The object to pickle
    path : str or Path
        The file path to save the pickle with.

    Returns
    -------
    success : bool
        Whether pickling was successful.
    """
    logging.info(f"Saving pickle file to '{str(path)}'")
    try:
        with open(path, "wb" if overwrite else "xb") as f_out:
            cloudpickle.dump(obj, f_out)
            return True

    except Exception as e:
        logging.error(f"Pickling failed with exception '{e}'")
        return False


def load_pickle(path: str | Path) -> object:
    with open(path, "rb") as f_in:
        return pickle.load(f_in)


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
