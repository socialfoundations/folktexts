from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Collection
from pathlib import Path

import cloudpickle


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
