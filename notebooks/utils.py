"""A collection of utils to accompany folktexts notebooks."""
import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime


def get_current_timestamp() -> str:
    """Return a timestamp representing the current time up to the second."""
    return datetime.now().strftime("%Y.%m.%d-%H.%M.%S")


def load_json(path: str | Path) -> object:
    """Loads a JSON file from disk and returns the deserialized object."""
    with open(path, "r") as f_in:
        return json.load(f_in)


def find_files(root_folder, pattern):
    """Iteratively yield file paths that match the given pattern."""
    # Compile the regular expression pattern
    regex = re.compile(pattern)

    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if regex.match(filename):
                # If the filename matches the pattern, add it to the list
                yield os.path.join(dirpath, filename)


def num_features_helper(feature_subset_val, max_features_return=-1):
    """Returns how many features the given subset corresponds to."""
    if isinstance(feature_subset_val, str):
        assert feature_subset_val.lower() == "full"
        return max_features_return    # by deafult return -1 for all features
    try:
        assert isinstance(feature_subset_val, (list, tuple, set))
        return len(feature_subset_val)
    except Exception:
        return max_features_return


def parse_model_name(name: str) -> str:
    name = name[name.find("--")+2:]
    return name


def get_non_instruction_tuned_name(name):
    """Returns the name of the equivalent base model."""
    name = name.replace("-Instruct", "")    # Llamma / Mistral
    name = name.replace("-Chat", "")        # Yi
    name = name.replace("-it", "")          # Gemma
    name = name.replace("-1.1", "")         # Gemma version
    name = name.replace("-v0.2", "-v0.1")   # Mistral version
    return name


def prettify_model_name(name: str) -> str:
    """Get prettified version of the given model name."""
    dct = {
        "Meta-Llama-3-70B": "Llama 3 70B",
        "Meta-Llama-3-70B-Instruct": "Llama 3 70B (it)",
        "Meta-Llama-3-8B": "Llama 3 8B",
        "Meta-Llama-3-8B-Instruct": "Llama 3 8B (it)",
        "Mistral-7B-Instruct-v0.2": "Mistral 7B (it)",
        "Mistral-7B-v0.1": "Mistral 7B",
        "Mixtral-8x22B-Instruct-v0.1": "Mixtral 8x22B (it)",
        "Mixtral-8x22B-v0.1": "Mixtral 8x22B",
        "Mixtral-8x7B-Instruct-v0.1": "Mixtral 8x7B (it)",
        "Mixtral-8x7B-v0.1": "Mixtral 8x7B",
        "Yi-34B": "Yi 34B",
        "Yi-34B-Chat": "Yi 34B (it)",
        "gemma-1.1-2b-it": "Gemma 2B (it)",
        "gemma-1.1-7b-it": "Gemma 7B (it)",
        "gemma-2b": "Gemma 2B",
        "gemma-7b": "Gemma 7B",
        "gemma-2-9b": "Gemma 2 9B",
        "gemma-2-9b-it": "Gemma 2 9B (it)",
        "gemma-2-27b": "Gemma 2 27B",
        "gemma-2-27b-it": "Gemma 2 27B (it)",
        "openai/gpt-4o-mini": "GPT 4o mini (it)",
        "penai/gpt-4o-mini": "GPT 4o mini (it)",
        "openai/gpt-4o": "GPT 4o (it)",
        "penai/gpt-4o": "GPT 4o (it)",
    }

    if name in dct:
        return dct[name]
    else:
        logging.error(f"Couldn't find prettified name for {name}.")
        return name