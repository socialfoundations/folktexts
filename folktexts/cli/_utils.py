"""Utils for the folktexts cmd-line interface.
"""
from __future__ import annotations


def cmd_line_args_to_kwargs(cmdline_args: list) -> dict:
    """Converts a list of command-line arguments to a dictionary of keyword arguments."""
    def _handle_str_value(val: str) -> int | float | str | bool:
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
            split_idx = parsed_arg.index("=")
            key = parsed_arg[:split_idx]
            val = parsed_arg[split_idx + 1:]
            kwargs_dict[key] = _handle_str_value(val)
        else:
            kwargs_dict[parsed_arg] = True

    return kwargs_dict
