"""Helper functions for ACS data processing."""
import re
import logging
from pathlib import Path
from typing import Callable


def get_thresholded_column_name(column_name: str, threshold: float | int) -> str:
    """Standardizes naming of thresholded columns."""
    threshold_str = f"{threshold:.2f}".replace(".", "_") if isinstance(threshold, float) else str(threshold)
    return f"{column_name}_binary_{threshold_str}"


def parse_pums_code(
    value: int,
    file: str | Path,
    postprocess: Callable[[str], str] = None,
    cache={},
) -> str:
    # Check if file already loaded into cache
    if file not in cache:
        line_re = re.compile(r"(?P<code>\d+)\s+[.](?P<description>.+)$")

        file_cache = {}
        with open(file) as f:
            for line in f:
                m = line_re.match(line)
                if m is None:
                    logging.error(f"Could not parse line: {line}")
                    continue

                code, description = m.group("code"), m.group("description")
                file_cache[int(code)] = postprocess(description) if postprocess else description

        cache[file] = file_cache

    # Get file cache
    file_cache = cache[file]

    # Return the value from cache, or "N/A" if not found
    if value not in file_cache:
        logging.warning(f"Could not find code '{value}' in file '{file}'")
        return "N/A"

    return file_cache[value]
