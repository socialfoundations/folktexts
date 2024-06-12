"""Helper functions for ACS data processing."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable


def parse_pums_code(
    value: int,
    file: str | Path,
    postprocess: Callable[[str], str] | None = None,
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
