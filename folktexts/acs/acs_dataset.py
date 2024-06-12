"""Module to access ACS data using the folktables package.
"""
from __future__ import annotations

import logging
from pathlib import Path

from folktables import ACSDataSource
from folktables.load_acs import state_list

from ..dataset import Dataset
from .._utils import get_thresholded_column_name
from .acs_tasks import ACSTaskMetadata  # noqa # load ACS tasks

DEFAULT_ACS_DATA_DIR = Path("~/data").expanduser().resolve()
DEFAULT_SEED = 42

DEFAULT_SURVEY_YEAR = "2018"
DEFAULT_SURVEY_HORIZON = "1-Year"
DEFAULT_SURVEY_UNIT = "person"


class ACSDataset(Dataset):
    """Wrapper for ACS folktables datasets."""

    def __init__(
        self,
        task: str | ACSTaskMetadata,
        cache_dir: str | Path = None,
        survey_year: str = DEFAULT_SURVEY_YEAR,
        horizon: str = DEFAULT_SURVEY_HORIZON,
        survey: str = DEFAULT_SURVEY_UNIT,
        seed: int = DEFAULT_SEED,
        **kwargs,
    ):
        # Create "folktables" sub-folder under the given cache dir
        cache_dir = Path(cache_dir or DEFAULT_ACS_DATA_DIR).expanduser().resolve() / "folktables"
        if not cache_dir.exists():
            logging.warning(f"Creating cache directory '{cache_dir}' for ACS data.")
            cache_dir.mkdir(exist_ok=True, parents=False)

        # Load ACS data source
        print("Loading ACS data...")
        data_source = ACSDataSource(
            survey_year=survey_year, horizon=horizon, survey=survey,
            root_dir=cache_dir.as_posix(),
        )

        # Get ACS data in a pandas DF
        data = data_source.get_data(
            states=state_list, download=True, random_seed=seed,
        )

        # Get information on this ACS/folktables task
        task = ACSTaskMetadata.get_task(task) if isinstance(task, str) else task

        # Keep only rows used in this task
        data = task.folktables_obj._preprocess(data)

        # Threshold the target column if necessary
        # > use standardized ACS naming convention
        if task.target_threshold is not None:
            # TODO: the target should be thresholded in the task definition not here!
            thresholded_target = task.get_target()
            if thresholded_target not in data.columns:
                data[thresholded_target] = (data[task.target] >= task.target_threshold).astype(int)

        super().__init__(
            data=data,
            task=task,
            seed=seed,
            **kwargs,
        )
