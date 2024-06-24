"""Module to access ACS data using the folktables package.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from folktables import ACSDataSource
from folktables.load_acs import state_list

from ..dataset import Dataset
from .acs_tasks import ACSTaskMetadata

DEFAULT_DATA_DIR = Path("~/data").expanduser().resolve()
DEFAULT_TEST_SIZE = 0.1
DEFAULT_VAL_SIZE = None
DEFAULT_SEED = 42

DEFAULT_SURVEY_YEAR = "2018"
DEFAULT_SURVEY_HORIZON = "1-Year"
DEFAULT_SURVEY_UNIT = "person"


class ACSDataset(Dataset):
    """Wrapper for ACS folktables datasets."""

    def __init__(
        self,
        data: pd.DataFrame,
        full_acs_data: pd.DataFrame,
        task: ACSTaskMetadata,
        test_size: float = DEFAULT_TEST_SIZE,
        val_size: float = DEFAULT_VAL_SIZE,
        subsampling: float = None,
        seed: int = 42,
    ):
        self._full_acs_data = full_acs_data
        super().__init__(
            data=data,
            task=task,
            test_size=test_size,
            val_size=val_size,
            subsampling=subsampling,
            seed=seed,
        )

    @classmethod
    def make_from_task(
        cls,
        task: str | ACSTaskMetadata,
        cache_dir: str | Path = None,
        survey_year: str = DEFAULT_SURVEY_YEAR,
        horizon: str = DEFAULT_SURVEY_HORIZON,
        survey: str = DEFAULT_SURVEY_UNIT,
        seed: int = DEFAULT_SEED,
        **kwargs,
    ):
        """Construct an ACSDataset object using ACS survey parameters.

        Parameters
        ----------
        task : str | ACSTaskMetadata
            The name of the ACS task or the task object itself.
        cache_dir : str | Path, optional
            The directory where ACS data is (or will be) saved to, by default
            uses DEFAULT_DATA_DIR.
        survey_year : str, optional
            The year from which to load survey data, by default DEFAULT_SURVEY_YEAR.
        horizon : str, optional
            The time horizon of survey data to load, by default DEFAULT_SURVEY_HORIZON.
        survey : str, optional
            The name of the survey unit to load, by default DEFAULT_SURVEY_UNIT.
        seed : int, optional
            The random seed, by default DEFAULT_SEED.
        **kwargs
            Extra key-word arguments to be passed to the Dataset constructor.
        """
        # Create "folktables" sub-folder under the given cache dir
        cache_dir = Path(cache_dir or DEFAULT_DATA_DIR).expanduser().resolve() / "folktables"
        if not cache_dir.exists():
            logging.warning(f"Creating cache directory '{cache_dir}' for ACS data.")
            cache_dir.mkdir(exist_ok=True, parents=False)

        # Parse task if given a string
        task_obj = ACSTaskMetadata.get_task(task) if isinstance(task, str) else task

        # Load ACS data source
        print("Loading ACS data...")
        data_source = ACSDataSource(
            survey_year=survey_year, horizon=horizon, survey=survey,
            root_dir=cache_dir.as_posix(),
        )

        # Get full ACS dataset
        full_acs_data = data_source.get_data(
            states=state_list, download=True, random_seed=seed)

        # Parse data for this task
        parsed_data = cls._parse_task_data(full_acs_data, task_obj)

        return cls(
            data=parsed_data,
            full_acs_data=full_acs_data,
            task=task_obj,
            seed=seed,
            **kwargs,
        )

    @property
    def task(self) -> ACSTaskMetadata:
        return self._task

    @task.setter
    def task(self, new_task: ACSTaskMetadata):
        # Parse data rows for new ACS task
        self._data = self._parse_task_data(self._full_acs_data, new_task)

        # Re-Make train/test/val split
        self._train_indices, self._test_indices, self._val_indices = (
            self._make_train_test_val_split(
                self._data, self.test_size, self.val_size, self._rng)
        )

        # Check if task columns are in the data
        if not all(col in self.data.columns for col in (new_task.features + [new_task.get_target()])):
            raise ValueError(
                f"Task columns not found in dataset: "
                f"features={new_task.features}, target={new_task.get_target()}")

        self._task = new_task

    @classmethod
    def _parse_task_data(cls, full_df: pd.DataFrame, task: ACSTaskMetadata) -> pd.DataFrame:
        """Parse a DataFrame for compatibility with the given task object.

        Parameters
        ----------
        full_df : pd.DataFrame
            Full DataFrame. Some rows and/or columns may be discarded for each
            task.
        task : ACSTaskMetadata
            The task object used to parse the given data.

        Returns
        -------
        parsed_df : pd.DataFrame
            Parsed DataFrame in accordance with the given task.
        """
        if not isinstance(task, ACSTaskMetadata):
            logging.error(f"Expected task of type `ACSTaskMetadata` for {type(task)}")
            return full_df

        # Parse data
        parsed_df = task.folktables_obj._preprocess(full_df)

        # Threshold the target column if necessary
        if task.target_threshold is not None and task.get_target() not in parsed_df.columns:
            parsed_df[task.get_target()] = task.target_threshold.apply_to_column_data(parsed_df[task.target])

        return parsed_df
