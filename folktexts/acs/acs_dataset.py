"""Module for ACS datasets and tasks."""

from pathlib import Path

from folktables import ACSDataSource
from folktables.load_acs import state_list

from ..dataset import Dataset
from ..task import TaskMetadata

from ._utils import get_thresholded_column_name


DEFAULT_ACS_DATA_DIR = Path("~/data").expanduser().resolve()


class ACSDataset(Dataset):
    """Wrapper for ACS folktables datasets."""

    def __init__(
            self,
            task_name: str,
            cache_dir: str | Path = DEFAULT_ACS_DATA_DIR,
            **kwargs,
        ):

        # Create "folktables" sub-folder under the given cache dir
        cache_dir = Path(cache_dir).expanduser().resolve() / "folktables"
        cache_dir.mkdir(exist_ok=True, parents=False)

        # Load ACS data source
        data_source = ACSDataSource(
            survey_year='2018', horizon='1-Year', survey='person',
            root_dir=cache_dir.as_posix(),
        )

        # Get ACS data in a pandas DF
        data = data_source.get_data(states=state_list, download=True)

        # Get information on this ACS/folktables task
        task = TaskMetadata.get_task(task_name)

        # Keep only rows used in this task
        data = task.folktables_obj._preprocess(data)

        # Threshold the target column if necessary
        # > use standardized ACS naming convention
        if task.target_threshold is not None:
            thresholded_target = get_thresholded_column_name(task.target, task.target_threshold)
            data[thresholded_target] = (data[task.target] >= task.target_threshold).astype(int)
            task.target = thresholded_target

        super().__init__(
            data=data,
            task=task,
            **kwargs,
        )
