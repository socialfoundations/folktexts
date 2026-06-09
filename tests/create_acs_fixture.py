"""Script to generate a tiny ACS fixture used by the test suite.

Run once (requires network access to download ACS data):

    python tests/create_acs_fixture.py
"""

from pathlib import Path

TASK_NAME = "ACSIncome"
DATA_DIR = "data"
FIXTURE_PATH = Path(__file__).parent / "acs_income_10rows.csv"


def main():
    from folktexts.acs import ACSDataset, ACSTaskMetadata

    print(f"Loading task '{TASK_NAME}' and dataset from cache dir '{DATA_DIR}'...")

    task = ACSTaskMetadata.get_task(TASK_NAME, use_numeric_qa=False)
    dataset = ACSDataset.make_from_task(task=task, cache_dir=DATA_DIR)

    cols = task.features + [task.get_target()]
    df = dataset.data[cols]
    # Take 5 rows per label to ensure both classes are represented
    sample = (
        df.groupby(task.get_target(), group_keys=False)
        .apply(lambda g: g.head(5))
    )

    FIXTURE_PATH.parent.mkdir(exist_ok=True)
    sample.to_csv(FIXTURE_PATH)
    print(
        f"Saved {len(sample)} rows with columns {list(sample.columns)} to {FIXTURE_PATH}"
    )


if __name__ == "__main__":
    main()
