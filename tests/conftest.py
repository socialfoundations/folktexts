"""Pytest fixtures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TEST_CAUSAL_LMS = [
    "hf-internal-testing/tiny-random-gpt2",
    # "hf-internal-testing/tiny-random-MistralForCausalLM",
]

ACS_FIXTURE_PATH = Path(__file__).parent / "acs_income_10rows.csv"
ACS_TASK_NAME = "ACSIncome"


@pytest.fixture(params=[42])
def random_seed(request) -> int:
    return request.param


@pytest.fixture
def rng(random_seed: int) -> np.random.Generator:
    return np.random.default_rng(random_seed)


@pytest.fixture(scope="module", params=TEST_CAUSAL_LMS)
def causal_lm_name_or_path(request) -> str:
    """Name or path of the CausalLM used for testing."""
    return request.param


@pytest.fixture(scope="session")
def acs_income_task():
    from folktexts.acs import ACSTaskMetadata
    return ACSTaskMetadata.get_task(ACS_TASK_NAME, use_numeric_qa=False)


@pytest.fixture(scope="session")
def acs_income_dataset(acs_income_task):
    if not ACS_FIXTURE_PATH.exists():
        pytest.skip(
            f"ACS fixture not found at {ACS_FIXTURE_PATH}. "
            "Run `python tests/create_acs_fixture.py` to generate it."
        )
    from folktexts.dataset import Dataset
    df = pd.read_csv(ACS_FIXTURE_PATH, index_col=0)
    # 10 rows: test_size=0.3 → 7 train + 3 test, no val split
    return Dataset(
        data=df, task=acs_income_task, test_size=0.3, val_size=0.0, seed=42
    )


@pytest.fixture(scope="session")
def acs_row(acs_income_dataset):
    X, _ = acs_income_dataset.get_test()
    return X.iloc[0]
