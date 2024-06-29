"""Pytest fixtures.
"""
from __future__ import annotations

import numpy as np
import pytest

TEST_CAUSAL_LMS = [
    "hf-internal-testing/tiny-random-gpt2",
    # "hf-internal-testing/tiny-random-MistralForCausalLM",
]


@pytest.fixture(params=[42])
def random_seed(request) -> int:
    return request.param


@pytest.fixture
def rng(random_seed: int) -> np.random.Generator:
    return np.random.default_rng(random_seed)


@pytest.fixture(params=TEST_CAUSAL_LMS)
def causal_lm_name_or_path(request) -> str:
    """Name or path of the CausalLM used for testing."""
    return request.param
