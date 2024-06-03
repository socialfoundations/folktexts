"""A benchmark class for measuring and evaluating LLM calibration.
"""

from .classifier import LLMClassifier
from .dataset import Dataset


class CalibrationBenchmark:
    """A benchmark class for measuring and evaluating LLM calibration."""

    def __init__(self, llm_clf: LLMClassifier, dataset: Dataset | str):
        pass

    def visualize(self):
        pass
