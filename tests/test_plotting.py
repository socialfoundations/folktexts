"""Tests for folktexts/plotting.py.

Degenerate score distributions (e.g. every row at the 0.5 regex-fallback score,
or a maximally-confident API model) make scipy's gaussian_kde covariance
singular; `render_evaluation_plots` must skip the per-label KDE sub-plot
instead of raising.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("seaborn")

from folktexts.plotting import render_evaluation_plots, render_fairness_plots  # noqa: E402

NON_KDE_PLOTS = {"roc_curve_path", "calibration_curve_path", "score_distribution_path"}


@pytest.fixture
def y_true() -> np.ndarray:
    return np.array([0] * 60 + [1] * 40)


def test_render_evaluation_plots_constant_scores(y_true, tmp_path):
    """All-identical scores must not crash; the 3 non-KDE plots are still saved."""
    y_scores = np.full(len(y_true), 0.5)

    results = render_evaluation_plots(y_true, y_scores, imgs_dir=tmp_path)

    assert NON_KDE_PLOTS.issubset(results.keys())


def test_render_evaluation_plots_spread_scores(y_true, tmp_path):
    """Well-spread scores render all 4 plots, including the per-label KDE."""
    rng = np.random.default_rng(42)
    y_scores = np.clip(0.3 * y_true + rng.uniform(0, 0.7, len(y_true)), 0, 1)

    results = render_evaluation_plots(y_true, y_scores, imgs_dir=tmp_path)

    assert (NON_KDE_PLOTS | {"score_distribution_per_label_path"}).issubset(results.keys())


def test_render_fairness_plots(y_true, tmp_path):
    """Group-wise ROC/calibration curves render across sklearn versions."""
    rng = np.random.default_rng(7)
    y_scores = np.clip(0.3 * y_true + rng.uniform(0, 0.7, len(y_true)), 0, 1)
    sensitive_attribute = np.tile([1, 2], len(y_true) // 2)

    results = render_fairness_plots(
        y_true,
        y_scores,
        sensitive_attribute=sensitive_attribute,
        group_value_map=str,
        imgs_dir=tmp_path,
    )

    assert {"roc_curve_per_subgroup_path", "calibration_curve_per_subgroup_path"}.issubset(
        results.keys()
    )
