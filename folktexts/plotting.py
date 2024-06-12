"""Module to plot evaluation results.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import RocCurveDisplay

from ._utils import safe_division

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING_DEPS = True
    sns.set_style("whitegrid", rc={"grid.linestyle": "--"})
except ImportError:
    HAS_PLOTTING_DEPS = False

_error_msg = (
    "Plotting functions will not work as optional plotting dependencies are not installed. "
    "Please install `matplotlib` and `seaborn` to enable plotting."
)


# Minimum fraction of the dataset size to consider a group for plotting
GROUP_SIZE_THRESHOLD = 0.03


def _check_plotting_deps() -> bool:
    if not HAS_PLOTTING_DEPS:
        logging.error(_error_msg)
    return HAS_PLOTTING_DEPS


def save_fig(fig, fig_name: str, imgs_dir: str | Path, format: str = "pdf") -> str:
    """Helper to save a matplotlib figure to disk."""
    path = Path(imgs_dir) / f"{fig_name}"
    path = path.with_suffix(f".{format}").resolve()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    return path.as_posix()


def render_evaluation_plots(
    y_true: np.ndarray,
    y_pred_scores: np.ndarray,
    *,
    eval_results: dict = {},
    model_name: str = None,
    imgs_dir: str | Path = None,
    show_plots: bool = False,
) -> dict:
    """Renders evaluation plots for the given predictions."""
    # Check if plotting dependencies are available
    if _check_plotting_deps() is False:
        return {}

    # Initialize vars
    results = {}
    model_str = f" - {model_name}" if model_name else ""

    # Helper function to show or save plot
    def show_or_save(fig, fig_name: str):
        if show_plots:
            plt.show()
        if imgs_dir:
            results[f"{fig_name}_path"] = save_fig(fig, fig_name, imgs_dir)

    # ### ### ### ###
    # Plot ROC curve
    # ### ### ### ###
    disp = RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_pred_scores, plot_chance_level=True)
    disp.figure_.suptitle("ROC Curve" + model_str)

    # Plot ROC point
    if "fpr" in eval_results and "tpr" in eval_results and "threshold" in eval_results:
        fpr, tpr = eval_results["fpr"], eval_results["tpr"]
        plt.plot(
            fpr, tpr, "ro", markersize=5, lw=0,
            label=f"threshold={eval_results['threshold']:.2f}")
        plt.legend()

    show_or_save(disp.figure_, "roc_curve")

    # ### ### ### ### ### ###
    # Plot calibration curve
    # ### ### ### ### ### ###
    disp = CalibrationDisplay.from_predictions(y_true=y_true, y_prob=y_pred_scores, n_bins=5, strategy="quantile")
    disp.figure_.suptitle("Calibration Curve" + model_str)
    show_or_save(disp.figure_, "calibration_curve")

    # ### ### ### ### ### ###
    # Plot score distribution
    # ### ### ### ### ### ###
    sns.histplot(y_pred_scores, bins=10, stat="frequency", kde=False)
    plt.xlabel("Predicted Risk Score")
    plt.ylabel("Frequency")
    plt.gcf().suptitle("Score Distribution" + model_str)
    show_or_save(plt.gcf(), "score_distribution")

    # ### ### ### ### ### ### ### ### ### ###
    # Plot distribution of scores per label #
    # ### ### ### ### ### ### ### ### ### ###
    sns.kdeplot(
        data=pd.DataFrame(
            {"score": y_pred_scores, "label": y_true}
        ).reset_index(drop=True),
        x="score",
        hue="label",
        multiple="fill",
    )
    plt.xlim(y_pred_scores.min(), y_pred_scores.max())
    plt.xlabel("Predicted Risk Score")
    plt.gcf().suptitle("Score Distribution per Label" + (f" - {model_name}" if model_name else ""))
    show_or_save(plt.gcf(), "score_distribution_per_label")

    return results


def render_fairness_plots(  # noqa: C901
    y_true: np.ndarray,
    y_pred_scores: np.ndarray,
    *,
    sensitive_attribute: np.ndarray,
    eval_results: dict = {},
    model_name: str = None,
    group_value_map: Callable[[int], str],
    group_size_threshold: float = GROUP_SIZE_THRESHOLD,
    imgs_dir: str | Path = None,
    show_plots: bool = False,
) -> dict:
    """Renders fairness plots for the given predictions."""
    # Check if plotting dependencies are available
    if _check_plotting_deps() is False:
        return {}

    # Plot fairness plots if sensitive attribute is provided
    assert len(sensitive_attribute) == len(y_true) == len(y_pred_scores), \
        "All arrays should have the same length."

    # Initialize vars
    results = {}
    model_str = f" - {model_name}" if model_name else ""
    n_groups = len(np.unique(sensitive_attribute))
    assert n_groups > 1, "At least 2 groups are required for fairness plots."

    # Helper function to show or save plot
    def show_or_save(fig, fig_name: str):
        if show_plots:
            plt.show()
        if imgs_dir:
            results[f"{fig_name}_path"] = save_fig(fig, fig_name, imgs_dir)

    # Set group-wise colors and global color
    palette = sns.color_palette(n_colors=n_groups + 1)
    global_color = palette[0]
    group_colors = palette[1:]
    group_line_styles = ["--", ":", "-."] * (1 + n_groups // 3)

    # ###
    # Plot group-specific ROC curves
    # ###
    for idx, s_value in enumerate(np.unique(sensitive_attribute)):
        is_first_group = (idx == 0)
        is_last_group = (idx == n_groups - 1)

        # If it's the first group
        if is_first_group:
            fig, ax = plt.subplots(figsize=(5, 4))

            # Plot global ROC point
            if "fpr" in eval_results and "tpr" in eval_results:
                ax.plot(
                    eval_results["fpr"], eval_results["tpr"],
                    marker="P", markersize=5, lw=0,
                    color=global_color, label="Global")

        # Get group-specific data
        group_indices = np.argwhere(sensitive_attribute == s_value).flatten()
        group_y_true = y_true[group_indices]
        group_y_pred_scores = y_pred_scores[group_indices]

        # If the group is too small of a fraction, skip (curve will be too erratic)
        if len(group_indices) / len(sensitive_attribute) < group_size_threshold:
            logging.info(f"Skipping group '{group_value_map(s_value)}' as it's too small.")
            continue

        # Plot group-specific ROC curve
        RocCurveDisplay.from_predictions(
            y_true=group_y_true, y_pred=group_y_pred_scores,
            plot_chance_level=is_last_group,
            ax=ax,

            # Group-specific visuals
            linestyle=group_line_styles[idx],
            color=group_colors[idx],
            name=group_value_map(s_value),
        )

        # Plot group-specific ROC point
        if "threshold" in eval_results:
            threshold = eval_results["threshold"]
            tn, fp, fn, tp = metrics.confusion_matrix(
                group_y_true, (group_y_pred_scores >= threshold).astype(int),
                labels=(0, 1),
            ).ravel()
            group_fpr = safe_division(fp, fp + tn, worst_result=1)
            group_tpr = safe_division(tp, tp + fn, worst_result=0)

            ax.plot(
                group_fpr, group_tpr,
                marker="X", markersize=5, lw=0,
                color=group_colors[idx],
                # label=group_value_map(s_value),
            )

    plt.legend()
    plt.title("ROC curve per sub-group" + model_str)
    show_or_save(fig, "roc_curve_per_subgroup")

    # ###
    # Plot group-specific calibration curves
    # ###
    for idx, s_value in enumerate(np.unique(sensitive_attribute)):
        group_indices = np.argwhere(sensitive_attribute == s_value).flatten()
        group_y_true = y_true[group_indices]
        group_y_pred_scores = y_pred_scores[group_indices]
        is_first_group = (idx == 0)

        if is_first_group:
            fig, ax = plt.subplots(figsize=(5, 4))

        # If the group is too small of a fraction, skip (curve will be too erratic)
        if len(group_indices) / len(sensitive_attribute) < group_size_threshold:
            logging.warning(f"Skipping group {group_value_map(s_value)} plot as it's too small.")
            continue

        # Plot global calibration curve
        CalibrationDisplay.from_predictions(
            y_true=group_y_true, y_prob=group_y_pred_scores,
            n_bins=5, strategy="quantile",
            ax=ax,

            # Group-specific visuals
            linestyle=group_line_styles[idx],
            color=group_colors[idx],
            name=group_value_map(s_value),
        )

    plt.legend()
    plt.title("Calibration curve per sub-group" + model_str)
    show_or_save(fig, "calibration_curve_per_subgroup")

    # ###
    # Plot scores distribution per group
    # ###
    # TODO: make a decent score-distribution plot... # TODO: try score CDFs!
    # hist_bin_edges = np.histogram_bin_edges(y_pred_scores, bins=10)
    # for idx, s_value in enumerate(np.unique(sensitive_attribute)):
    #     group_indices = np.argwhere(sensitive_attribute == s_value).flatten()
    #     group_y_pred_scores = y_pred_scores[group_indices]
    #     is_first_group = (idx == 0)
    #     if is_first_group:
    #         fig, ax = plt.subplots()
    #     sns.histplot(
    #         group_y_pred_scores,
    #         bins=hist_bin_edges,
    #         stat="density",
    #         kde=False,
    #         color=group_colors[idx],
    #         label=group_value_map(s_value),
    #         ax=ax,
    #     )

    # plt.legend()
    # plt.title("Score distribution per sub-group" + model_str)
    # results["score_distribution_per_subgroup_path"] = save_fig(fig, "score_distribution_per_subgroup", imgs_dir)

    return results
