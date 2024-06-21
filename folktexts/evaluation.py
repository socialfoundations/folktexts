"""Module to map risk-estimates to a variety of evaluation metrics.

Notes
-----
Code based on the `error_parity.evaluation` module,
at: https://github.com/socialfoundations/error-parity/blob/main/error_parity/evaluation.py
"""
from __future__ import annotations

import logging
import statistics
from typing import Optional

import numpy as np
from netcal.metrics import ECE
from sklearn.metrics import brier_score_loss, confusion_matrix, log_loss, roc_auc_score, roc_curve

from ._utils import is_valid_number, join_dictionaries, safe_division


def evaluate_binary_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluates the provided binary predictions on common performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        The true class labels.
    y_pred : np.ndarray
        The binary predictions.

    Returns
    -------
    dict
        A dictionary with key-value pairs of (metric name, metric value).
    """
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=(0, 1)).ravel()

    total = tn + fp + fn + tp
    pred_pos = tp + fp
    pred_neg = tn + fn
    assert pred_pos + pred_neg == total

    label_pos = tp + fn
    label_neg = tn + fp
    assert label_pos + label_neg == total

    results = {}

    # Accuracy
    results["accuracy"] = (tp + tn) / total

    # True Positive Rate (Recall)
    results["tpr"] = safe_division(tp, label_pos, worst_result=0)

    # False Negative Rate (1 - TPR)
    results["fnr"] = safe_division(fn, label_pos, worst_result=1)
    assert results["tpr"] + results["fnr"] == 1

    # False Positive Rate
    results["fpr"] = safe_division(fp, label_neg, worst_result=1)

    # True Negative Rate
    results["tnr"] = safe_division(tn, label_neg, worst_result=0)
    assert results["tnr"] + results["fpr"] == 1

    # Balanced accuracy
    results["balanced_accuracy"] = 0.5 * (results["tpr"] + results["tnr"])

    # Precision
    results["precision"] = safe_division(tp, pred_pos, worst_result=0)

    # Positive Prediction Rate
    results["ppr"] = safe_division(pred_pos, total, worst_result=0)

    return results


def evaluate_binary_predictions_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attribute: np.ndarray,
    return_groupwise_metrics: Optional[bool] = False,
) -> dict:
    """Evaluates fairness of the given predictions.

    Fairness metrics are computed as the ratios between group-wise performance
    metrics.

    Parameters
    ----------
    y_true : np.ndarray
        The true class labels.
    y_pred : np.ndarray
        The discretized predictions.
    sensitive_attribute : np.ndarray
        The sensitive attribute (protected group membership).
    return_groupwise_metrics : Optional[bool], optional
        Whether to return group-wise performance metrics (bool: True) or only
        the ratios between these metrics (bool: False), by default False.

    Returns
    -------
    dict
        A dictionary with key-value pairs of (metric name, metric value).
    """
    # All unique values for the sensitive attribute
    unique_groups = np.unique(sensitive_attribute)

    results = {}
    groupwise_metrics = {}
    unique_metrics = set()

    # Helper to compute key/name of a group-wise metric
    def group_metric_name(metric_name, group_name):
        return f"{metric_name}_group={group_name}"

    if len(unique_groups) <= 1:
        logging.error(f"Found a single unique sensitive attribute: {unique_groups}")
        return {}

    for s_value in unique_groups:
        # Indices of samples that belong to the current group
        group_indices = np.argwhere(sensitive_attribute == s_value).flatten()

        # Filter labels and predictions for samples of the current group
        group_labels = y_true[group_indices]
        group_preds = y_pred[group_indices]

        # Evaluate group-wise performance
        curr_group_metrics = evaluate_binary_predictions(group_labels, group_preds)

        # Add group-wise metrics to the dictionary
        groupwise_metrics.update(
            {
                group_metric_name(metric_name, s_value): metric_value
                for metric_name, metric_value in curr_group_metrics.items()
            }
        )

        unique_metrics = unique_metrics.union(curr_group_metrics.keys())

    # Compute ratios and absolute diffs
    for metric_name in unique_metrics:
        curr_metric_results = [
            groupwise_metrics[group_metric_name(metric_name, group_name)]
            for group_name in unique_groups
        ]

        # Metrics' ratio
        ratio_name = f"{metric_name}_ratio"

        # NOTE: should this ratio be computed w.r.t. global performance?
        # - i.e., min(curr_metric_results) / global_curr_metric_result;
        # - same question for the absolute diff calculations;
        results[ratio_name] = safe_division(
            min(curr_metric_results), max(curr_metric_results),
            worst_result=0,
        )

        # Metrics' absolute difference
        diff_name = f"{metric_name}_diff"
        results[diff_name] = max(curr_metric_results) - min(curr_metric_results)

    # ** Equalized odds **
    # default value: use maximum constraint violation for TPR and FPR equality
    results["equalized_odds_ratio"] = min(
        results["fnr_ratio"],
        results["fpr_ratio"],
    )
    results["equalized_odds_diff"] = max(
        results["tpr_diff"],  # same as FNR diff
        results["fpr_diff"],  # same as TNR diff
    )

    # Optionally, return group-wise metrics as well
    if return_groupwise_metrics:
        results.update(groupwise_metrics)

    return results


def compute_best_threshold(
    y_true: np.ndarray,
    y_pred_scores: np.ndarray,
    *,
    false_pos_cost: float = 1.0,
    false_neg_cost: float = 1.0,
) -> float:
    """Computes the binarization threshold that maximizes accuracy.

    Parameters
    ----------
    y_true : np.ndarray
        The true class labels.
    y_pred_scores : np.ndarray
        The predicted risk scores.
    false_pos_cost : float, optional
        The cost of a false positive error, by default 1.0
    false_neg_cost : float, optional
        The cost of a false negative error, by default 1.0

    Returns
    -------
    best_threshold : float
        The threshold value that maximizes accuracy for the given predictions.
    """

    # Compute TPR and FPR for all possible thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_scores)

    # Compute the cost of each threshold
    costs = false_pos_cost * fpr + false_neg_cost * (1 - tpr)

    # Get the threshold that minimizes the cost
    best_threshold = thresholds[np.argmin(costs)]
    return best_threshold


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_scores: np.ndarray,
    *,
    sensitive_attribute: np.ndarray = None,
    threshold: float | str = "best",
    model_name: str = None,
) -> dict:
    """Evaluates predictions on common performance and fairness metrics.

    Parameters
    ----------
    y_true : np.ndarray
        The true class labels.
    y_pred_scores : np.ndarray
        The predicted scores.
    sensitive_attribute: np.ndarray, optional
        The sensitive attribute data. Will compute fairness metrics if provided.
    threshold : float | str, optional
        The threshold to use for binarizing the predictions, or "best" to infer
        which threshold maximizes accuracy.
    model_name : str, optional
        The name of the model to be used on the plots, by default None.

    Returns
    -------
    results : dict
        A dictionary with key-value pairs of (metric name, metric value).
    """
    # Compute threshold if necessary
    if threshold == "best":
        threshold = compute_best_threshold(y_true, y_pred_scores)
    assert is_valid_number(threshold) and 0 <= threshold <= 1, \
        f"Invalid threshold: {threshold}"

    # Save initial results' statistics
    results = {
        "threshold": threshold,
        "n_samples": len(y_true),
        "n_positives": np.sum(y_true).item(),
        "n_negatives": len(y_true) - np.sum(y_true).item(),
        "model_name": model_name,
    }

    # Compute binary predictions using the default threshold
    y_pred_binary = (y_pred_scores >= threshold).astype(int)

    # Evaluate binary predictions
    results.update(evaluate_binary_predictions(y_true, y_pred_binary))

    # Add loss functions as proxies for calibration
    results["log_loss"] = log_loss(y_true, y_pred_scores, labels=[0, 1])
    results["brier_score_loss"] = brier_score_loss(y_true, y_pred_scores)

    # Evaluate fairness metrics
    if sensitive_attribute is not None:
        results.update(evaluate_binary_predictions_fairness(y_true, y_pred_binary, sensitive_attribute))

    # Compute additional metrics
    results["roc_auc"] = roc_auc_score(y_true, y_pred_scores)

    # Compute Expected Calibration Error
    # TODO: re-implement ECE scorer to avoid including 10 other dependencies for this one metric...
    class_preds = np.stack([1 - y_pred_scores, y_pred_scores], axis=1)
    n_bins = 10
    results["ece"] = ECE(bins=n_bins, equal_intervals=True).measure(class_preds, y_true)

    # Try to compute ECE with quantile bins
    try:
        results["ece_quantile"] = ECE(bins=n_bins, equal_intervals=False).measure(class_preds, y_true)
    except Exception as err:
        logging.warning(f"Failed to compute ECE quantile: {err}")
        results["ece_quantile"] = None

    return results


def evaluate_predictions_bootstrap(
    y_true: np.ndarray,
    y_pred_scores: np.ndarray,
    *,
    sensitive_attribute: np.ndarray = None,
    threshold: float | str = "best",
    k: int = 200,
    confidence_pct: float = 95,
    seed: int = 42,
) -> dict:
    """Computes bootstrap estimates of several metrics for the given predictions.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred_scores : np.ndarray
        The score predictions.
    sensitive_attribute : np.ndarray, optional
        The sensitive attribute data. Will compute fairness metrics if provided.
    threshold : float | str, optional
        The threshold to use for binarizing the predictions, or "best" to infer
        which threshold maximizes accuracy, by default "best".
    k : int, optional
        How many bootstrap samples to draw, by default 200.
    confidence_pct : float, optional
        How large of a confidence interval to use when reporting lower and upper
        bounds, by default 95 (i.e., 2.5 to 97.5 percentile of results).
    seed : int, optional
        The random seed, by default 42.

    Returns
    -------
    results : dict
        A dictionary containing bootstrap estimates for a variety of metrics.
    """
    assert len(y_true) == len(y_pred_scores)
    rng = np.random.default_rng(seed=seed)

    # Draw k bootstrap samples with replacement
    results = []
    for _ in range(k):
        # Indices of current bootstrap sample
        indices = rng.choice(len(y_true), replace=True, size=len(y_true))

        # Evaluate predictions on this bootstrap sample
        results.append(
            evaluate_predictions(
                y_true=y_true[indices],
                y_pred_scores=y_pred_scores[indices],
                sensitive_attribute=sensitive_attribute[indices] if sensitive_attribute is not None else None,
                threshold=threshold,
            )
        )

    # Compute statistics from bootstrapped results
    all_metrics = set(results[0].keys())

    bt_mean = {}
    bt_stdev = {}
    bt_percentiles = {}

    low_percentile = (100 - confidence_pct) / 2
    confidence_percentiles = [low_percentile, 100 - low_percentile]

    for m in all_metrics:
        metric_values = [r[m] for r in results]

        bt_mean[m] = statistics.mean(metric_values)
        bt_stdev[m] = statistics.stdev(metric_values)
        bt_percentiles[m] = tuple(np.percentile(metric_values, confidence_percentiles))

    # Join bootstrap results for all metrics
    return join_dictionaries(
        *(
            {
                f"{metric}_mean": bt_mean[metric],
                f"{metric}_stdev": bt_stdev[metric],
                f"{metric}_low-percentile": bt_percentiles[metric][0],
                f"{metric}_high-percentile": bt_percentiles[metric][1],
            }
            for metric in sorted(bt_mean.keys())
        )
    )
