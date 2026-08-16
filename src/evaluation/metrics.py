"""
Statistical evaluation metrics for credit scoring (Discrimination and Calibration).
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def evaluate_discrimination(
    y_true: pd.Series, y_pred_proba: np.ndarray
) -> Dict[str, Any]:
    """
    Computes discriminatory power metrics for credit risk models:
    - ROC-AUC: Area Under the Receiver Operating Characteristic
    - Gini Coefficient: 2 * AUC - 1
    - Kolmogorov-Smirnov (KS) Statistic: max(TPR - FPR)

    Args:
        y_true: Binary ground truth Series (1 = Good, 0 = Bad).
        y_pred_proba: Predicted probability of Good (or score).

    Returns:
        Dict[str, Any]: Dictionary containing AUC, Gini, KS, and ROC curves.
    """
    y_true_arr = np.array(y_true)
    auc = float(roc_auc_score(y_true_arr, y_pred_proba))
    gini = float(2.0 * auc - 1.0)

    fpr, tpr, thresholds = roc_curve(y_true_arr, y_pred_proba)
    ks_values = tpr - fpr
    ks_max_idx = int(np.argmax(ks_values))
    ks_stat = float(ks_values[ks_max_idx])
    ks_thresh = float(thresholds[ks_max_idx])

    metrics = {
        "auc": auc,
        "gini": gini,
        "ks_stat": ks_stat,
        "ks_threshold": ks_thresh,
        "fpr": fpr,
        "tpr": tpr,
    }

    print("[METRICS] Discriminatory Performance:")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Gini    : {gini:.4f}")
    print(f"  KS Stat : {ks_stat:.4f} (at threshold {ks_thresh:.4f})")

    return metrics


def evaluate_calibration_metrics(
    y_true: pd.Series, y_calib_good_proba: np.ndarray
) -> Dict[str, float]:
    """
    Computes calibration accuracy metrics comparing predicted probabilities with empirical rates.

    Args:
        y_true: Ground truth binary target (1 = Good, 0 = Bad).
        y_calib_good_proba: Calibrated probability of Good.

    Returns:
        Dict[str, float]: Brier score, predicted mean, actual mean, difference.
    """
    y_true_arr = np.array(y_true)
    brier = float(brier_score_loss(y_true_arr, y_calib_good_proba))
    pred_mean = float(np.mean(y_calib_good_proba))
    actual_mean = float(np.mean(y_true_arr))

    calib_metrics = {
        "brier_score": brier,
        "pred_good_rate": pred_mean,
        "actual_good_rate": actual_mean,
        "pred_bad_rate (PD)": 1.0 - pred_mean,
        "actual_bad_rate": 1.0 - actual_mean,
        "mean_diff": abs(pred_mean - actual_mean),
    }

    print("[CALIBRATION] Calibration Accuracy Metrics:")
    print(f"  Brier Score Loss: {brier:.5f}")
    print(f"  Pred Mean (Good): {pred_mean:.4f} | Actual Good Rate: {actual_mean:.4f}")
    print(f"  Pred Mean (Bad) : {1.0 - pred_mean:.4f} | Actual Bad Rate : {1.0 - actual_mean:.4f}")

    return calib_metrics
