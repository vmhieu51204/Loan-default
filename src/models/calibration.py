"""
Probability Calibration module using Isotonic Regression.
Calibrates raw logistic regression probabilities using an independent validation partition.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class IsotonicProbabilityCalibrator:
    """
    Fits non-parametric isotonic regression on validation set probabilities to convert
    raw logit-derived scores into accurately calibrated empirical default probabilities.
    """

    def __init__(self, y_min: float = 0.0, y_max: float = 1.0, out_of_bounds: str = "clip"):
        """
        Args:
            y_min: Lower bound for calibrated probabilities.
            y_max: Upper bound for calibrated probabilities.
            out_of_bounds: Strategy for out-of-bounds inputs ('clip' or 'nan').
        """
        self.calibrator = IsotonicRegression(
            y_min=y_min, y_max=y_max, out_of_bounds=out_of_bounds
        )
        self.is_fitted = False

    def fit(self, raw_probas: np.ndarray, y_val: np.ndarray) -> "IsotonicProbabilityCalibrator":
        """
        Fits isotonic regression on validation raw predicted probabilities.

        Args:
            raw_probas: Predicted probabilities of Good (1) from Logistic Regression.
            y_val: Actual binary labels in validation set (1 = Good, 0 = Bad).

        Returns:
            self
        """
        print("Fitting Isotonic Probability Calibrator on Validation set...")
        self.calibrator.fit(raw_probas, y_val)
        self.is_fitted = True
        print("Calibrator successfully fitted.")
        return self

    def predict_good_proba(self, raw_probas: np.ndarray) -> np.ndarray:
        """
        Predicts calibrated probability of Good (P(Y=1)).
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before predicting.")
        return self.calibrator.predict(raw_probas)

    def predict_pd(self, raw_probas: np.ndarray) -> np.ndarray:
        """
        Predicts calibrated Probability of Default (PD = P(Y=0) = 1 - P(Y=1)).
        """
        p_good = self.predict_good_proba(raw_probas)
        return 1.0 - p_good

    def diagnose_calibration_bins(
        self, y_true: pd.Series, calib_probas: np.ndarray, n_bins: int = 10
    ) -> pd.DataFrame:
        """
        Creates detailed bin statistics to inspect calibration reliability.

        Args:
            y_true: Ground truth target (1=Good, 0=Bad).
            calib_probas: Calibrated probabilities of Good.
            n_bins: Number of probability bins.

        Returns:
            pd.DataFrame: Table with counts, bad counts, actual prob, avg predicted prob.
        """
        check_df = pd.DataFrame({
            "actual_target": np.array(y_true),
            "calibrated_prob": calib_probas,
        })
        bins = np.linspace(0, 1, n_bins + 1)
        check_df["bin"] = pd.cut(check_df["calibrated_prob"], bins=bins, include_lowest=True)

        bin_stats = check_df.groupby("bin", observed=False).agg(
            count=("actual_target", "count"),
            good_count=("actual_target", "sum"),
            bad_count=("actual_target", lambda x: (1 - x).sum()),
            actual_good_rate=("actual_target", "mean"),
            avg_pred_good_prob=("calibrated_prob", "mean"),
        )
        bin_stats["actual_bad_rate"] = 1.0 - bin_stats["actual_good_rate"]
        return bin_stats
