"""
Weight of Evidence (WoE) and Information Value (IV) transformation module.
Supports tree-based optimal binning for continuous features and categorical binning.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class WoEBinning:
    """
    Automated Weight of Evidence (WoE) Transformer and Information Value (IV) calculator.
    Uses DecisionTreeClassifier for optimal numeric binning and calculates Laplace-smoothed WoE.
    """

    def __init__(self, max_bins: int = 6, min_samples: float = 0.05, random_state: int = 42):
        """
        Args:
            max_bins: Maximum number of bins (leaf nodes in decision tree).
            min_samples: Minimum fraction of samples per leaf bin.
            random_state: Seed for reproducible decision tree splits.
        """
        self.max_bins = max_bins
        self.min_samples = min_samples
        self.random_state = random_state
        self.woe_maps: Dict[str, Dict[Any, Any]] = {}
        self.iv_values: Dict[str, float] = {}
        self.summary_tables: Dict[str, pd.DataFrame] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEBinning":
        """
        Fits WoE binning on training features and binary target (1=Good, 0=Bad).

        Args:
            X: Training features DataFrame.
            y: Training target Series (1=Good, 0=Bad).

        Returns:
            self
        """
        self.woe_maps = {}
        self.iv_values = {}
        self.summary_tables = {}

        for col in X.columns:
            try:
                if pd.api.types.is_numeric_dtype(X[col]):
                    # Fit shallow decision tree to find monotonic/optimal splits
                    # Handle constant columns gracefully
                    if X[col].nunique() <= 1:
                        bins = [-np.inf, np.inf]
                    else:
                        dt = DecisionTreeClassifier(
                            max_leaf_nodes=self.max_bins,
                            min_samples_leaf=self.min_samples,
                            random_state=self.random_state,
                        )
                        dt.fit(X[[col]], y)
                        thresholds = np.sort(dt.tree_.threshold[dt.tree_.threshold != -2])
                        bins = [-np.inf] + list(thresholds) + [np.inf]
                    self._calculate_woe(X[col], y, col, bins=bins, is_numeric=True)
                else:
                    self._calculate_woe(X[col], y, col, is_numeric=False)
            except Exception as e:
                print(f"Warning: Could not compute WoE for column '{col}': {e}")
                self.iv_values[col] = 0.0

        return self

    def _calculate_woe(
        self,
        feature_series: pd.Series,
        target: pd.Series,
        col_name: str,
        bins: Optional[List[float]] = None,
        is_numeric: bool = True,
    ) -> None:
        """Internal helper to calculate WoE, IV, and summary tables per bin."""
        df_temp = pd.DataFrame({"feature": feature_series, "target": target})

        if is_numeric:
            df_temp["bin"] = pd.cut(df_temp["feature"], bins=bins, duplicates="drop")
        else:
            df_temp["bin"] = df_temp["feature"].astype(str)

        grouped = df_temp.groupby("bin", observed=False)["target"].agg(["count", "sum"])
        grouped.rename(columns={"count": "total", "sum": "good"}, inplace=True)
        grouped["bad"] = grouped["total"] - grouped["good"]

        total_good = grouped["good"].sum()
        total_bad = grouped["bad"].sum()

        if total_good == 0 or total_bad == 0:
            self.iv_values[col_name] = 0.0
            return

        # Laplace smoothing (0.5) to avoid log(0) or division by zero
        grouped["dist_good"] = (grouped["good"] + 0.5) / (total_good + 0.5 * len(grouped))
        grouped["dist_bad"] = (grouped["bad"] + 0.5) / (total_bad + 0.5 * len(grouped))

        # WoE = ln(dist_good / dist_bad)  (Higher WoE -> Lower Risk / Higher probability of Good)
        grouped["woe"] = np.log(grouped["dist_good"] / grouped["dist_bad"])
        grouped["iv"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]
        grouped["bad_rate"] = grouped["bad"] / grouped["total"].replace(0, np.nan)

        self.summary_tables[col_name] = grouped.copy()
        self.woe_maps[col_name] = grouped["woe"].to_dict()
        self.iv_values[col_name] = float(grouped["iv"].sum())

        if is_numeric:
            self.woe_maps[col_name]["bins"] = bins

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms feature DataFrame to Weight of Evidence (WoE) numerical values.

        Args:
            X: Input DataFrame.

        Returns:
            pd.DataFrame: WoE-transformed DataFrame.
        """
        data_dict = {}

        for col in self.woe_maps:
            if col not in X.columns:
                continue

            mapping = self.woe_maps[col]
            if "bins" in mapping:
                bins = mapping["bins"]
                binned_series = pd.cut(X[col], bins=bins, duplicates="drop")
                # Map bin intervals to WoE float values, cast to float first then fillna
                data_dict[col] = binned_series.map(mapping).astype(float).fillna(0.0)
            else:
                data_dict[col] = X[col].astype(str).map(mapping).astype(float).fillna(0.0)

        return pd.DataFrame(data_dict, index=X.index)

    def get_iv_summary(self) -> pd.DataFrame:
        """
        Returns DataFrame containing Information Value (IV) of all fitted features, sorted descending.
        """
        iv_df = pd.DataFrame.from_dict(
            self.iv_values, orient="index", columns=["IV"]
        ).sort_values(by="IV", ascending=False)
        return iv_df

    def get_bin_summary(self, col_name: str) -> Optional[pd.DataFrame]:
        """
        Returns binning summary table (Total, Good, Bad, Dist Good, Dist Bad, WoE, IV) for a specific feature.
        """
        return self.summary_tables.get(col_name)
