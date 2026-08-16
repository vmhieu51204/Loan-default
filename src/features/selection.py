"""
Feature selection module based on Information Value (IV) and multi-collinearity filtering.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def filter_by_iv(
    iv_df: pd.DataFrame, min_iv: float = 0.02, max_iv: float = 0.55
) -> List[str]:
    """
    Filters features based on Information Value (IV) rules:
    - IV < 0.02: Unpredictive (dropped)
    - 0.02 <= IV <= 0.55: Useful predictive signal (retained)
    - IV > 0.55: Suspiciously high predictive power / potential leakage (dropped)

    Args:
        iv_df: DataFrame with feature names as index and 'IV' column.
        min_iv: Lower threshold for IV.
        max_iv: Upper threshold for IV.

    Returns:
        List[str]: Selected feature names.
    """
    selected = iv_df[(iv_df["IV"] >= min_iv) & (iv_df["IV"] <= max_iv)].index.tolist()
    print(
        f"IV Filtering ({min_iv} <= IV <= {max_iv}): {len(selected)} of {len(iv_df)} features retained."
    )
    return selected


def filter_collinear_features(
    df_woe: pd.DataFrame, iv_df: pd.DataFrame, threshold: float = 0.70
) -> List[str]:
    """
    Identifies pairs of collinear features (|correlation| > threshold) and drops
    the one with lower Information Value (IV).

    Args:
        df_woe: WoE-transformed features DataFrame.
        iv_df: DataFrame with 'IV' column indexed by feature name.
        threshold: Absolute correlation threshold.

    Returns:
        List[str]: List of feature names to drop.
    """
    if df_woe.empty or len(df_woe.columns) <= 1:
        return []

    corr_matrix = df_woe.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = set()
    for col in upper_tri.columns:
        correlated_cols = upper_tri.index[upper_tri[col] > threshold].tolist()
        for row in correlated_cols:
            iv_col = iv_df.loc[col, "IV"] if col in iv_df.index else 0.0
            iv_row = iv_df.loc[row, "IV"] if row in iv_df.index else 0.0

            if iv_col > iv_row:
                to_drop.add(row)
            else:
                to_drop.add(col)

    drop_list = list(to_drop)
    print(
        f"Collinearity Filtering (|r| > {threshold}): Dropping {len(drop_list)} correlated features: {drop_list}"
    )
    return drop_list


def select_features(
    df_woe: pd.DataFrame,
    iv_df: pd.DataFrame,
    min_iv: float = 0.02,
    max_iv: float = 0.55,
    corr_threshold: float = 0.70,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Executes full feature selection pipeline:
    1. Filter features by IV range.
    2. Filter out collinear features with lower IV.

    Args:
        df_woe: WoE-transformed training DataFrame.
        iv_df: Information Value DataFrame.
        min_iv: Minimum IV threshold.
        max_iv: Maximum IV threshold.
        corr_threshold: Maximum allowed correlation.

    Returns:
        Tuple of (final_selected_features_list, filtered_iv_df).
    """
    # 1. IV Filter
    iv_selected = filter_by_iv(iv_df, min_iv=min_iv, max_iv=max_iv)

    if not iv_selected:
        print("Warning: No features passed IV filter. Falling back to top 5 IV features.")
        iv_selected = iv_df.head(5).index.tolist()

    # 2. Correlation Filter
    to_drop_corr = filter_collinear_features(
        df_woe[iv_selected], iv_df, threshold=corr_threshold
    )
    final_features = [f for f in iv_selected if f not in to_drop_corr]

    print(f"Final Selected Features ({len(final_features)}): {final_features}")
    return final_features, iv_df.loc[final_features]
