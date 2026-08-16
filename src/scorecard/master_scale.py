"""
Master Scale and Score Decile Table creation module for portfolio credit risk management.
"""

from typing import Optional
import numpy as np
import pandas as pd


def create_master_scale(
    scores: pd.Series,
    calibrated_pds: np.ndarray,
    min_score: int = 300,
    max_score: int = 900,
    bin_width: int = 50,
) -> pd.DataFrame:
    """
    Constructs the Master Scale table mapping standard Score Bands to Average Score,
    Volume, and Calibrated Probability of Default (PD).

    Args:
        scores: Series of predicted integer credit scores.
        calibrated_pds: Array of calibrated default probabilities (PD).
        min_score: Lower bound of scorecard master range.
        max_score: Upper bound of scorecard master range.
        bin_width: Step size for score bands (e.g., 50 points).

    Returns:
        pd.DataFrame: Formatted Master Scale table.
    """
    df_view = pd.DataFrame({
        "score": scores.values,
        "calibrated_pd": calibrated_pds,
    })

    bins = list(range(min_score, max_score + bin_width, bin_width))
    df_view["Score_Band"] = pd.cut(df_view["score"], bins=bins)

    master_scale = (
        df_view.groupby("Score_Band", observed=False)
        .agg(
            Volume=("score", "count"),
            Average_Score=("score", "mean"),
            Calibrated_PD=("calibrated_pd", "mean"),
        )
        .reset_index()
    )

    # Filter out bands with zero volume
    master_scale = master_scale[master_scale["Volume"] > 0].copy()

    total_volume = master_scale["Volume"].sum()
    master_scale["Volume_Share"] = (master_scale["Volume"] / total_volume).apply(
        lambda x: f"{x:.2%}"
    )
    master_scale["Average_Score"] = master_scale["Average_Score"].round(1)
    master_scale["Calibrated_PD_Formatted"] = master_scale["Calibrated_PD"].apply(
        lambda x: f"{x:.2%}"
    )

    print("\n--- MASTER SCALE TABLE ---")
    print(
        master_scale[
            ["Score_Band", "Average_Score", "Volume", "Volume_Share", "Calibrated_PD_Formatted"]
        ].to_string(index=False)
    )

    return master_scale


def create_decile_table(
    scores: pd.Series, actual_targets: pd.Series, num_deciles: int = 10
) -> pd.DataFrame:
    """
    Creates score decile risk performance table (Lower decile = higher risk).

    Args:
        scores: Series of credit scores.
        actual_targets: Binary target Series (1 = Good, 0 = Bad).
        num_deciles: Number of quantile buckets (default 10).

    Returns:
        pd.DataFrame: Summary table by score decile.
    """
    df_eval = pd.DataFrame({
        "score": scores.values,
        "actual": actual_targets.values,
    })

    df_eval["decile"] = pd.qcut(
        df_eval["score"], num_deciles, labels=False, duplicates="drop"
    )

    decile_summary = (
        df_eval.groupby("decile")
        .agg(
            min_score=("score", "min"),
            max_score=("score", "max"),
            volume=("score", "count"),
            good_count=("actual", "sum"),
            bad_count=("actual", lambda x: (1 - x).sum()),
            actual_bad_rate=("actual", lambda x: (1 - x).mean()),
        )
        .reset_index()
    )

    total_bads = decile_summary["bad_count"].sum()
    total_goods = decile_summary["good_count"].sum()

    decile_summary["cum_bad_rate"] = (
        decile_summary["bad_count"].cumsum() / total_bads if total_bads > 0 else 0
    )
    decile_summary["cum_good_rate"] = (
        decile_summary["good_count"].cumsum() / total_goods if total_goods > 0 else 0
    )
    decile_summary["ks_decile"] = (
        decile_summary["cum_bad_rate"] - decile_summary["cum_good_rate"]
    ).abs()

    print("\n--- SCORECARD DECILE TABLE ---")
    print(decile_summary.to_string(index=False))

    return decile_summary
