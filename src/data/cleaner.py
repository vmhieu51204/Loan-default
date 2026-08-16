"""
Data cleaner module for defining binary targets and removing future/leakage columns.
"""

from typing import Dict, List, Optional
import pandas as pd


def map_target(
    df: pd.DataFrame,
    target_col: str = "loan_status",
    target_mapping: Optional[Dict[str, int]] = None,
    binary_col_name: str = "loan_status_binary",
) -> pd.DataFrame:
    """
    Maps categorical loan status into binary target (1 = Good, 0 = Bad).
    Drops rows where the target status is unknown or not mapped.

    Args:
        df: Input DataFrame.
        target_col: Source column containing loan status strings.
        target_mapping: Dictionary mapping status strings to 0 or 1.
        binary_col_name: Name of the output binary column.

    Returns:
        pd.DataFrame: Cleaned DataFrame with binary target column added.
    """
    df = df.copy()
    if target_mapping is None:
        raise ValueError("target_mapping dictionary must be provided.")

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    df[binary_col_name] = df[target_col].map(target_mapping)
    initial_len = len(df)
    df = df.dropna(subset=[binary_col_name]).copy()
    df[binary_col_name] = df[binary_col_name].astype(int)

    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with unmapped target statuses.")

    print(
        f"Target distribution for '{binary_col_name}':\n"
        f"{df[binary_col_name].value_counts(normalize=True).to_dict()}"
    )
    return df


def drop_leakage_features(
    df: pd.DataFrame,
    leakage_cols: List[str],
) -> pd.DataFrame:
    """
    Drops data leakage columns and identifiers that must not be used at origination.

    Args:
        df: Input DataFrame.
        leakage_cols: List of column names to drop.

    Returns:
        pd.DataFrame: DataFrame with leakage columns removed.
    """
    df = df.copy()
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped {len(cols_to_drop)} leakage/ID columns. Remaining shape: {df.shape}")
    return df


def clean_target_and_leakage(
    df: pd.DataFrame,
    target_col: str,
    target_mapping: Dict[str, int],
    leakage_cols: List[str],
    binary_col_name: str = "loan_status_binary",
) -> pd.DataFrame:
    """
    Complete target mapping and leakage removal workflow.

    Args:
        df: Input DataFrame.
        target_col: Raw loan status column.
        target_mapping: Status mapping dictionary.
        leakage_cols: Leakage columns to drop.
        binary_col_name: Name of resulting target column.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for feature engineering.
    """
    df_cleaned = map_target(
        df,
        target_col=target_col,
        target_mapping=target_mapping,
        binary_col_name=binary_col_name,
    )
    df_cleaned = drop_leakage_features(df_cleaned, leakage_cols=leakage_cols)
    return df_cleaned
