"""
Feature engineering and dataset partitioning module.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs standard preprocessing and feature engineering:
    - Computes credit history length in months from issue_d and earliest_cr_line.
    - Parses emp_length string to integer years.
    - Parses term string to numeric months.
    - Imputes missing values (median for numeric, mode for categorical).

    Args:
        df: Input DataFrame after leakage cleaning.

    Returns:
        pd.DataFrame: Feature-engineered DataFrame.
    """
    df = df.copy()

    # 1. Date Processing
    if "issue_d" in df.columns and "earliest_cr_line" in df.columns:
        df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        df["earliest_cr_line"] = pd.to_datetime(
            df["earliest_cr_line"], format="%b-%Y", errors="coerce"
        )
        # Drop records where dates are missing
        df.dropna(subset=["earliest_cr_line", "issue_d"], inplace=True)
        df["credit_hist_months"] = (
            (df["issue_d"] - df["earliest_cr_line"]).dt.days / 30.4375
        ).astype(int)
        df.drop(columns=["issue_d", "earliest_cr_line"], inplace=True)

    # 2. Employment Length numeric parsing
    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].fillna("< 1 year").astype(str)
        df["emp_length"] = (
            df["emp_length"].str.extract(r"(\d+)").fillna(0).astype(int)
        )

    # 3. Term numeric parsing
    if "term" in df.columns:
        df["term_int"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
        df.drop("term", axis=1, inplace=True)

    # 4. Missing value imputation
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns

    for col in num_cols:
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0
        df[col] = df[col].fillna(median_val)

    for col in cat_cols:
        if not df[col].empty and len(df[col].dropna()) > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    print(f"Feature engineering completed. Resulting shape: {df.shape}")
    return df


def train_val_test_split_stratified(
    df: pd.DataFrame,
    target_col: str = "loan_status_binary",
    test_size: float = 0.20,
    val_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Performs a 3-way stratified partition:
    - Test set: 20% of total
    - Val set: 20% of total (val_size * 80% = 20%)
    - Train set: 60% of total (75% of remaining 80% = 60%)

    Args:
        df: Input DataFrame containing features and target column.
        target_col: Target column name.
        test_size: Proportion for test split.
        val_size: Proportion of remaining data for validation split.
        random_state: Random state seed.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    X = df.drop([target_col], axis=1, errors="ignore")
    y = df[target_col]

    # Split Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Split Train and Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )

    print("Stratified Dataset Split:")
    print(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(df):.1%})")
    print(f"  Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(df):.1%})")
    print(f"  Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(df):.1%})")

    return X_train, y_train, X_val, y_val, X_test, y_test
