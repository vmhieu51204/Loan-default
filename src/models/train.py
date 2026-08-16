"""
Model training module for fitting Logistic Regression on WoE features.
"""

from typing import List, Optional
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_logistic_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    c_penalty: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
) -> LogisticRegression:
    """
    Fits a Logistic Regression model on WoE-transformed training data.

    Args:
        X_train: Training features (WoE encoded).
        y_train: Target series (1 = Good, 0 = Bad).
        c_penalty: Inverse of regularization strength.
        solver: Optimization algorithm (default: 'lbfgs').
        max_iter: Maximum solver iterations.
        class_weight: Class balancing ('balanced' or None).
        random_state: Random state seed.

    Returns:
        LogisticRegression: Trained scikit-learn model.
    """
    print(f"Training Logistic Regression model on {X_train.shape[1]} features...")
    model = LogisticRegression(
        C=c_penalty,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    print(f"Model Intercept (beta_0): {model.intercept_[0]:.4f}")
    coef_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)
    print("Model Coefficients:")
    print(coef_df.to_string(index=False))

    return model


def get_coefficients_summary(
    model: LogisticRegression, feature_names: List[str]
) -> pd.DataFrame:
    """
    Returns a formatted summary table of model coefficients and odds ratios.
    """
    import numpy as np

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0],
        "Odds_Ratio": np.exp(model.coef_[0]),
    }).sort_values(by="Coefficient", ascending=False)
    return coef_df


def save_model(model: LogisticRegression, filepath: str) -> None:
    """Saves trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Saved model to {filepath}")


def load_model(filepath: str) -> LogisticRegression:
    """Loads saved model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    return joblib.load(filepath)
