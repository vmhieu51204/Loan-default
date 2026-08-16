"""
Scorecard scaling and point allocation module.
Converts logistic regression log-odds into standard integer credit scores.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.features.woe import WoEBinning


class ScorecardScaler:
    """
    Translates model log-odds (logits) into scaled credit scores using the standard
    PDO (Points to Double the Odds), Base Score, and Base Odds scaling formulas.
    """

    def __init__(self, pdo: float = 50.0, base_score: float = 600.0, base_odds: float = 50.0):
        """
        Args:
            pdo: Points to Double the Odds.
            base_score: Benchmark score at the base odds.
            base_odds: Benchmark Good:Bad odds (e.g., 50:1).
        """
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds

        # Compute Factor and Offset
        self.factor = self.pdo / np.log(2.0)
        self.offset = self.base_score - (self.factor * np.log(self.base_odds))

    def compute_logits(
        self, X_woe: pd.DataFrame, model: LogisticRegression, feature_names: List[str]
    ) -> pd.Series:
        """
        Computes raw logit (log-odds) for each observation:
        logit = beta_0 + sum(beta_i * WoE_i)
        """
        intercept = model.intercept_[0]
        logits = intercept + np.dot(X_woe[feature_names], model.coef_.T).ravel()
        return pd.Series(logits, index=X_woe.index, name="logit")

    def compute_scores_from_logits(self, logits: pd.Series) -> pd.Series:
        """
        Converts logits directly to scaled credit scores:
        Score = Offset + Factor * logit
        """
        scores = self.offset + (self.factor * logits)
        return scores.round().astype(int)

    def calculate_scores(
        self, X_woe: pd.DataFrame, model: LogisticRegression, feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Calculates both logits and final integer credit scores for input data.

        Args:
            X_woe: WoE-transformed features DataFrame.
            model: Trained LogisticRegression model.
            feature_names: List of model feature names in exact order.

        Returns:
            pd.DataFrame: Contains 'logit' and 'score' columns.
        """
        df_result = pd.DataFrame(index=X_woe.index)
        df_result["logit"] = self.compute_logits(X_woe, model, feature_names)
        df_result["score"] = self.compute_scores_from_logits(df_result["logit"])
        return df_result

    def generate_scorecard_points_table(
        self,
        model: LogisticRegression,
        feature_names: List[str],
        woe_encoder: WoEBinning,
    ) -> pd.DataFrame:
        """
        Generates the standard additive credit scorecard points lookup table.
        Score = Base_Points + sum(Points(feature_i, bin_j))

        Base_Points = Offset + Factor * beta_0
        Points(feature_i, bin_j) = Factor * beta_i * WoE(feature_i, bin_j)
        """
        intercept = model.intercept_[0]
        base_points = round(self.offset + (self.factor * intercept))

        rows = []
        # Add Base Points record
        rows.append({
            "Feature": "Base Points (Intercept)",
            "Bin": "All",
            "WoE": np.nan,
            "Coefficient": intercept,
            "Points": base_points,
        })

        num_features = len(feature_names)
        for i, feature in enumerate(feature_names):
            coef = model.coef_[0][i]
            mapping = woe_encoder.woe_maps.get(feature, {})

            for bin_key, woe_val in mapping.items():
                if bin_key == "bins":
                    continue
                # Points for specific bin
                bin_points = round(self.factor * coef * woe_val)
                rows.append({
                    "Feature": feature,
                    "Bin": str(bin_key),
                    "WoE": round(woe_val, 4),
                    "Coefficient": round(coef, 4),
                    "Points": bin_points,
                })

        scorecard_df = pd.DataFrame(rows)
        return scorecard_df
