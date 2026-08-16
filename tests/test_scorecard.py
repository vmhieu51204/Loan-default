"""
Unit tests for scorecard scaling and score calculation.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.scorecard.scaling import ScorecardScaler
from src.features.woe import WoEBinning


class TestScorecardScaling(unittest.TestCase):

    def setUp(self):
        self.scaler = ScorecardScaler(pdo=50.0, base_score=600.0, base_odds=50.0)

        # Train a mock logistic model
        np.random.seed(42)
        X = pd.DataFrame({"w1": np.array([0.5, -0.2, 0.1, -0.8]), "w2": np.array([0.2, 0.4, -0.1, -0.5])})
        y = pd.Series([1, 1, 0, 0])
        self.model = LogisticRegression().fit(X, y)
        self.feature_names = ["w1", "w2"]
        self.X = X

    def test_factor_and_offset_math(self):
        # Factor = 50 / ln(2) ~ 72.13475
        expected_factor = 50.0 / np.log(2.0)
        expected_offset = 600.0 - (expected_factor * np.log(50.0))

        self.assertAlmostEqual(self.scaler.factor, expected_factor, places=5)
        self.assertAlmostEqual(self.scaler.offset, expected_offset, places=5)

    def test_calculate_scores(self):
        scores_df = self.scaler.calculate_scores(self.X, self.model, self.feature_names)
        self.assertIn("logit", scores_df.columns)
        self.assertIn("score", scores_df.columns)
        self.assertEqual(len(scores_df), len(self.X))
        # Scores should be integers
        self.assertTrue(np.issubdtype(scores_df["score"].dtype, np.integer))

    def test_generate_scorecard_points_table(self):
        woe_encoder = WoEBinning()
        woe_encoder.woe_maps = {
            "w1": {"[-inf, 0.0]": -0.5, "(0.0, inf]": 0.5},
            "w2": {"[-inf, 0.0]": -0.3, "(0.0, inf]": 0.3},
        }

        points_df = self.scaler.generate_scorecard_points_table(
            self.model, self.feature_names, woe_encoder
        )
        self.assertIn("Feature", points_df.columns)
        self.assertIn("Bin", points_df.columns)
        self.assertIn("Points", points_df.columns)
        self.assertEqual(points_df.iloc[0]["Feature"], "Base Points (Intercept)")


if __name__ == "__main__":
    unittest.main()
