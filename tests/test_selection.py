"""
Unit tests for feature selection routines (IV filter and collinearity filter).
"""

import unittest
import numpy as np
import pandas as pd
from src.features.selection import filter_by_iv, filter_collinear_features, select_features


class TestFeatureSelection(unittest.TestCase):

    def setUp(self):
        # IV test DataFrame
        self.iv_df = pd.DataFrame(
            {"IV": [0.01, 0.05, 0.25, 0.40, 0.70]},
            index=["feat_weak", "feat_med", "feat_strong", "feat_vstrong", "feat_leak"],
        )

        # Collinear features DataFrame
        np.random.seed(42)
        n = 100
        f1 = np.random.randn(n)
        f2 = f1 + np.random.normal(0, 0.05, n)  # highly correlated with f1 (r ~ 0.99)
        f3 = np.random.randn(n)  # independent

        self.df_woe = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3})
        self.corr_iv_df = pd.DataFrame({"IV": [0.20, 0.35, 0.15]}, index=["f1", "f2", "f3"])

    def test_filter_by_iv(self):
        selected = filter_by_iv(self.iv_df, min_iv=0.02, max_iv=0.55)
        self.assertEqual(selected, ["feat_med", "feat_strong", "feat_vstrong"])

    def test_filter_collinear_features(self):
        to_drop = filter_collinear_features(self.df_woe, self.corr_iv_df, threshold=0.7)
        # f1 and f2 are collinear, f2 has higher IV (0.35 > 0.20), so f1 should be dropped
        self.assertIn("f1", to_drop)
        self.assertNotIn("f2", to_drop)
        self.assertNotIn("f3", to_drop)

    def test_select_features_pipeline(self):
        final_feats, final_iv = select_features(
            self.df_woe, self.corr_iv_df, min_iv=0.02, max_iv=0.55, corr_threshold=0.7
        )
        self.assertIn("f2", final_feats)
        self.assertIn("f3", final_feats)
        self.assertNotIn("f1", final_feats)


if __name__ == "__main__":
    unittest.main()
