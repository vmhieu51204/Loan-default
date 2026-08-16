"""
Unit tests for Weight of Evidence (WoE) and Information Value (IV) calculation.
"""

import unittest
import numpy as np
import pandas as pd
from src.features.woe import WoEBinning


class TestWoEBinning(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 200
        # Continuous feature correlated with target
        x1 = np.random.normal(loc=0, scale=1, size=n)
        # Categorical feature
        x2 = np.random.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
        # Binary target (1=Good, 0=Bad)
        prob = 1 / (1 + np.exp(-(0.8 * x1 + (x2 == "A") * 0.5)))
        y = (np.random.rand(n) < prob).astype(int)

        self.df = pd.DataFrame({"num_feat": x1, "cat_feat": x2})
        self.y = pd.Series(y, name="target")

    def test_woe_fit_and_transform(self):
        woe = WoEBinning(max_bins=4, min_samples=0.05)
        woe.fit(self.df, self.y)

        # Ensure mappings exist
        self.assertIn("num_feat", woe.woe_maps)
        self.assertIn("cat_feat", woe.woe_maps)
        self.assertIn("bins", woe.woe_maps["num_feat"])

        # Transform dataframe
        df_woe = woe.transform(self.df)
        self.assertEqual(df_woe.shape, self.df.shape)
        self.assertTrue(np.issubdtype(df_woe["num_feat"].dtype, np.floating))
        self.assertTrue(np.issubdtype(df_woe["cat_feat"].dtype, np.floating))
        self.assertFalse(df_woe.isna().any().any())

    def test_iv_summary(self):
        woe = WoEBinning(max_bins=4, min_samples=0.05)
        woe.fit(self.df, self.y)
        iv_df = woe.get_iv_summary()

        self.assertIn("IV", iv_df.columns)
        self.assertEqual(len(iv_df), 2)
        # IV should be non-negative
        self.assertTrue((iv_df["IV"] >= 0).all())

    def test_bin_summary_table(self):
        woe = WoEBinning(max_bins=4, min_samples=0.05)
        woe.fit(self.df, self.y)
        summary = woe.get_bin_summary("cat_feat")

        self.assertIsNotNone(summary)
        self.assertIn("total", summary.columns)
        self.assertIn("good", summary.columns)
        self.assertIn("bad", summary.columns)
        self.assertIn("woe", summary.columns)
        self.assertIn("iv", summary.columns)


if __name__ == "__main__":
    unittest.main()
