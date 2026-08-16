"""
Unit tests for data cleaning and feature engineering.
"""

import unittest
import pandas as pd
import numpy as np

from src.data.cleaner import map_target, drop_leakage_features, clean_target_and_leakage
from src.features.engineering import engineer_features, train_val_test_split_stratified


class TestDataCleanerAndEngineering(unittest.TestCase):

    def setUp(self):
        self.mapping = {
            "Fully Paid": 1,
            "Charged Off": 0,
            "Current": 1,
            "Default": 0,
        }
        self.leakage_cols = ["total_pymnt", "recoveries", "last_pymnt_d"]

        # Synthetic raw loan dataframe
        self.raw_df = pd.DataFrame({
            "loan_status": ["Fully Paid", "Charged Off", "Current", "Default", "Unknown Status"],
            "total_pymnt": [1000, 200, 500, 100, 0],
            "recoveries": [0, 50, 0, 100, 0],
            "emp_length": ["10+ years", "< 1 year", "3 years", None, "5 years"],
            "term": [" 36 months", " 60 months", " 36 months", " 60 months", " 36 months"],
            "issue_d": ["Jan-2015", "Mar-2016", "Jul-2017", "Dec-2014", "Feb-2015"],
            "earliest_cr_line": ["Jan-2000", "Mar-2006", "Jul-2007", "Dec-1994", "Feb-2005"],
            "annual_inc": [60000.0, 45000.0, np.nan, 80000.0, 50000.0],
            "home_ownership": ["RENT", "MORTGAGE", "RENT", None, "RENT"],
        })

    def test_map_target(self):
        cleaned = map_target(self.raw_df, target_col="loan_status", target_mapping=self.mapping)
        self.assertEqual(len(cleaned), 4)  # "Unknown Status" dropped
        self.assertIn("loan_status_binary", cleaned.columns)
        self.assertListEqual(cleaned["loan_status_binary"].tolist(), [1, 0, 1, 0])

    def test_drop_leakage_features(self):
        cleaned = drop_leakage_features(self.raw_df, self.leakage_cols)
        for col in self.leakage_cols:
            self.assertNotIn(col, cleaned.columns)

    def test_engineer_features(self):
        cleaned = clean_target_and_leakage(
            self.raw_df,
            target_col="loan_status",
            target_mapping=self.mapping,
            leakage_cols=self.leakage_cols,
        )
        engineered = engineer_features(cleaned)

        self.assertIn("credit_hist_months", engineered.columns)
        self.assertNotIn("issue_d", engineered.columns)
        self.assertNotIn("earliest_cr_line", engineered.columns)
        self.assertIn("term_int", engineered.columns)
        self.assertNotIn("term", engineered.columns)

        # Check imputation
        self.assertFalse(engineered["annual_inc"].isna().any())
        self.assertFalse(engineered["home_ownership"].isna().any())
        self.assertEqual(engineered["emp_length"].dtype, int)

    def test_train_val_test_split(self):
        # Create larger dataset for split testing
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "loan_status_binary": [1] * 80 + [0] * 20,
        })
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_stratified(
            df, target_col="loan_status_binary", test_size=0.2, val_size=0.25, random_state=42
        )
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(X_val), 20)
        self.assertEqual(len(X_train), 60)


if __name__ == "__main__":
    unittest.main()
