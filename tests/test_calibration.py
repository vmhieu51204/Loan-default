"""
Unit tests for isotonic probability calibration.
"""

import unittest
import numpy as np
import pandas as pd
from src.models.calibration import IsotonicProbabilityCalibrator


class TestCalibration(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 100
        self.raw_probas = np.linspace(0.05, 0.95, n)
        # Monotonically increasing true probabilities
        self.y_val = (np.random.rand(n) < self.raw_probas).astype(int)
        self.calibrator = IsotonicProbabilityCalibrator()

    def test_fit_and_predict(self):
        self.calibrator.fit(self.raw_probas, self.y_val)
        self.assertTrue(self.calibrator.is_fitted)

        good_probas = self.calibrator.predict_good_proba(self.raw_probas)
        pds = self.calibrator.predict_pd(self.raw_probas)

        self.assertEqual(len(good_probas), len(self.raw_probas))
        self.assertTrue((good_probas >= 0.0).all() and (good_probas <= 1.0).all())
        self.assertTrue((pds >= 0.0).all() and (pds <= 1.0).all())
        # PD + Good Proba should sum to 1
        np.testing.assert_allclose(good_probas + pds, 1.0)

    def test_diagnose_calibration_bins(self):
        self.calibrator.fit(self.raw_probas, self.y_val)
        calib_probas = self.calibrator.predict_good_proba(self.raw_probas)
        bin_df = self.calibrator.diagnose_calibration_bins(
            pd.Series(self.y_val), calib_probas, n_bins=5
        )
        self.assertIn("count", bin_df.columns)
        self.assertIn("good_count", bin_df.columns)
        self.assertIn("actual_bad_rate", bin_df.columns)


if __name__ == "__main__":
    unittest.main()
