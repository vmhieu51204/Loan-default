import unittest
import numpy as np
import pandas as pd
from src.scorecard.simulation_engine import PolicySimulationEngine, generate_benchmark_portfolio, get_simulation_engine


class TestPolicySimulationEngine(unittest.TestCase):
    def setUp(self):
        self.engine = get_simulation_engine()

    def test_benchmark_portfolio_shape(self):
        df = generate_benchmark_portfolio(n_samples=1000)
        self.assertIn("score", df.columns)
        self.assertIn("pd", df.columns)
        self.assertIn("target", df.columns)
        self.assertEqual(len(df), 678192)

    def test_evaluate_cutoff_benchmark(self):
        m = self.engine.evaluate_cutoff(cutoff=480, lgd=0.50, interest_margin=0.15, ead=20000.0)
        self.assertAlmostEqual(m["approval_rate"], 0.73, delta=0.03)
        self.assertGreater(m["approved_count"], 480000)
        self.assertLess(m["approved_count"], 510000)
        self.assertGreater(m["expected_loss"], 0)
        self.assertGreater(m["gross_revenue"], 0)

    def test_cutoff_bounds(self):
        # Extreme high cutoff (approve none)
        m_high = self.engine.evaluate_cutoff(cutoff=999)
        self.assertEqual(m_high["approved_count"], 0)
        self.assertEqual(m_high["approval_rate"], 0.0)
        self.assertEqual(m_high["net_profit"], 0.0)

        # Extreme low cutoff (approve all)
        m_low = self.engine.evaluate_cutoff(cutoff=100)
        self.assertEqual(m_low["approved_count"], self.engine.total_applicants)
        self.assertEqual(m_low["approval_rate"], 1.0)

    def test_lgd_clamping(self):
        # Test LGD bounds
        m_clamped = self.engine.evaluate_cutoff(cutoff=480, lgd=1.50) # should clamp to 1.0
        m_100 = self.engine.evaluate_cutoff(cutoff=480, lgd=1.00)
        self.assertEqual(m_clamped["expected_loss"], m_100["expected_loss"])

    def test_simulate_sweep(self):
        sweep = self.engine.simulate_sweep(lgd=0.50, interest_margin=0.10, cutoff_min=380, cutoff_max=650, cutoff_step=10)
        self.assertGreater(len(sweep), 20)
        self.assertIn("Net_Profit", sweep.columns)
        self.assertIn("Bad_Rate", sweep.columns)
        self.assertIn("Approved_Count", sweep.columns)

    def test_bucket_breakdown(self):
        df_b, totals = self.engine.get_score_bucket_breakdown(lgd=0.50, ead=20000.0)
        self.assertEqual(len(df_b), 5)
        self.assertEqual(totals["Total_Count"], self.engine.total_applicants)
        self.assertGreater(totals["Total_Expected_Loss"], 0)


if __name__ == "__main__":
    unittest.main()
