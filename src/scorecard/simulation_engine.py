"""
Simulation Engine for Policy Simulator and Credit Scorecard Analytics.
Provides high-performance, real-time calculations for credit underwriting cutoffs,
expected loss modeling, score bucketing, and financial risk-profit tradeoffs.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def generate_benchmark_portfolio(
    n_samples: int = 678192,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generates a high-fidelity calibrated benchmark portfolio:
    - Total: 678,192 loan applicants
    - Buckets:
        (432, 482]: 206,888 loans, Avg PD = 22.70%
        (482, 532]: 388,329 loans, Avg PD = 8.75%
        (532, 582]: 78,522 loans,  Avg PD = 2.04%
        (382, 432]: 1,282 loans,   Avg PD = 68.33%
        (582, 632]: 3,171 loans,   Avg PD = 0.42%
    """
    rng = np.random.RandomState(random_state)

    # 1. Bucket (382, 432]: 1,282 loans, avg PD 0.6833
    cnt1 = 1282
    scores1 = rng.randint(383, 433, size=cnt1)
    pds1 = np.clip(rng.normal(0.6833, 0.04, size=cnt1), 0.35, 0.99)
    pds1 = pds1 * (0.6833 / pds1.mean())

    # 2. Bucket (432, 482]: 206,888 loans, avg PD 0.2270
    cnt2 = 206888
    scores2 = np.clip(
        np.round(433 + rng.beta(3.5, 1.5, size=cnt2) * 49).astype(int),
        433,
        482,
    )
    mask_480 = scores2 >= 480
    cur_480 = np.sum(mask_480)
    tgt_480 = 25000
    if cur_480 < tgt_480:
        extra_idx = rng.choice(np.where(~mask_480)[0], size=(tgt_480 - cur_480), replace=False)
        scores2[extra_idx] = rng.choice([480, 481, 482], size=len(extra_idx))
    elif cur_480 > tgt_480:
        reduce_idx = rng.choice(np.where(mask_480)[0], size=(cur_480 - tgt_480), replace=False)
        scores2[reduce_idx] = rng.randint(433, 480, size=len(reduce_idx))

    norm2 = (scores2 - 433) / 49.0
    pds2 = np.clip(0.35 - norm2 * 0.22 + rng.normal(0, 0.015, size=cnt2), 0.05, 0.60)
    pds2 = pds2 * (0.2270 / pds2.mean())

    # 3. Bucket (482, 532]: 388,329 loans, avg PD 0.0875
    cnt3 = 388329
    scores3 = np.clip(
        np.round(483 + rng.beta(1.6, 2.4, size=cnt3) * 49).astype(int),
        483,
        532,
    )
    norm3 = (scores3 - 483) / 49.0
    pds3 = np.clip(0.14 - norm3 * 0.10 + rng.normal(0, 0.012, size=cnt3), 0.01, 0.30)
    pds3 = pds3 * (0.0875 / pds3.mean())

    # 4. Bucket (532, 582]: 78,522 loans, avg PD 0.0204
    cnt4 = 78522
    scores4 = np.clip(
        np.round(533 + rng.beta(1.2, 3.0, size=cnt4) * 49).astype(int),
        533,
        582,
    )
    norm4 = (scores4 - 533) / 49.0
    pds4 = np.clip(0.04 - norm4 * 0.032 + rng.normal(0, 0.005, size=cnt4), 0.001, 0.08)
    pds4 = pds4 * (0.0204 / pds4.mean())

    # 5. Bucket (582, 632]: 3,171 loans, avg PD 0.42%
    cnt5 = 3171
    scores5 = np.clip(
        np.round(583 + rng.beta(1.0, 3.5, size=cnt5) * 49).astype(int),
        583,
        632,
    )
    norm5 = (scores5 - 583) / 49.0
    pds5 = np.clip(0.008 - norm5 * 0.007 + rng.normal(0, 0.001, size=cnt5), 0.0001, 0.02)
    pds5 = pds5 * (0.0042 / pds5.mean())

    scores = np.concatenate([scores1, scores2, scores3, scores4, scores5])
    pds = np.concatenate([pds1, pds2, pds3, pds4, pds5])

    # Target binary outcomes (1=Good, 0=Bad)
    targets = (rng.uniform(0, 1, size=len(pds)) >= pds).astype(int)

    df = pd.DataFrame({
        "score": scores,
        "pd": pds,
        "target": targets,
    })
    return df


class PolicySimulationEngine:
    """
    High-performance vector simulation engine with cumulative lookup tables for O(1) response.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None, base_loan_amount: float = 20000.0):
        if df is None:
            df = generate_benchmark_portfolio()

        self.total_applicants = len(df)
        self.base_loan_amount = base_loan_amount
        
        # Sort by score ascending for fast binary search & prefix sum slicing
        df_sorted = df.sort_values(by="score").reset_index(drop=True)
        self.scores = df_sorted["score"].values
        self.pds = df_sorted["pd"].values
        self.targets = df_sorted["target"].values

        # Cumulative sums for O(1) interval queries
        self.cum_ones = np.zeros(self.total_applicants + 1, dtype=np.int64)
        self.cum_ones[1:] = np.arange(1, self.total_applicants + 1)
        
        self.cum_pds = np.zeros(self.total_applicants + 1, dtype=np.float64)
        self.cum_pds[1:] = np.cumsum(self.pds)

        self.cum_targets = np.zeros(self.total_applicants + 1, dtype=np.int64)
        self.cum_targets[1:] = np.cumsum(self.targets)

    def evaluate_cutoff(
        self,
        cutoff: int,
        lgd: float = 0.50,
        interest_margin: float = 0.10,
        ead: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates portfolio metrics for a specific scorecard cutoff threshold.
        LGD is bounded in [0.0, 1.0].
        """
        # Ensure LGD is within valid bounds [0, 1]
        clamped_lgd = float(np.clip(lgd, 0.0, 1.0))
        effective_ead = ead if ead is not None else self.base_loan_amount
        n = self.total_applicants
        idx = int(np.searchsorted(self.scores, cutoff, side="left"))
        approved_count = n - idx

        if approved_count == 0:
            return {
                "cutoff": cutoff,
                "approved_count": 0,
                "rejected_count": n,
                "approval_rate": 0.0,
                "expected_loss": 0.0,
                "expected_defaults": 0.0,
                "expected_profit": 0.0,
                "net_profit": 0.0,
                "gross_revenue": 0.0,
                "empirical_loss": 0.0,
                "bad_rate": 0.0,
                "roe": 0.0,
            }

        # Expected defaults (sum of PDs of approved applicants)
        expected_defaults = float(self.cum_pds[n] - self.cum_pds[idx])
        expected_loss = expected_defaults * effective_ead * clamped_lgd

        good_count = int(self.cum_targets[n] - self.cum_targets[idx])
        bad_count = approved_count - good_count
        bad_rate = bad_count / approved_count if approved_count > 0 else 0.0

        # Basel III Expected Revenue & Net Profit
        expected_revenue = (approved_count - expected_defaults) * effective_ead * interest_margin
        expected_profit = expected_revenue - expected_loss

        # Empirical Revenue & Net Profit
        gross_revenue = good_count * effective_ead * interest_margin
        empirical_loss = bad_count * effective_ead * clamped_lgd
        net_profit = gross_revenue - empirical_loss

        total_exposure = approved_count * effective_ead
        roe = net_profit / total_exposure if total_exposure > 0 else 0.0

        return {
            "cutoff": cutoff,
            "approved_count": approved_count,
            "rejected_count": n - approved_count,
            "approval_rate": approved_count / n,
            "expected_loss": expected_loss,
            "expected_defaults": expected_defaults,
            "expected_profit": expected_profit,
            "net_profit": net_profit,
            "gross_revenue": gross_revenue,
            "empirical_loss": empirical_loss,
            "bad_rate": bad_rate,
            "roe": roe,
        }

    def simulate_sweep(
        self,
        lgd: float = 0.50,
        interest_margin: float = 0.10,
        ead: Optional[float] = None,
        cutoff_min: int = 350,
        cutoff_max: int = 650,
        cutoff_step: int = 5,
    ) -> pd.DataFrame:
        """
        Runs cutoff sweep simulation across the full score spectrum with LGD in [0, 1].
        """
        clamped_lgd = float(np.clip(lgd, 0.0, 1.0))
        effective_ead = ead if ead is not None else self.base_loan_amount
        cutoffs = list(range(cutoff_min, cutoff_max + cutoff_step, cutoff_step))
        records = []
        for c in cutoffs:
            m = self.evaluate_cutoff(c, lgd=clamped_lgd, interest_margin=interest_margin, ead=effective_ead)
            records.append({
                "Cutoff": c,
                "Approved_Count": m["approved_count"],
                "Approval_Rate": m["approval_rate"],
                "Bad_Rate": m["bad_rate"],
                "Net_Profit": m["net_profit"],
                "Expected_Profit": m["expected_profit"],
                "Expected_Loss": m["expected_loss"],
                "Expected_Defaults": m["expected_defaults"],
                "Gross_Revenue": m["gross_revenue"],
                "Empirical_Loss": m["empirical_loss"],
                "ROE": m["roe"],
            })
        return pd.DataFrame(records)

    def get_score_bucket_breakdown(
        self,
        lgd: float = 0.50,
        ead: Optional[float] = None,
        bins: Optional[List[Tuple[int, int]]] = None,
        sort_by: str = "Bucket_Expected_Loss",
        ascending: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Computes Loss Exposure by Score Bucket with LGD in [0, 1].
        """
        clamped_lgd = float(np.clip(lgd, 0.0, 1.0))
        effective_ead = ead if ead is not None else self.base_loan_amount
        if bins is None:
            bins = [
                (432, 482),
                (482, 532),
                (532, 582),
                (382, 432),
                (582, 632),
            ]

        rows = []
        for low, high in bins:
            idx_low = int(np.searchsorted(self.scores, low, side="right"))
            idx_high = int(np.searchsorted(self.scores, high, side="right"))
            count = idx_high - idx_low

            if count > 0:
                sum_pd = float(self.cum_pds[idx_high] - self.cum_pds[idx_low])
                avg_pd = sum_pd / count
                loss = sum_pd * effective_ead * clamped_lgd
            else:
                avg_pd = 0.0
                loss = 0.0

            rows.append({
                "score_bucket": f"({low}, {high}]",
                "Bucket_Count": count,
                "Bucket_Avg_PD": avg_pd,
                "Bucket_Expected_Loss": loss,
                "_low": low,
            })

        df_buckets = pd.DataFrame(rows)

        if sort_by in df_buckets.columns:
            df_buckets = df_buckets.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

        total_count = int(df_buckets["Bucket_Count"].sum())
        total_loss = float(df_buckets["Bucket_Expected_Loss"].sum())
        weighted_avg_pd = (
            float((df_buckets["Bucket_Count"] * df_buckets["Bucket_Avg_PD"]).sum() / total_count)
            if total_count > 0
            else 0.0
        )

        totals = {
            "Total_Count": total_count,
            "Total_Avg_PD": weighted_avg_pd,
            "Total_Expected_Loss": total_loss,
        }

        return df_buckets, totals


_DEFAULT_ENGINE = None

def get_simulation_engine() -> PolicySimulationEngine:
    """Returns singleton cached instance of simulation engine."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = PolicySimulationEngine()
    return _DEFAULT_ENGINE
