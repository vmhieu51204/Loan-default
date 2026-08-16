"""
Settings and Configuration for Credit Scorecard Pipeline.
Centralizes data mappings, leakage lists, hyper-parameters, and scorecard scaling constants.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ScorecardConfig:
    """Configuration class for the end-to-end credit scorecard pipeline."""

    # --- Data Paths ---
    data_path: str = "ac.gz"
    output_dir: str = "outputs"

    # --- Target Mapping (1 = Good / Non-Default, 0 = Bad / Default) ---
    target_col: str = "loan_status"
    target_binary_col: str = "loan_status_binary"
    loan_status_mapping: Dict[str, int] = field(
        default_factory=lambda: {
            "Fully Paid": 1,
            "Current": 1,
            "In Grace Period": 1,
            "Late (16-30 days)": 0,
            "Late (31-120 days)": 0,
            "Charged Off": 0,
            "Default": 0,
            "Does not meet the credit policy. Status:Fully Paid": 1,
            "Does not meet the credit policy. Status:Charged Off": 0,
        }
    )

    # --- Data Leakage Columns to Drop ---
    leakage_cols: List[str] = field(
        default_factory=lambda: [
            "out_prncp",
            "out_prncp_inv",
            "total_pymnt",
            "total_pymnt_inv",
            "total_rec_prncp",
            "total_rec_int",
            "total_rec_late_fee",
            "recoveries",
            "collection_recovery_fee",
            "last_pymnt_d",
            "last_pymnt_amnt",
            "next_pymnt_d",
            "pymnt_plan",
            "last_fico_range_high",
            "last_fico_range_low",
            "last_credit_pull_d",
            "hardship_flag",
            "hardship_type",
            "hardship_reason",
            "hardship_status",
            "deferral_term",
            "hardship_amount",
            "hardship_start_date",
            "hardship_end_date",
            "payment_plan_start_date",
            "hardship_length",
            "hardship_dpd",
            "hardship_loan_status",
            "orig_projected_additional_accrued_interest",
            "hardship_payoff_balance_amount",
            "hardship_last_payment_amount",
            "debt_settlement_flag",
            "debt_settlement_flag_date",
            "settlement_status",
            "settlement_date",
            "settlement_amount",
            "settlement_percentage",
            "settlement_term",
            "loan_status",
            "url",
            "id",
            "member_id",
        ]
    )

    # --- Split Ratios (60% Train, 20% Val, 20% Test) ---
    test_size: float = 0.20
    val_size: float = 0.25  # 0.25 of 80% remaining = 20% of total
    random_state: int = 42

    # --- WoE Binning Parameters ---
    max_bins: int = 6
    min_samples_leaf: float = 0.05

    # --- Feature Selection Parameters ---
    iv_min: float = 0.02
    iv_max: float = 0.55
    corr_threshold: float = 0.70

    # --- Logistic Regression Hyperparameters ---
    c_penalty: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 1000
    class_weight: Optional[str] = "balanced"

    # --- Scorecard Scaling Constants ---
    pdo: float = 50.0  # Points to Double the Odds
    base_score: float = 600.0  # Base credit score benchmark
    base_odds: float = 50.0  # Target odds at base score (50:1 Good-to-Bad)

    # --- Master Scale Settings ---
    master_scale_bin_width: int = 50
    master_scale_min_score: int = 300
    master_scale_max_score: int = 900

    # --- Financial Simulation & Expected Loss Parameters ---
    ead: float = 20000.0  # Average Exposure at Default per loan ($)
    interest_margin: float = 0.10  # 10% profit margin per 'Good' loan (r)
    lgd_baseline: float = 0.50  # Baseline Loss Given Default (50% recovery / 50% loss severity)
    lgd_stressed: float = 1.00  # Stressed Loss Given Default (100% loss severity / 0% recovery)

    # --- Cutoff Optimization Settings ---
    cutoff_min: int = 380  # Score cutoff sweep lower bound
    cutoff_max: int = 650  # Score cutoff sweep upper bound
    cutoff_step: int = 5  # Score cutoff step size
