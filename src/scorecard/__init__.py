"""
Credit scorecard scaling, points allocation, and master scale generation sub-package.
"""

from src.scorecard.scaling import ScorecardScaler
from src.scorecard.master_scale import create_master_scale, create_decile_table
from src.scorecard.strategy import (
    calculate_expected_loss,
    simulate_cutoff_strategy,
    find_optimal_cutoff,
    find_risk_constrained_cutoff,
    compare_lgd_scenarios,
)

__all__ = [
    "ScorecardScaler",
    "create_master_scale",
    "create_decile_table",
    "calculate_expected_loss",
    "simulate_cutoff_strategy",
    "find_optimal_cutoff",
    "find_risk_constrained_cutoff",
    "compare_lgd_scenarios",
]
