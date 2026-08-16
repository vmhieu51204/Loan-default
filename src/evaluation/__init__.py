"""
Evaluation metrics and plotting sub-package for credit scorecard validation.
"""

from src.evaluation.metrics import (
    evaluate_discrimination,
    evaluate_calibration_metrics,
)
from src.evaluation.plots import (
    plot_roc_curve,
    plot_calibration_curve,
    plot_score_distribution,
    plot_decile_calibration,
    plot_master_scale,
    plot_top_iv_features,
    plot_cutoff_profit_curve,
    plot_acceptance_vs_bad_rate,
    plot_strategy_dashboard_profit_vs_risk,
)

__all__ = [
    "evaluate_discrimination",
    "evaluate_calibration_metrics",
    "plot_roc_curve",
    "plot_calibration_curve",
    "plot_score_distribution",
    "plot_decile_calibration",
    "plot_master_scale",
    "plot_top_iv_features",
    "plot_cutoff_profit_curve",
    "plot_acceptance_vs_bad_rate",
    "plot_strategy_dashboard_profit_vs_risk",
]
