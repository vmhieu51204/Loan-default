"""
Credit scorecard scaling, points allocation, and master scale generation sub-package.
"""

from src.scorecard.scaling import ScorecardScaler
from src.scorecard.master_scale import create_master_scale, create_decile_table

__all__ = ["ScorecardScaler", "create_master_scale", "create_decile_table"]
