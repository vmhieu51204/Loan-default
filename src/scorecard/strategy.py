"""
Credit Policy Strategy, Basel III Expected Loss, and Net Profit Cutoff Optimization Module.
Implements financial portfolio simulation under various Loss Given Default (LGD) scenarios.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def calculate_expected_loss(
    pds: Union[np.ndarray, pd.Series],
    ead: float = 20000.0,
    lgd: float = 0.45,
) -> np.ndarray:
    """
    Calculates Basel III Expected Loss (EL) per loan:
    EL_i = PD_i * EAD_i * LGD_i

    Args:
        pds: Calibrated Probability of Default array (PD_i in [0, 1]).
        ead: Exposure at Default ($ amount).
        lgd: Loss Given Default (loss severity ratio in [0, 1]).

    Returns:
        np.ndarray: Expected Loss in dollars for each loan.
    """
    pd_arr = np.asarray(pds)
    return pd_arr * ead * lgd


def simulate_cutoff_strategy(
    scores: Union[np.ndarray, pd.Series],
    targets: Union[np.ndarray, pd.Series],
    pds: Union[np.ndarray, pd.Series],
    ead: float = 20000.0,
    interest_margin: float = 0.10,
    lgd: float = 0.45,
    cutoff_min: int = 380,
    cutoff_max: int = 650,
    cutoff_step: int = 5,
    scenario_name: str = "Baseline (LGD=45%)",
) -> pd.DataFrame:
    """
    Simulates portfolio financial performance across a continuous sweep of credit score cutoffs.
    
    For each cutoff C:
      - Applicants with Score >= C are approved.
      - Revenue = sum_{approved, Good} (EAD * r)
      - Loss = sum_{approved, Bad} (EAD * LGD)
      - Net Profit Pi(C) = Revenue - Loss
      - Expected Profit = sum_{approved} [ EAD * r * (1 - PD_i) - EAD * LGD * PD_i ]

    Args:
        scores: Predicted integer credit scores.
        targets: Actual ground-truth binary targets (1=Good, 0=Bad).
        pds: Calibrated default probabilities (PD).
        ead: Average Exposure at Default per loan.
        interest_margin: Interest profit margin per Good borrower (r).
        lgd: Loss Given Default ratio.
        cutoff_min: Lower bound for score cutoff sweep.
        cutoff_max: Upper bound for score cutoff sweep.
        cutoff_step: Step increment for cutoff iteration.
        scenario_name: Label for the simulation scenario.

    Returns:
        pd.DataFrame: Cutoff strategy simulation performance table.
    """
    s_arr = np.asarray(scores)
    y_arr = np.asarray(targets)
    pd_arr = np.asarray(pds)
    total_applicants = len(s_arr)

    cutoffs = list(range(cutoff_min, cutoff_max + cutoff_step, cutoff_step))
    rows = []

    for cutoff in cutoffs:
        approved_mask = s_arr >= cutoff
        approved_count = int(np.sum(approved_mask))

        if approved_count == 0:
            rows.append({
                "Scenario": scenario_name,
                "Cutoff": cutoff,
                "Approved_Count": 0,
                "Rejected_Count": total_applicants,
                "Acceptance_Rate": 0.0,
                "Good_Count": 0,
                "Bad_Count": 0,
                "Portfolio_Bad_Rate": 0.0,
                "Gross_Revenue": 0.0,
                "Total_Empirical_Loss": 0.0,
                "Total_Expected_Loss": 0.0,
                "Net_Profit": 0.0,
                "Expected_Net_Profit": 0.0,
                "Net_Profit_Per_Loan": 0.0,
                "Return_On_Exposure": 0.0,
            })
            continue

        good_count = int(np.sum(approved_mask & (y_arr == 1)))
        bad_count = int(np.sum(approved_mask & (y_arr == 0)))
        bad_rate = bad_count / approved_count
        acceptance_rate = approved_count / total_applicants

        # Empirical Revenue and Loss
        gross_revenue = good_count * ead * interest_margin
        empirical_loss = bad_count * ead * lgd
        net_profit = gross_revenue - empirical_loss

        # Model Expected Loss based on calibrated PD
        expected_loss = np.sum(pd_arr[approved_mask]) * ead * lgd
        expected_revenue = np.sum(1.0 - pd_arr[approved_mask]) * ead * interest_margin
        expected_net_profit = expected_revenue - expected_loss

        total_exposure = approved_count * ead
        net_profit_per_loan = net_profit / approved_count if approved_count > 0 else 0.0
        return_on_exposure = net_profit / total_exposure if total_exposure > 0 else 0.0

        rows.append({
            "Scenario": scenario_name,
            "Cutoff": cutoff,
            "Approved_Count": approved_count,
            "Rejected_Count": total_applicants - approved_count,
            "Acceptance_Rate": acceptance_rate,
            "Good_Count": good_count,
            "Bad_Count": bad_count,
            "Portfolio_Bad_Rate": bad_rate,
            "Gross_Revenue": gross_revenue,
            "Total_Empirical_Loss": empirical_loss,
            "Total_Expected_Loss": expected_loss,
            "Net_Profit": net_profit,
            "Expected_Net_Profit": expected_net_profit,
            "Net_Profit_Per_Loan": net_profit_per_loan,
            "Return_On_Exposure": return_on_exposure,
        })

    sim_df = pd.DataFrame(rows)
    return sim_df


def find_optimal_cutoff(sim_df: pd.DataFrame) -> pd.Series:
    """
    Finds the score cutoff that strictly maximizes portfolio Net Profit.
    """
    if sim_df.empty or sim_df["Approved_Count"].sum() == 0:
        raise ValueError("Simulation DataFrame is empty or contains no approved loans.")
    max_idx = sim_df["Net_Profit"].idxmax()
    return sim_df.loc[max_idx]


def find_risk_constrained_cutoff(
    sim_df: pd.DataFrame, max_bad_rate: float = 0.05
) -> Optional[pd.Series]:
    """
    Finds the optimal cutoff subject to a maximum risk appetite constraint (e.g. Bad Rate <= 5.0%).
    """
    eligible = sim_df[(sim_df["Portfolio_Bad_Rate"] <= max_bad_rate) & (sim_df["Approved_Count"] > 0)]
    if eligible.empty:
        return None
    max_idx = eligible["Net_Profit"].idxmax()
    return eligible.loc[max_idx]


def compare_lgd_scenarios(
    scores: Union[np.ndarray, pd.Series],
    targets: Union[np.ndarray, pd.Series],
    pds: Union[np.ndarray, pd.Series],
    ead: float = 20000.0,
    interest_margin: float = 0.10,
    scenarios: Optional[Dict[str, float]] = None,
    cutoff_min: int = 380,
    cutoff_max: int = 650,
    cutoff_step: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Runs multi-scenario stress-testing across different Loss Given Default (LGD) assumptions.

    Args:
        scores: Predicted credit scores.
        targets: Actual loan outcomes.
        pds: Calibrated default probabilities.
        ead: Exposure at Default.
        interest_margin: Interest Margin (r).
        scenarios: Dict of {scenario_name: lgd_value}.
        cutoff_min: Cutoff minimum.
        cutoff_max: Cutoff maximum.
        cutoff_step: Step increment.

    Returns:
        Tuple of (combined_sim_df, scenario_summaries_dict).
    """
    if scenarios is None:
        scenarios = {
            "Baseline (LGD=45%)": 0.45,
            "Stressed (LGD=70%)": 0.70,
        }

    all_dfs = []
    summaries = {}

    for name, lgd in scenarios.items():
        sim = simulate_cutoff_strategy(
            scores=scores,
            targets=targets,
            pds=pds,
            ead=ead,
            interest_margin=interest_margin,
            lgd=lgd,
            cutoff_min=cutoff_min,
            cutoff_max=cutoff_max,
            cutoff_step=cutoff_step,
            scenario_name=name,
        )
        all_dfs.append(sim)

        optimal_row = find_optimal_cutoff(sim)
        constrained_row_3pct = find_risk_constrained_cutoff(sim, max_bad_rate=0.03)
        constrained_row_5pct = find_risk_constrained_cutoff(sim, max_bad_rate=0.05)

        summaries[name] = {
            "lgd": lgd,
            "simulation_df": sim,
            "optimal_cutoff": int(optimal_row["Cutoff"]),
            "max_profit": float(optimal_row["Net_Profit"]),
            "optimal_acceptance_rate": float(optimal_row["Acceptance_Rate"]),
            "optimal_bad_rate": float(optimal_row["Portfolio_Bad_Rate"]),
            "optimal_approved_count": int(optimal_row["Approved_Count"]),
            "constrained_3pct": constrained_row_3pct.to_dict() if constrained_row_3pct is not None else None,
            "constrained_5pct": constrained_row_5pct.to_dict() if constrained_row_5pct is not None else None,
        }

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df, summaries
