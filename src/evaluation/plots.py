"""
Plotting and visualization module for scorecard analytics and diagnostics.
"""

from typing import Any, Dict, List, Optional
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve

# Configure visual styling
sns.set_theme(style="whitegrid")


def _save_or_show(fig: plt.Figure, save_path: Optional[str] = None, show: bool = True) -> None:
    """Helper to save or display figure."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved figure to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_top_iv_features(
    iv_df: pd.DataFrame,
    top_n: int = 15,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plots horizontal bar chart of top features by Information Value (IV)."""
    top_df = iv_df.head(top_n).copy().sort_values(by="IV", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(top_df))
    ax.barh(top_df.index, top_df["IV"], color=colors)

    # Reference lines for IV rules of thumb
    ax.axvline(0.02, color="gray", linestyle="--", alpha=0.7, label="Min useful (0.02)")
    ax.axvline(0.1, color="blue", linestyle="--", alpha=0.7, label="Medium signal (0.10)")
    ax.axvline(0.3, color="green", linestyle="--", alpha=0.7, label="Strong signal (0.30)")
    ax.axvline(0.55, color="red", linestyle="--", alpha=0.7, label="Suspicious / Leakage (0.55)")

    ax.set_title(f"Top {top_n} Features by Information Value (IV)", fontsize=14, pad=12)
    ax.set_xlabel("Information Value (IV)", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.legend(loc="lower right")
    fig.tight_layout()

    _save_or_show(fig, save_path, show)


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plots the ROC Curve against random chance."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2.5, label=f"Scorecard Logistic (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Random Chance (AUC = 0.50)")

    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()

    _save_or_show(fig, save_path, show)


def plot_score_distribution(
    scores: pd.Series,
    y_true: pd.Series,
    bins: int = 30,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plots overlapping histograms and KDEs of credit scores for Good vs Bad borrowers."""
    df_plot = pd.DataFrame({"score": scores, "status": y_true.map({1: "Good (Non-Default)", 0: "Bad (Default)"})})

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        data=df_plot,
        x="score",
        hue="status",
        bins=bins,
        kde=True,
        palette={"Good (Non-Default)": "#2ca02c", "Bad (Default)": "#d62728"},
        alpha=0.5,
        ax=ax,
        stat="density",
        common_norm=False,
    )

    ax.set_title("Credit Score Distribution: Good vs Bad Borrowers", fontsize=14, pad=12)
    ax.set_xlabel("Credit Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    fig.tight_layout()

    _save_or_show(fig, save_path, show)


def plot_calibration_curve(
    y_true: pd.Series,
    raw_probas: np.ndarray,
    calib_probas: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plots reliability curves comparing Raw Logistic probabilities vs Isotonic Calibrated probabilities."""
    y_arr = np.array(y_true)
    prob_true_raw, prob_pred_raw = calibration_curve(y_arr, raw_probas, n_bins=n_bins)
    prob_true_cal, prob_pred_cal = calibration_curve(y_arr, calib_probas, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(prob_pred_raw, prob_true_raw, marker="o", linestyle="--", color="#ff7f0e", label="Raw Logistic Probs")
    ax.plot(prob_pred_cal, prob_true_cal, marker="s", linewidth=2.5, color="#2ca02c", label="Calibrated Probs (Isotonic)")
    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")

    ax.set_xlabel("Mean Predicted Probability of Good", fontsize=12)
    ax.set_ylabel("Empirical Fraction of Positives (Good)", fontsize=12)
    ax.set_title("Probability Calibration Reliability Diagram (Test Set)", fontsize=14, pad=12)
    ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout()

    _save_or_show(fig, save_path, show)


def plot_decile_calibration(
    scores: pd.Series,
    y_true: pd.Series,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plots dual-axis chart showing decile account volume (bars) and empirical bad rate (line).
    """
    df_eval = pd.DataFrame({"score": scores.values, "target": y_true.values})
    df_eval["decile"] = pd.qcut(df_eval["score"], 10, labels=False, duplicates="drop")

    agg_calib = df_eval.groupby("decile").agg(
        volume=("score", "count"),
        actual_bad_rate=("target", lambda x: 1.0 - x.mean()),
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Volume bars
    ax1.bar(agg_calib.index, agg_calib["volume"], color="#bcbd22", alpha=0.5, label="Volume")
    ax1.set_xlabel("Score Decile (0 = Riskiest, 9 = Safest)", fontsize=12)
    ax1.set_ylabel("Volume (Count)", fontsize=12)
    ax1.set_xticks(agg_calib.index)

    # Bad rate line
    ax2 = ax1.twinx()
    ax2.plot(agg_calib.index, agg_calib["actual_bad_rate"], color="#d62728", marker="o", linewidth=2.5, label="Empirical Bad Rate")
    ax2.set_ylabel("Default / Bad Rate", color="#d62728", fontsize=12)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.grid(False)

    plt.title("Scorecard Calibration: Volume vs Actual Bad Rate by Decile", fontsize=14, pad=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=11)
    fig.tight_layout()

    _save_or_show(fig, save_path, show)


def plot_master_scale(
    master_scale_df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plots Master Scale risk (Calibrated PD line) and volume (bars) across standard score bands."""
    df_plot = master_scale_df.copy()
    df_plot["Score_Band_Str"] = df_plot["Score_Band"].astype(str)
    df_plot = df_plot[df_plot["Volume"] > 0]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart (Volume)
    sns.barplot(data=df_plot, x="Score_Band_Str", y="Volume", color="#aec7e8", ax=ax1, label="Volume")
    ax1.set_xlabel("Score Band", fontsize=12)
    ax1.set_ylabel("Volume (Count)", fontsize=12)
    ax1.tick_params(axis="x", rotation=45)

    # Line chart (Calibrated PD)
    ax2 = ax1.twinx()
    sns.lineplot(
        data=df_plot,
        x="Score_Band_Str",
        y="Calibrated_PD",
        color="#d62728",
        marker="o",
        linewidth=3,
        ax=ax2,
        label="Calibrated PD",
    )
    ax2.set_ylabel("Probability of Default (PD)", color="#d62728", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.grid(False)

    top_limit = df_plot["Calibrated_PD"].max()
    if pd.isna(top_limit) or top_limit == 0:
        top_limit = 1.0
    else:
        top_limit *= 1.2
    ax2.set_ylim(0, top_limit)

    plt.title("Scorecard Master Scale: Calibrated PD vs Account Volume", fontsize=14, pad=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=11)
    fig.tight_layout()

    _save_or_show(fig, save_path, show)


def plot_cutoff_profit_curve(
    sim_combined_df: pd.DataFrame,
    scenario_summaries: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plots Net Economic Value / Net Profit curves across score cutoffs for different LGD scenarios.
    Highlights the optimal cutoff point (peak profit) for each scenario.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    palette = {"Baseline (LGD=45%)": "#1f77b4", "Stressed (LGD=70%)": "#d62728"}
    markers = {"Baseline (LGD=45%)": "o", "Stressed (LGD=70%)": "s"}

    scenarios = sim_combined_df["Scenario"].unique()
    for sc in scenarios:
        sc_data = sim_combined_df[sim_combined_df["Scenario"] == sc]
        color = palette.get(sc, None)
        marker = markers.get(sc, "o")

        ax.plot(
            sc_data["Cutoff"],
            sc_data["Net_Profit"] / 1e6,
            marker=marker,
            linewidth=2.5,
            markersize=5,
            label=f"{sc} Net Profit",
            color=color,
        )

        # Highlight optimal cutoff peak
        opt_row = sc_data.loc[sc_data["Net_Profit"].idxmax()]
        opt_cutoff = int(opt_row["Cutoff"])
        max_profit_mil = opt_row["Net_Profit"] / 1e6

        ax.plot(opt_cutoff, max_profit_mil, marker="*", markersize=14, color=color, markeredgecolor="black")
        ax.annotate(
            f"Optimal C* = {opt_cutoff}\nMax Profit: ${max_profit_mil:.2f}M\n(Bad Rate: {opt_row['Portfolio_Bad_Rate']:.1%})",
            xy=(opt_cutoff, max_profit_mil),
            xytext=(opt_cutoff + 10, max_profit_mil * 0.92),
            arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.5, alpha=0.9),
        )

    ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
    ax.set_title("Credit Policy Optimization: Net Profit Frontier across Score Cutoffs", fontsize=14, pad=12)
    ax.set_xlabel("Credit Score Cutoff (Approve if Score >= Cutoff)", fontsize=12)
    ax.set_ylabel("Portfolio Net Profit ($ Millions)", fontsize=12)
    ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()

    _save_or_show(fig, save_path, show)


def plot_acceptance_vs_bad_rate(
    sim_df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plots dual-axis trade-off chart showing Acceptance Rate vs Portfolio Bad Rate across score cutoffs.
    """
    # Use baseline scenario if multiple present
    if "Scenario" in sim_df.columns:
        scenarios = sim_df["Scenario"].unique()
        df_plot = sim_df[sim_df["Scenario"] == scenarios[0]].copy()
        title_suffix = f" ({scenarios[0]})"
    else:
        df_plot = sim_df.copy()
        title_suffix = ""

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Acceptance Rate (Blue)
    ax1.plot(df_plot["Cutoff"], df_plot["Acceptance_Rate"], color="#1f77b4", linewidth=2.5, marker="o", label="Acceptance Rate (%)")
    ax1.set_xlabel("Credit Score Cutoff", fontsize=12)
    ax1.set_ylabel("Portfolio Acceptance / Approval Rate", color="#1f77b4", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Bad Rate (Red)
    ax2 = ax1.twinx()
    ax2.plot(df_plot["Cutoff"], df_plot["Portfolio_Bad_Rate"], color="#d62728", linewidth=2.5, marker="s", linestyle="--", label="Approved Bad Rate (%)")
    ax2.set_ylabel("Approved Portfolio Bad Rate (%)", color="#d62728", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.grid(False)

    plt.title(f"Credit Strategy Trade-off: Acceptance Rate vs Portfolio Bad Rate{title_suffix}", fontsize=14, pad=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=11)
    fig.tight_layout()

    _save_or_show(fig, save_path, show)


def plot_strategy_dashboard_profit_vs_risk(
    sim_df: pd.DataFrame,
    title: str = "Strategy Dashboard: Profit vs. Risk",
    bar_color: str = "#a1d99b",
    line_color: str = "red",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Generates the dual-axis Strategy Dashboard matching output2.png:
    - Left Axis (Green): Projected Net Profit ($) as vertical bars.
    - Right Axis (Red): Portfolio Bad Rate (%) as a bold line.
    """
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Calculate bar width based on step size
    cutoffs = sim_df["Cutoff"].values
    step = np.median(np.diff(cutoffs)) if len(cutoffs) > 1 else 5
    bar_width = step * 0.75

    # 1. Left Axis: Projected Profit (Bars)
    profits = sim_df["Net_Profit"].values
    ax1.bar(
        cutoffs,
        profits,
        width=bar_width,
        color=bar_color,
        alpha=0.8,
        label="Projected Profit ($)",
        edgecolor="none",
    )
    ax1.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("Score Cutoff", fontsize=12, labelpad=8)
    ax1.set_ylabel("Projected Profit ($)", color="#2e7d32", fontsize=12, labelpad=8)
    ax1.tick_params(axis="y", labelcolor="#2e7d32", labelsize=11)
    ax1.tick_params(axis="x", labelsize=11)

    min_p = np.nanmin(profits) if len(profits) > 0 else 0
    max_p = np.nanmax(profits) if len(profits) > 0 else 0
    if min_p < 0:
        ax1.set_ylim(min_p * 1.10, max_p * 1.10)
    else:
        ax1.set_ylim(0, max_p * 1.10)

    # 2. Right Axis: Bad Rate (Red Line)
    ax2 = ax1.twinx()
    bad_rates = sim_df["Portfolio_Bad_Rate"].values
    ax2.plot(
        cutoffs,
        bad_rates,
        color=line_color,
        linewidth=3.0,
        label="Bad Rate",
    )
    ax2.set_ylabel("Bad Rate", color="#d32f2f", fontsize=12, labelpad=8)
    ax2.tick_params(axis="y", labelcolor="#d32f2f", labelsize=11)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.grid(False)

    top_bad = np.nanmax(bad_rates) if len(bad_rates) > 0 else 0.15
    ax2.set_ylim(0, top_bad * 1.15 if top_bad > 0 else 0.20)

    plt.title(title, fontsize=14, pad=14)
    fig.tight_layout()

    _save_or_show(fig, save_path, show)
