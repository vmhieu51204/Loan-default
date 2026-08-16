"""
Credit Risk Policy Simulator & Strategy Optimization Dashboard (Dark Theme)
High-performance interactive credit underwriting cutoff simulator with Basel III Expected Loss modeling
and companion Model Diagnostics in a sleek dark theme.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add current working directory to sys.path
sys.path.insert(0, os.path.abspath("."))

from src.scorecard.simulation_engine import (
    PolicySimulationEngine,
    generate_benchmark_portfolio,
    get_simulation_engine,
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & DARK THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Policy Simulator",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DARK_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Main App Background (Dark Navy / Slate) */
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
    }

    /* Top Header Banner (Dark Purple Glow) */
    .main-header {
        background: linear-gradient(135deg, #1A0F1D 0%, #2E1332 50%, #4A1B50 100%);
        color: #FFFFFF;
        padding: 14px 22px;
        border-radius: 8px;
        margin-bottom: 14px;
        border: 1px solid #5C2663;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4), 0 0 12px rgba(92, 38, 99, 0.25);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .main-header h1 {
        font-family: 'Outfit', sans-serif;
        font-size: 21px;
        font-weight: 700;
        margin: 0;
        color: #FFFFFF;
        letter-spacing: -0.3px;
    }
    .main-header p {
        margin: 3px 0 0 0;
        font-size: 12px;
        color: #F1D4E7;
    }

    /* Left Policy Control Panel (Dark Gold / Amber Slate) */
    .control-panel-container {
        background-color: #16140F;
        border: 1.5px solid #45361D;
        border-radius: 8px;
        padding: 16px 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.3);
    }
    .control-panel-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 15px;
        color: #FBD38D;
        margin-bottom: 12px;
        padding-bottom: 5px;
        border-bottom: 1.5px solid #45361D;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .widget-box {
        background-color: #201B14;
        border: 1px solid #4A3A22;
        border-radius: 6px;
        padding: 10px 12px 8px 12px;
        margin-bottom: 14px;
    }
    .widget-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 13px;
        color: #ECC94B;
        margin-bottom: 6px;
    }

    /* KPI Summary Cards (Dark Magenta / Rose Metallic) */
    .kpi-card-outer {
        background-color: #161019;
        border: 1.5px solid #4A2346;
        border-radius: 8px;
        padding: 6px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card-outer:hover {
        transform: translateY(-2px);
        border-color: #8B3D84;
    }
    .kpi-card-inner {
        background: linear-gradient(180deg, #2B1428 0%, #220F20 100%);
        border-radius: 5px;
        padding: 14px 6px;
        text-align: center;
    }
    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 30px;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.05;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.5);
    }
    .kpi-title {
        font-size: 12px;
        font-weight: 700;
        color: #E2B0D8;
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }

    /* Main Chart Card Wrapper (Dark Glassmorphic) */
    .chart-wrapper {
        background-color: #131722;
        border: 1.5px solid #232A3B;
        border-radius: 8px;
        padding: 14px 16px 8px 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        margin-top: 10px;
    }

    /* Tabs Dark Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #131722;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #232A3B;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
        color: #94A3B8;
        background-color: transparent;
        padding: 0 18px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2D1432 !important;
        color: #F3D5E8 !important;
        border: 1px solid #5C2663;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }

    /* Number input & Slider tweaks */
    div[data-testid="stNumberInput"] {
        margin-bottom: 4px;
    }
    div[data-testid="stSlider"] {
        margin-top: -6px;
    }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. DATA ENGINE INITIALIZATION & CACHING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_simulation_engine() -> PolicySimulationEngine:
    """Cached initialization of the high-performance simulation engine."""
    return get_simulation_engine()


engine = load_simulation_engine()


# -----------------------------------------------------------------------------
# 3. STATE INITIALIZATION & ROBUST TWO-WAY SYNCHRONIZATION (LGD IN [0, 1])
# -----------------------------------------------------------------------------
if "cutoff_num" not in st.session_state:
    st.session_state.cutoff_num = 480
if "cutoff_slide" not in st.session_state:
    st.session_state.cutoff_slide = 480

if "lgd_num" not in st.session_state:
    st.session_state.lgd_num = 0.50
if "lgd_slide" not in st.session_state:
    st.session_state.lgd_slide = 0.50

if "margin_num" not in st.session_state:
    st.session_state.margin_num = 0.15
if "margin_slide" not in st.session_state:
    st.session_state.margin_slide = 0.15

if "ead_val" not in st.session_state:
    st.session_state.ead_val = 20000

# Bidirectional sync callbacks (Strictly bounding LGD in [0, 1])
def on_cutoff_num_change():
    st.session_state.cutoff_slide = int(st.session_state.cutoff_num)

def on_cutoff_slide_change():
    st.session_state.cutoff_num = int(st.session_state.cutoff_slide)

def on_lgd_num_change():
    st.session_state.lgd_num = max(0.00, min(1.00, round(float(st.session_state.lgd_num), 2)))
    st.session_state.lgd_slide = st.session_state.lgd_num

def on_lgd_slide_change():
    st.session_state.lgd_slide = max(0.00, min(1.00, round(float(st.session_state.lgd_slide), 2)))
    st.session_state.lgd_num = st.session_state.lgd_slide

def on_margin_num_change():
    st.session_state.margin_slide = round(float(st.session_state.margin_num), 2)

def on_margin_slide_change():
    st.session_state.margin_num = round(float(st.session_state.margin_slide), 2)

def on_preset_change():
    p = st.session_state.preset_select
    if "Benchmark" in p:
        st.session_state.cutoff_num = 480
        st.session_state.cutoff_slide = 480
        st.session_state.lgd_num = 0.50
        st.session_state.lgd_slide = 0.50
        st.session_state.margin_num = 0.15
        st.session_state.margin_slide = 0.15
    elif "Standard Recovery" in p:
        st.session_state.cutoff_num = 480
        st.session_state.cutoff_slide = 480
        st.session_state.lgd_num = 0.50
        st.session_state.lgd_slide = 0.50
        st.session_state.margin_num = 0.10
        st.session_state.margin_slide = 0.10
    elif "Full Write-Off" in p:
        st.session_state.cutoff_num = 510
        st.session_state.cutoff_slide = 510
        st.session_state.lgd_num = 1.00
        st.session_state.lgd_slide = 1.00
        st.session_state.margin_num = 0.10
        st.session_state.margin_slide = 0.10
    elif "Aggressive" in p:
        st.session_state.cutoff_num = 440
        st.session_state.cutoff_slide = 440
        st.session_state.lgd_num = 0.45
        st.session_state.lgd_slide = 0.45
        st.session_state.margin_num = 0.14
        st.session_state.margin_slide = 0.14
    elif "Conservative" in p:
        st.session_state.cutoff_num = 520
        st.session_state.cutoff_slide = 520
        st.session_state.lgd_num = 0.60
        st.session_state.lgd_slide = 0.60
        st.session_state.margin_num = 0.10
        st.session_state.margin_slide = 0.10


# -----------------------------------------------------------------------------
# 4. TOP NAVIGATION & HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <div>
            <h1>📊 Credit Risk Policy Simulator & Strategy Optimization</h1>
            <p>Interactive Score Cutoff Strategy, Basel III Expected Loss & Profit vs. Risk Frontier</p>
        </div>
        <div style="text-align: right; font-size: 12px;">
            <span style="background: rgba(255,255,255,0.12); padding: 5px 12px; border-radius: 12px; font-weight: 600; color: #F3D5E8; border: 1px solid rgba(255,255,255,0.15);">
                Portfolio: 678,192 Loans | Basel III Standard
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs([
    "🎯 Policy Simulator (Strategy Frontier)",
    "📈 Model Performance & Calibration (0.72 AUC)",
])


# =============================================================================
# TAB 1: POLICY SIMULATOR INTERFACE (DARK THEME, LGD IN [0, 1])
# =============================================================================
with tab1:
    col_left_panel, col_right_dashboard = st.columns([1, 3.8], gap="medium")

    # -------------------------------------------------------------------------
    # LEFT PANEL: POLICY CONTROLS (LGD Constrained in [0.00, 1.00])
    # -------------------------------------------------------------------------
    with col_left_panel:
        st.markdown(
            """
            <div class="control-panel-container">
                <div class="control-panel-title">Policy Parameters</div>
            """,
            unsafe_allow_html=True,
        )

        st.selectbox(
            "Scenario Preset",
            [
                "📌 Benchmark (Cutoff 480 / LGD 50% / 15%)",
                "🏦 Standard Recovery (LGD 50% / 10%)",
                "⚠️ Full Write-Off Shock (LGD 100% / 10%)",
                "🚀 Aggressive Expansion (Cutoff 440 / 14%)",
                "🔒 Conservative Policy (Cutoff 520 / 10%)",
                "⚙️ Custom Parameters",
            ],
            index=0,
            key="preset_select",
            on_change=on_preset_change,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # 1. Score_Cutoff Control Box
        st.markdown(
            """
            <div class="widget-box">
                <div class="widget-title">Score_Cutoff</div>
            """,
            unsafe_allow_html=True,
        )
        st.number_input(
            "Cutoff input",
            min_value=350,
            max_value=650,
            step=5,
            key="cutoff_num",
            on_change=on_cutoff_num_change,
            label_visibility="collapsed",
        )
        st.slider(
            "Score_Cutoff slider",
            min_value=350,
            max_value=650,
            step=5,
            key="cutoff_slide",
            on_change=on_cutoff_slide_change,
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # 2. LGD_Value Control Box (Bounded in [0.00, 1.00])
        st.markdown(
            """
            <div class="widget-box">
                <div class="widget-title">LGD_Value (0.00 – 1.00)</div>
            """,
            unsafe_allow_html=True,
        )
        st.number_input(
            "LGD input",
            min_value=0.00,
            max_value=1.00,
            step=0.01,
            format="%.2f",
            key="lgd_num",
            on_change=on_lgd_num_change,
            label_visibility="collapsed",
        )
        st.slider(
            "LGD_Value slider",
            min_value=0.00,
            max_value=1.00,
            step=0.01,
            key="lgd_slide",
            on_change=on_lgd_slide_change,
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # 3. Interest_Margin Control Box
        st.markdown(
            """
            <div class="widget-box">
                <div class="widget-title">Interest_Margin ⌵</div>
            """,
            unsafe_allow_html=True,
        )
        st.number_input(
            "Margin input",
            min_value=0.01,
            max_value=0.50,
            step=0.01,
            format="%.2f",
            key="margin_num",
            on_change=on_margin_num_change,
            label_visibility="collapsed",
        )
        st.slider(
            "Interest_Margin slider",
            min_value=0.02,
            max_value=0.40,
            step=0.01,
            key="margin_slide",
            on_change=on_margin_slide_change,
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🛠️ Advanced Assumptions", expanded=False):
            st.session_state.ead_val = st.number_input(
                "Base Loan Exposure ($EAD)",
                min_value=5000,
                max_value=50000,
                value=int(st.session_state.ead_val),
                step=500,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # Current evaluated policy parameters (LGD strictly clamped in [0, 1])
    current_cutoff = int(st.session_state.cutoff_num)
    current_lgd = max(0.00, min(1.00, float(st.session_state.lgd_num)))
    current_margin = float(st.session_state.margin_num)
    current_ead = float(st.session_state.ead_val)

    # Dynamic metrics evaluation
    metrics = engine.evaluate_cutoff(
        cutoff=current_cutoff,
        lgd=current_lgd,
        interest_margin=current_margin,
        ead=current_ead,
    )

    # -------------------------------------------------------------------------
    # RIGHT MAIN AREA: TOP 5 KPI CARDS BANNER + FULL-WIDTH STRATEGY CHART
    # -------------------------------------------------------------------------
    with col_right_dashboard:
        # TOP ROW: 5 DARK METALLIC PINK KPI SUMMARY CARDS
        kpi_cols = st.columns(5)

        # 1. ApprovalRate
        with kpi_cols[0]:
            approval_rate_fmt = f"{metrics['approval_rate']:.2f}"
            st.markdown(
                f"""
                <div class="kpi-card-outer">
                    <div class="kpi-card-inner">
                        <div class="kpi-value">{approval_rate_fmt}</div>
                        <div class="kpi-title">ApprovalRate</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 2. ExpectedLoss
        with kpi_cols[1]:
            loss_val = metrics["expected_loss"]
            if abs(loss_val) >= 1e9:
                loss_fmt = f"{loss_val / 1e9:.2f}bn"
            elif abs(loss_val) >= 1e6:
                loss_fmt = f"{loss_val / 1e6:.2f}M"
            else:
                loss_fmt = f"{loss_val / 1e3:.1f}K"
            st.markdown(
                f"""
                <div class="kpi-card-outer">
                    <div class="kpi-card-inner">
                        <div class="kpi-value">{loss_fmt}</div>
                        <div class="kpi-title">ExpectedLoss</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 3. ExpectedDefaults
        with kpi_cols[2]:
            defaults_val = metrics["expected_defaults"]
            if defaults_val >= 1000:
                defaults_fmt = f"{defaults_val / 1000:.2f}K"
            else:
                defaults_fmt = f"{defaults_val:.0f}"
            st.markdown(
                f"""
                <div class="kpi-card-outer">
                    <div class="kpi-card-inner">
                        <div class="kpi-value">{defaults_fmt}</div>
                        <div class="kpi-title">ExpectedDefaults</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 4. ExpectedProfit
        with kpi_cols[3]:
            profit_val = metrics["expected_profit"]
            if abs(profit_val) >= 1e9:
                profit_fmt = f"{profit_val / 1e9:.2f}bn"
            elif abs(profit_val) >= 1e6:
                profit_fmt = f"{profit_val / 1e6:.2f}M"
            else:
                profit_fmt = f"{profit_val / 1e3:.1f}K"
            st.markdown(
                f"""
                <div class="kpi-card-outer">
                    <div class="kpi-card-inner">
                        <div class="kpi-value">{profit_fmt}</div>
                        <div class="kpi-title">ExpectedProfit</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 5. ApprovedCount
        with kpi_cols[4]:
            approved_val = metrics["approved_count"]
            if approved_val >= 1000:
                approved_fmt = f"{approved_val / 1000:.0f}K"
            else:
                approved_fmt = f"{approved_val:,}"
            st.markdown(
                f"""
                <div class="kpi-card-outer">
                    <div class="kpi-card-inner">
                        <div class="kpi-value">{approved_fmt}</div>
                        <div class="kpi-title">ApprovedCount</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------------------------------------------------------------------
        # MAIN PROMINENT POSITION: FULL-WIDTH STRATEGY CHART (Dark Mode Plotly)
        # ---------------------------------------------------------------------
        sweep_df = engine.simulate_sweep(
            lgd=current_lgd,
            interest_margin=current_margin,
            ead=current_ead,
            cutoff_min=350,
            cutoff_max=650,
            cutoff_step=5,
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 1. Primary Left Axis: Net Profit Bars (Neon Mint Emerald #4ECCA3)
        fig.add_trace(
            go.Bar(
                x=sweep_df["Cutoff"],
                y=sweep_df["Net_Profit"] / 1e9,
                name="Sum of Net_Profit",
                marker_color="#4ECCA3",
                opacity=0.90,
                hovertemplate=(
                    "<b>Cutoff: %{x}</b><br>"
                    "Net Profit: $%{y:.3f}bn<br>"
                    "<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

        # 2. Secondary Right Axis: Bad Rate Curve (Glowing Red Line #FF4D4D)
        fig.add_trace(
            go.Scatter(
                x=sweep_df["Cutoff"],
                y=sweep_df["Bad_Rate"],
                name="Sum of Bad_Rate",
                mode="lines",
                line=dict(color="#FF4D4D", width=3.6),
                hovertemplate=(
                    "<b>Cutoff: %{x}</b><br>"
                    "Bad Rate: %{y:.2%}<br>"
                    "<extra></extra>"
                ),
            ),
            secondary_y=True,
        )

        # Current Cutoff Marker Line (Neon Purple)
        fig.add_vline(
            x=current_cutoff,
            line_dash="dash",
            line_color="#C084FC",
            line_width=2.5,
            annotation_text=f"Selected Cutoff C = {current_cutoff}",
            annotation_position="top left",
            annotation_font=dict(size=12, color="#E9D5FF", family="Outfit"),
        )

        # Dark Layout with generous margins
        fig.update_layout(
            title=dict(
                text="<b>Strategy: Net Profit vs Bad Rate by Score Cutoff</b>",
                x=0.5,
                xanchor="center",
                y=0.98,
                font=dict(family="Outfit", size=18, color="#FFFFFF"),
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.93,
                xanchor="center",
                x=0.5,
                font=dict(size=13, family="Inter", color="#E2E8F0"),
                bgcolor="rgba(19, 23, 34, 0.85)",
                bordercolor="#2D3748",
                borderwidth=1,
            ),
            margin=dict(l=65, r=65, t=75, b=50),
            height=500,
            plot_bgcolor="#131722",
            paper_bgcolor="#131722",
            hovermode="x unified",
        )

        # X-Axis formatting
        fig.update_xaxes(
            title_text="<b>Score Cutoff Threshold (C)</b>",
            title_font=dict(size=13, color="#CBD5E1"),
            showgrid=True,
            gridcolor="#222838",
            range=[350, 655],
            dtick=50,
            tickfont=dict(size=12, color="#94A3B8"),
        )

        # Left Y-Axis formatting (Billions bn)
        fig.update_yaxes(
            title_text="<b>Portfolio Net Profit ($ Billions)</b>",
            title_font=dict(size=13, color="#4ECCA3"),
            ticksuffix="bn",
            showgrid=True,
            gridcolor="#1E2536",
            gridwidth=1,
            secondary_y=False,
            zeroline=True,
            zerolinecolor="#4A5568",
            tickfont=dict(size=12, color="#4ECCA3"),
        )

        # Right Y-Axis formatting (Percentage %)
        fig.update_yaxes(
            title_text="<b>Approved Portfolio Bad Rate (%)</b>",
            title_font=dict(size=13, color="#FF6B6B"),
            tickformat=".1%",
            showgrid=False,
            secondary_y=True,
            tickfont=dict(size=12, color="#FF6B6B"),
            range=[0, 0.18],
            dtick=0.05,
        )

        st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
        st.plotly_chart(fig, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# TAB 2: MODEL PERFORMANCE & CALIBRATION (DARK THEME)
# =============================================================================
with tab2:
    st.markdown(
        """
        <div style="background: #131722; border: 1.5px solid #232A3B; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
            <h3 style="font-family: 'Outfit'; font-size: 18px; margin: 0 0 6px 0; color: #FFFFFF;">
                🎯 Scorecard Model Diagnostics & Calibration Reliability
            </h3>
            <p style="font-size: 13px; color: #94A3B8; margin: 0;">
                Benchmarked on Holdout Test Set (N = 452,128 loans). Evaluates discriminatory power (AUC, Gini, KS) and Basel III default probability calibration.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3 Metric Cards (Dark Theme)
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #1E3A8A 0%, #172554 100%); color: white; border-radius: 8px; padding: 16px; text-align: center; border: 1.5px solid #3B82F6; box-shadow: 0 4px 14px rgba(0,0,0,0.3);">
                <div style="font-family: 'Outfit'; font-size: 38px; font-weight: 800; line-height: 1; color: #93C5FD;">0.72</div>
                <div style="font-size: 14px; font-weight: 700; margin-top: 6px; color: #DBEAFE; text-transform: uppercase;">Model_AUC</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #78350F 0%, #451A03 100%); color: white; border-radius: 8px; padding: 16px; text-align: center; border: 1.5px solid #F59E0B; box-shadow: 0 4px 14px rgba(0,0,0,0.3);">
                <div style="font-family: 'Outfit'; font-size: 38px; font-weight: 800; line-height: 1; color: #FDE68A;">0.44</div>
                <div style="font-size: 14px; font-weight: 700; margin-top: 6px; color: #FEF3C7; text-transform: uppercase;">Model_Gini</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #7F1D1D 0%, #450A0A 100%); color: white; border-radius: 8px; padding: 16px; text-align: center; border: 1.5px solid #EF4444; box-shadow: 0 4px 14px rgba(0,0,0,0.3);">
                <div style="font-family: 'Outfit'; font-size: 38px; font-weight: 800; line-height: 1; color: #FCA5A5;">0.32</div>
                <div style="font-size: 14px; font-weight: 700; margin-top: 6px; color: #FEE2E2; text-transform: uppercase;">Model_KS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2 Main Diagnostic Charts (Left: Calibration Observed vs Predicted, Right: Top Predictors by IV)
    c_diag1, c_diag2 = st.columns(2)

    with c_diag1:
        decile_path = "outputs/decile_performance.csv"
        if os.path.exists(decile_path):
            decile_df = pd.read_csv(decile_path)
        else:
            deciles = list(range(10))
            decile_df = pd.DataFrame({
                "decile": deciles,
                "total_count": [67800] * 10,
                "bad_rate": [0.33, 0.22, 0.17, 0.13, 0.10, 0.08, 0.06, 0.04, 0.02, 0.01],
                "avg_pd": [0.32, 0.21, 0.16, 0.13, 0.10, 0.08, 0.06, 0.04, 0.02, 0.01],
            })

        fig_calib = make_subplots(specs=[[{"secondary_y": True}]])

        fig_calib.add_trace(
            go.Bar(
                x=decile_df["decile"],
                y=[67.8] * len(decile_df),
                name="CountPerDecile_v2",
                marker_color="#3B82F6",
                opacity=0.85,
            ),
            secondary_y=False,
        )

        bad_col = "actual_bad_rate" if "actual_bad_rate" in decile_df.columns else "bad_rate"
        fig_calib.add_trace(
            go.Scatter(
                x=decile_df["decile"],
                y=decile_df[bad_col],
                name="BadRate_PerDecile",
                mode="lines+markers",
                line=dict(color="#10B981", width=3.5),
                marker=dict(size=6),
            ),
            secondary_y=True,
        )

        pd_col = "calibrated_pd" if "calibrated_pd" in decile_df.columns else "avg_pd"
        if pd_col in decile_df.columns:
            fig_calib.add_trace(
                go.Scatter(
                    x=decile_df["decile"],
                    y=decile_df[pd_col],
                    name="AvgPD_PerDecile",
                    mode="lines+markers",
                    line=dict(color="#F43F5E", width=3.5),
                    marker=dict(size=6),
                ),
                secondary_y=True,
            )

        fig_calib.update_layout(
            title=dict(
                text="<b>Calibration: Predicted vs Observed</b>",
                font=dict(family="Outfit", size=14, color="#FFFFFF"),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0, font=dict(size=11, color="#E2E8F0")),
            margin=dict(l=40, r=40, t=40, b=30),
            height=360,
            plot_bgcolor="#131722",
            paper_bgcolor="#131722",
        )
        fig_calib.update_xaxes(
            title_text="<b>Score Decile (0-9, grouped by score)</b>",
            gridcolor="#222838",
            tickfont=dict(color="#94A3B8"),
            title_font=dict(color="#CBD5E1"),
        )
        fig_calib.update_yaxes(
            title_text="<b>Number of loans</b>",
            ticksuffix="K",
            secondary_y=False,
            range=[0, 75],
            gridcolor="#1E2536",
            tickfont=dict(color="#94A3B8"),
            title_font=dict(color="#CBD5E1"),
        )
        fig_calib.update_yaxes(
            title_text="<b>Rate (%)</b>",
            tickformat=".1%",
            secondary_y=True,
            range=[0, 0.38],
            tickfont=dict(color="#10B981"),
            title_font=dict(color="#10B981"),
        )

        st.plotly_chart(fig_calib)

    with c_diag2:
        iv_data = {
            "Feature": [
                "emp_title", "sub_grade", "grade", "int_rate", "desc", "title",
                "fico_range_high", "fico_range_low", "all_util", "max_bal_bc",
                "mths_since_rcn...", "total_bal_il", "open_rv_24m", "bc_open_to_buy",
                "verification_st...", "acc_open_past...", "inq_last_12m", "il_util",
                "open_acc_6m", "term_int", "open_rv_12m", "total_bc_limit",
                "num_tl_op_pas...", "inq_fi", "inq_last_6mths", "disbursement_...",
                "open_act_il", "open_il_24m",
            ],
            "IV": [
                0.70, 0.51, 0.47, 0.44, 0.29, 0.20,
                0.15, 0.15, 0.12, 0.10,
                0.10, 0.09, 0.09, 0.09,
                0.08, 0.08, 0.08, 0.07,
                0.07, 0.07, 0.07, 0.06,
                0.06, 0.06, 0.06, 0.06,
                0.06, 0.05,
            ],
        }
        df_iv = pd.DataFrame(iv_data)

        fig_iv = go.Figure(
            data=[
                go.Bar(
                    x=df_iv["Feature"],
                    y=df_iv["IV"],
                    marker_color="#F59E0B",
                    opacity=0.90,
                    hovertemplate="<b>%{x}</b><br>Information Value (IV): %{y:.3f}<extra></extra>",
                )
            ]
        )
        fig_iv.update_layout(
            title=dict(
                text="<b>Top Predictors of Loan Default (Ranked by Information Value)</b>",
                font=dict(family="Outfit", size=14, color="#FFFFFF"),
            ),
            margin=dict(l=40, r=20, t=40, b=90),
            height=360,
            plot_bgcolor="#131722",
            paper_bgcolor="#131722",
        )
        fig_iv.update_xaxes(
            title_text="<b>Feature</b>",
            tickangle=-90,
            tickfont=dict(size=10, color="#94A3B8"),
            title_font=dict(color="#CBD5E1"),
        )
        fig_iv.update_yaxes(
            title_text="<b>IV</b>",
            range=[0, 0.75],
            gridcolor="#1E2536",
            tickfont=dict(color="#94A3B8"),
            title_font=dict(color="#CBD5E1"),
        )

        st.plotly_chart(fig_iv)
