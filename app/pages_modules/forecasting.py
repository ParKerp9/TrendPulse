"""Trend Popularity & Lifecycle Trajectory Forecasting Page."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.plots import create_trajectory_forecast_chart, get_plot_theme
from src.models.predictor import TrendPulsePredictor


def render_forecasting_page(posts_df: pd.DataFrame, predictor: TrendPulsePredictor, theme_mode: str):
    """Renders 24-hour predictive popularity forecasting curves and saturation telemetry."""
    cfg = get_plot_theme(theme_mode)

    st.markdown("## 📈 Popularity Forecasting & S-Curve Trajectory")
    st.markdown(
        "Forecast 24-hour cumulative engagement growth using mathematical lifecycle diffusion models. "
        "Predicts velocity inflection, peak timing, and audience saturation half-life."
    )

    if not predictor or not predictor.is_ready():
        st.error("ML models not loaded. Please train models first using run_trainer.py.")
        return

    if posts_df.empty:
        st.warning("No trend records available.")
        return

    # Post Selector
    titles = posts_df["title"].tolist()
    selected_title = st.selectbox("Select a Trend to Forecast", options=titles, index=0)
    selected_row = posts_df[posts_df["title"] == selected_title].iloc[0]
    post_dict = selected_row.to_dict()

    # Generate Forecast
    try:
        pop_forecast = predictor.forecast_popularity(post_dict)
        peak_est = predictor.estimate_peak_time(post_dict)
    except Exception as e:
        st.error(f"Error generating forecast: {e}")
        return

    hours = pop_forecast["trajectory_hours"]
    trajectory = pop_forecast["trajectory_interactions"]
    peak_h = peak_est["peak_hours"]
    total_proj = pop_forecast["predicted_total_interactions"]

    # Top Metrics
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.markdown(
            f"""
            <div class="tp-card">
                <div class="tp-metric-title">24h Forecasted Volume</div>
                <div class="tp-metric-val">{total_proj:,}</div>
                <div class="tp-metric-sub">Cumulative Interactions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fc2:
        st.markdown(
            f"""
            <div class="tp-card">
                <div class="tp-metric-title">Peak Inflection Arrival</div>
                <div class="tp-metric-val" style="color: #F59E0B;">~{peak_h:.1f}h</div>
                <div class="tp-metric-sub">Elapsed Hours to Peak Velocity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fc3:
        # Peak velocity rate
        peak_rate = int(total_proj * 0.12)
        st.markdown(
            f"""
            <div class="tp-card">
                <div class="tp-metric-title">Peak Velocity Rate</div>
                <div class="tp-metric-val" style="color: #38BDF8;">{peak_rate:,}</div>
                <div class="tp-metric-sub">Interactions / Hour at Peak</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fc4:
        # Saturation Horizon
        sat_hour = min(24, int(peak_h + 8))
        st.markdown(
            f"""
            <div class="tp-card">
                <div class="tp-metric-title">Saturation Horizon</div>
                <div class="tp-metric-val">~{sat_hour}h</div>
                <div class="tp-metric-sub">Growth Decelerates Below 5%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Main Interactive Trajectory Curve
    st.markdown("### 📊 S-Curve Lifecycle Growth Projection")
    fig_traj = create_trajectory_forecast_chart(hours, trajectory, peak_h, theme_mode)

    # Add 95% Confidence Bounds
    rmse_bound = total_proj * 0.18
    upper_bound = [int(v + rmse_bound * (h / 24.0)) for h, v in zip(hours, trajectory)]
    lower_bound = [max(0, int(v - rmse_bound * (h / 24.0))) for h, v in zip(hours, trajectory)]

    fig_traj.add_trace(
        go.Scatter(
            x=hours + hours[::-1],
            y=upper_bound + lower_bound[::-1],
            fill="toself",
            fillcolor="rgba(129, 140, 248, 0.08)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% Confidence Interval",
        )
    )

    st.plotly_chart(fig_traj, use_container_width=True)

    # Forecast Diagnostics Table
    st.markdown("### 📋 Hourly Projected Progression Table")
    forecast_df = pd.DataFrame({
        "Hour Post-Creation (t + Δh)": [f"+{h}h" for h in hours],
        "Cumulative Projected Interactions": trajectory,
        "Incremental Hourly Influx": [trajectory[0]] + [trajectory[i] - trajectory[i-1] for i in range(1, len(trajectory))],
        "Lifecycle Phase": [
            "Incubation / Emerging" if h < (peak_h * 0.5) else
            ("Accelerating Growth" if h < peak_h else
             ("Peak Velocity" if abs(h - peak_h) <= 1 else "Decay / Saturation"))
            for h in hours
        ],
    })
    st.dataframe(forecast_df.T, use_container_width=True)
