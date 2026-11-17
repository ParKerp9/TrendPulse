"""Historical Trends & Longitudinal Analytics Page for TrendPulse."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from components.plots import create_temporal_heatmap, get_plot_theme


def render_historical_page(posts_df: pd.DataFrame, theme_mode: str):
    """Renders longitudinal performance, posting heatmaps, and correlation diagnostics."""
    cfg = get_plot_theme(theme_mode)

    st.markdown("## 📜 Historical Trends & Deep-Dive Analytics")
    st.markdown(
        "Examine longitudinal pattern dynamics, identify optimal posting schedule windows, "
        "and audit feature correlations with long-tail virality."
    )

    if posts_df.empty:
        st.warning("No trend records available.")
        return

    df = posts_df.copy()
    if "hour_of_day" not in df.columns and "timestamp_utc" in df.columns:
        dt = pd.to_datetime(df["timestamp_utc"], unit="s", utc=True)
        df["hour_of_day"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek

    # Heatmap Section: Hour of Day vs Day of Week
    st.markdown("### 🕒 Optimal Timing: Engagement Heatmap")
    fig_heat = create_temporal_heatmap(df, theme_mode)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Correlation Matrix
    st.markdown("### 🔗 Correlation Matrix with Engagement")
    numeric_cols = [
        c for c in [
            "score", "comment_count", "sentiment_polarity", "sentiment_subjectivity",
            "title_word_count", "uppercase_ratio", "upvote_ratio", "has_question"
        ] if c in df.columns
    ]

    if len(numeric_cols) > 3:
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r" if theme_mode == "dark" else "Blues",
            template=cfg["template"],
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            font=dict(color=cfg["text_color"]),
            margin=dict(l=15, r=15, t=25, b=15),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Statistical Summary Table
    st.markdown("### 📐 Community Engagement Percentile Distribution")
    if "score" in df.columns and "subreddit" in df.columns:
        summary_table = (
            df.groupby("subreddit")["score"]
            .agg(
                Count="count",
                Mean="mean",
                Median="median",
                P75=lambda x: np.percentile(x, 75),
                P90=lambda x: np.percentile(x, 90),
                Max="max",
            )
            .round(1)
        )
        st.dataframe(summary_table, use_container_width=True)
