"""Overview Page for TrendPulse Dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.plots import get_plot_theme
from data_loader import DataSourceInfo, DataSourceType


def render_overview_page(posts_df: pd.DataFrame, hourly_df: pd.DataFrame, ds_info: DataSourceInfo, theme_mode: str):
    """Renders executive summary, real-time KPI metrics, community distributions, and live radar."""
    cfg = get_plot_theme(theme_mode)

    # Header & Live/Simulated Data Banner
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.markdown("## 📊 Social Media Trend Lifecycle Overview")
        st.markdown(
            "Predictive intelligence telemetry across digital communities. Tracking real-time kinetic momentum, "
            "algorithmic virality breakout signals, and saturation decay."
        )
    with col_t2:
        badge_class = "tp-badge-live" if ds_info.is_live else "tp-badge-sim"
        badge_label = "🟢 LIVE DATA" if ds_info.is_live else "🟡 SIMULATED BENCHMARK"
        st.markdown(
            f"""
            <div style="text-align: right; margin-top: 10px;">
                <span class="tp-badge {badge_class}">{badge_label}</span>
                <div style="font-size: 11px; color: gray; margin-top: 4px;">{ds_info.description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if posts_df.empty:
        st.warning("No trend records found in data lake. Please run the collector or verify connection.")
        return

    # Calculate Top KPIs
    total_posts = len(posts_df)
    unique_subs = posts_df["subreddit"].nunique() if "subreddit" in posts_df.columns else 1
    total_score = int(pd.to_numeric(posts_df["score"], errors="coerce").sum())
    total_comments = int(pd.to_numeric(posts_df["comment_count"], errors="coerce").sum())
    avg_score = int(pd.to_numeric(posts_df["score"], errors="coerce").mean())

    # High virality candidates (> 80th percentile)
    viral_cutoff = float(posts_df["score"].quantile(0.80)) if not posts_df.empty else 4000
    viral_candidates = len(posts_df[posts_df["score"] >= viral_cutoff])

    # KPI Metric Cards Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="tp-card">
                <div class="tp-metric-title">Trends Monitored</div>
                <div class="tp-metric-val">{total_posts:,}</div>
                <div class="tp-metric-sub">Across {unique_subs} Active Communities</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="tp-card">
                <div class="tp-metric-title">Total Interactions</div>
                <div class="tp-metric-val">{total_score + total_comments:,}</div>
                <div class="tp-metric-sub">{total_score:,} Upvotes • {total_comments:,} Comments</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="tp-card">
                <div class="tp-metric-title">Viral Breakouts</div>
                <div class="tp-metric-val" style="color: #EF4444;">{viral_candidates}</div>
                <div class="tp-metric-sub">Top 20th Percentile (> {viral_cutoff:,.0f} pts)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="tp-card">
                <div class="tp-metric-title">Mean Engagement</div>
                <div class="tp-metric-val">{avg_score:,}</div>
                <div class="tp-metric-sub">Points per Submission</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Mid Section: Hourly Growth Dynamics & Community Distribution
    col_chart1, col_chart2 = st.columns([2, 1])

    with col_chart1:
        st.markdown("### 📈 Real-Time Hourly Interaction Velocity")
        if not hourly_df.empty and "hourly_bucket" in hourly_df.columns:
            fig_hourly = px.line(
                hourly_df,
                x="hourly_bucket",
                y="total_interactions",
                color="subreddit",
                markers=True,
                title="",
                template=cfg["template"],
                color_discrete_sequence=cfg["accent_palette"],
            )
            fig_hourly.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=340,
                xaxis=dict(title="Timestamp (UTC)", gridcolor=cfg["grid_color"]),
                yaxis=dict(title="Hourly Interactions", gridcolor=cfg["grid_color"]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=20, b=20),
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        else:
            st.info("Hourly aggregate data will display here after running run_eda.py.")

    with col_chart2:
        st.markdown("### 🌐 Community Share")
        if "subreddit" in posts_df.columns:
            sub_counts = posts_df["subreddit"].value_counts()
            fig_sub = px.pie(
                values=sub_counts.values,
                names=[f"r/{s}" for s in sub_counts.index],
                hole=0.55,
                template=cfg["template"],
                color_discrete_sequence=cfg["accent_palette"],
            )
            fig_sub.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=340,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(l=10, r=10, t=20, b=20),
            )
            st.plotly_chart(fig_sub, use_container_width=True)

    # Lower Section: Live Momentum Radar (Top Breakouts)
    st.markdown("### 🚀 Top Trending Radar (Highest Kinetic Momentum)")
    top_radar = posts_df.sort_values(by="score", ascending=False).head(5)

    for _, row in top_radar.iterrows():
        title = row.get("title", "")
        sub = row.get("subreddit", "general")
        score = int(row.get("score", 0))
        comments = int(row.get("comment_count", 0))
        url = row.get("url", "#")
        sentiment = row.get("sentiment_label", "neutral")
        sent_color = "#10B981" if sentiment == "positive" else ("#EF4444" if sentiment == "negative" else "#64748B")

        st.markdown(
            f"""
            <div class="tp-card" style="padding: 14px 18px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <span class="tp-badge tp-badge-growth">r/{sub}</span>
                        <span class="tp-badge" style="background: rgba(255,255,255,0.08); color: {sent_color};">{sentiment.upper()}</span>
                        <a href="{url}" target="_blank" style="text-decoration: none; font-weight: 600; font-size: 15px; margin-left: 6px;">
                            {title}
                        </a>
                    </div>
                    <div style="text-align: right; min-width: 160px;">
                        <span style="font-size: 18px; font-weight: 700; color: #38BDF8;">{score:,}</span> <span style="font-size: 12px; color: gray;">upvotes</span>
                        <div style="font-size: 12px; color: gray;">💬 {comments:,} comments</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
