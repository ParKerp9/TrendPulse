"""Live ML Predictions & Sandbox Analyzer for TrendPulse."""

import streamlit as st
import pandas as pd

from components.plots import create_feature_driver_chart, create_virality_gauge
from src.models.predictor import TrendPulsePredictor


def render_predictions_page(posts_df: pd.DataFrame, predictor: TrendPulsePredictor, theme_mode: str):
    """Renders interactive multi-model prediction sandbox without placeholder mocks."""
    st.markdown("## 🔮 Virality & Lifecycle Prediction Sandbox")
    st.markdown(
        "Evaluate any social media headline or draft content using the trained **TrendPulse Multi-Model Ensemble**. "
        "Outputs calibrated virality odds, lifecycle categorization, estimated peak velocity time, and underlying feature attributions."
    )

    if not predictor or not predictor.is_ready():
        st.error("ML model artifacts are not loaded. Please train models first using run_trainer.py.")
        return

    # Choose Input Mode
    mode = st.radio(
        "Choose Input Source",
        options=["Select from Ingested Trends", "Draft Custom Headline (Live Simulator)"],
        horizontal=True,
    )

    input_data = {}

    if mode == "Select from Ingested Trends":
        if posts_df.empty:
            st.warning("No ingested posts available to select.")
            return

        titles = posts_df["title"].tolist()
        selected_title = st.selectbox("Select a Trend to Analyze", options=titles)
        selected_row = posts_df[posts_df["title"] == selected_title].iloc[0]
        input_data = selected_row.to_dict()

        st.caption(
            f"Selected: **r/{input_data.get('subreddit', 'general')}** | "
            f"Observed Score: {input_data.get('score', 0):,} | "
            f"Comments: {input_data.get('comment_count', 0):,}"
        )

    else:
        # Custom Simulator
        c1, c2 = st.columns([2.5, 1.5])
        with c1:
            custom_title = st.text_input(
                "Post Title / Headline",
                value="Breakthrough: Open-Source AI Model Rivals Frontier Commercial Models on Code Benchmarks",
            )
            custom_body = st.text_area(
                "Content Body (Optional)",
                value="New architecture leverages selective state spaces and 4-bit quantization with minimal loss.",
                height=90,
            )
        with c2:
            default_sub_list = [
                "technology", "artificial", "science", "wallstreetbets",
                "MachineLearning", "futurology", "programming", "gadgets",
                "dataisbeautiful", "stocks", "worldnews", "gaming", "general"
            ]
            if not posts_df.empty and "subreddit" in posts_df.columns:
                available_subs = sorted(list(set(default_sub_list).union(posts_df["subreddit"].dropna().unique())))
            else:
                available_subs = default_sub_list

            custom_sub = st.selectbox(
                "Target Community / Subreddit",
                options=available_subs,
                index=available_subs.index("technology") if "technology" in available_subs else 0,
            )
            custom_hour = st.slider("Hour of Posting (UTC)", min_value=0, max_value=23, value=14)
            custom_upvote_ratio = st.slider("Initial Upvote Ratio", min_value=0.50, max_value=1.00, value=0.94, step=0.01)

        input_data = {
            "title": custom_title,
            "text": custom_body,
            "subreddit": custom_sub,
            "score": 150.0,
            "comment_count": 25.0,
            "hour_of_day": custom_hour,
            "day_of_week": 2,
            "upvote_ratio": custom_upvote_ratio,
        }

    # Execute Prediction
    if st.button("🚀 Run TrendPulse Prediction Engine", type="primary"):
        with st.spinner("Executing 5 ML inference pipelines and computing feature drivers..."):
            try:
                results = predictor.predict_all(input_data)
            except Exception as e:
                st.error(f"Inference error: {e}")
                return

        st.success("✅ Multi-Model Analysis Complete!")

        # Top Results Cards
        v = results["virality"]
        lc = results["lifecycle"]
        pop = results["popularity"]
        pk = results["peak_time"]
        sent = results["sentiment"]

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="tp-card">
                    <div class="tp-metric-title">Virality Probability</div>
                    <div class="tp-metric-val" style="color: {v['tier_color']};">{v['virality_probability']}%</div>
                    <div class="tp-metric-sub">{v['virality_tier']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="tp-card">
                    <div class="tp-metric-title">Projected Lifecycle</div>
                    <div class="tp-metric-val" style="font-size: 20px; color: #38BDF8;">{lc['stage_name']}</div>
                    <div class="tp-metric-sub">Phase Index #{lc['stage_id']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="tp-card">
                    <div class="tp-metric-title">Est. Peak Time</div>
                    <div class="tp-metric-val" style="color: #F59E0B;">{pk['peak_hours']:.1f}h</div>
                    <div class="tp-metric-sub">CI: {pk['confidence_interval'][0]:.1f}h - {pk['confidence_interval'][1]:.1f}h</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""
                <div class="tp-card">
                    <div class="tp-metric-title">24h Engagement</div>
                    <div class="tp-metric-val">{pop['predicted_total_interactions']:,}</div>
                    <div class="tp-metric-sub">~{pop['predicted_score']:,} pts • ~{pop['predicted_comments']:,} cmts</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Visual Diagnostics Row: Gauge & Feature Drivers
        col_g, col_drv = st.columns([1, 1.2])

        with col_g:
            st.markdown("### 🎯 Virality Odds Gauge")
            fig_g = create_virality_gauge(v["virality_probability"], v["virality_tier"], v["tier_color"], theme_mode)
            st.plotly_chart(fig_g, use_container_width=True)

            st.markdown(f"**Lifecycle Assessment:** {lc['stage_description']}")

        with col_drv:
            st.markdown("### 🧠 Model Explainability Drivers")
            fig_drv = create_feature_driver_chart(results.get("feature_drivers", []), theme_mode)
            st.plotly_chart(fig_drv, use_container_width=True)

        # Lifecycle Probabilities Table
        st.markdown("### 📊 Lifecycle State Probabilities")
        p_df = pd.DataFrame([lc["stage_probabilities"]]).T.rename(columns={0: "Confidence (%)"})
        st.bar_chart(p_df)
