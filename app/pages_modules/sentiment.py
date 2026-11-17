"""Sentiment Analysis & Linguistic Intelligence Page for TrendPulse."""

import pandas as pd
import plotly.express as px
import streamlit as st

from components.plots import create_sentiment_donut_chart, get_plot_theme
from src.models.predictor import TrendPulsePredictor
from src.preprocessing.text_cleaner import TextCleaner


def render_sentiment_page(posts_df: pd.DataFrame, predictor: TrendPulsePredictor, theme_mode: str):
    """Renders sentiment distribution diagnostics, emotion metrics, and interactive text tester."""
    cfg = get_plot_theme(theme_mode)

    st.markdown("## 💬 Sentiment & Linguistic Intelligence")
    st.markdown(
        "Analyze emotional tone, linguistic sentiment polarity, and psychological controversy drivers "
        "across online social discussions."
    )

    if posts_df.empty:
        st.warning("No trend records available.")
        return

    # Aggregate Sentiment Counts
    sentiment_counts = posts_df["sentiment_label"].value_counts().to_dict() if "sentiment_label" in posts_df.columns else {
        "positive": 25, "neutral": 20, "negative": 11
    }

    # Top Visual Row: Donut Chart & Polarity Boxplots
    col_donut, col_box = st.columns([1, 1.4])

    with col_donut:
        st.plotly_chart(create_sentiment_donut_chart(sentiment_counts, theme_mode), use_container_width=True)

    with col_box:
        st.markdown("### 📊 Polarity Spread Across Communities")
        if "sentiment_polarity" in posts_df.columns and "subreddit" in posts_df.columns:
            fig_box = px.box(
                posts_df,
                x="subreddit",
                y="sentiment_polarity",
                color="subreddit",
                points="all",
                template=cfg["template"],
                color_discrete_sequence=cfg["accent_palette"],
            )
            fig_box.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                xaxis=dict(title="Community", gridcolor=cfg["grid_color"]),
                yaxis=dict(title="Polarity (-1.0 to +1.0)", gridcolor=cfg["grid_color"]),
                showlegend=False,
                margin=dict(l=15, r=15, t=35, b=15),
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # Mid Section: Correlation between Sentiment and Virality
    st.markdown("### ⚡ Sentiment Polarity vs. Virality Odds")
    if "sentiment_polarity" in posts_df.columns and "score" in posts_df.columns:
        fig_scatter = px.scatter(
            posts_df,
            x="sentiment_polarity",
            y="score",
            color="subreddit",
            size="comment_count",
            hover_name="title",
            template=cfg["template"],
            color_discrete_sequence=cfg["accent_palette"],
            labels={"sentiment_polarity": "Sentiment Polarity", "score": "Total Score (Upvotes)"},
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
            xaxis=dict(gridcolor=cfg["grid_color"]),
            yaxis=dict(gridcolor=cfg["grid_color"], type="log"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Interactive Sentiment Analyzer Sandbox
    st.markdown("### 🧪 Real-Time Text Sentiment Analyzer")
    user_text = st.text_input(
        "Enter any headline or post comment to evaluate live sentiment",
        value="Researchers report revolutionary discovery with unprecedented accuracy and low cost!",
    )

    if user_text:
        ling_res = TextCleaner.extract_linguistic_features(user_text)
        pol = ling_res["sentiment_polarity"]
        subj = ling_res["sentiment_subjectivity"]
        lbl = ling_res["sentiment_label"]
        col = "#10B981" if lbl == "positive" else ("#EF4444" if lbl == "negative" else "#64748B")

        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(
                f"""
                <div class="tp-card">
                    <div class="tp-metric-title">Sentiment Classification</div>
                    <div class="tp-metric-val" style="color: {col};">{lbl.upper()}</div>
                    <div class="tp-metric-sub">NLP Lexicon & Context Match</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                f"""
                <div class="tp-card">
                    <div class="tp-metric-title">Polarity Score</div>
                    <div class="tp-metric-val">{pol:+.3f}</div>
                    <div class="tp-metric-sub">Range: [-1.0 Negative, +1.0 Positive]</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                f"""
                <div class="tp-card">
                    <div class="tp-metric-title">Subjectivity Index</div>
                    <div class="tp-metric-val">{subj:.2f}</div>
                    <div class="tp-metric-sub">0.0 (Factual) to 1.0 (Opinionated)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
