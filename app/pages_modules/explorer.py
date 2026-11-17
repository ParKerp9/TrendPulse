"""Trend Explorer Page: Search, Filter, Inspect, and Run Predictions on Real Social Media Trends."""

import pandas as pd
import streamlit as st

from src.models.predictor import TrendPulsePredictor


def render_explorer_page(posts_df: pd.DataFrame, predictor: TrendPulsePredictor, theme_mode: str):
    """Renders interactive trend discovery grid with multi-filter search and post inspector."""
    st.markdown("## 🔍 Social Media Trend Explorer")
    st.markdown("Inspect real-time community submissions, filter by momentum signals, and audit underlying feature attributes.")

    if posts_df.empty:
        st.warning("No trend records available to explore.")
        return

    # Filter Bar
    f1, f2, f3, f4 = st.columns([2, 1.2, 1.2, 1.2])

    with f1:
        search_query = st.text_input("🔎 Search by Headline / Keyword", placeholder="e.g. LLM, semiconductor, battery...")

    with f2:
        all_subs = sorted(list(posts_df["subreddit"].dropna().unique()))
        selected_subs = st.multiselect("Subreddits", options=all_subs, default=all_subs)

    with f3:
        max_score = int(posts_df["score"].max()) if not posts_df.empty else 1000
        min_score = st.slider("Min Upvotes", min_value=0, max_value=max(100, max_score), value=0, step=50)

    with f4:
        sort_by = st.selectbox(
            "Sort Order",
            options=["Score: High to Low", "Score: Low to High", "Comments: High to Low", "Most Recent"],
        )

    # Apply Filters
    filtered = posts_df.copy()

    if selected_subs:
        filtered = filtered[filtered["subreddit"].isin(selected_subs)]

    if min_score > 0:
        filtered = filtered[filtered["score"] >= min_score]

    if search_query:
        q = search_query.lower()
        filtered = filtered[
            filtered["title"].str.lower().str.contains(q, na=False)
            | filtered.get("text", pd.Series([""] * len(filtered))).str.lower().str.contains(q, na=False)
        ]

    # Sort
    if sort_by == "Score: High to Low":
        filtered = filtered.sort_values(by="score", ascending=False)
    elif sort_by == "Score: Low to High":
        filtered = filtered.sort_values(by="score", ascending=True)
    elif sort_by == "Comments: High to Low":
        filtered = filtered.sort_values(by="comment_count", ascending=False)
    elif sort_by == "Most Recent" and "timestamp_utc" in filtered.columns:
        filtered = filtered.sort_values(by="timestamp_utc", ascending=False)

    st.markdown(f"**Showing {len(filtered):,} of {len(posts_df):,} total trends**")

    # Render Posts Cards & Expanders
    for _, row in filtered.head(25).iterrows():
        title = row.get("title", "Untitled")
        sub = row.get("subreddit", "general")
        score = int(row.get("score", 0))
        comments = int(row.get("comment_count", 0))
        post_id = row.get("post_id", "")
        author = row.get("author", "anonymous")
        created_iso = row.get("timestamp_iso", "N/A")
        url = row.get("url", "#")
        sentiment = row.get("sentiment_label", "neutral")
        disc_ratio = comments / (score + 1)

        # Quick prediction run if predictor is ready
        pred_summary = ""
        if predictor and predictor.is_ready():
            try:
                virality = predictor.predict_virality(row.to_dict())
                lc = predictor.predict_lifecycle(row.to_dict())
                pred_summary = f"""
                <span class="tp-badge" style="background: rgba(56,189,248,0.15); color: #38BDF8;">{lc['stage_name']}</span>
                <span class="tp-badge" style="background: rgba(239,68,68,0.15); color: {virality['tier_color']};">Virality: {virality['virality_probability']}%</span>
                """
            except Exception:
                pass

        with st.expander(f"r/{sub} • {score:,} pts • {title[:90]}..."):
            st.markdown(
                f"""
                <div style="margin-bottom: 12px;">
                    {pred_summary}
                    <span class="tp-badge" style="background: rgba(255,255,255,0.08); color: gray;">Author: u/{author}</span>
                    <span class="tp-badge" style="background: rgba(255,255,255,0.08); color: gray;">Published: {created_iso}</span>
                </div>
                <div style="font-size: 15px; font-weight: 600; margin-bottom: 8px;">{title}</div>
                <p style="font-size: 13px; color: gray;">{row.get('text', '') or 'No body text provided.'}</p>
                <div style="display: flex; gap: 16px; font-size: 13px; margin-top: 10px;">
                    <div>👍 <b>Score:</b> {score:,}</div>
                    <div>💬 <b>Comments:</b> {comments:,}</div>
                    <div>🔥 <b>Discussion Density:</b> {disc_ratio:.3f}</div>
                    <div>🔗 <a href="{url}" target="_blank">View on Reddit</a></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
