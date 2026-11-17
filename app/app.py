"""TrendPulse - Social Media Trend Lifecycle Prediction Platform.
Main Streamlit Application Controller.
"""

import sys
from pathlib import Path

# Ensure project root and app directory are on sys.path BEFORE any local imports.
# This must happen at the module top-level, before Streamlit or any app module is loaded.
_ROOT_DIR = str(Path(__file__).resolve().parent.parent)
_APP_DIR = str(Path(__file__).resolve().parent)

if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import streamlit as st

# Configure page layout and metadata
st.set_page_config(
    page_title="TrendPulse - Social Media Trend Lifecycle Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Direct sibling imports — these work because _APP_DIR is on sys.path.
# NEVER use "from app.components..." here: when Streamlit runs app/app.py,
# Python treats "app" as this script's own module, not the package directory.
from components.theme import get_theme_css
from data_loader import DashboardDataLoader, DataSourceType
from pages_modules.explorer import render_explorer_page
from pages_modules.forecasting import render_forecasting_page
from pages_modules.historical import render_historical_page
from pages_modules.overview import render_overview_page
from pages_modules.predictions import render_predictions_page
from pages_modules.sentiment import render_sentiment_page


# Session State Initialization
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"


def main():
    # -------------------------------------------------------------
    # Sidebar: Brand, Theme Switcher & Navigation
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span style="font-size: 30px;">🔥</span>
                <div>
                    <div style="font-size: 20px; font-weight: 800; letter-spacing: -0.02em;">TrendPulse</div>
                    <div style="font-size: 11px; color: gray;">Lifecycle Prediction Platform</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Theme Toggle (Dark Mode vs Light Mode)
        theme_choice = st.radio(
            "🎨 Interface Theme",
            options=["Dark", "Light"],
            index=0 if st.session_state.theme_mode == "Dark" else 1,
            horizontal=True,
        )
        st.session_state.theme_mode = theme_choice

        # Apply CSS dynamically
        st.markdown(get_theme_css(st.session_state.theme_mode), unsafe_allow_html=True)

        st.markdown("---")

        # Navigation Selector
        st.markdown("**Navigation**")
        page = st.radio(
            "Go to",
            options=[
                "Overview",
                "Trend Explorer",
                "Predictions",
                "Forecasting",
                "Sentiment Analysis",
                "Historical Trends",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Data Source Status Widget
        loader = DashboardDataLoader()
        posts_df, hourly_df, ds_info = loader.load_data()
        predictor = loader.get_predictor()

        is_live = ds_info.is_live
        badge_style = "tp-badge-live" if is_live else "tp-badge-sim"
        badge_text = "🟢 LIVE DATA" if is_live else "🟡 BENCHMARK"

        st.markdown(
            f"""
            <div class="tp-card" style="padding: 12px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: gray;">Data Source</span>
                    <span class="tp-badge {badge_style}">{badge_text}</span>
                </div>
                <div style="font-size: 13px; font-weight: 600; margin-top: 6px;">{ds_info.source_type.value}</div>
                <div style="font-size: 11px; color: gray;">{ds_info.record_count:,} records ingested</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Model Status Widget
        models_ready = predictor.is_ready()
        model_badge = "tp-badge-live" if models_ready else "tp-badge-viral"
        model_text = "5/5 LOADED" if models_ready else "UNLOADED"

        st.markdown(
            f"""
            <div class="tp-card" style="padding: 12px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: gray;">ML Pipeline</span>
                    <span class="tp-badge {model_badge}">{model_text}</span>
                </div>
                <div style="font-size: 11px; color: gray; margin-top: 6px;">
                    Virality • Lifecycle • Popularity • Peak Time • Sentiment
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Refresh
        if st.button("🔄 Refresh Data & Cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown(
            """
            <div style="font-size: 11px; color: gray; margin-top: 20px; text-align: center;">
                TrendPulse ML v1.0.0<br>Built with Python, Streamlit & XGBoost
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # Main Content Router
    # -------------------------------------------------------------
    theme = st.session_state.theme_mode.lower()

    if page == "Overview":
        render_overview_page(posts_df, hourly_df, ds_info, theme)
    elif page == "Trend Explorer":
        render_explorer_page(posts_df, predictor, theme)
    elif page == "Predictions":
        render_predictions_page(posts_df, predictor, theme)
    elif page == "Forecasting":
        render_forecasting_page(posts_df, predictor, theme)
    elif page == "Sentiment Analysis":
        render_sentiment_page(posts_df, predictor, theme)
    elif page == "Historical Trends":
        render_historical_page(posts_df, theme)


if __name__ == "__main__":
    main()
