"""Interactive Plotly Visualizations Engine for TrendPulse Streamlit Application."""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def get_plot_theme(theme_mode: str = "dark") -> Dict[str, Any]:
    """Returns color schemes and templates based on active theme."""
    is_dark = theme_mode.lower() == "dark"
    return {
        "template": "plotly_dark" if is_dark else "plotly_white",
        "bg_color": "#151C2C" if is_dark else "#FFFFFF",
        "paper_bg": "#0B0F19" if is_dark else "#F8FAFC",
        "text_color": "#F8FAFC" if is_dark else "#0F172A",
        "grid_color": "#232F48" if is_dark else "#E2E8F0",
        "accent_palette": ["#38BDF8", "#818CF8", "#10B981", "#F59E0B", "#EF4444", "#EC4899"],
    }


def create_virality_gauge(probability: float, tier: str, color: str, theme_mode: str = "dark") -> go.Figure:
    """Builds a gauge chart displaying virality probability and risk tier."""
    cfg = get_plot_theme(theme_mode)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            number={"suffix": "%", "font": {"size": 42, "color": cfg["text_color"]}},
            title={"text": f"<b>{tier}</b>", "font": {"size": 16, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": cfg["text_color"]},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": cfg["bg_color"],
                "borderwidth": 1,
                "bordercolor": cfg["grid_color"],
                "steps": [
                    {"range": [0, 40], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [40, 70], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [70, 100], "color": "rgba(239, 68, 68, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#EF4444", "width": 4},
                    "thickness": 0.75,
                    "value": 75,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=25, r=25, t=35, b=25),
        font=dict(color=cfg["text_color"]),
    )
    return fig


def create_trajectory_forecast_chart(
    hours: List[int],
    interactions: List[int],
    peak_hour: float,
    theme_mode: str = "dark",
) -> go.Figure:
    """Builds an interactive growth forecast curve with peak popularity annotation."""
    cfg = get_plot_theme(theme_mode)

    fig = go.Figure()

    # Projected Area Curve
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=interactions,
            mode="lines+markers",
            name="Forecasted Interactions",
            line=dict(color="#38BDF8", width=3, shape="spline"),
            marker=dict(size=6, color="#38BDF8"),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.12)",
            hovertemplate="Hour +%{x}h<br>Projected Total: %{y:,}<extra></extra>",
        )
    )

    # Peak Time Marker
    peak_val = np.interp(peak_hour, hours, interactions) if hours else 0
    fig.add_trace(
        go.Scatter(
            x=[peak_hour],
            y=[peak_val],
            mode="markers+text",
            name="Peak Velocity",
            marker=dict(size=14, color="#F59E0B", symbol="star"),
            text=[f"Peak (~{peak_hour:.1f}h)"],
            textposition="top center",
            textfont=dict(color="#F59E0B", size=12),
        )
    )

    fig.update_layout(
        title="<b>24-Hour Cumulative Trajectory & Lifecycle S-Curve</b>",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        font=dict(color=cfg["text_color"]),
        xaxis=dict(
            title="Elapsed Hours Post-Creation (t + Δh)",
            gridcolor=cfg["grid_color"],
            dtick=2,
        ),
        yaxis=dict(
            title="Total Estimated Engagement",
            gridcolor=cfg["grid_color"],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_feature_driver_chart(drivers: List[Dict[str, Any]], theme_mode: str = "dark") -> go.Figure:
    """Builds a horizontal bar chart displaying key ML feature importance drivers."""
    cfg = get_plot_theme(theme_mode)

    if not drivers:
        return go.Figure()

    df = pd.DataFrame(drivers).sort_values(by="importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df["importance"],
            y=df["feature"],
            orientation="h",
            marker=dict(
                color=df["importance"],
                colorscale="Blues" if theme_mode == "light" else "Viridis",
                showscale=False,
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Key Predictive Drivers (Model Attribution)</b>",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        font=dict(color=cfg["text_color"]),
        xaxis=dict(title="Relative Importance (%)", gridcolor=cfg["grid_color"]),
        yaxis=dict(gridcolor=cfg["grid_color"]),
        margin=dict(l=10, r=10, t=35, b=25),
    )
    return fig


def create_sentiment_donut_chart(counts: Dict[str, int], theme_mode: str = "dark") -> go.Figure:
    """Builds a donut chart showcasing sentiment polarity proportions."""
    cfg = get_plot_theme(theme_mode)

    labels = list(counts.keys())
    values = list(counts.values())
    colors = {
        "positive": "#10B981",
        "neutral": "#64748B",
        "negative": "#EF4444",
    }
    marker_colors = [colors.get(k.lower(), "#38BDF8") for k in labels]

    fig = go.Figure(
        go.Pie(
            labels=[l.capitalize() for l in labels],
            values=values,
            hole=0.55,
            marker=dict(colors=marker_colors),
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b>: %{value} posts (%{percent})<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Sentiment Distribution Breakdown</b>",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        font=dict(color=cfg["text_color"]),
        showlegend=False,
        margin=dict(l=15, r=15, t=40, b=15),
    )
    return fig


def create_temporal_heatmap(df: pd.DataFrame, theme_mode: str = "dark") -> go.Figure:
    """Creates a 2D Heatmap of Posting Hour vs Day of Week for Engagement."""
    cfg = get_plot_theme(theme_mode)

    if df.empty or "hour_of_day" not in df.columns or "day_of_week" not in df.columns:
        return go.Figure()

    grouped = (
        df.groupby(["day_of_week", "hour_of_day"])["score"]
        .mean()
        .unstack(fill_value=0)
    )

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    y_labels = [days[int(d)] for d in grouped.index if d < 7]

    fig = go.Figure(
        go.Heatmap(
            z=grouped.values,
            x=list(grouped.columns),
            y=y_labels,
            colorscale="Viridis" if theme_mode == "dark" else "Blues",
            hovertemplate="Day: %{y}<br>Hour: %{x}:00<br>Avg Score: %{z:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Historical Popularity Heatmap (Hour vs Day of Week)</b>",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        font=dict(color=cfg["text_color"]),
        xaxis=dict(title="Hour of Day (UTC)", dtick=2),
        yaxis=dict(title="Day of Week"),
        margin=dict(l=20, r=20, t=40, b=25),
    )
    return fig
