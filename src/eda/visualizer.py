"""Interactive Plotly Visualization Engine for Trend Growth, Velocity, and Sentiment."""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.logger import setup_logger

logger = setup_logger("Visualizer")


class EDAVisualizer:
    """Creates interactive Plotly figures for trend growth, engagement velocity, and sentiment."""

    def __init__(self, output_dir: str = "data/eda_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Professional modern color palette
        self.theme_colors = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4"]

    def plot_trend_growth(
        self,
        hourly_df: pd.DataFrame,
        save_html: bool = True,
        filename: str = "trend_growth.html",
    ) -> go.Figure:
        """
        Visualizes trend growth over time:
        - Post volume (bar)
        - Total interactions & rolling 6h trajectory (line)
        """
        if hourly_df.empty:
            logger.warning("Empty DataFrame passed to plot_trend_growth.")
            return go.Figure()

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Cumulative & Community Interactions Over Time", "Hourly Post Volume"),
            row_heights=[0.65, 0.35],
        )

        subreddits = hourly_df["subreddit"].unique() if "subreddit" in hourly_df.columns else ["all"]

        for i, sub in enumerate(subreddits):
            color = self.theme_colors[i % len(self.theme_colors)]
            sub_data = hourly_df[hourly_df["subreddit"] == sub] if "subreddit" in hourly_df.columns else hourly_df

            # Line plot for interactions
            fig.add_trace(
                go.Scatter(
                    x=sub_data["hourly_bucket"],
                    y=sub_data["total_interactions"],
                    mode="lines+markers",
                    name=f"r/{sub} Interactions",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=5),
                    hovertemplate="<b>r/%{text}</b><br>Time: %{x}<br>Interactions: %{y:,}<extra></extra>",
                    text=[sub] * len(sub_data),
                ),
                row=1,
                col=1,
            )

            # Bar plot for post volume
            fig.add_trace(
                go.Bar(
                    x=sub_data["hourly_bucket"],
                    y=sub_data["post_volume"],
                    name=f"r/{sub} Volume",
                    marker=dict(color=color, opacity=0.6),
                    hovertemplate="<b>r/%{text}</b><br>Volume: %{y}<extra></extra>",
                    text=[sub] * len(sub_data),
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            title="<b>Trend Growth Trajectory & Hourly Post Volume</b>",
            template="plotly_white",
            height=700,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            barmode="stack",
        )
        fig.update_yaxes(title_text="Total Interactions", row=1, col=1)
        fig.update_yaxes(title_text="Post Count", row=2, col=1)
        fig.update_xaxes(title_text="Timestamp (UTC)", row=2, col=1)

        if save_html:
            out_file = self.output_dir / filename
            fig.write_html(out_file)
            logger.info(f"Saved trend growth visualization to: {out_file}")

        return fig

    def plot_engagement_velocity(
        self,
        features_df: pd.DataFrame,
        save_html: bool = True,
        filename: str = "engagement_velocity.html",
    ) -> go.Figure:
        """
        Visualizes kinetic engagement velocity (dScore/dt) and acceleration over time.
        Highlights positive acceleration (surge) and decay phases.
        """
        if features_df.empty or "velocity_interactions" not in features_df.columns:
            logger.warning("Velocity features not present in DataFrame.")
            return go.Figure()

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Engagement Velocity (dInteractions/dt)", "Acceleration (d²Interactions/dt²)"),
            row_heights=[0.55, 0.45],
        )

        subreddits = features_df["subreddit"].unique() if "subreddit" in features_df.columns else ["all"]

        for i, sub in enumerate(subreddits):
            color = self.theme_colors[i % len(self.theme_colors)]
            sub_data = features_df[features_df["subreddit"] == sub] if "subreddit" in features_df.columns else features_df

            # Velocity Trace
            fig.add_trace(
                go.Scatter(
                    x=sub_data["hourly_bucket"],
                    y=sub_data["velocity_interactions"],
                    mode="lines",
                    name=f"r/{sub} Velocity",
                    line=dict(color=color, width=2),
                    hovertemplate="Time: %{x}<br>Velocity: %{y:+.1f}/hr<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Acceleration Trace (Bar)
            fig.add_trace(
                go.Bar(
                    x=sub_data["hourly_bucket"],
                    y=sub_data["acceleration_interactions"],
                    name=f"r/{sub} Acceleration",
                    marker=dict(color=color, opacity=0.7),
                    hovertemplate="Time: %{x}<br>Acceleration: %{y:+.1f}/hr²<extra></extra>",
                ),
                row=2,
                col=1,
            )

        # Zero reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        fig.update_layout(
            title="<b>Kinetic Engagement Dynamics: Velocity & Acceleration</b>",
            template="plotly_white",
            height=700,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_yaxes(title_text="Velocity (Interactions/hr)", row=1, col=1)
        fig.update_yaxes(title_text="Acceleration (Rate of Change)", row=2, col=1)
        fig.update_xaxes(title_text="Timestamp (UTC)", row=2, col=1)

        if save_html:
            out_file = self.output_dir / filename
            fig.write_html(out_file)
            logger.info(f"Saved engagement velocity visualization to: {out_file}")

        return fig

    def plot_sentiment_distribution(
        self,
        posts_df: pd.DataFrame,
        save_html: bool = True,
        filename: str = "sentiment_distribution.html",
    ) -> go.Figure:
        """
        Visualizes sentiment distribution:
        - Box plot of polarity score across subreddits
        - Sentiment categorical breakdown (Positive, Neutral, Negative)
        """
        if posts_df.empty or "sentiment_polarity" not in posts_df.columns:
            logger.warning("Sentiment features missing from posts DataFrame.")
            return go.Figure()

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Sentiment Polarity Distribution by Subreddit", "Sentiment Sentiment Proportions"),
            specs=[[{"type": "box"}, {"type": "bar"}]],
        )

        # 1. Box plot of polarity per subreddit
        subreddits = posts_df["subreddit"].unique()
        for i, sub in enumerate(subreddits):
            color = self.theme_colors[i % len(self.theme_colors)]
            sub_posts = posts_df[posts_df["subreddit"] == sub]
            fig.add_trace(
                go.Box(
                    y=sub_posts["sentiment_polarity"],
                    name=f"r/{sub}",
                    marker_color=color,
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                ),
                row=1,
                col=1,
            )

        # 2. Categorical breakdown bar chart
        counts = posts_df.groupby(["subreddit", "sentiment_label"]).size().unstack(fill_value=0)
        for label, col_color in [("positive", "#10B981"), ("neutral", "#6B7280"), ("negative", "#EF4444")]:
            if label in counts.columns:
                fig.add_trace(
                    go.Bar(
                        x=counts.index,
                        y=counts[label],
                        name=label.capitalize(),
                        marker_color=col_color,
                    ),
                    row=1,
                    col=2,
                )

        fig.update_layout(
            title="<b>Linguistic Sentiment Analysis Across Communities</b>",
            template="plotly_white",
            height=500,
            barmode="stack",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_yaxes(title_text="Polarity (-1.0 to +1.0)", row=1, col=1)
        fig.update_yaxes(title_text="Number of Posts", row=1, col=2)

        if save_html:
            out_file = self.output_dir / filename
            fig.write_html(out_file)
            logger.info(f"Saved sentiment visualization to: {out_file}")

        return fig

    def plot_historical_popularity(
        self,
        posts_df: pd.DataFrame,
        save_html: bool = True,
        filename: str = "historical_popularity.html",
    ) -> go.Figure:
        """
        Visualizes historical popularity landscape:
        - Scatter plot of score vs comment count (log-scale) with hover details
        - Hourly posting density distribution
        """
        if posts_df.empty:
            logger.warning("Empty posts DataFrame.")
            return go.Figure()

        df = posts_df.copy()
        df["upvote_ratio"] = pd.to_numeric(df.get("upvote_ratio", 0.9), errors="coerce").fillna(0.9)
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
        df["comment_count"] = pd.to_numeric(df["comment_count"], errors="coerce").fillna(0)

        # Scatter plot with size representing upvote ratio
        fig = px.scatter(
            df,
            x="score",
            y="comment_count",
            color="subreddit",
            size=np.clip(df["upvote_ratio"] * 15, 6, 20),
            hover_name="title",
            hover_data={
                "author": True,
                "score": ":,",
                "comment_count": ":,",
                "upvote_ratio": ":.2f",
                "timestamp_iso": True,
            },
            title="<b>Historical Popularity: Engagement Depth vs Upvote Volume</b>",
            template="plotly_white",
            color_discrete_sequence=self.theme_colors,
        )

        fig.update_layout(
            height=600,
            xaxis=dict(title="Total Score (Upvotes)", type="log"),
            yaxis=dict(title="Total Comments", type="log"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        if save_html:
            out_file = self.output_dir / filename
            fig.write_html(out_file)
            logger.info(f"Saved historical popularity visualization to: {out_file}")

        return fig

    def generate_all_visualizations(
        self,
        posts_df: pd.DataFrame,
        hourly_df: pd.DataFrame,
        features_df: pd.DataFrame,
    ) -> Dict[str, go.Figure]:
        """Runs all visualization routines and generates individual + index HTML files."""
        figs = {}
        figs["trend_growth"] = self.plot_trend_growth(hourly_df, save_html=True)
        figs["engagement_velocity"] = self.plot_engagement_velocity(features_df, save_html=True)
        figs["sentiment_distribution"] = self.plot_sentiment_distribution(posts_df, save_html=True)
        figs["historical_popularity"] = self.plot_historical_popularity(posts_df, save_html=True)

        # Generate index.html showcasing all links
        index_path = self.output_dir / "index.html"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(
                """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>TrendPulse - Exploratory Data Analysis & Visualizations</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; background: #0f172a; color: #f8fafc; }
        h1 { color: #38bdf8; font-size: 28px; }
        p { color: #94a3b8; font-size: 16px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-top: 30px; }
        .card { background: #1e293b; border-radius: 12px; padding: 24px; border: 1px solid #334155; transition: transform 0.2s; text-decoration: none; color: inherit; }
        .card:hover { transform: translateY(-4px); border-color: #38bdf8; }
        .card h2 { font-size: 20px; color: #f1f5f9; margin-top: 0; }
        .card p { font-size: 14px; margin-bottom: 0; }
    </style>
</head>
<body>
    <h1>🚀 TrendPulse: Exploratory Data Analysis Suite</h1>
    <p>Interactive diagnostic and predictive telemetry extracted from social media streams.</p>
    <div class="grid">
        <a class="card" href="trend_growth.html" target="_blank">
            <h2>📈 Trend Growth</h2>
            <p>Hourly post volume, community interactions, and trajectory dynamics over time.</p>
        </a>
        <a class="card" href="engagement_velocity.html" target="_blank">
            <h2>⚡ Engagement Velocity</h2>
            <p>Kinetic momentum (dScore/dt) and acceleration curves identifying explosive virality.</p>
        </a>
        <a class="card" href="sentiment_distribution.html" target="_blank">
            <h2>💬 Sentiment Distribution</h2>
            <p>Linguistic polarity, subjective density, and categorical sentiment breakdown across subreddits.</p>
        </a>
        <a class="card" href="historical_popularity.html" target="_blank">
            <h2>🔥 Historical Popularity</h2>
            <p>Log-log distribution of upvotes versus comments, and content engagement depth.</p>
        </a>
    </div>
</body>
</html>
"""
            )
        logger.info(f"Generated master EDA index at: {index_path}")
        return figs
