"""
Professional Plotly-based EDA visualizations for the app.
"""

from collections import Counter
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.feature_engineering import build_skill_set_per_job

# Light theme palette
COLORS = {
    "primary": "#6366f1",
    "secondary": "#0ea5e9",
    "accent": "#22c55e",
    "bar1": "#6366f1",
    "bar2": "#0ea5e9",
    "bar3": "#22c55e",
    "bg": "rgba(15, 23, 42, 0.02)",
}

LAYOUT = {
    "font": {"family": "Plus Jakarta Sans, sans-serif", "size": 12},
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "margin": {"t": 50, "b": 50, "l": 50, "r": 30},
    "showlegend": True,
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    "xaxis": {"gridcolor": "rgba(0,0,0,0.06)", "zeroline": False},
    "yaxis": {"gridcolor": "rgba(0,0,0,0.06)", "zeroline": False},
}


def plot_role_frequency_plotly(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of top job roles."""
    col = "title" if "title" in df.columns else df.columns[0]
    counts = df[col].value_counts().head(top_n)
    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        color=counts.values,
        color_continuous_scale=["#6366f1", "#0ea5e9"],
        labels={"x": "Number of Postings", "y": ""},
        title="Top Job Roles by Frequency",
    )
    fig.update_layout(**LAYOUT, height=450)
    fig.update_coloraxes(showscale=False)
    fig.update_traces(marker_line_width=0)
    return fig


def plot_top_skills_plotly(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of top skills."""
    skill_series = build_skill_set_per_job(df)
    counter: Counter = Counter()
    for skills in skill_series:
        counter.update(skills)
    top = counter.most_common(top_n)
    labels = [t[0] for t in top]
    values = [t[1] for t in top]
    fig = px.bar(
        x=values,
        y=labels,
        orientation="h",
        color=values,
        color_continuous_scale=["#0ea5e9", "#22c55e"],
        labels={"x": "Frequency", "y": ""},
        title="Top Skills in Job Market",
    )
    fig.update_layout(**LAYOUT, height=500)
    fig.update_coloraxes(showscale=False)
    fig.update_traces(marker_line_width=0)
    return fig


def plot_location_plotly(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Bar chart of job distribution by location."""
    col = "location" if "location" in df.columns else None
    if col not in df.columns or df[col].isna().all():
        fig = go.Figure()
        fig.add_annotation(text="No location data", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(**LAYOUT, height=300)
        return fig
    counts = df[col].value_counts().head(top_n)
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        color=counts.values,
        color_continuous_scale=["#f59e0b", "#ef4444"],
        labels={"x": "Location", "y": "Count"},
        title="Job Distribution by Location",
    )
    fig.update_layout(**LAYOUT, height=400)
    fig.update_coloraxes(showscale=False)
    return fig


def plot_trend_plotly(df: pd.DataFrame, date_col: str = "posted_date") -> go.Figure:
    """Line chart of job posting trend over time."""
    if date_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No date data", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(**LAYOUT, height=300)
        return fig
    ts = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if ts.empty:
        fig = go.Figure()
        fig.add_annotation(text="No valid dates", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(**LAYOUT, height=300)
        return fig
    agg = ts.dt.to_period("M").value_counts().sort_index()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=agg.index.astype(str),
            y=agg.values,
            mode="lines+markers",
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(size=10),
            fill="tozeroy",
            fillcolor="rgba(99, 102, 241, 0.2)",
        )
    )
    fig.update_layout(
        **LAYOUT,
        height=350,
        title="Job Posting Trend Over Time",
        xaxis_title="Month",
        yaxis_title="Postings",
    )
    return fig


def plot_skill_match_gaps_plotly(gap_df, top_n: int = 10) -> go.Figure:
    """Grouped bar chart of matched vs missing skills per role."""
    if gap_df is None or gap_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No skill gap data yet. Run Career Analysis first.", x=0.5, y=0.5, showarrow=False, font={"size": 14})
        fig.update_layout(**LAYOUT, height=300)
        return fig
    df = gap_df.head(top_n)
    col_title = "job_title" if "job_title" in df.columns else "title" if "title" in df.columns else df.columns[0]
    matched = df["matched_skills"].apply(len)
    missing = df["missing_skills"].apply(len)
    roles = [str(t)[:30] + "..." if len(str(t)) > 30 else str(t) for t in df[col_title]]
    fig = go.Figure(data=[
        go.Bar(name="Matched Skills", x=roles, y=matched, marker_color=COLORS["bar1"]),
        go.Bar(name="Missing Skills", x=roles, y=missing, marker_color="#ef4444"),
    ])
    fig.update_layout(
        **LAYOUT,
        barmode="group",
        height=420,
        title="Skill Match vs Gaps per Role",
        xaxis_title="Role",
        yaxis_title="Count",
    )
    return fig
