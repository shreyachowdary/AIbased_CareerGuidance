"""
EDA and data mapping visualization: role frequency, skills, location, trends.
"""

from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from config.settings import ARTIFACTS_DIR, FIGURE_DPI, PLOT_STYLE
from src.feature_engineering import build_skill_set_per_job
from utils.logging_config import get_logger

logger = get_logger("eda_visualization")


def _setup_style():
    """Apply consistent plot style."""
    if not HAS_MATPLOTLIB:
        return
    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = FIGURE_DPI


def plot_role_frequency(
    df: pd.DataFrame,
    top_n: int = 20,
    title_col: str = "title",
    save_path: Optional[Path] = None,
):
    """
    Bar chart of most frequent job titles/roles.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not installed; skipping plot")
        return None
    _setup_style()
    col = title_col if title_col in df.columns else df.columns[0]
    counts = df[col].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="barh", ax=ax, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xlabel("Count")
    ax.set_ylabel("Job Title")
    ax.set_title("Top Job Roles by Frequency")
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_top_skills(
    df: pd.DataFrame,
    top_n: int = 25,
    save_path: Optional[Path] = None,
):
    """
    Bar chart of most frequent skills/keywords across jobs.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not installed; skipping plot")
        return None
    _setup_style()
    skill_series = build_skill_set_per_job(df)
    counter: Counter = Counter()
    for skills in skill_series:
        counter.update(skills)
    top = dict(counter.most_common(top_n))
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = list(top.keys())
    values = list(top.values())
    ax.barh(range(len(labels)), values, color="teal", alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Frequency")
    ax.set_title("Top Skills/Keywords in Job Postings")
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_location_distribution(
    df: pd.DataFrame,
    top_n: int = 15,
    location_col: str = "location",
    save_path: Optional[Path] = None,
):
    """
    Bar chart of job distribution by location.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not installed; skipping plot")
        return None
    _setup_style()
    col = location_col if location_col in df.columns else "location"
    if col not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))  # type: ignore
        ax.text(0.5, 0.5, "No location data", ha="center", va="center")
        return fig
    counts = df[col].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="barh", ax=ax, color="coral", alpha=0.8)
    ax.set_xlabel("Count")
    ax.set_ylabel("Location")
    ax.set_title("Job Distribution by Location")
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_trend_over_time(
    df: pd.DataFrame,
    date_col: str = "posted_date",
    freq: str = "M",
    save_path: Optional[Path] = None,
):
    """
    Line plot of job posting trend over time.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not installed; skipping plot")
        return None
    _setup_style()
    col = date_col if date_col in df.columns else None
    if col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[col]):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No date data available", ha="center", va="center")
        return fig
    ts = df[col].dropna()
    ts = pd.to_datetime(ts, errors="coerce").dropna()
    if ts.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No valid dates", ha="center", va="center")
        return fig
    agg = ts.dt.to_period(freq).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(agg.index.astype(str), agg.values, marker="o", color="darkgreen", linewidth=2)
    ax.set_xlabel("Period")
    ax.set_ylabel("Job Postings Count")
    ax.set_title("Job Posting Trend Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
    return fig


def run_full_eda(df: pd.DataFrame, output_dir: Optional[Path] = None) -> dict:
    """
    Run all EDA visualizations and save to output_dir.
    """
    out = output_dir or ARTIFACTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["role_freq"] = out / "eda_role_frequency.png"
    paths["top_skills"] = out / "eda_top_skills.png"
    paths["location"] = out / "eda_location_distribution.png"
    paths["trend"] = out / "eda_trend_over_time.png"
    plot_role_frequency(df, save_path=paths["role_freq"])
    plot_top_skills(df, save_path=paths["top_skills"])
    plot_location_distribution(df, save_path=paths["location"])
    plot_trend_over_time(df, save_path=paths["trend"])
    logger.info("EDA plots saved to %s", out)
    return paths
