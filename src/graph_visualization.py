"""
Network graph visualization using Plotly.
Uses circular layout for node positions (no external graph lib).
"""

import math
from typing import Dict, List, Tuple

import plotly.graph_objects as go

from src.graph import Graph


def _circular_layout(nodes: List[str], radius: float = 1.0) -> Dict[str, Tuple[float, float]]:
    """Place nodes on a circle. Returns {node: (x, y)}."""
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: (0.0, 0.0)}
    pos = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n - math.pi / 2  # Start from top
        pos[node] = (radius * math.cos(angle), radius * math.sin(angle))
    return pos


def plot_skill_graph_plotly(
    g: Graph,
    highlight_skills: List[str] = None,
    title: str = "Skill Graph",
) -> go.Figure:
    """
    Plot network graph: nodes = skills, edges = co-occurrence.
    Uses Plotly scatter for nodes and edges.
    """
    nodes = g.nodes()
    if not nodes:
        fig = go.Figure()
        fig.add_annotation(text="No graph data", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=title, height=600)
        return fig

    pos = _circular_layout(nodes, radius=1.2)
    highlight_set = set((highlight_skills or [])[:20])

    # Edge traces
    edge_x, edge_y = [], []
    for u in nodes:
        for v in g.neighbors(u):
            if u < v:  # Undirected: avoid duplicate
                x0, y0 = pos.get(u, (0, 0))
                x1, y1 = pos.get(v, (0, 0))
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Node traces
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    colors = ["#6366f1" if n in highlight_set else "#94a3b8" for n in nodes]
    sizes = [16 if n in highlight_set else 12 for n in nodes]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[n[:15] + "..." if len(n) > 15 else n for n in nodes],
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=1, color="white"),
        ),
        hoverinfo="text",
        hovertext=[f"Skill: {n}" for n in nodes],
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=600,
    )
    return fig
