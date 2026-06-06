# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""Chart helper components for the dashboard."""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Optional


def create_time_series(
    x: pd.DatetimeIndex,
    y: np.ndarray,
    name: str = "Value",
    color: str = "#4ecdc4",
    fill: bool = False,
) -> go.Figure:
    """Create a styled time series chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y, name=name,
            line=dict(color=color, width=2),
            fill="tozeroy" if fill else None,
            fillcolor=f"{color}15" if fill else None,
        )
    )
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=40, r=20, t=20, b=40))
    return fig


def create_comparison_bar(
    categories: List[str],
    values_a: List[float],
    values_b: List[float],
    name_a: str = "Fixed",
    name_b: str = "Dynamic",
) -> go.Figure:
    """Create a grouped bar comparison chart."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=values_a, name=name_a, marker_color="#ff6b6b"))
    fig.add_trace(go.Bar(x=categories, y=values_b, name=name_b, marker_color="#4ecdc4"))
    fig.update_layout(template="plotly_dark", barmode="group", height=400)
    return fig


def create_heatmap(
    data: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    colorscale: str = "YlOrRd",
) -> go.Figure:
    """Create a styled heatmap."""
    fig = px.imshow(
        data, x=x_labels, y=y_labels,
        color_continuous_scale=colorscale,
    )
    fig.update_layout(template="plotly_dark", height=400)
    return fig
