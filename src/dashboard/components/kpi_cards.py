"""
KPI Card components for the dashboard.
"""
import streamlit as st
from typing import Optional


def render_kpi_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_positive: bool = True,
    gradient: str = "linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%)",
) -> None:
    """Render a styled KPI card."""
    delta_class = "kpi-delta-positive" if delta_positive else "kpi-delta-negative"
    delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""

    st.markdown(
        f"""
        <div class="kpi-card" style="background: {gradient};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(label: str, status: str = "active") -> None:
    """Render a status badge."""
    colors = {
        "active": "#4ecdc4",
        "warning": "#ffd93d",
        "error": "#ff6b6b",
        "inactive": "#6c757d",
    }
    color = colors.get(status, "#6c757d")
    st.markdown(
        f"""
        <span style="
            background-color: {color}20;
            color: {color};
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        ">● {label}</span>
        """,
        unsafe_allow_html=True,
    )
