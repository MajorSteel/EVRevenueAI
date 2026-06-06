# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""
Overview Page - Comprehensive system overview with KPIs and summary charts.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

st.title("📊 System Overview")
st.markdown("Comprehensive view of the EV charging network performance and agent system status.")

# Generate sample data for demonstration
np.random.seed(42)
dates = pd.date_range("2022-06-19", periods=720, freq="1h")

# KPI Section
st.markdown("### 🎯 Key Performance Indicators")
cols = st.columns(6)
kpi_data = [
    ("Total Revenue", "₹2.41M", "+18.5%", True),
    ("Avg Utilization", "72.3%", "+8.2%", True),
    ("Congestion Rate", "12.1%", "-23.4%", True),
    ("Off-Peak Uplift", "31.2%", "+31.2%", True),
    ("Pricing Efficiency", "₹16.8/kWh", "+12.0%", True),
    ("Wait Time Proxy", "2.3 min", "-41.5%", True),
]
for col, (label, value, delta, is_positive) in zip(cols, kpi_data):
    col.metric(label=label, value=value, delta=delta)

st.markdown("---")

# Revenue trend
st.subheader("💰 Revenue Trend — Fixed vs Dynamic Pricing")
revenue_fixed = np.cumsum(np.random.uniform(2000, 5000, len(dates)))
revenue_dynamic = np.cumsum(np.random.uniform(2500, 6000, len(dates)))

fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=revenue_fixed, name="Fixed (₹15/kWh)", line=dict(color="#ff6b6b", width=2)))
fig.add_trace(go.Scatter(x=dates, y=revenue_dynamic, name="Dynamic Pricing", line=dict(color="#4ecdc4", width=2), fill="tonexty", fillcolor="rgba(78, 205, 196, 0.1)"))
fig.update_layout(template="plotly_dark", height=400, xaxis_title="Time", yaxis_title="Cumulative Revenue (₹)", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
st.plotly_chart(fig, use_container_width=True)

# Utilization and Congestion
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Hourly Utilization Pattern")
    hours = list(range(24))
    util_mean = [0.15, 0.12, 0.10, 0.08, 0.09, 0.12, 0.25, 0.55, 0.78, 0.82, 0.75, 0.68, 0.72, 0.65, 0.60, 0.58, 0.62, 0.75, 0.85, 0.80, 0.65, 0.45, 0.30, 0.20]
    colors = ["#ff6b6b" if u > 0.8 else "#ffd93d" if u > 0.5 else "#4ecdc4" for u in util_mean]
    fig2 = go.Figure(go.Bar(x=hours, y=util_mean, marker_color=colors))
    fig2.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="Surge (80%)")
    fig2.add_hline(y=0.3, line_dash="dash", line_color="green", annotation_text="Discount (30%)")
    fig2.update_layout(template="plotly_dark", height=350, xaxis_title="Hour of Day", yaxis_title="Utilization Rate")
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("🚦 Congestion Distribution")
    congestion_data = pd.DataFrame({
        "Status": ["Normal (<50%)", "Moderate (50-80%)", "Congested (>80%)"],
        "Count": [145, 72, 31]
    })
    fig3 = px.pie(congestion_data, values="Count", names="Status", color="Status",
                  color_discrete_map={"Normal (<50%)": "#4ecdc4", "Moderate (50-80%)": "#ffd93d", "Congested (>80%)": "#ff6b6b"})
    fig3.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig3, use_container_width=True)

# Agent Status Table
st.markdown("---")
st.subheader("🤖 Agent Performance Summary")
agent_df = pd.DataFrame({
    "Agent": ["Demand Prediction", "Congestion Prediction", "Tariff Pricing (PPO)", "GNN Spatial", "Monitoring"],
    "Status": ["✅ Active", "✅ Active", "✅ Active", "✅ Active", "✅ Active"],
    "Primary Metric": ["R² = 0.847", "AUC = 0.912", "Rev Gain = 18.5%", "R² = 0.863", "Drift: None"],
    "Last Updated": ["2 min ago", "2 min ago", "5 min ago", "10 min ago", "Live"],
    "Model": ["LightGBM", "XGBoost", "PPO (SB3)", "GCN (PyG)", "Rule-based"],
})
st.dataframe(agent_df, use_container_width=True, hide_index=True)
