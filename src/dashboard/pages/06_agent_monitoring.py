# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""
Agent Monitoring Page - Feedback loop performance and drift detection.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Agent Monitoring", page_icon="🤖", layout="wide")
st.title("🤖 Agent Monitoring & Learning")
st.markdown("Real-time tracking of agent performance, drift detection, and feedback loop status.")

np.random.seed(42)

# Feedback Loop Status
st.subheader("🔄 Feedback Loop Status")
cols = st.columns(4)
cols[0].metric("Loop Iterations", "1,247", "+12 today")
cols[1].metric("Drift Detected", "0", "No anomalies")
cols[2].metric("Model Retrains", "3", "Last: 2h ago")
cols[3].metric("Pricing Efficiency", "₹16.82/kWh", "+1.8/kWh vs start")

st.markdown("---")

# Metric tracking over episodes
st.subheader("📈 Performance Metrics Over Time")
episodes = np.arange(1, 1248)
tab1, tab2, tab3, tab4 = st.tabs(["Revenue Gain %", "Utilization", "Congestion Rate", "Pricing Efficiency"])

with tab1:
    rev_gain = 5 + 13.5 * (1 - np.exp(-episodes / 400)) + np.random.normal(0, 1.5, len(episodes))
    fig = go.Figure(go.Scatter(x=episodes, y=rev_gain, mode="lines", line=dict(color="#4ecdc4", width=1.5)))
    fig.add_hline(y=18.5, line_dash="dash", line_color="gold", annotation_text="Current: 18.5%")
    fig.update_layout(template="plotly_dark", height=350, xaxis_title="Episode", yaxis_title="Revenue Gain (%)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    util = 0.55 + 0.17 * (1 - np.exp(-episodes / 300)) + np.random.normal(0, 0.02, len(episodes))
    fig2 = go.Figure(go.Scatter(x=episodes, y=util, mode="lines", line=dict(color="#ffd93d", width=1.5)))
    fig2.add_hline(y=0.723, line_dash="dash", line_color="gold", annotation_text="Current: 72.3%")
    fig2.update_layout(template="plotly_dark", height=350, xaxis_title="Episode", yaxis_title="Utilization Rate")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    cong = 0.20 - 0.08 * (1 - np.exp(-episodes / 500)) + np.random.normal(0, 0.01, len(episodes))
    fig3 = go.Figure(go.Scatter(x=episodes, y=cong, mode="lines", line=dict(color="#ff6b6b", width=1.5)))
    fig3.add_hline(y=0.121, line_dash="dash", line_color="gold", annotation_text="Current: 12.1%")
    fig3.update_layout(template="plotly_dark", height=350, xaxis_title="Episode", yaxis_title="Congestion Rate")
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    eff = 14.5 + 2.3 * (1 - np.exp(-episodes / 350)) + np.random.normal(0, 0.3, len(episodes))
    fig4 = go.Figure(go.Scatter(x=episodes, y=eff, mode="lines", line=dict(color="#45b7aa", width=1.5)))
    fig4.add_hline(y=16.82, line_dash="dash", line_color="gold", annotation_text="Current: ₹16.82")
    fig4.update_layout(template="plotly_dark", height=350, xaxis_title="Episode", yaxis_title="₹/kWh")
    st.plotly_chart(fig4, use_container_width=True)

# Drift Detection
st.subheader("🔍 Drift Detection Dashboard")
drift_df = pd.DataFrame({
    "Metric": ["Revenue Gain %", "Utilization Rate", "Congestion Rate", "Wait Time Proxy", "Pricing Efficiency"],
    "Current Value": [18.5, 0.723, 0.121, 2.3, 16.82],
    "Rolling Mean": [18.2, 0.718, 0.124, 2.4, 16.75],
    "Z-Score": [0.42, 0.31, -0.28, -0.15, 0.38],
    "Status": ["✅ Normal", "✅ Normal", "✅ Normal", "✅ Normal", "✅ Normal"],
    "Threshold": ["±2.0σ", "±2.0σ", "±2.0σ", "±2.0σ", "±2.0σ"],
})
st.dataframe(drift_df, use_container_width=True, hide_index=True)

# Customer Response
st.subheader("👥 Customer Response Rate")
col1, col2 = st.columns(2)
with col1:
    price_changes = np.linspace(-20, 30, 6)
    demand_response = [-8, -4, 0, -3, -7, -12]  # % change in demand
    fig5 = go.Figure(go.Bar(x=[f"{int(p)}%" for p in price_changes], y=demand_response, marker_color=["#4ecdc4", "#45b7aa", "gray", "#ffd93d", "#ff6b6b", "#ff3333"]))
    fig5.update_layout(template="plotly_dark", height=300, xaxis_title="Price Change", yaxis_title="Demand Response (%)")
    st.plotly_chart(fig5, use_container_width=True)

with col2:
    st.markdown("**Demand Elasticity Summary**")
    st.markdown("""
    - **Price ↑ 10%** → Demand ↓ 3.0% (elasticity: -0.30)
    - **Price ↑ 20%** → Demand ↓ 7.0% (elasticity: -0.35)
    - **Price ↑ 30%** → Demand ↓ 12.0% (elasticity: -0.40)
    - **Price ↓ 10%** → Demand ↑ 4.0% (elasticity: -0.40)
    - **Price ↓ 20%** → Demand ↑ 8.0% (elasticity: -0.40)
    
    > Average elasticity: **-0.37** (inelastic demand, favorable for surge pricing)
    """)
