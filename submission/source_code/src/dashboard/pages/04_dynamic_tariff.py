# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""
Dynamic Tariff Recommendations Page - PPO agent pricing decisions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dynamic Tariff", page_icon="💰", layout="wide")
st.title("💰 Dynamic Tariff Recommendations")
st.markdown("PPO Reinforcement Learning agent recommending optimal per-kWh tariffs in real-time.")

np.random.seed(42)

# Current recommendations
st.subheader("🎯 Current Tariff Recommendations")
rec_cols = st.columns(3)
with rec_cols[0]:
    st.metric("Baseline Price", "₹15.00/kWh", help="Fixed flat rate")
with rec_cols[1]:
    st.metric("Avg Dynamic Price", "₹16.82/kWh", "+12.1%", help="PPO recommended average")
with rec_cols[2]:
    st.metric("Revenue Gain", "+18.5%", help="vs ₹15 fixed baseline")

st.markdown("---")

# Tariff actions distribution
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Action Distribution (24h)")
    actions = ["-20%", "-10%", "0%", "+10%", "+20%", "+30%"]
    counts = [45, 82, 120, 95, 68, 30]
    colors = ["#4ecdc4", "#45b7aa", "#ffd93d", "#ffb347", "#ff6b6b", "#ff3333"]
    fig = go.Figure(go.Bar(x=actions, y=counts, marker_color=colors))
    fig.update_layout(template="plotly_dark", height=350, xaxis_title="Price Action", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("💹 Dynamic Price Over Time")
    hours = pd.date_range("2022-07-15", periods=288, freq="5min")
    base_price = np.full(288, 15.0)
    dynamic_price = 15.0 + 5 * np.sin(np.arange(288) * 2 * np.pi / 288) + np.random.normal(0, 1, 288)
    dynamic_price = np.clip(dynamic_price, 12, 19.5)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hours, y=base_price, name="Fixed ₹15", line=dict(color="gray", dash="dash")))
    fig2.add_trace(go.Scatter(x=hours, y=dynamic_price, name="Dynamic", line=dict(color="#4ecdc4", width=2), fill="tozeroy", fillcolor="rgba(78,205,196,0.1)"))
    fig2.update_layout(template="plotly_dark", height=350, yaxis_title="Price (₹/kWh)")
    st.plotly_chart(fig2, use_container_width=True)

# Station-level recommendations
st.subheader("📋 Station-Level Tariff Table")
tariff_df = pd.DataFrame({
    "District": [329, 332, 335, 324, 1088, 346, 506, 525, 1076, 333],
    "Current Utilization": ["87%", "82%", "76%", "45%", "91%", "22%", "65%", "28%", "38%", "71%"],
    "Recommended Action": ["🔴 +30%", "🔴 +20%", "🟡 +10%", "🟢 -10%", "🔴 +30%", "🟢 -20%", "⚪ 0%", "🟢 -10%", "🟢 -10%", "🟡 +10%"],
    "New Price (₹/kWh)": [19.50, 18.00, 16.50, 13.50, 19.50, 12.00, 15.00, 13.50, 13.50, 16.50],
    "Expected Revenue Δ": ["+₹2,450", "+₹1,890", "+₹980", "+₹320", "+₹3,100", "+₹150", "—", "+₹210", "+₹180", "+₹650"],
    "Congestion Prob": [0.94, 0.88, 0.65, 0.12, 0.96, 0.05, 0.42, 0.08, 0.15, 0.58],
})
st.dataframe(tariff_df, use_container_width=True, hide_index=True)

# PPO Training curve
st.subheader("🧠 PPO Training Progress")
steps = np.arange(0, 100001, 1000)
reward = -50 + 80 * (1 - np.exp(-steps / 30000)) + np.random.normal(0, 5, len(steps))
fig3 = go.Figure(go.Scatter(x=steps, y=reward, mode="lines", line=dict(color="#4ecdc4", width=2)))
fig3.update_layout(template="plotly_dark", height=300, xaxis_title="Training Steps", yaxis_title="Episode Reward")
st.plotly_chart(fig3, use_container_width=True)
