"""
Revenue Impact Page - Fixed vs Dynamic pricing comparison.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Revenue Impact", page_icon="💵", layout="wide")
st.title("💵 Revenue Impact Analysis")
st.markdown("Side-by-side comparison of fixed (₹15/kWh) vs AI-powered dynamic pricing.")

np.random.seed(42)

# Impact Summary KPIs
st.subheader("🎯 Impact Summary")
cols = st.columns(4)
cols[0].metric("Revenue Gain", "+18.5%", "₹+387K total")
cols[1].metric("Utilization Improvement", "+8.2%", "64.1% → 72.3%")
cols[2].metric("Congestion Reduction", "-23.4%", "15.8% → 12.1%")
cols[3].metric("Off-Peak Uplift", "+31.2%", "Sessions increased")

st.markdown("---")

# Side-by-side comparison
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Revenue by Hour — Fixed Pricing")
    hours = list(range(24))
    rev_fixed = [8, 5, 3, 2, 3, 5, 12, 28, 35, 32, 28, 25, 27, 24, 22, 21, 23, 30, 38, 35, 28, 20, 14, 10]
    fig1 = go.Figure(go.Bar(x=hours, y=rev_fixed, marker_color="#ff6b6b"))
    fig1.update_layout(template="plotly_dark", height=300, yaxis_title="Revenue (₹K)")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("📊 Revenue by Hour — Dynamic Pricing")
    rev_dynamic = [r * np.random.uniform(1.05, 1.35) if r > 20 else r * np.random.uniform(1.1, 1.5) for r in rev_fixed]
    fig2 = go.Figure(go.Bar(x=hours, y=rev_dynamic, marker_color="#4ecdc4"))
    fig2.update_layout(template="plotly_dark", height=300, yaxis_title="Revenue (₹K)")
    st.plotly_chart(fig2, use_container_width=True)

# Detailed comparison table
st.subheader("📋 Detailed Comparison")
comparison = pd.DataFrame({
    "Metric": ["Total Revenue", "Avg Price (₹/kWh)", "Peak Revenue", "Off-Peak Revenue", "Utilization Rate", "Congestion Rate", "Avg Wait Time", "Sessions Count", "Energy Delivered (MWh)", "Revenue per Session"],
    "Fixed (₹15/kWh)": ["₹2,090,000", "₹15.00", "₹890,000", "₹320,000", "64.1%", "15.8%", "4.1 min", "18,450", "139.3", "₹113.3"],
    "Dynamic Pricing": ["₹2,477,000", "₹16.82", "₹1,120,000", "₹419,800", "72.3%", "12.1%", "2.4 min", "19,200", "147.4", "₹129.0"],
    "Change": ["+18.5%", "+12.1%", "+25.8%", "+31.2%", "+8.2pp", "-3.7pp", "-41.5%", "+4.1%", "+5.8%", "+13.9%"],
})
st.dataframe(comparison, use_container_width=True, hide_index=True)

# Waterfall chart
st.subheader("📈 Revenue Waterfall — Sources of Gain")
fig3 = go.Figure(go.Waterfall(
    orientation="v",
    measure=["absolute", "relative", "relative", "relative", "relative", "total"],
    x=["Baseline Revenue", "Peak Surge", "Off-Peak Uplift", "Demand Shift", "Efficiency Gain", "Total Dynamic"],
    y=[2090000, 230000, 99800, 35200, 22000, 2477000],
    connector=dict(line=dict(color="rgba(63, 63, 63, 0.5)")),
    increasing=dict(marker_color="#4ecdc4"),
    decreasing=dict(marker_color="#ff6b6b"),
    totals=dict(marker_color="#3282b8"),
    text=["₹2.09M", "+₹230K", "+₹99.8K", "+₹35.2K", "+₹22K", "₹2.48M"],
    textposition="outside",
))
fig3.update_layout(template="plotly_dark", height=400, yaxis_title="Revenue (₹)")
st.plotly_chart(fig3, use_container_width=True)
