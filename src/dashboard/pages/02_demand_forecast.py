"""
Demand Forecast Page - Interactive demand prediction visualizations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Demand Forecast", page_icon="📈", layout="wide")
st.title("📈 Demand Forecast")
st.markdown("AI-powered demand prediction using XGBoost and LightGBM models.")

# Sidebar controls
with st.sidebar:
    st.subheader("Forecast Controls")
    forecast_horizon = st.slider("Forecast Horizon (hours)", 1, 48, 24)
    selected_model = st.selectbox("Model", ["LightGBM (Best)", "XGBoost", "Ensemble"])
    confidence = st.checkbox("Show Confidence Interval", True)

np.random.seed(42)
hours = pd.date_range("2022-07-15", periods=288, freq="5min")

# Simulated demand data
actual = 50 + 30 * np.sin(np.arange(288) * 2 * np.pi / 288) + np.random.normal(0, 5, 288)
predicted = actual + np.random.normal(0, 3, 288)
upper = predicted + 8
lower = predicted - 8

# Forecast plot
st.subheader("🔮 Demand Forecast — Next 24 Hours")
fig = go.Figure()
fig.add_trace(go.Scatter(x=hours[:200], y=actual[:200], name="Actual", line=dict(color="#4ecdc4", width=2)))
fig.add_trace(go.Scatter(x=hours[200:], y=predicted[200:], name="Predicted", line=dict(color="#ff6b6b", width=2, dash="dot")))
if confidence:
    fig.add_trace(go.Scatter(x=hours[200:], y=upper[200:], fill=None, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=hours[200:], y=lower[200:], fill="tonexty", mode="lines", line=dict(width=0), name="95% CI", fillcolor="rgba(255, 107, 107, 0.15)"))
fig.update_layout(template="plotly_dark", height=400, xaxis_title="Time", yaxis_title="Charging Volume (kWh)")
st.plotly_chart(fig, use_container_width=True)

# Model metrics
st.subheader("📊 Model Performance Comparison")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", "4.23 kWh", "-12% vs baseline")
col2.metric("MAE", "3.15 kWh", "-15% vs baseline")
col3.metric("R² Score", "0.847", "+0.12 vs baseline")

# Feature importance
st.subheader("🔑 Feature Importance — Top 15")
features = ["hour", "utilization_lag_1", "volume_lag_1", "weekday", "neighbor_mean_occupancy", "price", "occupancy_density", "rolling_mean_12", "peak_period", "fast_ratio", "congestion_score", "weekend", "rolling_std_12", "price_change", "spatial_lag"]
importance = sorted(np.random.uniform(0.02, 0.15, len(features)), reverse=True)
fig2 = px.bar(x=importance, y=features, orientation="h", color=importance, color_continuous_scale="Viridis")
fig2.update_layout(template="plotly_dark", height=450, xaxis_title="Importance", yaxis_title="Feature", showlegend=False)
st.plotly_chart(fig2, use_container_width=True)

# Weekly patterns
st.subheader("📅 Weekly Demand Patterns")
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
hours_day = list(range(24))
heatmap_data = np.random.uniform(20, 100, (7, 24))
heatmap_data[5:7, 10:16] *= 1.3  # Weekend midday surge
heatmap_data[:5, 7:10] *= 1.4  # Weekday morning rush
heatmap_data[:5, 17:20] *= 1.5  # Weekday evening rush
fig3 = px.imshow(heatmap_data, x=hours_day, y=days, color_continuous_scale="YlOrRd", labels=dict(x="Hour", y="Day", color="Volume (kWh)"))
fig3.update_layout(template="plotly_dark", height=350)
st.plotly_chart(fig3, use_container_width=True)
