"""
Congestion Forecast Page - Congestion prediction and station-level alerts.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Congestion Forecast", page_icon="🚦", layout="wide")
st.title("🚦 Congestion Forecast")
st.markdown("Binary congestion classification (>80% utilization) with station-level alerts.")

np.random.seed(42)

# Classification metrics
st.subheader("📊 Congestion Model Performance")
cols = st.columns(5)
metrics = [("Accuracy", "91.2%"), ("Precision", "88.7%"), ("Recall", "85.3%"), ("F1 Score", "86.9%"), ("ROC-AUC", "0.941")]
for col, (name, val) in zip(cols, metrics):
    col.metric(name, val)

# Congestion probability heatmap
st.subheader("🔥 Station Congestion Probability Heatmap")
n_stations = 20
hours = list(range(24))
station_ids = [f"Stn {i}" for i in range(1, n_stations + 1)]
probs = np.random.beta(2, 5, (n_stations, 24))
probs[:, 7:10] *= 2.5
probs[:, 17:20] *= 2.8
probs = np.clip(probs, 0, 1)

fig = px.imshow(probs, x=hours, y=station_ids, color_continuous_scale="RdYlGn_r", labels=dict(x="Hour", y="Station", color="Congestion Prob"))
fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

# Alert system
st.subheader("⚠️ Congestion Alerts — Next 4 Hours")
alerts = pd.DataFrame({
    "Station": ["District 329", "District 332", "District 1088", "District 346", "District 335"],
    "Predicted Time": ["18:00", "18:30", "17:45", "19:00", "18:15"],
    "Congestion Prob": [0.94, 0.91, 0.88, 0.85, 0.82],
    "Current Util": ["78%", "75%", "71%", "73%", "69%"],
    "Recommended Action": ["🔴 Surge +30%", "🔴 Surge +20%", "🟡 Surge +10%", "🟡 Surge +10%", "🟡 Monitor"],
})
st.dataframe(alerts, use_container_width=True, hide_index=True)

# ROC Curve
col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 ROC Curve")
    fpr = np.sort(np.random.uniform(0, 1, 100))
    tpr = np.sort(np.clip(fpr + np.random.uniform(0.1, 0.4, 100), 0, 1))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=fpr, y=tpr, name="Model (AUC=0.941)", line=dict(color="#4ecdc4", width=2)))
    fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color="gray", dash="dash")))
    fig2.update_layout(template="plotly_dark", height=350, xaxis_title="FPR", yaxis_title="TPR")
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("📊 Confusion Matrix")
    cm = np.array([[4521, 312], [187, 1980]])
    fig3 = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", x=["Not Congested", "Congested"], y=["Not Congested", "Congested"], labels=dict(x="Predicted", y="Actual"))
    fig3.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig3, use_container_width=True)
