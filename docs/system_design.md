# System Design Document

## 1. Overview

This system implements an **Agentic AI framework** for dynamic tariff optimization in EV charging networks. It uses a multi-agent architecture where specialized AI agents collaborate to predict demand, detect congestion, recommend optimal pricing, simulate revenue outcomes, and continuously improve through feedback.

## 2. Design Principles

1. **Modularity**: Each agent is independently trainable, testable, and deployable
2. **Reproducibility**: MLflow tracking + DVC pipeline + deterministic configs
3. **Scalability**: Agent-based design allows adding new agents without modifying existing ones
4. **Transparency**: All assumptions documented; no causal claims without justification
5. **Continuous Learning**: Monitoring agent detects drift and triggers retraining

## 3. Data Pipeline

### 3.1 Ingestion
- **ACN Data**: Excel file → pandas DataFrame → datetime parsing → schema validation
- **UrbanEV Data**: 9 CSV files → wide-format matrices → long-format DataFrames → graph structures

### 3.2 Feature Engineering
| Category | Features | Source |
|----------|----------|--------|
| Temporal | hour, weekday, weekend, peak_period, off_peak | Timestamps |
| Demand | utilization_rate, revenue, revenue_per_kwh | occupancy, volume, price |
| Congestion | queue_length_proxy, occupancy_density, congestion_score | occupancy, capacity, area |
| Pricing | price_change, demand_elasticity | price series |
| Spatial | neighbor_mean, weighted_neighbor_mean | adj.csv, distance.csv |

### 3.3 Preprocessing
- **Missing values**: Linear interpolation (numeric), forward-fill (categorical), median (sparse)
- **Outliers**: IQR method (1.5×IQR bounds)
- **Normalization**: MinMax scaling to [0, 1] for neural networks

## 4. Agent Design

### 4.1 Demand Prediction Agent
- **Task**: Regression — predict next-timestep volume/utilization
- **Models**: XGBoost, LightGBM (best selected by R²)
- **Features**: 15+ engineered features including temporal, demand, spatial lags
- **Evaluation**: RMSE, MAE, R²

### 4.2 Congestion Agent
- **Task**: Binary classification — utilization > 80% threshold
- **Models**: LightGBM, XGBoost classifiers with class_weight='balanced'
- **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC

### 4.3 GNN Spatial Agent
- **Architecture**: 2-layer GCN (PyTorch Geometric)
- **Graph**: 248 nodes (districts), edges from adj.csv, weights from distance.csv
- **Training**: Sliding window over time series, early stopping

### 4.4 Tariff Pricing Agent (PPO)
- **Environment**: Custom Gymnasium env with 8-dim observation, 6 discrete actions
- **Actions**: Price multipliers [-20%, -10%, 0%, +10%, +20%, +30%]
- **Reward**: α·revenue_gain + β·utilization_balance − γ·congestion_penalty
- **Baseline**: ₹15/kWh fixed pricing

### 4.5 Monitoring Agent
- **Tracking**: Revenue Gain %, Utilization, Congestion, Wait Time, Pricing Efficiency
- **Drift Detection**: Z-score method (window=100, threshold=2σ)
- **Storage**: SQLite database for persistent metrics

## 5. Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| ML Models | XGBoost, LightGBM |
| RL Agent | Stable-Baselines3 (PPO) |
| GNN | PyTorch Geometric |
| Dashboard | Streamlit |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |
| Database | SQLite |
| Config | YAML |
| Validation | Pydantic v2 |

## 6. Assumptions & Limitations

1. **Demand elasticity** is approximated from historical volume-price correlation
2. **Queue length** is a proxy (max(0, occupancy − capacity)), not measured directly
3. **Cross-dataset alignment** (ACN + UrbanEV) is by temporal features only — different geographies
4. **PPO training** uses simulated demand response, not live market interaction
5. **Price range** bounded to ±30% of baseline to avoid unrealistic tariffs
