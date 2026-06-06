<!--
Name: Vivek Kumar
Enroll: 23125038
Email: vivek_k@mfs.iitr.ac.in
-->
# Evaluation Report

## 1. Demand Prediction Agent

### Model Comparison
| Model | RMSE | MAE | R² | MAPE |
|-------|------|-----|-----|------|
| XGBoost | 4.56 | 3.28 | 0.823 | 8.7% |
| **LightGBM** | **4.23** | **3.15** | **0.847** | **7.9%** |

**Best Model**: LightGBM (selected by R² score)

### Feature Importance (Top 10)
| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | hour | 0.142 |
| 2 | utilization_lag_1 | 0.131 |
| 3 | volume_lag_1 | 0.118 |
| 4 | weekday | 0.095 |
| 5 | neighbor_mean_occupancy | 0.087 |
| 6 | price | 0.076 |
| 7 | occupancy_density | 0.068 |
| 8 | rolling_mean_12 | 0.062 |
| 9 | peak_period | 0.054 |
| 10 | fast_ratio | 0.048 |

## 2. Congestion Prediction Agent

### Classification Performance
| Metric | XGBoost | LightGBM |
|--------|---------|----------|
| Accuracy | 89.8% | **91.2%** |
| Precision | 86.5% | **88.7%** |
| Recall | 83.1% | **85.3%** |
| F1 Score | 84.8% | **86.9%** |
| ROC-AUC | 0.928 | **0.941** |

### Congestion Distribution
- Normal (<50% utilization): 58.5% of timesteps
- Moderate (50-80%): 29.0%
- Congested (>80%): 12.5%

## 3. Tariff Pricing Agent (PPO)

### Revenue Impact vs ₹15/kWh Baseline
| Metric | Fixed | Dynamic | Change |
|--------|-------|---------|--------|
| Total Revenue | ₹2,090,000 | ₹2,477,000 | **+18.5%** |
| Avg Price | ₹15.00/kWh | ₹16.82/kWh | +12.1% |
| Peak Revenue | ₹890,000 | ₹1,120,000 | +25.8% |
| Off-Peak Revenue | ₹320,000 | ₹419,800 | **+31.2%** |

### Utilization Impact
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Avg Utilization | 64.1% | 72.3% | +8.2pp |
| Congestion Rate | 15.8% | 12.1% | -3.7pp |
| Avg Wait Time | 4.1 min | 2.4 min | **-41.5%** |

### PPO Training Performance
- Training steps: 100,000
- Final episode reward: 28.4 (converged from -50)
- Convergence achieved at ~60,000 steps

## 4. GNN Spatial Agent

### Spatial Demand Forecasting
| Metric | Non-Spatial Baseline | GCN Model |
|--------|---------------------|-----------|
| RMSE | 5.12 | **4.38** |
| MAE | 3.89 | **3.21** |
| R² | 0.798 | **0.863** |

### Spatial Correlation
- Average spatial autocorrelation: 0.67
- CBD districts show 2.3x higher peak demand
- Adjacent districts exhibit 0.45 correlation in demand shifts

## 5. Monitoring Agent

### Feedback Loop Performance
| Metric | Value | Status |
|--------|-------|--------|
| Loop Iterations | 1,247 | Active |
| Drift Events | 0 | No drift detected |
| Auto-Retrains | 3 | Healthy |
| Pricing Efficiency Trend | ↑ 12.1% | Improving |

### Demand Elasticity Estimates
| Price Change | Demand Response | Elasticity |
|-------------|----------------|------------|
| +10% | -3.0% | -0.30 |
| +20% | -7.0% | -0.35 |
| +30% | -12.0% | -0.40 |
| -10% | +4.0% | -0.40 |
| -20% | +8.0% | -0.40 |
| **Average** | | **-0.37** |

> **Note**: Inelastic demand (|ε| < 1) is favorable for surge pricing — modest price increases generate significant revenue without proportional demand loss.

## 6. Limitations & Caveats

1. Results are based on simulated demand response within the PPO environment
2. Revenue projections assume stable demand patterns over the evaluation period
3. Elasticity estimates are derived from historical correlation, not controlled experiments
4. Cross-validation across temporal splits shows consistent but not identical metrics
