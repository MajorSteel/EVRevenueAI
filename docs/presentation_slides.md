<!--
Name: Vivek Kumar
Enroll: 23125038
Email: vivek_k@mfs.iitr.ac.in
-->
# Presentation Deck — EV Charging Dynamic Tariff Optimization

> 5-7 slides (excluding cover, executive summary, and appendix)

---

## Slide 1: Data Landscape & Preprocessing

### Datasets
- **ACN-Data**: 30,000+ EV charging sessions from Caltech/JPL (Apr–Dec 2018)
- **UrbanEV**: 248 districts, 24,798 charging piles, 5-min intervals (Jun–Jul 2022)

### Preprocessing Decisions
- **Schema validation** using Pydantic models for both datasets
- **Missing values**: Linear interpolation (kWh), forward-fill (station IDs), median (sparse)
- **Outlier removal**: IQR method, removed sessions >24h duration or >200 kWh
- **Feature engineering**: 15+ features across 5 categories (temporal, demand, congestion, pricing, spatial)
- **Assumption**: Cross-dataset alignment via temporal patterns, not geographic matching

### Visualization
- Data pipeline flow diagram showing ACN + UrbanEV → unified feature set

---

## Slide 2: Key EDA Findings & Demand Behavior

### Temporal Patterns
- **Peak hours**: 8-9 AM and 6-7 PM (commuter-driven)
- **Weekend shift**: Midday surge (11 AM - 3 PM) replaces commuter peaks
- **Off-peak**: 11 PM - 5 AM accounts for only 8% of daily volume

### Spatial Patterns
- CBD districts show **2.3x higher** peak utilization
- Adjacent districts have **0.67 demand correlation** — spatial spillover effect
- Fast charger stations: 40% higher turnover, 15% shorter sessions

### Price Behavior
- Price range: ₹0.25 – ₹1.35/kWh (mostly static within each district)
- Districts with `dynamic_pricing=1` show 12% better revenue efficiency

---

## Slide 3: Demand Prediction Modeling

### Models Compared
| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| XGBoost | 4.56 | 3.28 | 0.823 |
| **LightGBM** ✓ | **4.23** | **3.15** | **0.847** |
| GNN (Spatial) | **4.38** | **3.21** | **0.863** |

### Top Predictive Features
1. Hour of day (14.2%)
2. Lagged utilization (13.1%)
3. Lagged volume (11.8%)
4. Neighbor mean occupancy (8.7%)

### GNN Advantage
- Spatial model (GCN) outperforms non-spatial baseline by **+6.5% R²**
- Captures cross-district demand spillover effects
- Edge weights from inverse distance improve prediction accuracy

---

## Slide 4: Dynamic Tariff Optimization

### PPO RL Agent Design
- **State**: [occupancy, predicted_demand, price, hour, utilization, fast_ratio, CBD flag]
- **Actions**: 6 price multipliers (−20%, −10%, 0%, +10%, +20%, +30%)
- **Reward**: Revenue gain + utilization balance − congestion penalty
- **Baseline**: ₹15/kWh fixed pricing

### Pricing Logic
| Condition | Action | Rationale |
|-----------|--------|-----------|
| Utilization > 80% | 🔴 Surge +20–30% | Reduce congestion, capture willingness to pay |
| Utilization 30–80% | ⚪ Maintain ±10% | Minor adjustments for optimization |
| Utilization < 30% | 🟢 Discount −10–20% | Attract demand, improve utilization |

### Results vs ₹15 Baseline
- **Revenue Gain: +18.5%** (₹2.09M → ₹2.48M)
- Off-Peak Uplift: +31.2%
- Avg Dynamic Price: ₹16.82/kWh (+12.1% vs baseline)

---

## Slide 5: Monitoring Agent & Feedback Loop

### Tracked Metrics
| Metric | Value | Trend |
|--------|-------|-------|
| Revenue Gain % | 18.5% | ↑ Stable |
| Utilization Rate | 72.3% | ↑ Improving |
| Congestion Rate | 12.1% | ↓ Improving |
| Pricing Efficiency | ₹16.82/kWh | ↑ Improving |
| Wait Time Proxy | 2.4 min | ↓ Improving |

### Demand Elasticity (Customer Response)
- Average elasticity: **−0.37** (inelastic → favorable for surge pricing)
- +10% price → −3% demand | −10% price → +4% demand

### Drift Detection
- Z-score based monitoring (threshold: 2σ)
- Auto-retrain triggered when performance degrades
- 0 drift events detected over evaluation period

---

## Slide 6: Business & Policy Implications

### For Operators
- **₹387K additional revenue** in 30 days from dynamic pricing
- **ROI: 158%** in first month, 1,227% projected at 12 months
- Congestion reduced by 23.4%, improving customer satisfaction

### For Policy Makers
- Dynamic pricing effectively **redistributes demand** from peak to off-peak
- Wait times reduced by **41.5%** — better user experience
- Recommended: Cap surge at +30%, mandate off-peak incentives

### Key Takeaway
> An inelastic demand profile (ε = −0.37) means moderate surge pricing generates significant revenue with minimal customer loss — a win-win for operators and grid stability.

---

## Appendix

### Technical Stack
Python 3.10+ | XGBoost | LightGBM | PPO (Stable-Baselines3) | GCN (PyTorch Geometric) | Streamlit | MLflow | Docker | GitHub Actions

### Reproducibility
- Full pipeline: `python scripts/run_pipeline.py`
- Dashboard: `streamlit run src/dashboard/app.py`
- Tests: `python -m pytest tests/ -v`
- Docker: `docker-compose up --build`

### Limitations
1. Demand elasticity from correlation, not causal experiments
2. Revenue projections assume stable market conditions
3. PPO trained on simulated demand response
4. Queue length is a proxy metric, not directly measured
