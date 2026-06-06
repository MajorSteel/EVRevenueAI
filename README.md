# Agentic AI-Based Dynamic Tariff Optimization for EV Charging Networks

[![CI/CD](https://github.com/ev-charging-tariff-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/ev-charging-tariff-optimization/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **OP'26 Analytics** — Society of Business  
> A self-improving pricing engine that autonomously predicts demand, recommends dynamic tariffs, and continuously learns from outcomes.

---

## 🎯 Objective

Build an end-to-end **multi-agent AI system** that:
1. **Predicts** EV charging demand and station utilization
2. **Forecasts** congestion across time and location
3. **Recommends** optimal dynamic tariffs (vs ₹15/kWh baseline)
4. **Simulates** revenue outcomes under different pricing strategies
5. **Continuously learns** via a monitoring feedback loop

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Data Ingestion                     │
│         ACN (30K+ sessions) + UrbanEV (248          │
│         districts, 8641 timesteps)                   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Feature Engineering                     │
│    Temporal │ Demand │ Congestion │ Pricing │ Spatial│
└──────────────────────┬──────────────────────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
┌───▼───┐    ┌────────▼────────┐   ┌─────▼─────┐
│Demand │    │   Congestion    │   │   GNN     │
│Agent  │    │   Agent         │   │  Spatial  │
│XGB/LGB│    │   (>80% util)   │   │  Agent    │
└───┬───┘    └────────┬────────┘   └─────┬─────┘
    │                 │                   │
    └────────┬────────┘───────────────────┘
             │
    ┌────────▼────────┐
    │  Tariff Agent   │
    │  (PPO RL)       │
    │  ₹15 baseline   │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │    Revenue      │
    │   Simulator     │
    │ Fixed vs Dynamic│
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │   Monitoring    │
    │   & Learning    │
    │   Agent         │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │   Streamlit     │
    │   Dashboard     │
    │   (7 pages)     │
    └─────────────────┘
```

## 📊 Datasets

| Dataset | Source | Coverage | Format |
|---------|--------|----------|--------|
| **ACN-Data** | [Caltech ACN](https://ev.caltech.edu/dataset.html) | 30,000+ sessions (Apr-Dec 2018) | XLSX |
| **UrbanEV** | [ST-EVCDP](https://github.com/IntelligentSystemsLab/ST-EVCDP) | 24,798 piles, 248 districts, 5-min intervals | CSV |

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.10
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/your-repo/ev-charging-tariff-optimization.git
cd ev-charging-tariff-optimization
pip install -e .
```

### Run Full Pipeline
```bash
python scripts/run_pipeline.py
```

### Run Individual Stages
```bash
python scripts/run_pipeline.py --stage preprocess
python scripts/train_demand.py
python scripts/train_congestion.py
python scripts/train_gnn.py
python scripts/train_tariff.py
python scripts/run_simulation.py
```

### Launch Dashboard
```bash
streamlit run src/dashboard/app.py
```

### Run Tests
```bash
python -m pytest tests/ -v --tb=short
```

## 🐳 Docker

```bash
# Build and run
docker-compose up --build

# Access services
# Dashboard: http://localhost:8501
# MLflow UI: http://localhost:5000
```

## 📈 Evaluation Metrics

### Demand Prediction Agent
| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| R² | Coefficient of Determination |

### Tariff Pricing Agent
| Metric | Description |
|--------|-------------|
| Revenue Gain % | ((Dynamic − Fixed) / Fixed) × 100 vs ₹15/kWh |
| Charger Utilization Rate | Charging Time / Total Available Time |
| Off-Peak Uplift | Session increase during <30% utilization |

### Monitoring & Learning Agent
| Metric | Description |
|--------|-------------|
| Avg Wait Time Reduction | Queue length decrease at peak |
| Customer Response Rate | Demand elasticity proxy |
| Pricing Efficiency Score | Revenue per kWh delivered |

## 📁 Project Structure

```
├── config/config.yaml          # Centralized configuration
├── src/
│   ├── data/                   # Data loaders & preprocessors
│   ├── features/               # Feature engineering modules
│   ├── agents/                 # AI agents (demand, congestion, tariff, GNN, monitoring)
│   ├── models/                 # ML models (XGBoost, LightGBM, PPO, GNN)
│   ├── evaluation/             # Metrics & evaluation
│   ├── dashboard/              # Streamlit dashboard (7 pages)
│   └── utils/                  # Logging, config, DB, MLflow
├── scripts/                    # Training & pipeline scripts
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── notebooks/                  # EDA notebooks
├── Dockerfile                  # Container build
├── docker-compose.yml          # Multi-service orchestration
├── dvc.yaml                    # DVC pipeline
└── requirements.txt            # Dependencies
```

## ⚠️ Important Notes

- Causal claims are avoided unless clearly justified
- All assumptions and limitations are transparently documented
- Missing value handling strategies are documented at each stage

## 📜 License

MIT License

## 👥 Team

OP'26 Analytics — Society of Business
