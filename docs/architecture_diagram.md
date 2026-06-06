# System Architecture

## High-Level Architecture Diagram

```mermaid
graph TD
    subgraph Data Layer
        A1["ACN Dataset<br>30K+ Sessions"] --> DI["Data Ingestion"]
        A2["UrbanEV Dataset<br>248 Districts"] --> DI
        DI --> SV["Schema Validator<br>Pydantic"]
        SV --> PP["Preprocessor<br>Missing Values, Outliers"]
    end

    subgraph Feature Engineering
        PP --> TF["Temporal Features<br>hour, weekday, peak"]
        PP --> DF["Demand Features<br>utilization, revenue"]
        PP --> CF["Congestion Features<br>queue, density"]
        PP --> PF["Pricing Features<br>elasticity, change"]
        PP --> SF["Spatial Features<br>neighbor mean, GNN"]
    end

    subgraph Agent Layer
        TF & DF & CF & PF & SF --> DA["Demand Prediction Agent<br>XGBoost / LightGBM"]
        TF & DF & CF & PF & SF --> CA["Congestion Agent<br>Binary Classification"]
        SF --> GA["GNN Spatial Agent<br>PyTorch Geometric"]
        DA & CA & GA --> TA["Tariff Pricing Agent<br>PPO RL (SB3)"]
        TA --> RS["Revenue Simulator<br>Fixed vs Dynamic"]
        RS --> MA["Monitoring Agent<br>Feedback Loop"]
        MA -->|"retrain signal"| DA
        MA -->|"retrain signal"| TA
    end

    subgraph Infrastructure
        DA & CA & GA & TA --> ML["MLflow<br>Experiment Tracking"]
        MA --> DB["SQLite<br>Metrics Storage"]
        DA & CA & TA & RS & MA --> SD["Streamlit Dashboard<br>7 Pages"]
    end

    subgraph DevOps
        CI["GitHub Actions<br>CI/CD"] --> DK["Docker<br>Container"]
        DVC["DVC<br>Data Versioning"] --> CI
    end
```

## Component Interaction Flow

```mermaid
sequenceDiagram
    participant DI as Data Ingestion
    participant FE as Feature Engine
    participant DA as Demand Agent
    participant CA as Congestion Agent
    participant GA as GNN Agent
    participant TA as Tariff Agent
    participant RS as Revenue Simulator
    participant MA as Monitoring Agent

    DI->>FE: Raw data
    FE->>DA: Engineered features
    FE->>CA: Engineered features
    FE->>GA: Graph-structured features
    DA->>TA: Demand predictions
    CA->>TA: Congestion probabilities
    GA->>TA: Spatial forecasts
    TA->>RS: Dynamic tariff decisions
    RS->>MA: Revenue comparison results
    MA-->>DA: Retrain trigger (if drift)
    MA-->>TA: Retrain trigger (if drift)
```

## Agent Communication

| Agent | Input | Output | Downstream |
|-------|-------|--------|------------|
| Demand Prediction | Temporal + demand features | Predicted utilization, volume | Tariff Agent |
| Congestion | Temporal + congestion features | Congestion probability | Tariff Agent |
| GNN Spatial | Graph features + adj matrix | Spatial demand forecast | Tariff Agent |
| Tariff Pricing (PPO) | State: [util, demand, price, hour] | Price multiplier action | Revenue Simulator |
| Revenue Simulator | Fixed & dynamic prices | Revenue comparison | Monitoring Agent |
| Monitoring | All agent outputs | Drift alerts, feedback | All agents (retrain) |
