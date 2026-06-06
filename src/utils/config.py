"""
Project-wide configuration for the EV Charging Tariff Optimization project.

Provides a ``ProjectConfig`` dataclass with all paths and business-rule
constants referenced by downstream modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Resolve project root (two levels up from this file)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ProjectConfig:
    """Immutable project configuration container.

    Attributes cover data paths, business-rule thresholds, and model
    hyper-parameter defaults.  Override individual values by passing keyword
    arguments to the constructor.
    """

    # ---- project paths ----
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = _PROJECT_ROOT / "data"
    models_dir: Path = _PROJECT_ROOT / "models"
    reports_dir: Path = _PROJECT_ROOT / "reports"
    logs_dir: Path = _PROJECT_ROOT / "logs"

    # ---- raw data paths ----
    acn_data_path: Path = Path(
        r"c:\Users\vivam\Downloads\socbiz2"
        r"\ACN Data_ 25 April 2018 to 16 Dec 2018-20260606T164559Z-3-001"
        r"\ACN Data_ 25 April 2018 to 16 Dec 2018"
        r"\acndata_sessions.json.xlsx"
    )
    urbanev_data_dir: Path = Path(
        r"c:\Users\vivam\Downloads\socbiz2"
        r"\UrbanEV_ SZ_districts-20260606T164609Z-3-001"
        r"\UrbanEV_ SZ_districts"
    )

    # ---- business-rule constants ----
    baseline_price_per_kwh: float = 15.0          # ₹15 / kWh
    surge_utilization_threshold: float = 0.80     # ≥ 80 %
    discount_utilization_threshold: float = 0.30  # ≤ 30 %

    # ---- model defaults ----
    test_size: float = 0.20
    random_state: int = 42
    early_stopping_rounds: int = 50

    # ---- MLflow ----
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "ev-tariff-optimisation"

    # ---- derived (populated in __post_init__) ----
    urbanev_files: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Create output directories and resolve UrbanEV file map."""
        for d in (self.data_dir, self.models_dir, self.reports_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

        _urbanev_names = [
            "occupancy", "volume", "duration", "price",
            "adj", "distance", "information", "stations", "time",
        ]
        self.urbanev_files = {
            name: self.urbanev_data_dir / f"{name}.csv"
            for name in _urbanev_names
        }

    # ------------------------------------------------------------------
    # Revenue helpers
    # ------------------------------------------------------------------
    @staticmethod
    def revenue_gain_pct(new_revenue: float, old_revenue: float) -> float:
        """Compute revenue gain as a percentage.

        ``((new − old) / old) × 100``
        """
        if old_revenue == 0:
            return float("inf") if new_revenue > 0 else 0.0
        return ((new_revenue - old_revenue) / old_revenue) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise config to a plain dictionary (for MLflow logging)."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }
