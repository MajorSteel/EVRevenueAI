# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""
Demand Prediction Agent for the EV Charging Tariff Optimization project.

``DemandPredictionAgent`` trains *both* XGBoost and LightGBM regressors on
temporal EV-charging features and selects the best model by R².

Targets
-------
  • ``future_volume``         – predicted session volume
  • ``future_utilization``    – predicted utilization ratio
  • ``future_energy_demand``  – predicted kWh demand

Pipeline
--------
1. ``prepare_data``   – temporal train/test split (last 20 %)
2. ``train``          – fit both tree models with early stopping
3. ``evaluate``       – compute RMSE / MAE / R² / MAPE per model
4. ``select_best_model`` – pick the model with highest R²
5. ``predict``        – inference with the best model
6. ``save_best_model``– persist best model to disk

Everything is logged to MLflow for experiment tracking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.evaluation.metrics import ModelMetrics
from src.models.tree_models import LightGBMModel, XGBoostModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Target columns this agent knows how to predict.
_SUPPORTED_TARGETS: List[str] = [
    "future_volume",
    "future_utilization",
    "future_energy_demand",
]


class DemandPredictionAgent:
    """End-to-end demand prediction pipeline.

    Parameters
    ----------
    config : dict or object
        Project configuration.  Must expose at least:
        ``test_size``, ``random_state``, ``models_dir``,
        ``mlflow_tracking_uri``, ``mlflow_experiment_name``.
        A plain ``dict`` is also accepted.
    target : str
        One of ``'future_volume'``, ``'future_utilization'``,
        ``'future_energy_demand'``.
    """

    def __init__(
        self,
        config: Any,
        target: str = "future_volume",
    ) -> None:
        # Accept both dict and object (e.g. ProjectConfig dataclass)
        if isinstance(config, dict):
            self._cfg = config
        else:
            self._cfg = config.__dict__ if hasattr(config, "__dict__") else {}

        if target not in _SUPPORTED_TARGETS:
            raise ValueError(
                f"Unsupported target '{target}'. "
                f"Choose from {_SUPPORTED_TARGETS}."
            )
        self.target = target

        # Model containers
        self._xgb_model: Optional[XGBoostModel] = None
        self._lgbm_model: Optional[LightGBMModel] = None
        self._best_model: Optional[Union[XGBoostModel, LightGBMModel]] = None
        self._best_model_name: Optional[str] = None
        self._metrics: Dict[str, Dict[str, float]] = {}

        logger.info(
            "DemandPredictionAgent initialised (target=%s)", self.target
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Temporal train / test split (last 20 % of rows).

        The dataframe is assumed to be sorted chronologically (or by a
        meaningful time index).  No shuffling is applied so as to avoid
        look-ahead bias.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset with feature columns **and** the target column
            (``self.target``).
        feature_cols : list[str], optional
            Explicit list of feature column names.  If ``None``, every
            numeric column except the target is used.

        Returns
        -------
        X_train, y_train, X_test, y_test
        """
        if self.target not in df.columns:
            raise KeyError(
                f"Target column '{self.target}' not found in dataframe. "
                f"Available columns: {list(df.columns)}"
            )

        # Derive feature columns if not specified
        if feature_cols is None:
            feature_cols = [
                c
                for c in df.select_dtypes(include=[np.number]).columns
                if c != self.target
            ]
        logger.info(
            "Using %d feature columns for target '%s'.",
            len(feature_cols),
            self.target,
        )

        X = df[feature_cols].copy()
        y = df[self.target].copy()

        # Replace infinities and fill NaNs
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0.0, inplace=True)
        y.fillna(0.0, inplace=True)

        test_size = self._cfg.get("test_size", 0.20) if isinstance(self._cfg, dict) else getattr(self._cfg, "test_size", 0.20)
        split_idx = int(len(X) * (1 - test_size))

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(
            "Data split → train=%d, test=%d (test_size=%.0f%%)",
            len(X_train),
            len(X_test),
            test_size * 100,
        )
        return X_train, y_train, X_test, y_test

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        """Train both XGBoost and LightGBM regressors.

        If no explicit validation set is provided, the last 10 % of the
        training data is carved out as a validation set for early stopping.
        """
        # Carve validation set if needed
        if X_val is None or y_val is None:
            val_split = int(len(X_train) * 0.9)
            X_val = X_train.iloc[val_split:]
            y_val = y_train.iloc[val_split:]
            X_train = X_train.iloc[:val_split]
            y_train = y_train.iloc[:val_split]
            logger.info(
                "Auto-carved validation set: train=%d, val=%d",
                len(X_train),
                len(X_val),
            )

        # --- XGBoost ---
        logger.info("Training XGBoost regressor …")
        self._xgb_model = XGBoostModel(task="regression")
        self._xgb_model.fit(
            X_train, y_train, X_val=X_val, y_val=y_val, early_stopping=True
        )

        # --- LightGBM ---
        logger.info("Training LightGBM regressor …")
        self._lgbm_model = LightGBMModel(task="regression")
        self._lgbm_model.fit(
            X_train, y_train, X_val=X_val, y_val=y_val, early_stopping=True
        )

        logger.info("Both models trained successfully.")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate both models on the held-out test set.

        Returns
        -------
        dict
            ``{"XGBoost": {…}, "LightGBM": {…}}``
        """
        results: Dict[str, Dict[str, float]] = {}

        for name, model in [("XGBoost", self._xgb_model), ("LightGBM", self._lgbm_model)]:
            if model is None:
                logger.warning("Model '%s' not trained – skipping evaluation.", name)
                continue
            y_pred = model.predict(X_test)
            metrics = ModelMetrics.regression_metrics(y_test, y_pred)
            results[name] = metrics
            ModelMetrics.print_report(metrics, title=f"{name} – {self.target}")

        self._metrics = results
        self._log_to_mlflow(results)
        return results

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    def select_best_model(self) -> Union[XGBoostModel, LightGBMModel]:
        """Select the model with the highest R² score.

        Returns
        -------
        The best model instance.
        """
        if not self._metrics:
            raise RuntimeError(
                "No evaluation results found. Call .evaluate() first."
            )

        best_name: Optional[str] = None
        best_r2: float = -np.inf

        for name, m in self._metrics.items():
            r2 = m.get("R2", -np.inf)
            if r2 > best_r2:
                best_r2 = r2
                best_name = name

        if best_name == "XGBoost":
            self._best_model = self._xgb_model
        else:
            self._best_model = self._lgbm_model

        self._best_model_name = best_name
        logger.info(
            "Best model selected: %s (R²=%.6f)", best_name, best_r2
        )
        return self._best_model  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the best model.

        If ``select_best_model`` has not been called yet, it is invoked
        automatically.
        """
        if self._best_model is None:
            logger.info("Best model not selected yet – auto-selecting …")
            self.select_best_model()
        assert self._best_model is not None
        preds = self._best_model.predict(X)
        logger.info("Generated %d predictions via %s.", len(preds), self._best_model_name)
        return preds

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_best_model(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save the best model to disk.

        Parameters
        ----------
        path : str or Path, optional
            Destination file.  Defaults to
            ``<models_dir>/demand_<target>_best.pkl``.

        Returns
        -------
        Path
            The absolute path to the saved file.
        """
        if self._best_model is None:
            self.select_best_model()

        if path is None:
            models_dir = self._cfg.get("models_dir", "models") if isinstance(self._cfg, dict) else getattr(self._cfg, "models_dir", "models")
            path = Path(models_dir) / f"demand_{self.target}_best.pkl"

        path = Path(path)
        assert self._best_model is not None
        self._best_model.save(path)
        logger.info("Best demand model saved → %s", path)
        return path

    # ------------------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------------------
    def _log_to_mlflow(self, results: Dict[str, Dict[str, float]]) -> None:
        """Log metrics and parameters to MLflow (best-effort)."""
        try:
            import mlflow

            tracking_uri = (
                self._cfg.get("mlflow_tracking_uri", "mlruns")
                if isinstance(self._cfg, dict)
                else getattr(self._cfg, "mlflow_tracking_uri", "mlruns")
            )
            experiment_name = (
                self._cfg.get("mlflow_experiment_name", "ev-tariff-optimisation")
                if isinstance(self._cfg, dict)
                else getattr(self._cfg, "mlflow_experiment_name", "ev-tariff-optimisation")
            )

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

            for model_name, metrics in results.items():
                with mlflow.start_run(run_name=f"demand_{self.target}_{model_name}"):
                    mlflow.log_param("agent", "DemandPredictionAgent")
                    mlflow.log_param("target", self.target)
                    mlflow.log_param("model", model_name)
                    for k, v in metrics.items():
                        mlflow.log_metric(k, v)
                    logger.info("MLflow run logged for %s.", model_name)

        except ImportError:
            logger.warning("mlflow not installed – skipping experiment tracking.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow logging failed: %s", exc)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DemandPredictionAgent(target={self.target!r}, "
            f"best_model={self._best_model_name})"
        )


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """Quick smoke-test with synthetic data."""
    from src.utils.config import ProjectConfig

    logger.info("Running DemandPredictionAgent standalone smoke-test …")

    config = ProjectConfig()

    # Synthetic DataFrame
    np.random.seed(42)
    n = 2000
    df = pd.DataFrame(
        {
            "hour": np.random.randint(0, 24, n),
            "day_of_week": np.random.randint(0, 7, n),
            "occupancy": np.random.uniform(0, 50, n),
            "volume_lag1": np.random.uniform(0, 100, n),
            "price": np.random.uniform(10, 20, n),
            "future_volume": np.random.uniform(0, 100, n),
        }
    )

    agent = DemandPredictionAgent(config, target="future_volume")
    X_train, y_train, X_test, y_test = agent.prepare_data(df)
    agent.train(X_train, y_train)
    results = agent.evaluate(X_test, y_test)
    best = agent.select_best_model()
    preds = agent.predict(X_test)
    saved_path = agent.save_best_model()
    logger.info("Smoke-test complete. Saved model → %s", saved_path)
