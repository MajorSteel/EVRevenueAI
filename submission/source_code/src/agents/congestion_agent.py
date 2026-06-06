# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""
Congestion Prediction Agent for the EV Charging Tariff Optimization project.

``CongestionPredictionAgent`` trains *both* XGBoost and LightGBM classifiers
to predict whether a charging station / district will become **congested**
(utilization > threshold).

Pipeline
--------
1. ``prepare_data``      – binarise utilization → congestion label, split
2. ``handle_imbalance``  – adjust class weights (``class_weight='balanced'``)
3. ``train``             – fit both classifiers with early stopping
4. ``evaluate``          – Accuracy / Precision / Recall / F1 / ROC-AUC
5. ``predict_proba``     – congestion probability per sample
6. ``save_best_model``   – persist best model to disk

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


class CongestionPredictionAgent:
    """End-to-end congestion classification pipeline.

    Parameters
    ----------
    config : dict or object
        Project configuration.  Must expose at least:
        ``test_size``, ``random_state``, ``models_dir``,
        ``surge_utilization_threshold``,
        ``mlflow_tracking_uri``, ``mlflow_experiment_name``.
        A plain ``dict`` is also accepted.
    """

    def __init__(self, config: Any) -> None:
        if isinstance(config, dict):
            self._cfg = config
        else:
            self._cfg = config.__dict__ if hasattr(config, "__dict__") else {}

        self._threshold: float = (
            self._cfg.get("surge_utilization_threshold", 0.80)
            if isinstance(self._cfg, dict)
            else getattr(config, "surge_utilization_threshold", 0.80)
        )

        # Model containers
        self._xgb_model: Optional[XGBoostModel] = None
        self._lgbm_model: Optional[LightGBMModel] = None
        self._best_model: Optional[Union[XGBoostModel, LightGBMModel]] = None
        self._best_model_name: Optional[str] = None
        self._metrics: Dict[str, Dict[str, float]] = {}
        self._use_balanced_weights: bool = False

        logger.info(
            "CongestionPredictionAgent initialised (threshold=%.2f)",
            self._threshold,
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def prepare_data(
        self,
        df: pd.DataFrame,
        utilization_col: str = "utilization",
        threshold: Optional[float] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Create a binary congestion target and perform a temporal split.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset – must contain ``utilization_col``.
        utilization_col : str
            Column name that holds the utilization ratio [0, 1].
        threshold : float, optional
            Override for the congestion threshold.  Defaults to the
            value from the config (``surge_utilization_threshold``).
        feature_cols : list[str], optional
            Explicit list of feature column names.

        Returns
        -------
        X_train, y_train, X_test, y_test
        """
        thresh = threshold if threshold is not None else self._threshold

        if utilization_col not in df.columns:
            raise KeyError(
                f"Column '{utilization_col}' not in dataframe. "
                f"Available: {list(df.columns)}"
            )

        # Binary target
        target_col = "congested"
        df = df.copy()
        df[target_col] = (df[utilization_col] > thresh).astype(int)
        logger.info(
            "Congestion target created (threshold=%.2f): "
            "%d congested / %d total (%.1f%%)",
            thresh,
            df[target_col].sum(),
            len(df),
            df[target_col].mean() * 100,
        )

        # Feature columns
        if feature_cols is None:
            exclude = {utilization_col, target_col}
            feature_cols = [
                c
                for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]
        logger.info("Using %d feature columns.", len(feature_cols))

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Clean up
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0.0, inplace=True)

        test_size = (
            self._cfg.get("test_size", 0.20)
            if isinstance(self._cfg, dict)
            else getattr(self._cfg, "test_size", 0.20)
        )
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
    # Imbalance handling
    # ------------------------------------------------------------------
    def handle_imbalance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "class_weight",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance.

        Currently supports ``'class_weight'`` which instructs downstream
        model constructors to use ``scale_pos_weight`` (XGBoost) or
        ``class_weight='balanced'`` (LightGBM).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (returned unchanged).
        y : pd.Series
            Binary target vector (returned unchanged).
        method : str
            Strategy.  Only ``'class_weight'`` is supported.

        Returns
        -------
        X, y  (unchanged – side-effect is setting an internal flag)
        """
        if method != "class_weight":
            raise ValueError(
                f"Unsupported imbalance method '{method}'. "
                "Only 'class_weight' is currently supported."
            )

        pos = int(y.sum())
        neg = int(len(y) - pos)
        ratio = neg / max(pos, 1)
        logger.info(
            "Imbalance handling (%s): pos=%d, neg=%d, ratio=%.2f",
            method,
            pos,
            neg,
            ratio,
        )
        self._use_balanced_weights = True
        self._scale_pos_weight = ratio
        return X, y

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
        """Train both XGBoost and LightGBM classifiers.

        If no explicit validation set is provided, the last 10 % of
        training data is carved out.
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

        # Model hyper-param overrides for imbalance
        xgb_extra: Dict[str, Any] = {}
        lgbm_extra: Dict[str, Any] = {}
        if self._use_balanced_weights:
            xgb_extra["scale_pos_weight"] = self._scale_pos_weight
            lgbm_extra["class_weight"] = "balanced"
            logger.info(
                "Applying balanced weights (XGB scale_pos_weight=%.2f).",
                self._scale_pos_weight,
            )

        # --- LightGBM ---
        logger.info("Training LightGBM classifier …")
        self._lgbm_model = LightGBMModel(task="classification", **lgbm_extra)
        self._lgbm_model.fit(
            X_train, y_train, X_val=X_val, y_val=y_val, early_stopping=True
        )

        # --- XGBoost ---
        logger.info("Training XGBoost classifier …")
        self._xgb_model = XGBoostModel(task="classification", **xgb_extra)
        self._xgb_model.fit(
            X_train, y_train, X_val=X_val, y_val=y_val, early_stopping=True
        )

        logger.info("Both classifiers trained successfully.")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate both classifiers on the held-out test set.

        Returns
        -------
        dict
            ``{"LightGBM": {…}, "XGBoost": {…}}``
        """
        results: Dict[str, Dict[str, float]] = {}

        for name, model in [
            ("LightGBM", self._lgbm_model),
            ("XGBoost", self._xgb_model),
        ]:
            if model is None:
                logger.warning("Model '%s' not trained – skipping.", name)
                continue

            y_pred = model.predict(X_test)
            # For classification, LightGBM may return float probabilities
            y_pred_labels = (np.asarray(y_pred) >= 0.5).astype(int)

            try:
                y_prob = model.predict_proba(X_test)
            except Exception:  # noqa: BLE001
                y_prob = None

            metrics = ModelMetrics.classification_metrics(
                y_test, y_pred_labels, y_prob=y_prob
            )
            results[name] = metrics
            ModelMetrics.print_report(metrics, title=f"{name} – Congestion")

        self._metrics = results
        self._log_to_mlflow(results)
        return results

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return congestion probability per sample using the best model.

        If ``select_best_model()`` has not been called, it is invoked
        automatically.
        """
        if self._best_model is None:
            self._select_best_model()
        assert self._best_model is not None
        proba = self._best_model.predict_proba(X)
        logger.info(
            "Generated congestion probabilities for %d samples via %s.",
            len(proba),
            self._best_model_name,
        )
        return proba

    # ------------------------------------------------------------------
    # Model selection (internal)
    # ------------------------------------------------------------------
    def _select_best_model(self) -> Union[XGBoostModel, LightGBMModel]:
        """Select the best model by F1 score (or ROC-AUC as tie-breaker)."""
        if not self._metrics:
            raise RuntimeError(
                "No evaluation results found. Call .evaluate() first."
            )

        best_name: Optional[str] = None
        best_f1: float = -np.inf

        for name, m in self._metrics.items():
            f1 = m.get("F1", -np.inf)
            if f1 > best_f1:
                best_f1 = f1
                best_name = name

        if best_name == "XGBoost":
            self._best_model = self._xgb_model
        else:
            self._best_model = self._lgbm_model

        self._best_model_name = best_name
        logger.info("Best classifier selected: %s (F1=%.6f)", best_name, best_f1)
        return self._best_model  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_best_model(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save the best classifier to disk.

        Parameters
        ----------
        path : str or Path, optional
            Destination file.  Defaults to
            ``<models_dir>/congestion_best.pkl``.

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        if self._best_model is None:
            self._select_best_model()

        if path is None:
            models_dir = (
                self._cfg.get("models_dir", "models")
                if isinstance(self._cfg, dict)
                else getattr(self._cfg, "models_dir", "models")
            )
            path = Path(models_dir) / "congestion_best.pkl"

        path = Path(path)
        assert self._best_model is not None
        self._best_model.save(path)
        logger.info("Best congestion model saved → %s", path)
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
                with mlflow.start_run(run_name=f"congestion_{model_name}"):
                    mlflow.log_param("agent", "CongestionPredictionAgent")
                    mlflow.log_param("threshold", self._threshold)
                    mlflow.log_param("model", model_name)
                    mlflow.log_param("balanced_weights", self._use_balanced_weights)
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
            f"CongestionPredictionAgent(threshold={self._threshold}, "
            f"best_model={self._best_model_name})"
        )


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """Quick smoke-test with synthetic data."""
    from src.utils.config import ProjectConfig

    logger.info("Running CongestionPredictionAgent standalone smoke-test …")

    config = ProjectConfig()

    # Synthetic DataFrame
    np.random.seed(42)
    n = 2000
    utilization = np.random.uniform(0, 1, n)
    df = pd.DataFrame(
        {
            "hour": np.random.randint(0, 24, n),
            "day_of_week": np.random.randint(0, 7, n),
            "occupancy": np.random.uniform(0, 50, n),
            "volume_lag1": np.random.uniform(0, 100, n),
            "price": np.random.uniform(10, 20, n),
            "utilization": utilization,
        }
    )

    agent = CongestionPredictionAgent(config)
    X_train, y_train, X_test, y_test = agent.prepare_data(df)
    X_train, y_train = agent.handle_imbalance(X_train, y_train)
    agent.train(X_train, y_train)
    results = agent.evaluate(X_test, y_test)
    proba = agent.predict_proba(X_test)
    saved_path = agent.save_best_model()
    logger.info("Smoke-test complete. Saved model → %s", saved_path)
