# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""
MLflow tracking utilities for the EV Charging Tariff Optimization project.

Provides thin wrappers around common MLflow operations so that every module
records experiments consistently.

Usage::

    from src.utils.mlflow_utils import setup_mlflow, log_model_metrics, log_model_params

    setup_mlflow()
    log_model_params({"lr": 0.05, "depth": 8})
    log_model_metrics({"rmse": 0.12, "mae": 0.08}, step=1)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger(__name__)


def setup_mlflow(
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> None:
    """Initialise MLflow tracking using values from *config.yaml*.

    Parameters
    ----------
    experiment_name : str | None
        Override the experiment name from config.
    tracking_uri : str | None
        Override the tracking URI from config.
    """
    try:
        import mlflow  # deferred import – mlflow is optional
    except ImportError:
        logger.warning(
            "mlflow is not installed. Tracking calls will be no-ops. "
            "Install with: pip install mlflow"
        )
        return

    cfg = ConfigLoader()
    mlflow_cfg = cfg.get("mlflow", {})

    uri = tracking_uri or mlflow_cfg.get("tracking_uri", "mlruns")
    # If the URI is a relative path, anchor it to the project root
    if not uri.startswith(("http://", "https://", "file://")):
        project_root = Path(__file__).resolve().parents[2]
        uri = str(project_root / uri)

    exp_name = experiment_name or mlflow_cfg.get(
        "experiment_name", "ev_tariff_optimization"
    )

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp_name)
    logger.info(
        "MLflow initialised – tracking_uri=%s, experiment=%s", uri, exp_name
    )

    # Optional: enable autologging
    if mlflow_cfg.get("autolog", False):
        try:
            mlflow.autolog(log_models=mlflow_cfg.get("log_models", True))
            logger.info("MLflow autologging enabled.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not enable MLflow autolog: %s", exc)


def log_model_params(params: Dict[str, Any]) -> None:
    """Log a dict of hyper-parameters to the active MLflow run.

    Creates a new run if one is not already active.
    """
    try:
        import mlflow
    except ImportError:
        logger.debug("mlflow not installed – skipping param logging.")
        return

    if mlflow.active_run() is None:
        mlflow.start_run()
        logger.debug("Started new MLflow run for param logging.")

    mlflow.log_params(params)
    logger.debug("Logged %d params to MLflow.", len(params))


def log_model_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
) -> None:
    """Log a dict of metrics to the active MLflow run.

    Parameters
    ----------
    metrics : dict[str, float]
        Metric name → value pairs.
    step : int | None
        Optional training step / epoch number.
    """
    try:
        import mlflow
    except ImportError:
        logger.debug("mlflow not installed – skipping metric logging.")
        return

    if mlflow.active_run() is None:
        mlflow.start_run()
        logger.debug("Started new MLflow run for metric logging.")

    mlflow.log_metrics(metrics, step=step)
    logger.debug("Logged %d metrics to MLflow (step=%s).", len(metrics), step)


def end_run() -> None:
    """End the currently active MLflow run, if any."""
    try:
        import mlflow
    except ImportError:
        return

    if mlflow.active_run() is not None:
        mlflow.end_run()
        logger.debug("Ended active MLflow run.")
