# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""
Evaluation metrics for the EV Charging Tariff Optimization project.

Provides a single ``ModelMetrics`` class with static / class methods that
cover:
  • Regression evaluation  (RMSE, MAE, R², MAPE)
  • Classification evaluation  (Accuracy, Precision, Recall, F1, ROC-AUC)
  • Domain-specific KPIs  (Revenue Gain %, Utilization Rate,
    Off-Peak Uplift, Pricing Efficiency)
  • Pretty-printed console reports

Usage
-----
>>> from src.evaluation.metrics import ModelMetrics
>>> metrics = ModelMetrics.regression_metrics(y_true, y_pred)
>>> ModelMetrics.print_report(metrics, title="XGBoost Regression")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class ModelMetrics:
    """Collection of static evaluation helpers.

    All public methods are ``@staticmethod`` or ``@classmethod`` so the class
    never needs to be instantiated – it acts as a namespace.
    """

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------
    @staticmethod
    def regression_metrics(
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> Dict[str, float]:
        """Compute standard regression metrics.

        Parameters
        ----------
        y_true : array-like
            Ground-truth target values.
        y_pred : array-like
            Model predictions.

        Returns
        -------
        dict
            Keys: ``RMSE``, ``MAE``, ``R2``, ``MAPE``.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        # MAPE – guard against zero actuals
        mask = y_true != 0
        if mask.sum() == 0:
            mape = float("inf")
            logger.warning("All true values are zero – MAPE is undefined.")
        else:
            mape = float(
                np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
            )

        metrics: Dict[str, float] = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MAPE": mape,
        }
        logger.info("Regression metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    @staticmethod
    def classification_metrics(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        y_prob: Optional[ArrayLike] = None,
    ) -> Dict[str, float]:
        """Compute standard classification metrics.

        Parameters
        ----------
        y_true : array-like
            Ground-truth binary labels.
        y_pred : array-like
            Predicted binary labels.
        y_prob : array-like, optional
            Predicted probabilities for the positive class (needed for
            ROC-AUC).

        Returns
        -------
        dict
            Keys: ``Accuracy``, ``Precision``, ``Recall``, ``F1``,
            ``ROC_AUC``.
        """
        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)

        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        roc_auc: float = 0.0
        if y_prob is not None:
            y_prob_arr = np.asarray(y_prob, dtype=np.float64)
            try:
                roc_auc = float(roc_auc_score(y_true, y_prob_arr))
            except ValueError:
                logger.warning(
                    "ROC-AUC could not be computed (e.g. single class present)."
                )
                roc_auc = float("nan")
        else:
            logger.info(
                "y_prob not provided – ROC-AUC set to 0.0. "
                "Pass predict_proba output for a meaningful value."
            )

        metrics: Dict[str, float] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
        }
        logger.info("Classification metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Domain-specific KPIs
    # ------------------------------------------------------------------
    @staticmethod
    def revenue_gain_pct(new_revenue: float, old_revenue: float) -> float:
        """Compute revenue gain percentage.

        Formula: ``((new_revenue − old_revenue) / old_revenue) × 100``

        Parameters
        ----------
        new_revenue : float
            Revenue under the new / optimised tariff.
        old_revenue : float
            Revenue under the baseline tariff.

        Returns
        -------
        float
            Percentage revenue change (positive = gain).
        """
        if old_revenue == 0:
            logger.warning("old_revenue is zero – returning inf or 0.0.")
            return float("inf") if new_revenue > 0 else 0.0
        gain = ((new_revenue - old_revenue) / old_revenue) * 100.0
        logger.info(
            "Revenue gain: %.2f%% (new=%.2f, old=%.2f)",
            gain,
            new_revenue,
            old_revenue,
        )
        return gain

    @staticmethod
    def utilization_rate(
        charging_time: float,
        total_time: float,
    ) -> float:
        """Compute charger utilization rate.

        Parameters
        ----------
        charging_time : float
            Total time (hours / minutes) chargers were in use.
        total_time : float
            Total available time window.

        Returns
        -------
        float
            Utilization ratio in [0, 1].
        """
        if total_time <= 0:
            logger.warning("total_time is <= 0 – returning 0.0.")
            return 0.0
        rate = charging_time / total_time
        rate = float(np.clip(rate, 0.0, 1.0))
        return rate

    @staticmethod
    def off_peak_uplift(
        sessions_before: int | float,
        sessions_after: int | float,
    ) -> float:
        """Percentage increase in off-peak sessions.

        Parameters
        ----------
        sessions_before : int or float
            Number of off-peak sessions *before* intervention.
        sessions_after : int or float
            Number of off-peak sessions *after* intervention.

        Returns
        -------
        float
            Percentage uplift.
        """
        if sessions_before <= 0:
            logger.warning("sessions_before is <= 0 – returning inf or 0.0.")
            return float("inf") if sessions_after > 0 else 0.0
        uplift = ((sessions_after - sessions_before) / sessions_before) * 100.0
        logger.info("Off-peak uplift: %.2f%%", uplift)
        return uplift

    @staticmethod
    def pricing_efficiency(
        revenue: float,
        kwh_delivered: float,
    ) -> float:
        """Revenue per kWh delivered.

        Parameters
        ----------
        revenue : float
            Total revenue (₹).
        kwh_delivered : float
            Total energy delivered (kWh).

        Returns
        -------
        float
            Revenue per kWh (₹ / kWh).
        """
        if kwh_delivered <= 0:
            logger.warning("kwh_delivered is <= 0 – returning 0.0.")
            return 0.0
        efficiency = revenue / kwh_delivered
        logger.info("Pricing efficiency: ₹%.4f / kWh", efficiency)
        return efficiency

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @staticmethod
    def print_report(
        metrics_dict: Dict[str, Any],
        title: str = "Evaluation Report",
    ) -> None:
        """Print a nicely formatted metrics report to the console.

        Parameters
        ----------
        metrics_dict : dict
            Metric-name → value mapping.
        title : str
            Section header printed above the table.
        """
        border = "=" * 55
        print(f"\n{border}")
        print(f"  {title}")
        print(border)
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                print(f"  {key:<25s} : {value:>12.6f}")
            else:
                print(f"  {key:<25s} : {str(value):>12s}")
        print(f"{border}\n")
        logger.info("Printed evaluation report: %s", title)
