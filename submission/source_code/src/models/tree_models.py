# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""
Gradient-boosted tree model wrappers for the EV Charging Tariff project.

Provides two OOP wrappers – ``XGBoostModel`` and ``LightGBMModel`` – that
share the same public interface:

    model.fit(X_train, y_train, X_val, y_val, early_stopping)
    model.predict(X)
    model.predict_proba(X)          # classification only
    model.feature_importance()
    model.save(path)
    Model.load(path)                # classmethod

Both wrappers support *regression* and *classification* tasks via the
``task`` constructor argument.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XGBoost wrapper
# ---------------------------------------------------------------------------

class XGBoostModel:
    """Thin OOP wrapper around ``xgboost.XGBRegressor`` / ``XGBClassifier``.

    Parameters
    ----------
    task : str
        ``'regression'`` or ``'classification'``.
    **params
        Additional XGBoost hyper-parameters forwarded to the underlying
        estimator constructor (e.g. ``n_estimators``, ``max_depth``,
        ``learning_rate``).
    """

    _TASK_MAP = {
        "regression": "xgboost.XGBRegressor",
        "classification": "xgboost.XGBClassifier",
    }

    def __init__(self, task: str = "regression", **params: Any) -> None:
        import xgboost as xgb  # lazy import – fail fast if missing

        self.task = task.lower()
        if self.task not in self._TASK_MAP:
            raise ValueError(
                f"Unsupported task '{self.task}'. Choose 'regression' or 'classification'."
            )

        # Sensible defaults that can be overridden via **params
        defaults: Dict[str, Any] = {
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        if self.task == "classification":
            defaults["eval_metric"] = "logloss"
            defaults["use_label_encoder"] = False
        else:
            defaults["eval_metric"] = "rmse"

        merged = {**defaults, **params}

        if self.task == "regression":
            self.model: Any = xgb.XGBRegressor(**merged)
        else:
            self.model = xgb.XGBClassifier(**merged)

        self._is_fitted: bool = False
        self._feature_names: list[str] = []
        logger.info(
            "XGBoostModel created (task=%s, params=%s)", self.task, merged
        )

    # ---- training --------------------------------------------------------
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: ArrayLike,
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[ArrayLike] = None,
        early_stopping: bool = True,
    ) -> "XGBoostModel":
        """Fit the model.

        Parameters
        ----------
        X_train, y_train:
            Training features and target.
        X_val, y_val:
            Optional validation set (required when *early_stopping* is
            ``True``).
        early_stopping:
            If ``True`` and a validation set is provided, training stops when
            the validation metric has not improved for 50 consecutive rounds.

        Returns
        -------
        self
        """
        fit_params: Dict[str, Any] = {}

        if early_stopping and X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False
            # XGBoost ≥ 2.0 uses 'callbacks' for early stopping;
            # the sklearn wrapper still accepts early_stopping_rounds.
            try:
                import xgboost as xgb
                callback = xgb.callback.EarlyStopping(
                    rounds=50,
                    save_best=True,
                    maximize=False,
                )
                fit_params["callbacks"] = [callback]
            except (AttributeError, TypeError):
                # Fallback for older xgboost versions
                fit_params["early_stopping_rounds"] = 50
        elif X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        if isinstance(X_train, pd.DataFrame):
            self._feature_names = list(X_train.columns)
        else:
            self._feature_names = [f"f{i}" for i in range(X_train.shape[1])]

        logger.info(
            "Training XGBoost (%s) on %d samples, %d features …",
            self.task,
            X_train.shape[0],
            X_train.shape[1],
        )
        self.model.fit(X_train, y_train, **fit_params)
        self._is_fitted = True
        logger.info("XGBoost training complete.")
        return self

    # ---- inference -------------------------------------------------------
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return point predictions."""
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return class probabilities (classification only).

        Returns the probability of the **positive** class (column 1).
        """
        if self.task != "classification":
            raise RuntimeError(
                "predict_proba is only available for classification tasks."
            )
        self._check_fitted()
        probas = self.model.predict_proba(X)
        # Return positive-class probability
        if probas.ndim == 2 and probas.shape[1] == 2:
            return probas[:, 1]
        return probas

    # ---- feature importance ----------------------------------------------
    def feature_importance(self) -> Dict[str, float]:
        """Return a dict mapping feature names to importance scores.

        Uses XGBoost's built-in ``feature_importances_`` (gain-based).
        """
        self._check_fitted()
        importances = self.model.feature_importances_
        fi = dict(zip(self._feature_names, importances.tolist()))
        fi = dict(sorted(fi.items(), key=lambda kv: kv[1], reverse=True))
        return fi

    # ---- persistence -----------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Persist the model to disk (pickle)."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("XGBoostModel saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "XGBoostModel":
        """Load a previously saved ``XGBoostModel`` from disk."""
        path = Path(path)
        with open(path, "rb") as fh:
            obj = pickle.load(fh)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected XGBoostModel, got {type(obj).__name__}")
        logger.info("XGBoostModel loaded from %s", path)
        return obj

    # ---- internals -------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    def __repr__(self) -> str:  # pragma: no cover
        return f"XGBoostModel(task={self.task!r}, fitted={self._is_fitted})"


# ---------------------------------------------------------------------------
# LightGBM wrapper
# ---------------------------------------------------------------------------

class LightGBMModel:
    """Thin OOP wrapper around ``lightgbm.LGBMRegressor`` / ``LGBMClassifier``.

    Parameters
    ----------
    task : str
        ``'regression'`` or ``'classification'``.
    **params
        Additional LightGBM hyper-parameters forwarded to the underlying
        estimator constructor.
    """

    _TASK_MAP = {
        "regression": "lightgbm.LGBMRegressor",
        "classification": "lightgbm.LGBMClassifier",
    }

    def __init__(self, task: str = "regression", **params: Any) -> None:
        import lightgbm as lgb  # lazy import

        self.task = task.lower()
        if self.task not in self._TASK_MAP:
            raise ValueError(
                f"Unsupported task '{self.task}'. Choose 'regression' or 'classification'."
            )

        defaults: Dict[str, Any] = {
            "n_estimators": 1000,
            "max_depth": -1,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        if self.task == "classification":
            defaults["metric"] = "binary_logloss"
        else:
            defaults["metric"] = "rmse"

        merged = {**defaults, **params}

        if self.task == "regression":
            self.model: Any = lgb.LGBMRegressor(**merged)
        else:
            self.model = lgb.LGBMClassifier(**merged)

        self._is_fitted: bool = False
        self._feature_names: list[str] = []
        logger.info(
            "LightGBMModel created (task=%s, params=%s)", self.task, merged
        )

    # ---- training --------------------------------------------------------
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: ArrayLike,
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[ArrayLike] = None,
        early_stopping: bool = True,
    ) -> "LightGBMModel":
        """Fit the model.

        Parameters
        ----------
        X_train, y_train:
            Training features and target.
        X_val, y_val:
            Optional validation set.
        early_stopping:
            If ``True`` and a validation set is provided, training uses
            ``callbacks.early_stopping(50)``.

        Returns
        -------
        self
        """
        import lightgbm as lgb

        fit_params: Dict[str, Any] = {}

        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["eval_metric"] = (
                "binary_logloss" if self.task == "classification" else "rmse"
            )

            callbacks: list[Any] = [lgb.log_evaluation(period=0)]
            if early_stopping:
                callbacks.append(lgb.early_stopping(stopping_rounds=50))
            fit_params["callbacks"] = callbacks

        if isinstance(X_train, pd.DataFrame):
            self._feature_names = list(X_train.columns)
        else:
            self._feature_names = [f"f{i}" for i in range(X_train.shape[1])]

        logger.info(
            "Training LightGBM (%s) on %d samples, %d features …",
            self.task,
            X_train.shape[0],
            X_train.shape[1],
        )
        self.model.fit(X_train, y_train, **fit_params)
        self._is_fitted = True
        logger.info("LightGBM training complete.")
        return self

    # ---- inference -------------------------------------------------------
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return point predictions."""
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return class probabilities (classification only).

        Returns the probability of the **positive** class.
        """
        if self.task != "classification":
            raise RuntimeError(
                "predict_proba is only available for classification tasks."
            )
        self._check_fitted()
        probas = self.model.predict_proba(X)
        if probas.ndim == 2 and probas.shape[1] == 2:
            return probas[:, 1]
        return probas

    # ---- feature importance ----------------------------------------------
    def feature_importance(self) -> Dict[str, float]:
        """Return a dict mapping feature names to importance scores."""
        self._check_fitted()
        importances = self.model.feature_importances_
        fi = dict(zip(self._feature_names, importances.tolist()))
        fi = dict(sorted(fi.items(), key=lambda kv: kv[1], reverse=True))
        return fi

    # ---- persistence -----------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Persist the model to disk (pickle)."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("LightGBMModel saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LightGBMModel":
        """Load a previously saved ``LightGBMModel`` from disk."""
        path = Path(path)
        with open(path, "rb") as fh:
            obj = pickle.load(fh)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected LightGBMModel, got {type(obj).__name__}")
        logger.info("LightGBMModel loaded from %s", path)
        return obj

    # ---- internals -------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    def __repr__(self) -> str:  # pragma: no cover
        return f"LightGBMModel(task={self.task!r}, fitted={self._is_fitted})"
