"""
Data preprocessor – cleaning, outlier detection, normalisation, and
cross-dataset unification for ACN + UrbanEV data.

Usage::

    from src.data.preprocessor import DataPreprocessor

    preprocessor = DataPreprocessor()
    clean_df = preprocessor.handle_missing(df, strategy="interpolate")
    outliers = preprocessor.detect_outliers(df, columns=["kWhDelivered"])
    normed   = preprocessor.normalize(df, columns=["kWhDelivered"], method="minmax")
    unified  = preprocessor.create_unified_dataset()
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader
from src.data.acn_loader import ACNDataLoader
from src.data.urbanev_loader import UrbanEVDataLoader

logger = get_logger(__name__)

MissingStrategy = Literal["interpolate", "ffill", "bfill", "median", "mean", "drop"]
OutlierMethod = Literal["iqr", "zscore"]
NormMethod = Literal["minmax", "standard"]


class DataPreprocessor:
    """General-purpose preprocessor shared by both ACN and UrbanEV pipelines.

    The class is stateless by default – each method takes a DataFrame,
    transforms it, and returns a new copy.  ``create_unified_dataset``
    is the one convenience method that internally instantiates the
    two data loaders and produces a combined feature DataFrame.

    Assumptions
    -----------
    * ACN data is session-level; features are aggregated to hourly
      station-level rows before merging.
    * UrbanEV timeseries are already at 5-minute resolution; they are
      resampled to hourly averages for alignment with ACN.
    * Missing-value strategies are applied **column-wise**.
    * Outlier detection flags rows but does **not** remove them
      automatically – the caller decides.
    """

    def __init__(self) -> None:
        self._scalers: Dict[str, MinMaxScaler | StandardScaler] = {}
        logger.info("DataPreprocessor initialised.")

    # ==================================================================
    # Missing values
    # ==================================================================
    def handle_missing(
        self,
        df: pd.DataFrame,
        strategy: MissingStrategy = "interpolate",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fill or drop missing values.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        strategy : str
            One of ``"interpolate"``, ``"ffill"``, ``"bfill"``,
            ``"median"``, ``"mean"``, ``"drop"``.
        columns : list[str] | None
            Columns to apply the strategy to.  *None* → all columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values handled.
        """
        df = df.copy()
        cols = columns or list(df.columns)
        missing_before = df[cols].isnull().sum().sum()

        if strategy == "interpolate":
            df[cols] = df[cols].interpolate(method="linear", limit_direction="both")
        elif strategy == "ffill":
            df[cols] = df[cols].ffill()
        elif strategy == "bfill":
            df[cols] = df[cols].bfill()
        elif strategy == "median":
            for col in cols:
                if df[col].dtype.kind in "iufb":  # numeric
                    df[col] = df[col].fillna(df[col].median())
        elif strategy == "mean":
            for col in cols:
                if df[col].dtype.kind in "iufb":
                    df[col] = df[col].fillna(df[col].mean())
        elif strategy == "drop":
            df = df.dropna(subset=cols)
        else:
            raise ValueError(f"Unknown missing-value strategy: {strategy!r}")

        missing_after = df[cols].isnull().sum().sum() if strategy != "drop" else 0
        logger.info(
            "handle_missing(%s): %d → %d NaN values (cols=%d)",
            strategy,
            missing_before,
            missing_after,
            len(cols),
        )
        return df

    # ==================================================================
    # Outlier detection
    # ==================================================================
    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: OutlierMethod = "iqr",
        threshold: float = 1.5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Flag outlier rows without removing them.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        columns : list[str]
            Numeric columns to inspect.
        method : str
            ``"iqr"`` (inter-quartile range) or ``"zscore"``.
        threshold : float
            IQR multiplier (default 1.5) or z-score cutoff (default 3.0
            when *method* is ``"zscore"``).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(full_df_with_outlier_flag, outlier_rows_only)``
        """
        df = df.copy()
        outlier_mask = pd.Series(False, index=df.index)

        for col in columns:
            if col not in df.columns or df[col].dtype.kind not in "iuf":
                logger.warning("Skipping non-numeric or missing column: %s", col)
                continue

            if method == "iqr":
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                col_outliers = (df[col] < lower) | (df[col] > upper)
            elif method == "zscore":
                z = (df[col] - df[col].mean()) / df[col].std()
                z_thresh = threshold if threshold > 2.0 else 3.0
                col_outliers = z.abs() > z_thresh
            else:
                raise ValueError(f"Unknown outlier method: {method!r}")

            outlier_mask |= col_outliers
            logger.debug(
                "Outliers in '%s' (%s): %d rows", col, method, col_outliers.sum()
            )

        df["is_outlier"] = outlier_mask
        outlier_df = df.loc[df["is_outlier"]].copy()
        logger.info(
            "detect_outliers: %d / %d rows flagged (method=%s, threshold=%.2f)",
            len(outlier_df),
            len(df),
            method,
            threshold,
        )
        return df, outlier_df

    # ==================================================================
    # Normalisation
    # ==================================================================
    def normalize(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: NormMethod = "minmax",
    ) -> pd.DataFrame:
        """Scale numeric columns in-place.

        Fitted scalers are stored in ``self._scalers`` keyed by column name
        so they can be reused for inverse-transform or test-set scaling.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        columns : list[str]
            Columns to normalise.
        method : str
            ``"minmax"`` (0-1 range) or ``"standard"`` (zero-mean, unit-var).

        Returns
        -------
        pd.DataFrame
            DataFrame with normalised columns.
        """
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                logger.warning("Column '%s' not in DataFrame – skipping.", col)
                continue

            values = df[[col]].values.astype(np.float64)
            non_null = ~np.isnan(values).ravel()

            if method == "minmax":
                scaler = MinMaxScaler()
            elif method == "standard":
                scaler = StandardScaler()
            else:
                raise ValueError(f"Unknown normalize method: {method!r}")

            scaler.fit(values[non_null].reshape(-1, 1))
            transformed = values.copy()
            transformed[non_null] = scaler.transform(
                values[non_null].reshape(-1, 1)
            ).ravel()
            df[col] = transformed
            self._scalers[col] = scaler

        logger.info(
            "normalize(%s): scaled %d columns.", method, len(columns)
        )
        return df

    def inverse_normalize(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> pd.DataFrame:
        """Reverse a previous normalisation using stored scalers."""
        df = df.copy()
        for col in columns:
            if col not in self._scalers:
                logger.warning(
                    "No scaler stored for '%s' – skipping inverse.", col
                )
                continue
            scaler = self._scalers[col]
            df[col] = scaler.inverse_transform(df[[col]].values)
        return df

    # ==================================================================
    # Unified dataset
    # ==================================================================
    def create_unified_dataset(self) -> pd.DataFrame:
        """Merge ACN session features with UrbanEV timeseries features.

        Workflow
        --------
        1. Load and clean ACN sessions → aggregate to hourly station-level.
        2. Load UrbanEV timeseries → resample to hourly averages.
        3. Concatenate as a combined feature table (outer join on timestamp).

        Returns
        -------
        pd.DataFrame
            Unified DataFrame indexed by timestamp.

        Notes
        -----
        * The two datasets cover different time spans and geographies,
          so the merge is an **outer** join.  Downstream consumers
          should handle NaN columns from the "other" source.
        * ACN features: ``mean_kWh``, ``session_count``,
          ``mean_session_hours``.
        * UrbanEV features: ``mean_occupancy``, ``mean_volume``,
          ``mean_duration``, ``mean_price``.
        """
        logger.info("Creating unified dataset from ACN + UrbanEV …")

        # --- ACN aggregation ---
        try:
            acn = ACNDataLoader()
            acn_df = acn.get_sessions()

            acn_df = acn_df.set_index("connectionTime")
            acn_hourly = acn_df.resample("h").agg(
                mean_kWh=("kWhDelivered", "mean"),
                session_count=("kWhDelivered", "count"),
                mean_session_hours=(
                    "session_duration_hours",
                    "mean",
                ),
            )
            acn_hourly = acn_hourly.add_prefix("acn_")
            logger.info(
                "ACN hourly features: %d rows.", len(acn_hourly)
            )
        except Exception as exc:
            logger.error("Failed to load ACN data: %s", exc)
            acn_hourly = pd.DataFrame()

        # --- UrbanEV aggregation ---
        try:
            uev = UrbanEVDataLoader()

            metrics: Dict[str, pd.DataFrame] = {}
            for metric_name in ("occupancy", "volume", "duration", "price"):
                wide = getattr(uev, metric_name)
                # Mean across all districts per timestep
                ts = wide.mean(axis=1).rename(f"urbanev_mean_{metric_name}")
                metrics[metric_name] = ts.to_frame()

            urbanev_ts = pd.concat(metrics.values(), axis=1)
            # Resample to hourly
            urbanev_hourly = urbanev_ts.resample("h").mean()
            logger.info(
                "UrbanEV hourly features: %d rows.", len(urbanev_hourly)
            )
        except Exception as exc:
            logger.error("Failed to load UrbanEV data: %s", exc)
            urbanev_hourly = pd.DataFrame()

        # --- Merge ---
        if acn_hourly.empty and urbanev_hourly.empty:
            logger.warning("Both data sources returned empty – unified dataset is empty.")
            return pd.DataFrame()

        unified = pd.concat([acn_hourly, urbanev_hourly], axis=1, join="outer")
        unified.index.name = "timestamp"

        # Basic imputation on the merged frame
        unified = self.handle_missing(unified, strategy="interpolate")

        logger.info(
            "Unified dataset: %d rows × %d columns.",
            len(unified),
            len(unified.columns),
        )
        return unified
