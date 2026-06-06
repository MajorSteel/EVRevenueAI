"""Demand-side feature engineering for EV charging tariff optimization.

This module provides the :class:`DemandFeatureEngine` which computes
utilization rates, charging durations, revenue metrics, and
lagged / rolling-window features.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class DemandFeatureEngine:
    """Compute demand-related features for EV charging data.

    Parameters
    ----------
    capacity_col : str
        Column name that holds the station/district charger capacity.
        Defaults to ``"capacity"``.
    occupancy_col : str
        Column name for real-time occupancy counts.
        Defaults to ``"occupancy"``.
    volume_col : str
        Column name for energy volume (kWh).
        Defaults to ``"volume"``.
    price_col : str
        Column name for energy price (per kWh).
        Defaults to ``"price"``.
    """

    def __init__(
        self,
        capacity_col: str = "capacity",
        occupancy_col: str = "occupancy",
        volume_col: str = "volume",
        price_col: str = "price",
    ) -> None:
        self.capacity_col = capacity_col
        self.occupancy_col = occupancy_col
        self.volume_col = volume_col
        self.price_col = price_col
        logger.info(
            "DemandFeatureEngine initialised – capacity=%s, occupancy=%s, "
            "volume=%s, price=%s",
            capacity_col,
            occupancy_col,
            volume_col,
            price_col,
        )

    # ------------------------------------------------------------------
    # Core feature methods
    # ------------------------------------------------------------------
    def compute_utilization_rate(
        self,
        occupancy: pd.Series,
        capacity: pd.Series,
    ) -> pd.Series:
        """Compute utilization rate as ``occupancy / capacity``, clipped to [0, 1].

        Parameters
        ----------
        occupancy : pd.Series
            Current occupancy (number of active sessions or vehicles).
        capacity : pd.Series
            Maximum capacity (total chargers or slots).

        Returns
        -------
        pd.Series
            Utilization rate clipped to [0.0, 1.0].
        """
        if capacity.eq(0).any():
            logger.warning(
                "Zero capacity detected in %d rows – setting utilization to 0.",
                capacity.eq(0).sum(),
            )
        utilization = occupancy / capacity.replace(0, np.nan)
        utilization = utilization.fillna(0.0).clip(lower=0.0, upper=1.0)
        logger.debug(
            "Computed utilization rate – mean=%.3f, max=%.3f",
            utilization.mean(),
            utilization.max(),
        )
        return utilization.rename("utilization_rate")

    def compute_charging_duration(self, df: pd.DataFrame) -> pd.Series:
        """Compute charging duration in hours for ACN-style session data.

        Uses ``doneChargingTime - connectionTime``.  Falls back to
        ``disconnectTime - connectionTime`` when ``doneChargingTime``
        is missing.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``connectionTime`` and at least one of
            ``doneChargingTime`` / ``disconnectTime``.

        Returns
        -------
        pd.Series
            Duration in hours.  Negative values are set to ``NaN``.
        """
        df = df.copy()
        for col in ("connectionTime", "doneChargingTime", "disconnectTime"):
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(
                df[col]
            ):
                df[col] = pd.to_datetime(df[col], errors="coerce")

        if "doneChargingTime" in df.columns:
            end_col = "doneChargingTime"
        elif "disconnectTime" in df.columns:
            end_col = "disconnectTime"
            logger.warning(
                "'doneChargingTime' missing; using 'disconnectTime' as "
                "fallback for charging duration."
            )
        else:
            raise KeyError(
                "Neither 'doneChargingTime' nor 'disconnectTime' found "
                "in the DataFrame."
            )

        if "connectionTime" not in df.columns:
            raise KeyError("'connectionTime' column is required.")

        duration_hours: pd.Series = (
            (df[end_col] - df["connectionTime"]).dt.total_seconds() / 3600.0
        )
        n_neg = (duration_hours < 0).sum()
        if n_neg > 0:
            logger.warning(
                "%d negative durations detected – setting to NaN.", n_neg
            )
            duration_hours = duration_hours.where(duration_hours >= 0, np.nan)

        logger.debug(
            "Computed charging duration – mean=%.2fh, median=%.2fh",
            duration_hours.mean(),
            duration_hours.median(),
        )
        return duration_hours.rename("charging_duration_hours")

    def compute_revenue(
        self,
        volume: pd.Series,
        price: pd.Series,
    ) -> pd.Series:
        """Compute revenue per timestep as ``volume × price``.

        Parameters
        ----------
        volume : pd.Series
            Energy volume delivered (kWh).
        price : pd.Series
            Price per kWh.

        Returns
        -------
        pd.Series
            Revenue per timestep.
        """
        revenue = (volume * price).fillna(0.0)
        logger.debug(
            "Computed revenue – total=%.2f, mean=%.4f",
            revenue.sum(),
            revenue.mean(),
        )
        return revenue.rename("revenue")

    def compute_revenue_per_kwh(
        self,
        revenue: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """Compute revenue per kWh, safely handling division by zero.

        Parameters
        ----------
        revenue : pd.Series
            Revenue values.
        volume : pd.Series
            Energy volume (kWh).

        Returns
        -------
        pd.Series
            Revenue per kWh.  ``NaN`` where volume is zero.
        """
        safe_volume = volume.replace(0, np.nan)
        rev_per_kwh = revenue / safe_volume
        n_nan = rev_per_kwh.isna().sum()
        if n_nan > 0:
            logger.info(
                "revenue_per_kwh: %d NaN values due to zero volume.", n_nan
            )
        return rev_per_kwh.rename("revenue_per_kwh")

    # ------------------------------------------------------------------
    # Lagged & rolling features
    # ------------------------------------------------------------------
    def compute_lagged_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Create lagged (shifted) versions of specified columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        columns : list[str]
            Column names to lag.
        lags : list[int], optional
            Number of timesteps to lag.
            Defaults to ``[1, 3, 6, 12]``.

        Returns
        -------
        pd.DataFrame
            DataFrame with new ``{col}_lag_{k}`` columns appended.
        """
        lags = lags or [1, 3, 6, 12]
        df = df.copy()
        new_cols: List[str] = []
        for col in columns:
            if col not in df.columns:
                logger.warning("Column '%s' not found – skipping lag.", col)
                continue
            for lag in lags:
                col_name = f"{col}_lag_{lag}"
                df[col_name] = df[col].shift(lag)
                new_cols.append(col_name)

        logger.info(
            "Created %d lagged features from %d columns with lags %s.",
            len(new_cols),
            len(columns),
            lags,
        )
        return df

    def compute_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Create rolling mean and standard deviation features.

        Window sizes are expressed in timesteps.  At 5-minute intervals:

        * 12  → 1 hour
        * 36  → 3 hours
        * 288 → 1 day

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        columns : list[str]
            Column names to compute rolling stats for.
        windows : list[int], optional
            Rolling window sizes.
            Defaults to ``[12, 36, 288]``.

        Returns
        -------
        pd.DataFrame
            DataFrame with ``{col}_rolling_mean_{w}`` and
            ``{col}_rolling_std_{w}`` columns appended.
        """
        windows = windows or [12, 36, 288]
        df = df.copy()
        new_cols: List[str] = []
        for col in columns:
            if col not in df.columns:
                logger.warning(
                    "Column '%s' not found – skipping rolling.", col
                )
                continue
            for w in windows:
                mean_name = f"{col}_rolling_mean_{w}"
                std_name = f"{col}_rolling_std_{w}"
                rolling = df[col].rolling(window=w, min_periods=1)
                df[mean_name] = rolling.mean()
                df[std_name] = rolling.std()
                new_cols.extend([mean_name, std_name])

        logger.info(
            "Created %d rolling features from %d columns with windows %s.",
            len(new_cols),
            len(columns),
            windows,
        )
        return df

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all demand feature computations on *df*.

        Expects the DataFrame to contain the configured occupancy,
        capacity, volume, and price columns.  ACN-specific fields
        (``connectionTime``, ``doneChargingTime``) are processed only
        if present.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.

        Returns
        -------
        pd.DataFrame
            DataFrame with all demand features appended.
        """
        logger.info("Running full demand feature computation …")
        df = df.copy()

        # Utilization
        if self.occupancy_col in df.columns and self.capacity_col in df.columns:
            df["utilization_rate"] = self.compute_utilization_rate(
                df[self.occupancy_col], df[self.capacity_col]
            )
        else:
            logger.info(
                "Skipping utilization_rate – missing '%s' or '%s'.",
                self.occupancy_col,
                self.capacity_col,
            )

        # Charging duration (ACN)
        if "connectionTime" in df.columns:
            df["charging_duration_hours"] = self.compute_charging_duration(df)

        # Revenue
        if self.volume_col in df.columns and self.price_col in df.columns:
            df["revenue"] = self.compute_revenue(
                df[self.volume_col], df[self.price_col]
            )
            df["revenue_per_kwh"] = self.compute_revenue_per_kwh(
                df["revenue"], df[self.volume_col]
            )
        else:
            logger.info(
                "Skipping revenue – missing '%s' or '%s'.",
                self.volume_col,
                self.price_col,
            )

        # Lagged / rolling features on numeric columns that exist
        numeric_targets = [
            c
            for c in [
                "utilization_rate",
                self.volume_col,
                self.price_col,
                "revenue",
            ]
            if c in df.columns
        ]
        if numeric_targets:
            df = self.compute_lagged_features(df, numeric_targets)
            df = self.compute_rolling_features(df, numeric_targets)

        logger.info(
            "Demand feature computation complete – DataFrame now has %d "
            "columns.",
            len(df.columns),
        )
        return df
