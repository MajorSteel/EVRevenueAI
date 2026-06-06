# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""Temporal feature extraction for EV charging tariff optimization.

This module provides the :class:`TemporalFeatureEngine` which derives
calendar/time-of-day features from a configurable datetime column in a
``pandas.DataFrame``.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class TemporalFeatureEngine:
    """Extract temporal features from a datetime column.

    Parameters
    ----------
    datetime_col : str, optional
        Name of the datetime column in the input DataFrame.
        Defaults to ``"timestamp"``.
    peak_hours : list[tuple[int, int]], optional
        List of ``(start_hour, end_hour)`` ranges (inclusive start,
        exclusive end) that define *peak* periods.
        Defaults to ``[(7, 10), (17, 21)]`` (7-9 AM & 5-8 PM).
    off_peak_hours : list[tuple[int, int]], optional
        Same format as *peak_hours* for *off-peak* windows.
        Defaults to ``[(22, 24), (0, 7)]`` (10 PM – 6 AM).
    """

    def __init__(
        self,
        datetime_col: str = "timestamp",
        peak_hours: Optional[List[tuple[int, int]]] = None,
        off_peak_hours: Optional[List[tuple[int, int]]] = None,
    ) -> None:
        self.datetime_col = datetime_col
        self.peak_hours = peak_hours or [(7, 10), (17, 21)]
        self.off_peak_hours = off_peak_hours or [(22, 24), (0, 7)]
        logger.info(
            "TemporalFeatureEngine initialised – datetime_col=%s, "
            "peak_hours=%s, off_peak_hours=%s",
            self.datetime_col,
            self.peak_hours,
            self.off_peak_hours,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the datetime column exists and is of datetime dtype.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with the datetime column cast to ``datetime64``.

        Raises
        ------
        KeyError
            If ``self.datetime_col`` is not present in *df*.
        """
        if self.datetime_col not in df.columns:
            raise KeyError(
                f"Datetime column '{self.datetime_col}' not found in "
                f"DataFrame. Available columns: {list(df.columns)}"
            )
        if not pd.api.types.is_datetime64_any_dtype(df[self.datetime_col]):
            logger.warning(
                "Column '%s' is not datetime; attempting conversion.",
                self.datetime_col,
            )
            df = df.copy()
            df[self.datetime_col] = pd.to_datetime(
                df[self.datetime_col], errors="coerce"
            )
        return df

    @staticmethod
    def _hour_in_ranges(hour: int, ranges: List[tuple[int, int]]) -> bool:
        """Check whether *hour* falls in any of the given ``(lo, hi)`` ranges."""
        return any(lo <= hour < hi for lo, hi in ranges)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_hour(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add an ``hour`` column (0–23).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``self.datetime_col``.

        Returns
        -------
        pd.DataFrame
            DataFrame with an ``hour`` column appended.
        """
        df = self._ensure_datetime(df)
        df = df.copy()
        df["hour"] = df[self.datetime_col].dt.hour
        logger.debug("Extracted 'hour' column (%d rows).", len(df))
        return df

    def extract_weekday(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a ``weekday`` column (0 = Monday … 6 = Sunday).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``self.datetime_col``.

        Returns
        -------
        pd.DataFrame
            DataFrame with a ``weekday`` column appended.
        """
        df = self._ensure_datetime(df)
        df = df.copy()
        df["weekday"] = df[self.datetime_col].dt.dayofweek
        logger.debug("Extracted 'weekday' column (%d rows).", len(df))
        return df

    def extract_weekend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a boolean ``weekend`` column (True for Saturday/Sunday).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``self.datetime_col``.

        Returns
        -------
        pd.DataFrame
            DataFrame with a ``weekend`` column appended.
        """
        df = self._ensure_datetime(df)
        df = df.copy()
        df["weekend"] = df[self.datetime_col].dt.dayofweek >= 5
        logger.debug("Extracted 'weekend' column (%d rows).", len(df))
        return df

    def extract_peak_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a boolean ``peak_period`` column.

        A timestep is *peak* when its hour is inside any of the
        ``self.peak_hours`` ranges (default 7-9 AM, 5-8 PM).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``self.datetime_col``.

        Returns
        -------
        pd.DataFrame
            DataFrame with a ``peak_period`` column appended.
        """
        df = self._ensure_datetime(df)
        df = df.copy()
        hours = df[self.datetime_col].dt.hour
        df["peak_period"] = hours.apply(
            lambda h: self._hour_in_ranges(h, self.peak_hours)
        )
        logger.debug(
            "Extracted 'peak_period' column – %d / %d rows are peak.",
            df["peak_period"].sum(),
            len(df),
        )
        return df

    def extract_off_peak_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a boolean ``off_peak_period`` column.

        A timestep is *off-peak* when its hour is inside any of the
        ``self.off_peak_hours`` ranges (default 10 PM – 6 AM).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``self.datetime_col``.

        Returns
        -------
        pd.DataFrame
            DataFrame with an ``off_peak_period`` column appended.
        """
        df = self._ensure_datetime(df)
        df = df.copy()
        hours = df[self.datetime_col].dt.hour
        df["off_peak_period"] = hours.apply(
            lambda h: self._hour_in_ranges(h, self.off_peak_hours)
        )
        logger.debug(
            "Extracted 'off_peak_period' column – %d / %d rows are off-peak.",
            df["off_peak_period"].sum(),
            len(df),
        )
        return df

    def extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run **all** temporal feature extractions in sequence.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``self.datetime_col``.

        Returns
        -------
        pd.DataFrame
            DataFrame with ``hour``, ``weekday``, ``weekend``,
            ``peak_period``, and ``off_peak_period`` columns appended.
        """
        logger.info("Running full temporal feature extraction …")
        df = self._ensure_datetime(df)
        df = df.copy()

        df["hour"] = df[self.datetime_col].dt.hour
        df["weekday"] = df[self.datetime_col].dt.dayofweek
        df["weekend"] = df["weekday"] >= 5
        df["peak_period"] = df["hour"].apply(
            lambda h: self._hour_in_ranges(h, self.peak_hours)
        )
        df["off_peak_period"] = df["hour"].apply(
            lambda h: self._hour_in_ranges(h, self.off_peak_hours)
        )

        logger.info(
            "Temporal feature extraction complete – added 5 columns to "
            "%d rows.",
            len(df),
        )
        return df
