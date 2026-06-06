"""
ACN data loader – reads, validates, and cleans the ACN charging-session dataset.

Usage::

    from src.data.acn_loader import ACNDataLoader

    loader = ACNDataLoader()
    sessions = loader.get_sessions()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader
from src.data.schema_validator import ACNSessionSchema, validate_dataframe

logger = get_logger(__name__)


class ACNDataLoader:
    """Load, validate, and clean the ACN charging-sessions Excel file.

    Parameters
    ----------
    filepath : str | Path | None
        Path to the ``.xlsx`` file.  Defaults to ``data.acn.filepath``
        from *config.yaml*.
    """

    _DATETIME_COLS = ["connectionTime", "disconnectTime", "doneChargingTime"]

    def __init__(self, filepath: Optional[str | Path] = None) -> None:
        if filepath is None:
            cfg = ConfigLoader()
            filepath = cfg.data.acn.filepath
        self._filepath = Path(filepath)
        self._raw_df: Optional[pd.DataFrame] = None
        self._clean_df: Optional[pd.DataFrame] = None
        logger.info("ACNDataLoader initialised – file: %s", self._filepath)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Read the Excel file and parse datetime columns.

        Returns
        -------
        pd.DataFrame
            Raw DataFrame with datetime columns parsed.
        """
        if not self._filepath.exists():
            raise FileNotFoundError(
                f"ACN data file not found: {self._filepath}"
            )

        logger.info("Reading ACN data from %s …", self._filepath)
        cfg = ConfigLoader()
        sheet = cfg.data.acn.get("sheet_name", 0)
        self._raw_df = pd.read_excel(
            self._filepath,
            sheet_name=sheet,
            engine="openpyxl",
        )
        logger.info(
            "ACN raw data loaded: %d rows × %d columns",
            len(self._raw_df),
            len(self._raw_df.columns),
        )
        logger.info("Columns: %s", list(self._raw_df.columns))

        # Parse datetime columns
        for col in self._DATETIME_COLS:
            if col in self._raw_df.columns:
                self._raw_df[col] = pd.to_datetime(
                    self._raw_df[col], errors="coerce"
                )
                logger.debug(
                    "Parsed '%s' – %d NaT values",
                    col,
                    self._raw_df[col].isna().sum(),
                )

        # Log null summary
        null_summary = self._raw_df.isnull().sum()
        non_zero = null_summary[null_summary > 0]
        if not non_zero.empty:
            logger.info("Null counts:\n%s", non_zero.to_string())
        else:
            logger.info("No null values detected in raw data.")

        # Schema validation (sample for speed)
        val_result = validate_dataframe(
            self._raw_df, ACNSessionSchema, sample_size=500
        )
        logger.info("Schema validation summary: %s", val_result)

        return self._raw_df

    def clean(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Clean the ACN data.

        Strategy
        --------
        * **kWhDelivered**: interpolate missing values; clip negatives to 0.
        * **connectionTime / disconnectTime**: forward-fill, then drop
          rows still containing NaT.
        * **doneChargingTime**: forward-fill (nullable – don't drop).
        * **stationID / siteID**: fill with ``"UNKNOWN"``.
        * Remove duplicate rows.

        Parameters
        ----------
        df : pd.DataFrame | None
            DataFrame to clean.  If *None*, uses the previously loaded raw data.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame.
        """
        if df is None:
            if self._raw_df is None:
                self.load()
            df = self._raw_df.copy()  # type: ignore[union-attr]
        else:
            df = df.copy()

        initial_rows = len(df)
        logger.info("Cleaning ACN data (%d rows) …", initial_rows)

        # --- kWhDelivered ---
        if "kWhDelivered" in df.columns:
            df["kWhDelivered"] = (
                df["kWhDelivered"]
                .interpolate(method="linear", limit_direction="both")
                .clip(lower=0.0)
            )

        # --- Datetime columns ---
        for col in ["connectionTime", "disconnectTime"]:
            if col in df.columns:
                df[col] = df[col].ffill()

        if "doneChargingTime" in df.columns:
            df["doneChargingTime"] = df["doneChargingTime"].ffill()

        # Drop rows where critical datetimes are still NaT
        critical_dt = ["connectionTime", "disconnectTime"]
        existing_critical = [c for c in critical_dt if c in df.columns]
        if existing_critical:
            before = len(df)
            df = df.dropna(subset=existing_critical)
            dropped = before - len(df)
            if dropped:
                logger.info(
                    "Dropped %d rows with missing critical datetimes.", dropped
                )

        # --- ID columns ---
        for col in ["stationID", "siteID"]:
            if col in df.columns:
                df[col] = df[col].fillna("UNKNOWN").astype(str)

        # --- Duplicates ---
        before = len(df)
        df = df.drop_duplicates()
        dups = before - len(df)
        if dups:
            logger.info("Removed %d duplicate rows.", dups)

        # --- Derived: session duration (hours) ---
        if {"connectionTime", "disconnectTime"}.issubset(df.columns):
            df["session_duration_hours"] = (
                (df["disconnectTime"] - df["connectionTime"])
                .dt.total_seconds()
                / 3600.0
            )
            # Drop physically impossible sessions (negative or > 48 h)
            mask = df["session_duration_hours"].between(0, 48, inclusive="both")
            removed = (~mask).sum()
            if removed:
                logger.info(
                    "Removing %d sessions with duration outside [0, 48] h.",
                    removed,
                )
                df = df.loc[mask]

        self._clean_df = df.reset_index(drop=True)
        logger.info(
            "ACN cleaning complete: %d → %d rows.", initial_rows, len(self._clean_df)
        )
        return self._clean_df

    def get_sessions(self) -> pd.DataFrame:
        """Convenience method: load → clean → return.

        Returns
        -------
        pd.DataFrame
            Cleaned sessions DataFrame, ready for feature engineering.
        """
        if self._clean_df is not None:
            return self._clean_df
        self.load()
        self.clean()
        assert self._clean_df is not None
        return self._clean_df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return a summary dict of the loaded (raw) data."""
        if self._raw_df is None:
            self.load()
        assert self._raw_df is not None
        return {
            "rows": len(self._raw_df),
            "columns": list(self._raw_df.columns),
            "dtypes": self._raw_df.dtypes.astype(str).to_dict(),
            "null_counts": self._raw_df.isnull().sum().to_dict(),
            "kWh_stats": (
                self._raw_df["kWhDelivered"].describe().to_dict()
                if "kWhDelivered" in self._raw_df.columns
                else {}
            ),
        }
