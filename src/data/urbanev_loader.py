"""
UrbanEV data loader – loads all nine CSVs from the UrbanEV Shenzhen dataset.

The dataset contains 248 districts, 8 641 five-minute timesteps
(19 Jun 2022 – 18 Jul 2022), and graph topology / station metadata.

Usage::

    from src.data.urbanev_loader import UrbanEVDataLoader

    loader = UrbanEVDataLoader()
    loader.load_all()

    occ = loader.occupancy          # pd.DataFrame (8641 × 249)
    adj = loader.adj_matrix         # np.ndarray   (248 × 248)
    long = loader.get_long_format("occupancy")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader
from src.data.schema_validator import UrbanEVInfoSchema, validate_dataframe

logger = get_logger(__name__)

# Accepted metric names for get_long_format()
MetricName = Literal["occupancy", "volume", "duration", "price"]


class UrbanEVDataLoader:
    """Load and expose all components of the UrbanEV Shenzhen dataset.

    Parameters
    ----------
    data_dir : str | Path | None
        Directory containing the nine CSVs.  Defaults to
        ``data.urbanev.directory`` from *config.yaml*.
    """

    _TIMESERIES_FILES = ("occupancy", "volume", "duration", "price")

    def __init__(self, data_dir: Optional[str | Path] = None) -> None:
        if data_dir is None:
            cfg = ConfigLoader()
            data_dir = cfg.data.urbanev.directory
        self._data_dir = Path(data_dir)
        if not self._data_dir.exists():
            raise FileNotFoundError(
                f"UrbanEV data directory not found: {self._data_dir}"
            )

        cfg = ConfigLoader()
        self._file_map: Dict[str, str] = dict(cfg.data.urbanev.files)
        self._expected_timesteps: int = int(cfg.data.urbanev.expected_timesteps)
        self._expected_districts: int = int(cfg.data.urbanev.expected_districts)

        # Internal storage
        self._time_index: Optional[pd.DatetimeIndex] = None
        self._occupancy: Optional[pd.DataFrame] = None
        self._volume: Optional[pd.DataFrame] = None
        self._duration: Optional[pd.DataFrame] = None
        self._price: Optional[pd.DataFrame] = None
        self._adj_matrix: Optional[np.ndarray] = None
        self._dist_matrix: Optional[np.ndarray] = None
        self._station_info: Optional[pd.DataFrame] = None
        self._stations: Optional[pd.DataFrame] = None

        logger.info(
            "UrbanEVDataLoader initialised – dir: %s (%d expected timesteps)",
            self._data_dir,
            self._expected_timesteps,
        )

    # ==================================================================
    # Properties (lazy-safe)
    # ==================================================================
    @property
    def occupancy(self) -> pd.DataFrame:
        """Occupancy timeseries DataFrame (timestep × district)."""
        if self._occupancy is None:
            self._occupancy = self._load_timeseries("occupancy")
        return self._occupancy

    @property
    def volume(self) -> pd.DataFrame:
        """Volume (kWh) timeseries DataFrame."""
        if self._volume is None:
            self._volume = self._load_timeseries("volume")
        return self._volume

    @property
    def duration(self) -> pd.DataFrame:
        """Duration (hours) timeseries DataFrame."""
        if self._duration is None:
            self._duration = self._load_timeseries("duration")
        return self._duration

    @property
    def price(self) -> pd.DataFrame:
        """Price (yuan/kWh) timeseries DataFrame."""
        if self._price is None:
            self._price = self._load_timeseries("price")
        return self._price

    @property
    def adj_matrix(self) -> np.ndarray:
        """Binary adjacency matrix (248 × 248)."""
        if self._adj_matrix is None:
            self._adj_matrix = self._load_adjacency()
        return self._adj_matrix

    @property
    def dist_matrix(self) -> np.ndarray:
        """Sparse distance matrix (248 × 248, km)."""
        if self._dist_matrix is None:
            self._dist_matrix = self._load_distance()
        return self._dist_matrix

    @property
    def station_info(self) -> pd.DataFrame:
        """District-level information (248 rows)."""
        if self._station_info is None:
            self._load_station_info()
        assert self._station_info is not None
        return self._station_info

    @property
    def stations(self) -> pd.DataFrame:
        """Individual station records (1 707 rows)."""
        if self._stations is None:
            self._load_station_info()
        assert self._stations is not None
        return self._stations

    # ==================================================================
    # Public methods
    # ==================================================================
    def load_all(self) -> None:
        """Eagerly load every file in the dataset.

        This is handy for up-front validation; individual properties
        also load lazily on first access.
        """
        logger.info("Loading all UrbanEV files …")
        self._build_time_index()

        for metric in self._TIMESERIES_FILES:
            _ = getattr(self, metric)

        _ = self.adj_matrix
        _ = self.dist_matrix
        _ = self.station_info
        _ = self.stations

        logger.info("All UrbanEV files loaded successfully.")

    def get_long_format(self, metric: MetricName) -> pd.DataFrame:
        """Melt a wide-format timeseries into long format.

        Parameters
        ----------
        metric : str
            One of ``"occupancy"``, ``"volume"``, ``"duration"``, ``"price"``.

        Returns
        -------
        pd.DataFrame
            Columns: ``["timestamp", "station_id", "value"]``.
        """
        wide_df: pd.DataFrame = getattr(self, metric)

        # The index is a DatetimeIndex; station columns are all remaining
        value_cols = [c for c in wide_df.columns if c != "timestamp"]
        df_reset = wide_df.reset_index()

        # Determine the timestamp column name after reset
        ts_col = "timestamp" if "timestamp" in df_reset.columns else df_reset.columns[0]

        long = df_reset.melt(
            id_vars=[ts_col],
            value_vars=value_cols,
            var_name="station_id",
            value_name="value",
        )
        long = long.rename(columns={ts_col: "timestamp"})
        logger.info(
            "Melted '%s' to long format: %d rows × %d cols",
            metric,
            len(long),
            len(long.columns),
        )
        return long

    # ==================================================================
    # Internal loaders
    # ==================================================================
    def _build_time_index(self) -> pd.DatetimeIndex:
        """Load *time.csv* and build a ``DatetimeIndex``."""
        if self._time_index is not None:
            return self._time_index

        time_path = self._data_dir / self._file_map["time"]
        logger.info("Loading time index from %s …", time_path)
        time_df = pd.read_csv(time_path)

        # Expect columns: month, day, year, hour, minute, second
        self._time_index = pd.to_datetime(
            time_df[["year", "month", "day", "hour", "minute", "second"]]
        )
        logger.info(
            "Time index built: %d steps from %s to %s",
            len(self._time_index),
            self._time_index[0],
            self._time_index[-1],
        )

        if len(self._time_index) != self._expected_timesteps:
            logger.warning(
                "Expected %d timesteps but got %d.",
                self._expected_timesteps,
                len(self._time_index),
            )
        return self._time_index

    def _load_timeseries(self, metric: str) -> pd.DataFrame:
        """Load a wide-format timeseries CSV and attach the datetime index.

        Parameters
        ----------
        metric : str
            Key into ``_file_map`` (e.g. ``"occupancy"``).

        Returns
        -------
        pd.DataFrame
            DataFrame with a ``DatetimeIndex`` and one column per district.
        """
        filepath = self._data_dir / self._file_map[metric]
        logger.info("Loading timeseries '%s' from %s …", metric, filepath)
        df = pd.read_csv(filepath)

        # Drop any unnamed/index columns that may have crept in
        unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)

        # Validate row count
        if len(df) != self._expected_timesteps:
            logger.warning(
                "'%s' has %d rows (expected %d).",
                metric,
                len(df),
                self._expected_timesteps,
            )

        # If there is a 'timestamp' column already, try using it
        if "timestamp" in df.columns:
            df.index = pd.to_datetime(df.pop("timestamp"))
        else:
            # Use the time index built from time.csv
            time_idx = self._build_time_index()
            if len(df) == len(time_idx):
                df.index = time_idx
            else:
                logger.warning(
                    "Row count mismatch for '%s' – cannot attach time index.",
                    metric,
                )

        df.index.name = "timestamp"
        logger.info(
            "'%s' loaded: %d rows × %d district columns.",
            metric,
            len(df),
            len(df.columns),
        )
        return df

    def _load_adjacency(self) -> np.ndarray:
        """Load ``adj.csv`` as a NumPy array (248 × 248).

        The CSV has a ``node_id`` column followed by 248 binary columns.
        """
        filepath = self._data_dir / self._file_map["adj"]
        logger.info("Loading adjacency matrix from %s …", filepath)
        df = pd.read_csv(filepath)

        # Drop the identifier column (usually first column = node_id)
        if "node_id" in df.columns:
            df = df.drop(columns=["node_id"])
        elif df.columns[0] not in [str(i) for i in range(self._expected_districts)]:
            # Heuristic: first column is an ID column
            df = df.iloc[:, 1:]

        matrix = df.values.astype(np.float32)
        logger.info("Adjacency matrix shape: %s", matrix.shape)
        return matrix

    def _load_distance(self) -> np.ndarray:
        """Load ``distance.csv`` as a NumPy array (248 × 248, km)."""
        filepath = self._data_dir / self._file_map["distance"]
        logger.info("Loading distance matrix from %s …", filepath)
        df = pd.read_csv(filepath)

        # Drop identifier column similar to adjacency
        if df.columns[0].lower() in ("node_id", "unnamed: 0", ""):
            df = df.iloc[:, 1:]

        matrix = df.values.astype(np.float64)
        logger.info("Distance matrix shape: %s", matrix.shape)
        return matrix

    def _load_station_info(self) -> None:
        """Load ``information.csv`` and ``stations.csv``."""
        # --- information.csv (248 rows) ---
        info_path = self._data_dir / self._file_map["information"]
        logger.info("Loading station info from %s …", info_path)
        self._station_info = pd.read_csv(info_path)
        logger.info(
            "Station info: %d rows × %d cols – columns: %s",
            len(self._station_info),
            len(self._station_info.columns),
            list(self._station_info.columns),
        )

        # Validate with Pydantic schema (sample)
        val_result = validate_dataframe(
            self._station_info, UrbanEVInfoSchema, sample_size=248
        )
        logger.info("UrbanEV info validation: %s", val_result)

        # --- stations.csv (1707 rows) ---
        stations_path = self._data_dir / self._file_map["stations"]
        logger.info("Loading stations from %s …", stations_path)
        self._stations = pd.read_csv(stations_path)
        logger.info(
            "Stations: %d rows × %d cols – columns: %s",
            len(self._stations),
            len(self._stations.columns),
            list(self._stations.columns),
        )
