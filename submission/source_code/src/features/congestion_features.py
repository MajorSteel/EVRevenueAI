# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""Congestion feature engineering for EV charging tariff optimization.

This module provides the :class:`CongestionFeatureEngine` which derives
congestion-related metrics from occupancy, capacity, and area data.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

# Default weights for the composite congestion score
_DEFAULT_CONGESTION_WEIGHTS: Dict[str, float] = {
    "utilization": 0.5,
    "queue_proxy": 0.3,
    "occupancy_density": 0.2,
}


class CongestionFeatureEngine:
    """Compute congestion-related features for EV charging stations.

    Parameters
    ----------
    occupancy_col : str
        Column name for real-time occupancy counts.
        Defaults to ``"occupancy"``.
    capacity_col : str
        Column name for station capacity.
        Defaults to ``"capacity"``.
    area_col : str
        Column name for district area (km²).
        Defaults to ``"area"``.
    """

    def __init__(
        self,
        occupancy_col: str = "occupancy",
        capacity_col: str = "capacity",
        area_col: str = "area",
    ) -> None:
        self.occupancy_col = occupancy_col
        self.capacity_col = capacity_col
        self.area_col = area_col
        logger.info(
            "CongestionFeatureEngine initialised – occupancy=%s, "
            "capacity=%s, area=%s",
            occupancy_col,
            capacity_col,
            area_col,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_queue_length_proxy(
        self,
        occupancy: pd.Series,
        capacity: pd.Series,
    ) -> pd.Series:
        """Compute a proxy for queue length: ``max(0, occupancy − capacity)``.

        Parameters
        ----------
        occupancy : pd.Series
            Current occupancy count.
        capacity : pd.Series
            Maximum charger capacity.

        Returns
        -------
        pd.Series
            Non-negative queue-length proxy.
        """
        queue = (occupancy - capacity).clip(lower=0)
        logger.debug(
            "Queue length proxy – mean=%.2f, max=%.0f, nonzero=%d/%d",
            queue.mean(),
            queue.max(),
            (queue > 0).sum(),
            len(queue),
        )
        return queue.rename("queue_length_proxy")

    def compute_occupancy_density(
        self,
        occupancy: pd.Series,
        area: pd.Series,
    ) -> pd.Series:
        """Compute occupancy density: ``occupancy / area``.

        Parameters
        ----------
        occupancy : pd.Series
            Current occupancy count.
        area : pd.Series
            District area (km² or any consistent unit).

        Returns
        -------
        pd.Series
            Occupancy density.  ``NaN`` where area is zero.
        """
        safe_area = area.replace(0, np.nan)
        density = occupancy / safe_area
        n_nan = density.isna().sum()
        if n_nan > 0:
            logger.warning(
                "occupancy_density: %d NaN values due to zero/missing area.",
                n_nan,
            )
        logger.debug("Occupancy density – mean=%.4f", density.mean())
        return density.rename("occupancy_density")

    def compute_congestion_score(
        self,
        utilization: pd.Series,
        queue_proxy: pd.Series,
        occ_density: pd.Series,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """Compute a composite congestion score as a weighted sum.

        The three component series are min-max normalised individually
        before weighting so that they contribute on a comparable scale.

        Parameters
        ----------
        utilization : pd.Series
            Utilization rate (expected 0–1).
        queue_proxy : pd.Series
            Queue-length proxy (≥ 0).
        occ_density : pd.Series
            Occupancy density.
        weights : dict[str, float], optional
            Weights for ``"utilization"``, ``"queue_proxy"``, and
            ``"occupancy_density"``.  Defaults to ``{0.5, 0.3, 0.2}``.

        Returns
        -------
        pd.Series
            Congestion score in [0, 1].
        """
        weights = weights or _DEFAULT_CONGESTION_WEIGHTS

        def _min_max(s: pd.Series) -> pd.Series:
            s_min, s_max = s.min(), s.max()
            if s_max == s_min:
                return pd.Series(0.0, index=s.index)
            return (s - s_min) / (s_max - s_min)

        norm_util = _min_max(utilization.fillna(0))
        norm_queue = _min_max(queue_proxy.fillna(0))
        norm_density = _min_max(occ_density.fillna(0))

        score = (
            weights.get("utilization", 0.5) * norm_util
            + weights.get("queue_proxy", 0.3) * norm_queue
            + weights.get("occupancy_density", 0.2) * norm_density
        ).clip(lower=0.0, upper=1.0)

        logger.debug(
            "Congestion score – mean=%.3f, max=%.3f, weights=%s",
            score.mean(),
            score.max(),
            weights,
        )
        return score.rename("congestion_score")

    def compute_congestion_label(
        self,
        utilization: pd.Series,
        threshold: float = 0.8,
    ) -> pd.Series:
        """Compute a binary congestion label.

        Parameters
        ----------
        utilization : pd.Series
            Utilization rate (0–1).
        threshold : float
            Utilization value above which a station is labelled
            *congested* (``1``).  Defaults to ``0.8``.

        Returns
        -------
        pd.Series
            Binary label – ``1`` if ``utilization > threshold``, else ``0``.
        """
        label = (utilization > threshold).astype(int)
        n_congested = label.sum()
        logger.debug(
            "Congestion label (threshold=%.2f) – %d / %d rows congested "
            "(%.1f%%).",
            threshold,
            n_congested,
            len(label),
            100.0 * n_congested / max(len(label), 1),
        )
        return label.rename("congestion_label")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all congestion feature computations on *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain the configured occupancy, capacity, and
            (optionally) area columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with congestion features appended.
        """
        logger.info("Running full congestion feature computation …")
        df = df.copy()

        occupancy = df.get(self.occupancy_col)
        capacity = df.get(self.capacity_col)

        if occupancy is None or capacity is None:
            logger.error(
                "Cannot compute congestion features – missing '%s' or '%s'.",
                self.occupancy_col,
                self.capacity_col,
            )
            return df

        # Utilization (re-compute locally for independence from DemandFE)
        utilization = (occupancy / capacity.replace(0, np.nan)).fillna(0).clip(0, 1)
        df["utilization_rate"] = utilization

        # Queue proxy
        df["queue_length_proxy"] = self.compute_queue_length_proxy(
            occupancy, capacity
        )

        # Occupancy density
        if self.area_col in df.columns:
            df["occupancy_density"] = self.compute_occupancy_density(
                occupancy, df[self.area_col]
            )
        else:
            logger.info(
                "Skipping occupancy_density – '%s' column not found.",
                self.area_col,
            )
            df["occupancy_density"] = 0.0

        # Congestion score
        df["congestion_score"] = self.compute_congestion_score(
            df["utilization_rate"],
            df["queue_length_proxy"],
            df["occupancy_density"],
        )

        # Congestion label
        df["congestion_label"] = self.compute_congestion_label(
            df["utilization_rate"]
        )

        logger.info(
            "Congestion feature computation complete – added 5 columns."
        )
        return df
