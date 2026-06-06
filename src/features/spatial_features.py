"""Spatial feature engineering for EV charging tariff optimization.

This module provides the :class:`SpatialFeatureEngine` which derives
graph/spatial features from adjacency and distance matrices and
per-station metadata (UrbanEV dataset layout).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class SpatialFeatureEngine:
    """Compute spatial features using adjacency and distance matrices.

    Parameters
    ----------
    adj_matrix : pd.DataFrame or np.ndarray
        Binary adjacency matrix of shape ``(N, N)`` where ``N`` is the
        number of districts/nodes.  ``adj[i, j] == 1`` means district
        *i* and *j* are neighbours.
    dist_matrix : pd.DataFrame or np.ndarray
        Distance matrix of shape ``(N, N)`` (km).  Sparse – zero
        entries imply no direct connection.
    station_info : pd.DataFrame, optional
        Station/district information table.  Expected to contain at
        least ``count`` (total charger count), and optionally
        ``fast_count``, ``slow_count``, ``area``.
    """

    def __init__(
        self,
        adj_matrix: pd.DataFrame | np.ndarray,
        dist_matrix: pd.DataFrame | np.ndarray,
        station_info: Optional[pd.DataFrame] = None,
    ) -> None:
        self.adj: np.ndarray = (
            adj_matrix.values
            if isinstance(adj_matrix, pd.DataFrame)
            else np.asarray(adj_matrix)
        )
        self.dist: np.ndarray = (
            dist_matrix.values
            if isinstance(dist_matrix, pd.DataFrame)
            else np.asarray(dist_matrix)
        )
        self.station_info = station_info
        self.n_nodes: int = self.adj.shape[0]

        logger.info(
            "SpatialFeatureEngine initialised – %d nodes, adj shape=%s, "
            "dist shape=%s, station_info=%s",
            self.n_nodes,
            self.adj.shape,
            self.dist.shape,
            "provided" if station_info is not None else "None",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_station_capacity(
        self,
        station_info: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Return total charger count per district.

        Parameters
        ----------
        station_info : pd.DataFrame, optional
            Override for ``self.station_info``.  Must contain a
            ``count`` column.

        Returns
        -------
        pd.Series
            Charger capacity per district (indexed by district id/row).

        Raises
        ------
        ValueError
            If no station information is available.
        """
        info = station_info if station_info is not None else self.station_info
        if info is None:
            raise ValueError(
                "station_info not provided – cannot compute capacity."
            )

        if "count" not in info.columns:
            raise KeyError(
                "'count' column not found in station_info. Available: "
                f"{list(info.columns)}"
            )

        capacity = info["count"].astype(float)
        logger.debug(
            "Station capacity – total=%d, mean=%.1f, max=%d",
            int(capacity.sum()),
            capacity.mean(),
            int(capacity.max()),
        )
        return capacity.rename("station_capacity")

    def compute_neighbor_mean(
        self,
        values_matrix: np.ndarray | pd.DataFrame,
        adj_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute unweighted mean of adjacent nodes' values.

        Parameters
        ----------
        values_matrix : np.ndarray or pd.DataFrame
            2-D array of shape ``(T, N)`` where *T* is the number of
            timesteps and *N* the number of nodes, **or** a 1-D array
            of length *N* (single timestep).
        adj_matrix : np.ndarray, optional
            Override adjacency matrix.  Defaults to ``self.adj``.

        Returns
        -------
        np.ndarray
            Same shape as *values_matrix* with neighbour-mean values.
        """
        adj = adj_matrix if adj_matrix is not None else self.adj
        vals = (
            values_matrix.values
            if isinstance(values_matrix, pd.DataFrame)
            else np.asarray(values_matrix, dtype=float)
        )

        # Ensure 2-D
        squeeze = False
        if vals.ndim == 1:
            vals = vals.reshape(1, -1)
            squeeze = True

        # Degree per node (number of neighbours)
        degree = adj.sum(axis=1)  # shape (N,)
        safe_degree = np.where(degree == 0, 1, degree)  # avoid div-by-zero

        # vals @ adj.T gives sum of neighbour values for each node
        neighbor_sum = vals @ adj.T  # (T, N)
        neighbor_mean = neighbor_sum / safe_degree  # broadcast (T, N)

        if squeeze:
            neighbor_mean = neighbor_mean.squeeze(axis=0)

        logger.debug(
            "Neighbor mean computed – shape=%s", neighbor_mean.shape
        )
        return neighbor_mean

    def compute_weighted_neighbor_mean(
        self,
        values_matrix: np.ndarray | pd.DataFrame,
        adj_matrix: Optional[np.ndarray] = None,
        dist_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute inverse-distance-weighted mean of adjacent nodes' values.

        For each node *i* the weight on neighbour *j* is::

            w_ij = 1 / dist(i, j)   if adj(i, j) == 1 and dist > 0

        Non-adjacent entries and zero-distance entries are ignored.

        Parameters
        ----------
        values_matrix : np.ndarray or pd.DataFrame
            Shape ``(T, N)`` or ``(N,)`` – node values per timestep.
        adj_matrix : np.ndarray, optional
            Override adjacency.  Defaults to ``self.adj``.
        dist_matrix : np.ndarray, optional
            Override distance.  Defaults to ``self.dist``.

        Returns
        -------
        np.ndarray
            Same shape as *values_matrix*.
        """
        adj = adj_matrix if adj_matrix is not None else self.adj
        dist = dist_matrix if dist_matrix is not None else self.dist
        vals = (
            values_matrix.values
            if isinstance(values_matrix, pd.DataFrame)
            else np.asarray(values_matrix, dtype=float)
        )

        squeeze = False
        if vals.ndim == 1:
            vals = vals.reshape(1, -1)
            squeeze = True

        # Build inverse-distance weight matrix (only where adjacent)
        safe_dist = np.where((adj > 0) & (dist > 0), dist, np.nan)
        inv_dist = np.where(np.isnan(safe_dist), 0.0, 1.0 / safe_dist)

        # Normalise weights per node (row-wise)
        weight_sum = inv_dist.sum(axis=1)  # (N,)
        safe_weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)
        weights_norm = inv_dist / safe_weight_sum[:, np.newaxis]  # (N, N)

        weighted_mean = vals @ weights_norm.T  # (T, N)

        if squeeze:
            weighted_mean = weighted_mean.squeeze(axis=0)

        logger.debug(
            "Weighted neighbor mean computed – shape=%s",
            weighted_mean.shape,
        )
        return weighted_mean

    def compute_spatial_lag(
        self,
        df: pd.DataFrame,
        feature_col: str,
    ) -> pd.DataFrame:
        """Add spatial-lag features to a *long-format* DataFrame.

        Expects the DataFrame to have one row per ``(timestep, node)``
        combination with a column that identifies the node (district).

        If the DataFrame has a simple numeric index and one column per
        node (wide format), it is processed directly.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.  For **wide format**: columns represent nodes
            and rows represent timesteps.  For **long format**: must
            contain a ``node_id`` column.
        feature_col : str
            Column name (wide) or value column (long) to compute
            spatial lag for.

        Returns
        -------
        pd.DataFrame
            DataFrame with ``{feature_col}_spatial_lag`` and
            ``{feature_col}_spatial_lag_w`` (weighted) columns added.
        """
        lag_col = f"{feature_col}_spatial_lag"
        wlag_col = f"{feature_col}_spatial_lag_w"

        # --- Wide-format path (columns are nodes) ---
        if feature_col not in df.columns and df.shape[1] >= self.n_nodes:
            # Assume the whole df IS the wide matrix
            vals = df.values.astype(float)
            neighbor_mean = self.compute_neighbor_mean(vals)
            weighted_mean = self.compute_weighted_neighbor_mean(vals)

            result = df.copy()
            lag_df = pd.DataFrame(
                neighbor_mean, index=df.index, columns=df.columns
            )
            wlag_df = pd.DataFrame(
                weighted_mean, index=df.index, columns=df.columns
            )
            # Return stacked for convenience
            logger.info(
                "Spatial lag computed (wide format) – %d timesteps × %d "
                "nodes.",
                vals.shape[0],
                vals.shape[1],
            )
            return result, lag_df, wlag_df  # type: ignore[return-value]

        # --- Long-format path ---
        if feature_col in df.columns:
            logger.info(
                "Spatial lag requested for column '%s' in long format – "
                "pivoting to wide, computing, and un-pivoting.",
                feature_col,
            )
            if "node_id" not in df.columns:
                logger.error(
                    "Long-format spatial lag requires a 'node_id' column."
                )
                return df

            # Pivot → wide
            wide = df.pivot(
                columns="node_id", values=feature_col
            ).astype(float)
            vals = wide.values
            neighbor_mean = self.compute_neighbor_mean(vals)
            weighted_mean = self.compute_weighted_neighbor_mean(vals)

            wide_lag = pd.DataFrame(
                neighbor_mean, index=wide.index, columns=wide.columns
            )
            wide_wlag = pd.DataFrame(
                weighted_mean, index=wide.index, columns=wide.columns
            )

            # Un-pivot back → long
            df = df.copy()
            lag_long = wide_lag.stack().reset_index()
            lag_long.columns = ["_idx", "node_id", lag_col]
            wlag_long = wide_wlag.stack().reset_index()
            wlag_long.columns = ["_idx", "node_id", wlag_col]

            df = df.merge(lag_long, on="node_id", how="left")
            df = df.merge(wlag_long, on="node_id", how="left")

            # Clean up helper columns
            for tmp in ("_idx_x", "_idx_y", "_idx"):
                if tmp in df.columns:
                    df = df.drop(columns=[tmp])

            logger.info("Spatial lag features added to long-format df.")
            return df

        logger.warning(
            "Column '%s' not found and DataFrame shape (%s) doesn't "
            "match n_nodes=%d – returning unmodified.",
            feature_col,
            df.shape,
            self.n_nodes,
        )
        return df

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all spatial feature computations.

        This is a thin wrapper – because spatial features depend on
        which value columns are relevant, ``compute_all`` processes
        a pre-defined list of common feature names if they are present
        in *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Input data (wide or long format).

        Returns
        -------
        pd.DataFrame
            DataFrame enriched with spatial features.
        """
        logger.info("Running full spatial feature computation …")

        # Station capacity
        if self.station_info is not None:
            try:
                cap = self.compute_station_capacity()
                logger.info(
                    "Station capacity computed – %d districts.", len(cap)
                )
            except (ValueError, KeyError) as exc:
                logger.warning("Could not compute station capacity: %s", exc)

        # Spatial lag for common columns
        common_features = [
            "occupancy",
            "volume",
            "price",
            "utilization_rate",
        ]
        for feat in common_features:
            if feat in df.columns:
                logger.info("Computing spatial lag for '%s'.", feat)
                df = self.compute_spatial_lag(df, feat)

        logger.info("Spatial feature computation complete.")
        return df
