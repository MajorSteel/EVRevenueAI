# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""
Graph Neural Network for spatial demand prediction in EV charging.

This module provides:

* :class:`SpatialDemandGNN` – a GCN-based model that predicts per-node
  demand (occupancy or volume) given node features and a graph topology
  defined by an adjacency / distance matrix.
* :class:`GraphDataset` – a utility that converts tabular UrbanEV
  matrices into ``torch_geometric.data.Data`` objects with sliding-window
  temporal features.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

# ======================================================================
# Lazy import helper for torch_geometric (optional heavy dependency)
# ======================================================================

_TG_AVAILABLE: bool = False
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv

    _TG_AVAILABLE = True
except ImportError:
    logger.warning(
        "torch_geometric is not installed.  GNN features will be "
        "unavailable until it is installed."
    )


def _require_tg() -> None:
    """Raise if ``torch_geometric`` is missing."""
    if not _TG_AVAILABLE:
        raise ImportError(
            "torch_geometric is required for GNN models.  "
            "Install it with: pip install torch-geometric"
        )


# ======================================================================
# SpatialDemandGNN
# ======================================================================
class SpatialDemandGNN(nn.Module):
    """GCN-based model for per-node demand prediction.

    Architecture::

        GCNConv(in, hidden) → ReLU → Dropout
        → GCNConv(hidden, hidden) → ReLU → Dropout
        (…repeat for num_layers…)
        → Linear(hidden, output_dim)

    Parameters
    ----------
    num_features : int
        Number of input features per node.
    hidden_dim : int
        Hidden channel width of GCN layers.
    output_dim : int
        Output dimension per node (1 for scalar demand prediction).
    num_layers : int
        Number of GCN convolution layers.
    dropout : float
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        _require_tg()
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Build GCN convolution stack
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Final readout
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        logger.info(
            "SpatialDemandGNN created – features=%d, hidden=%d, "
            "output=%d, layers=%d, dropout=%.2f",
            num_features,
            hidden_dim,
            output_dim,
            num_layers,
            dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``(num_nodes, num_features)``.
        edge_index : torch.Tensor
            Graph connectivity in COO format ``(2, num_edges)``.
        edge_weight : torch.Tensor | None
            Optional edge weights of shape ``(num_edges,)``.

        Returns
        -------
        torch.Tensor
            Per-node predictions of shape ``(num_nodes, output_dim)``.
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out: torch.Tensor = self.fc_out(x)
        return out

    def __repr__(self) -> str:
        return (
            f"SpatialDemandGNN(in={self.num_features}, "
            f"hidden={self.hidden_dim}, out={self.output_dim}, "
            f"layers={self.num_layers}, drop={self.dropout})"
        )


# ======================================================================
# GraphDataset
# ======================================================================
class GraphDataset:
    """Convert UrbanEV matrices into ``torch_geometric.data.Data`` objects.

    The class builds sliding-window samples where each sample has:

    * **Node features** – a window of historical occupancy/volume plus
      optional temporal features (hour-of-day sin/cos, day-of-week).
    * **Target** – next-timestep occupancy or volume per node.
    * **Edge index / weight** – from the adjacency and (inverse) distance
      matrices.

    Parameters
    ----------
    occupancy_matrix : np.ndarray
        Shape ``(T, N)`` – occupancy counts per timestep per district.
    volume_matrix : np.ndarray
        Shape ``(T, N)`` – energy volume (kWh).
    adj_matrix : np.ndarray
        Shape ``(N, N)`` – binary adjacency matrix.
    dist_matrix : np.ndarray | None
        Shape ``(N, N)`` – distance matrix (km).  Used to compute
        edge weights as ``1 / (distance + ε)``.
    features_dict : dict | None
        Optional dict of extra per-timestep arrays, each ``(T,)``
        or ``(T, N)``.  Recognised keys: ``hour``, ``day_of_week``.
    window_size : int
        Number of historical timesteps per sample.
    horizon : int
        Prediction horizon (default 1 = next step).
    target : str
        Target variable – ``"occupancy"`` or ``"volume"``.
    """

    def __init__(
        self,
        occupancy_matrix: np.ndarray,
        volume_matrix: np.ndarray,
        adj_matrix: np.ndarray,
        dist_matrix: Optional[np.ndarray] = None,
        features_dict: Optional[Dict[str, np.ndarray]] = None,
        window_size: int = 12,
        horizon: int = 1,
        target: str = "occupancy",
    ) -> None:
        _require_tg()

        self.occupancy_matrix = occupancy_matrix.astype(np.float32)
        self.volume_matrix = volume_matrix.astype(np.float32)
        self.adj_matrix = adj_matrix.astype(np.float32)
        self.dist_matrix = (
            dist_matrix.astype(np.float32) if dist_matrix is not None else None
        )
        self.features_dict = features_dict or {}
        self.window_size = window_size
        self.horizon = horizon
        self.target = target

        self._T, self._N = occupancy_matrix.shape

        # Pre-compute edge_index and edge_weight
        self._edge_index, self._edge_weight = self._build_edges()

        logger.info(
            "GraphDataset initialised – T=%d, N=%d, window=%d, "
            "horizon=%d, edges=%d, target=%s",
            self._T,
            self._N,
            window_size,
            horizon,
            self._edge_index.shape[1],
            target,
        )

    # ------------------------------------------------------------------
    # Edge construction
    # ------------------------------------------------------------------
    def _build_edges(
        self,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Build COO edge_index and optional edge_weight from adj/dist."""
        rows, cols = np.where(self.adj_matrix > 0)
        edge_index = torch.tensor(
            np.stack([rows, cols], axis=0),
            dtype=torch.long,
        )

        edge_weight: Optional[torch.Tensor] = None
        if self.dist_matrix is not None:
            eps = 1e-3
            distances = self.dist_matrix[rows, cols]
            weights = 1.0 / (distances + eps)
            # Normalise to [0, 1]
            w_max = weights.max()
            if w_max > 0:
                weights = weights / w_max
            edge_weight = torch.tensor(weights, dtype=torch.float32)

        return edge_index, edge_weight

    # ------------------------------------------------------------------
    # Feature computation per sample
    # ------------------------------------------------------------------
    def _build_node_features(
        self,
        t_start: int,
        t_end: int,
    ) -> torch.Tensor:
        """Build node feature matrix for the window ``[t_start, t_end)``.

        Each node gets ``2 * window_size`` features (occupancy + volume
        over the window), plus optional temporal encodings.

        Returns shape ``(N, num_features)``.
        """
        occ_window = self.occupancy_matrix[t_start:t_end]  # (W, N)
        vol_window = self.volume_matrix[t_start:t_end]      # (W, N)

        # Flatten windows into per-node feature vectors
        # Transpose to (N, W) then concatenate
        occ_feats = occ_window.T  # (N, W)
        vol_feats = vol_window.T  # (N, W)
        node_feats = np.concatenate([occ_feats, vol_feats], axis=1)  # (N, 2W)

        # Optional temporal features (broadcast across all nodes)
        if "hour" in self.features_dict:
            hours = self.features_dict["hour"][t_start:t_end].astype(
                np.float32,
            )
            hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
            hour_cos = np.cos(2.0 * np.pi * hours / 24.0)
            # Use last timestep's temporal features
            h_sin = np.full((self._N, 1), hour_sin[-1], dtype=np.float32)
            h_cos = np.full((self._N, 1), hour_cos[-1], dtype=np.float32)
            node_feats = np.concatenate([node_feats, h_sin, h_cos], axis=1)

        if "day_of_week" in self.features_dict:
            dow = self.features_dict["day_of_week"][t_start:t_end].astype(
                np.float32,
            )
            dow_norm = dow[-1] / 6.0  # normalise to [0, 1]
            dow_col = np.full((self._N, 1), dow_norm, dtype=np.float32)
            node_feats = np.concatenate([node_feats, dow_col], axis=1)

        return torch.tensor(node_feats, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------
    def create_samples(self) -> List[Data]:
        """Generate a list of ``torch_geometric.data.Data`` samples.

        Each sample corresponds to one sliding-window position.

        Returns
        -------
        list[Data]
            List of graph data objects.
        """
        samples: List[Data] = []
        n_samples = self._T - self.window_size - self.horizon + 1
        if n_samples <= 0:
            logger.warning(
                "Not enough timesteps (%d) for window=%d + horizon=%d",
                self._T,
                self.window_size,
                self.horizon,
            )
            return samples

        target_matrix = (
            self.occupancy_matrix
            if self.target == "occupancy"
            else self.volume_matrix
        )

        for i in range(n_samples):
            t_start = i
            t_end = i + self.window_size
            t_target = t_end + self.horizon - 1

            x = self._build_node_features(t_start, t_end)
            y = torch.tensor(
                target_matrix[t_target],
                dtype=torch.float32,
            ).unsqueeze(-1)  # (N, 1)

            data = Data(
                x=x,
                y=y,
                edge_index=self._edge_index,
            )
            if self._edge_weight is not None:
                data.edge_weight = self._edge_weight

            samples.append(data)

        logger.info("Created %d graph samples.", len(samples))
        return samples

    @property
    def num_node_features(self) -> int:
        """Number of features per node (computed from window_size)."""
        base = 2 * self.window_size  # occupancy + volume windows
        if "hour" in self.features_dict:
            base += 2  # sin, cos
        if "day_of_week" in self.features_dict:
            base += 1
        return base

    def __len__(self) -> int:
        n = self._T - self.window_size - self.horizon + 1
        return max(n, 0)

    def __repr__(self) -> str:
        return (
            f"GraphDataset(T={self._T}, N={self._N}, "
            f"window={self.window_size}, horizon={self.horizon}, "
            f"features={self.num_node_features})"
        )
