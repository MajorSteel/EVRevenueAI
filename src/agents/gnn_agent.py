# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""
GNN-based spatial agent for EV charging demand prediction.

Orchestrates training, evaluation, and inference of
:class:`~src.models.gnn_model.SpatialDemandGNN` with early stopping,
learning-rate scheduling, and per-node metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.gnn_model import GraphDataset, SpatialDemandGNN
from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

# Optional – only needed at import time when torch_geometric is installed
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError:
    Data = None  # type: ignore[assignment,misc]
    PyGDataLoader = None  # type: ignore[assignment,misc]


class _EarlyStopping:
    """Simple early stopping tracker.

    Parameters
    ----------
    patience : int
        Number of epochs to wait after the last improvement.
    min_delta : float
        Minimum change in the monitored metric to qualify as improvement.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-5) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_score is None or val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class GNNSpatialAgent:
    """High-level agent for GNN-based spatial demand prediction.

    Parameters
    ----------
    config : dict | None
        Configuration dictionary.  Recognised keys under ``"gnn"``:
        ``hidden_channels``, ``num_layers``, ``dropout``,
        ``learning_rate``, ``weight_decay``, ``epochs``, ``patience``,
        ``batch_size``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self._gnn_cfg: Dict[str, Any] = self.config.get("gnn", {})
        self._model: Optional[SpatialDemandGNN] = None
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        logger.info(
            "GNNSpatialAgent initialised – device=%s", self._device,
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def prepare_graph_data(
        self,
        occupancy_matrix: np.ndarray,
        volume_matrix: np.ndarray,
        adj_matrix: np.ndarray,
        dist_matrix: Optional[np.ndarray] = None,
        features_dict: Optional[Dict[str, np.ndarray]] = None,
        window_size: int = 12,
        horizon: int = 1,
        target: str = "occupancy",
    ) -> GraphDataset:
        """Prepare a :class:`GraphDataset` from UrbanEV matrices.

        Parameters
        ----------
        occupancy_matrix : np.ndarray
            ``(T, N)`` occupancy counts.
        volume_matrix : np.ndarray
            ``(T, N)`` energy volume (kWh).
        adj_matrix : np.ndarray
            ``(N, N)`` binary adjacency.
        dist_matrix : np.ndarray | None
            ``(N, N)`` distance matrix (km).
        features_dict : dict | None
            Extra temporal features (``hour``, ``day_of_week``).
        window_size : int
            Historical window length.
        horizon : int
            Prediction horizon.
        target : str
            ``"occupancy"`` or ``"volume"``.

        Returns
        -------
        GraphDataset
        """
        dataset = GraphDataset(
            occupancy_matrix=occupancy_matrix,
            volume_matrix=volume_matrix,
            adj_matrix=adj_matrix,
            dist_matrix=dist_matrix,
            features_dict=features_dict,
            window_size=window_size,
            horizon=horizon,
            target=target,
        )
        logger.info(
            "Graph data prepared – %d samples, %d node features.",
            len(dataset),
            dataset.num_node_features,
        )
        return dataset

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        train_data: GraphDataset,
        val_data: Optional[GraphDataset] = None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        batch_size: Optional[int] = None,
        patience: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Train the GNN model.

        Parameters
        ----------
        train_data : GraphDataset
            Training graph dataset.
        val_data : GraphDataset | None
            Validation graph dataset.  Enables early stopping.
        epochs : int | None
            Number of training epochs (default from config or 100).
        lr : float | None
            Learning rate (default from config or 0.001).
        weight_decay : float | None
            L2 regularisation (default from config or 1e-4).
        batch_size : int | None
            Mini-batch size (default from config or 32).
        patience : int | None
            Early stopping patience (default from config or 20).

        Returns
        -------
        dict
            Training history with keys ``train_loss`` and optionally
            ``val_loss``.
        """
        if PyGDataLoader is None:
            raise ImportError("torch_geometric is required for GNN training.")

        # Resolve hyperparameters
        epochs = epochs or int(self._gnn_cfg.get("epochs", 100))
        lr = lr or float(self._gnn_cfg.get("learning_rate", 0.001))
        weight_decay = weight_decay or float(
            self._gnn_cfg.get("weight_decay", 1e-4),
        )
        batch_size = batch_size or int(self._gnn_cfg.get("batch_size", 32))
        patience = patience or int(self._gnn_cfg.get("patience", 20))

        # Build model
        num_features = train_data.num_node_features
        hidden_dim = int(self._gnn_cfg.get("hidden_channels", 64))
        num_layers = int(self._gnn_cfg.get("num_layers", 2))
        dropout = float(self._gnn_cfg.get("dropout", 0.1))

        self._model = SpatialDemandGNN(
            num_features=num_features,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self._device)

        optimizer = Adam(
            self._model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience // 2,
        )
        criterion = nn.MSELoss()
        early_stopping = _EarlyStopping(patience=patience)

        # DataLoaders
        train_samples = train_data.create_samples()
        train_loader = PyGDataLoader(
            train_samples, batch_size=batch_size, shuffle=True,
        )

        val_loader = None
        if val_data is not None:
            val_samples = val_data.create_samples()
            val_loader = PyGDataLoader(
                val_samples, batch_size=batch_size, shuffle=False,
            )

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        logger.info(
            "GNN training starting – epochs=%d, lr=%.1e, batch=%d, "
            "patience=%d, device=%s",
            epochs,
            lr,
            batch_size,
            patience,
            self._device,
        )

        best_model_state = None

        for epoch in range(1, epochs + 1):
            # ---- Train ----
            self._model.train()
            train_loss_sum = 0.0
            train_count = 0

            for batch in train_loader:
                batch = batch.to(self._device)
                optimizer.zero_grad()

                edge_weight = getattr(batch, "edge_weight", None)
                pred = self._model(batch.x, batch.edge_index, edge_weight)
                loss = criterion(pred, batch.y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), max_norm=1.0,
                )
                optimizer.step()

                train_loss_sum += loss.item() * batch.num_graphs
                train_count += batch.num_graphs

            epoch_train_loss = train_loss_sum / max(train_count, 1)
            history["train_loss"].append(epoch_train_loss)

            # ---- Validate ----
            epoch_val_loss: Optional[float] = None
            if val_loader is not None:
                epoch_val_loss = self._evaluate_loss(val_loader, criterion)
                history["val_loss"].append(epoch_val_loss)
                scheduler.step(epoch_val_loss)

                if early_stopping(epoch_val_loss):
                    logger.info(
                        "Early stopping at epoch %d (val_loss=%.6f).",
                        epoch,
                        epoch_val_loss,
                    )
                    break

                # Track best model
                if early_stopping.best_score == epoch_val_loss:
                    best_model_state = {
                        k: v.cpu().clone()
                        for k, v in self._model.state_dict().items()
                    }

            if epoch % max(1, epochs // 10) == 0 or epoch == 1:
                val_str = (
                    f", val_loss={epoch_val_loss:.6f}"
                    if epoch_val_loss is not None
                    else ""
                )
                logger.info(
                    "Epoch %d/%d – train_loss=%.6f%s",
                    epoch,
                    epochs,
                    epoch_train_loss,
                    val_str,
                )

        # Restore best model if we tracked it
        if best_model_state is not None:
            self._model.load_state_dict(best_model_state)
            logger.info("Restored best model (val_loss=%.6f).", early_stopping.best_score)

        logger.info("GNN training complete.")
        return history

    def _evaluate_loss(
        self,
        loader: Any,
        criterion: nn.Module,
    ) -> float:
        """Compute average loss on a data loader."""
        self._model.eval()  # type: ignore[union-attr]
        loss_sum = 0.0
        count = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)
                edge_weight = getattr(batch, "edge_weight", None)
                pred = self._model(batch.x, batch.edge_index, edge_weight)  # type: ignore[union-attr]
                loss = criterion(pred, batch.y)
                loss_sum += loss.item() * batch.num_graphs
                count += batch.num_graphs

        return loss_sum / max(count, 1)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        graph_data: Union[Data, List[Data]],
    ) -> np.ndarray:
        """Generate predictions for one or more graph snapshots.

        Parameters
        ----------
        graph_data : Data | list[Data]
            Single ``Data`` object or a list.

        Returns
        -------
        np.ndarray
            Predictions with shape ``(num_samples, num_nodes, output_dim)``.
        """
        if self._model is None:
            raise RuntimeError("Model not trained.  Call train() first.")

        if PyGDataLoader is None:
            raise ImportError("torch_geometric is required.")

        if isinstance(graph_data, list):
            samples = graph_data
        else:
            samples = [graph_data]

        loader = PyGDataLoader(samples, batch_size=len(samples), shuffle=False)
        self._model.eval()

        all_preds: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)
                edge_weight = getattr(batch, "edge_weight", None)
                pred = self._model(batch.x, batch.edge_index, edge_weight)
                all_preds.append(pred.cpu().numpy())

        return np.concatenate(all_preds, axis=0)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        test_data: GraphDataset,
    ) -> Dict[str, Any]:
        """Evaluate model on test data.

        Returns per-node and aggregate RMSE, MAE, R² metrics.

        Parameters
        ----------
        test_data : GraphDataset
            Test graph dataset.

        Returns
        -------
        dict
            Keys: ``rmse``, ``mae``, ``r2`` (aggregate), plus
            ``rmse_per_node``, ``mae_per_node``, ``r2_per_node``.
        """
        if self._model is None:
            raise RuntimeError("Model not trained.")

        if PyGDataLoader is None:
            raise ImportError("torch_geometric is required.")

        test_samples = test_data.create_samples()
        loader = PyGDataLoader(test_samples, batch_size=64, shuffle=False)
        self._model.eval()

        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)
                edge_weight = getattr(batch, "edge_weight", None)
                pred = self._model(batch.x, batch.edge_index, edge_weight)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)    # (total_nodes, 1)
        targets = np.concatenate(all_targets, axis=0)  # (total_nodes, 1)

        num_nodes = test_data._N
        n_samples = len(test_samples)

        # Reshape to (n_samples, N, 1) for per-node metrics
        try:
            preds_reshaped = preds.reshape(n_samples, num_nodes, -1)
            targets_reshaped = targets.reshape(n_samples, num_nodes, -1)
        except ValueError:
            logger.warning(
                "Could not reshape for per-node metrics; computing "
                "aggregate only."
            )
            preds_reshaped = None
            targets_reshaped = None

        # Aggregate metrics
        rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
        mae = float(np.mean(np.abs(preds - targets)))
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

        metrics: Dict[str, Any] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_samples": n_samples,
            "n_nodes": num_nodes,
        }

        # Per-node metrics
        if preds_reshaped is not None and targets_reshaped is not None:
            rmse_per_node = np.sqrt(
                np.mean((preds_reshaped - targets_reshaped) ** 2, axis=0),
            ).flatten()
            mae_per_node = np.mean(
                np.abs(preds_reshaped - targets_reshaped), axis=0,
            ).flatten()

            ss_res_node = np.sum(
                (targets_reshaped - preds_reshaped) ** 2, axis=0,
            ).flatten()
            ss_tot_node = np.sum(
                (targets_reshaped - np.mean(targets_reshaped, axis=0)) ** 2,
                axis=0,
            ).flatten()
            r2_per_node = 1.0 - ss_res_node / (ss_tot_node + 1e-8)

            metrics["rmse_per_node"] = rmse_per_node.tolist()
            metrics["mae_per_node"] = mae_per_node.tolist()
            metrics["r2_per_node"] = r2_per_node.tolist()

        logger.info(
            "GNN Evaluation – RMSE=%.4f, MAE=%.4f, R²=%.4f",
            rmse,
            mae,
            r2,
        )
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Save model state dict and config.

        Parameters
        ----------
        path : str | Path
            File path (without extension – ``.pt`` is appended).
        """
        if self._model is None:
            raise RuntimeError("No model to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_path = path.with_suffix(".pt")
        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "config": self.config,
            "model_config": {
                "num_features": self._model.num_features,
                "hidden_dim": self._model.hidden_dim,
                "output_dim": self._model.output_dim,
                "num_layers": self._model.num_layers,
                "dropout": self._model.dropout,
            },
        }
        torch.save(checkpoint, save_path)
        logger.info("GNN model saved to %s", save_path)

    def load(self, path: Union[str, Path]) -> None:
        """Load model state dict from checkpoint.

        Parameters
        ----------
        path : str | Path
            Path to the ``.pt`` checkpoint.
        """
        path = Path(path).with_suffix(".pt")
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self._device, weights_only=False)

        model_cfg = checkpoint.get("model_config", {})
        self._model = SpatialDemandGNN(
            num_features=model_cfg["num_features"],
            hidden_dim=model_cfg.get("hidden_dim", 64),
            output_dim=model_cfg.get("output_dim", 1),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.1),
        ).to(self._device)

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()
        logger.info("GNN model loaded from %s", path)

    def __repr__(self) -> str:
        status = "trained" if self._model is not None else "untrained"
        return f"GNNSpatialAgent(status={status}, device={self._device})"
