# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""
SQLite database manager for persisting agent metrics, pricing decisions,
and model performance records.

Usage::

    from src.utils.db_manager import DatabaseManager

    with DatabaseManager() as db:
        db.insert_pricing_decision({...})
        rows = db.query_agent_metrics(limit=100)
"""

from __future__ import annotations

import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger(__name__)


class DatabaseManager:
    """Lightweight SQLite manager with context-manager support.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the SQLite file.  Defaults to ``data.database_path`` from
        *config.yaml* (resolved relative to the project root).
    """

    # ------------------------------------------------------------------
    # DDL statements
    # ------------------------------------------------------------------
    _CREATE_AGENT_METRICS = """
    CREATE TABLE IF NOT EXISTS agent_metrics (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp     TEXT    NOT NULL,
        agent_name    TEXT    NOT NULL,
        metric_name   TEXT    NOT NULL,
        metric_value  REAL    NOT NULL,
        metadata      TEXT,
        created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
    );
    """

    _CREATE_PRICING_DECISIONS = """
    CREATE TABLE IF NOT EXISTS pricing_decisions (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp         TEXT    NOT NULL,
        station_id        TEXT    NOT NULL,
        utilization       REAL    NOT NULL,
        baseline_price    REAL    NOT NULL,
        optimized_price   REAL    NOT NULL,
        price_delta       REAL    NOT NULL,
        revenue_old       REAL,
        revenue_new       REAL,
        revenue_gain_pct  REAL,
        decision_source   TEXT,
        created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
    );
    """

    _CREATE_MODEL_PERFORMANCE = """
    CREATE TABLE IF NOT EXISTS model_performance (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp      TEXT    NOT NULL,
        model_name     TEXT    NOT NULL,
        dataset_split  TEXT    NOT NULL,
        metric_name    TEXT    NOT NULL,
        metric_value   REAL    NOT NULL,
        hyperparams    TEXT,
        created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
    );
    """

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        if db_path is None:
            cfg = ConfigLoader()
            project_root = Path(__file__).resolve().parents[2]
            db_path = project_root / cfg.data.database_path
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._create_tables()
        logger.info("DatabaseManager initialised – %s", self._db_path)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row

    def _create_tables(self) -> None:
        assert self._conn is not None
        cur = self._conn.cursor()
        cur.execute(self._CREATE_AGENT_METRICS)
        cur.execute(self._CREATE_PRICING_DECISIONS)
        cur.execute(self._CREATE_MODEL_PERFORMANCE)
        self._conn.commit()
        logger.debug("Database tables verified / created.")

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("Database connection closed.")

    # Context-manager protocol
    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Insert helpers
    # ------------------------------------------------------------------
    def insert_agent_metric(
        self,
        agent_name: str,
        metric_name: str,
        metric_value: float,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert a single agent-metric row and return the ``rowid``."""
        assert self._conn is not None
        ts = timestamp or datetime.utcnow().isoformat()
        meta_json = json.dumps(metadata) if metadata else None
        cur = self._conn.execute(
            """INSERT INTO agent_metrics
               (timestamp, agent_name, metric_name, metric_value, metadata)
               VALUES (?, ?, ?, ?, ?)""",
            (ts, agent_name, metric_name, metric_value, meta_json),
        )
        self._conn.commit()
        logger.debug(
            "Inserted agent_metric: %s / %s = %s",
            agent_name, metric_name, metric_value,
        )
        return cur.lastrowid  # type: ignore[return-value]

    def insert_pricing_decision(self, record: Dict[str, Any]) -> int:
        """Insert a pricing-decision row from a flat dict.

        Expected keys: ``timestamp``, ``station_id``, ``utilization``,
        ``baseline_price``, ``optimized_price``, ``price_delta``, and
        optionally ``revenue_old``, ``revenue_new``, ``revenue_gain_pct``,
        ``decision_source``.
        """
        assert self._conn is not None
        cur = self._conn.execute(
            """INSERT INTO pricing_decisions
               (timestamp, station_id, utilization, baseline_price,
                optimized_price, price_delta, revenue_old, revenue_new,
                revenue_gain_pct, decision_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record["timestamp"],
                record["station_id"],
                record["utilization"],
                record["baseline_price"],
                record["optimized_price"],
                record["price_delta"],
                record.get("revenue_old"),
                record.get("revenue_new"),
                record.get("revenue_gain_pct"),
                record.get("decision_source"),
            ),
        )
        self._conn.commit()
        logger.debug("Inserted pricing_decision for station %s", record["station_id"])
        return cur.lastrowid  # type: ignore[return-value]

    def insert_model_performance(
        self,
        model_name: str,
        dataset_split: str,
        metric_name: str,
        metric_value: float,
        timestamp: Optional[str] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert a model-performance row and return its ``rowid``."""
        assert self._conn is not None
        ts = timestamp or datetime.utcnow().isoformat()
        hp_json = json.dumps(hyperparams) if hyperparams else None
        cur = self._conn.execute(
            """INSERT INTO model_performance
               (timestamp, model_name, dataset_split, metric_name,
                metric_value, hyperparams)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ts, model_name, dataset_split, metric_name, metric_value, hp_json),
        )
        self._conn.commit()
        logger.debug(
            "Inserted model_performance: %s / %s / %s = %s",
            model_name, dataset_split, metric_name, metric_value,
        )
        return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def query_agent_metrics(
        self,
        agent_name: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Return agent-metric rows as a list of dicts."""
        assert self._conn is not None
        sql = "SELECT * FROM agent_metrics"
        params: list[Any] = []
        if agent_name:
            sql += " WHERE agent_name = ?"
            params.append(agent_name)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def query_pricing_decisions(
        self,
        station_id: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Return pricing-decision rows as a list of dicts."""
        assert self._conn is not None
        sql = "SELECT * FROM pricing_decisions"
        params: list[Any] = []
        if station_id:
            sql += " WHERE station_id = ?"
            params.append(station_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def query_model_performance(
        self,
        model_name: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Return model-performance rows as a list of dicts."""
        assert self._conn is not None
        sql = "SELECT * FROM model_performance"
        params: list[Any] = []
        if model_name:
            sql += " WHERE model_name = ?"
            params.append(model_name)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
