"""
Monitoring agent for tracking, drift detection, and feedback generation.

Uses a local SQLite database to persist per-episode metrics and exposes
Z-score-based drift detection to trigger model retraining.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

# Metrics tracked by the monitoring agent
TRACKED_METRICS: List[str] = [
    "revenue_gain_pct",
    "utilization_rate",
    "congestion_rate",
    "wait_time_proxy",
    "pricing_efficiency_score",
]


class MonitoringAgent:
    """Agent for tracking model performance and detecting drift.

    Parameters
    ----------
    config : dict | None
        Configuration dictionary.
    db_path : str | Path
        Path to the SQLite database file.  Parent directories are
        created automatically.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        db_path: Union[str, Path] = "data/models/monitoring.db",
    ) -> None:
        self.config: Dict[str, Any] = config or {}
        self.db_path: Path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        logger.info("MonitoringAgent initialised – db=%s", self.db_path)

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        """Create the metrics table if it does not exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episode_metrics (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,
                    episode         INTEGER,
                    revenue_gain_pct        REAL,
                    utilization_rate        REAL,
                    congestion_rate         REAL,
                    wait_time_proxy         REAL,
                    pricing_efficiency_score REAL,
                    extra_json      TEXT
                )
                """,
            )
        logger.debug("Database table ensured.")

    def _connect(self) -> sqlite3.Connection:
        """Return a new SQLite connection."""
        return sqlite3.connect(str(self.db_path))

    # ------------------------------------------------------------------
    # Track metrics
    # ------------------------------------------------------------------
    def track_metrics(
        self,
        episode_data: Dict[str, Any],
    ) -> int:
        """Store episode metrics in the database.

        Parameters
        ----------
        episode_data : dict
            Must contain a subset of :data:`TRACKED_METRICS` keys.
            An ``episode`` key is optional (auto-incremented if absent).
            Extra keys are serialised to ``extra_json``.

        Returns
        -------
        int
            Row ID of the inserted record.
        """
        import json

        ts = datetime.now(tz=timezone.utc).isoformat()
        episode = episode_data.get("episode")

        known_keys = set(TRACKED_METRICS) | {"episode"}
        extra = {k: v for k, v in episode_data.items() if k not in known_keys}
        extra_json = json.dumps(extra) if extra else None

        # Compute derived metrics if not provided
        revenue_gain = episode_data.get("revenue_gain_pct")
        utilization = episode_data.get("utilization_rate")
        congestion = episode_data.get("congestion_rate")
        wait_time = episode_data.get("wait_time_proxy")
        pricing_eff = episode_data.get("pricing_efficiency_score")

        # Pricing efficiency = revenue_gain / (congestion + 1)
        if pricing_eff is None and revenue_gain is not None and congestion is not None:
            pricing_eff = revenue_gain / (congestion + 1.0)

        # Wait-time proxy ~ congestion * average occupancy
        if wait_time is None and congestion is not None:
            avg_occ = episode_data.get("mean_utilization", congestion)
            wait_time = congestion * avg_occ * 10.0  # arbitrary scale

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO episode_metrics
                    (timestamp, episode, revenue_gain_pct,
                     utilization_rate, congestion_rate,
                     wait_time_proxy, pricing_efficiency_score,
                     extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    episode,
                    revenue_gain,
                    utilization,
                    congestion,
                    wait_time,
                    pricing_eff,
                    extra_json,
                ),
            )
            row_id: int = cursor.lastrowid  # type: ignore[assignment]

        logger.debug(
            "Tracked episode %s – revenue_gain=%.2f%%, util=%.3f, "
            "congestion=%.3f",
            episode,
            revenue_gain or 0.0,
            utilization or 0.0,
            congestion or 0.0,
        )
        return row_id

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------
    def detect_drift(
        self,
        metric_name: str,
        window: int = 100,
        threshold: float = 2.0,
    ) -> bool:
        """Detect distribution drift using a Z-score test.

        Compares the most recent ``window // 4`` values against the
        full window history.  Returns ``True`` if the mean of the
        recent segment is more than *threshold* standard deviations
        from the historical mean.

        Parameters
        ----------
        metric_name : str
            One of :data:`TRACKED_METRICS`.
        window : int
            Number of recent episodes to consider.
        threshold : float
            Z-score threshold for flagging drift.

        Returns
        -------
        bool
            ``True`` if drift is detected.
        """
        if metric_name not in TRACKED_METRICS:
            raise ValueError(
                f"Unknown metric '{metric_name}'.  "
                f"Choose from {TRACKED_METRICS}."
            )

        history = self.get_performance_history(metric_name, n_episodes=window)
        if len(history) < window // 2:
            logger.debug(
                "Not enough data (%d rows) for drift detection on '%s'.",
                len(history),
                metric_name,
            )
            return False

        values = history[metric_name].dropna().values
        if len(values) < 8:
            return False

        recent_size = max(len(values) // 4, 1)
        recent = values[-recent_size:]
        historical = values[:-recent_size]

        hist_mean = float(np.mean(historical))
        hist_std = float(np.std(historical))
        if hist_std < 1e-8:
            return False

        z_score = abs(float(np.mean(recent)) - hist_mean) / hist_std
        drifted = z_score > threshold

        if drifted:
            logger.warning(
                "Drift detected on '%s' – z=%.2f (threshold=%.1f), "
                "recent_mean=%.4f, hist_mean=%.4f",
                metric_name,
                z_score,
                threshold,
                float(np.mean(recent)),
                hist_mean,
            )
        else:
            logger.debug(
                "No drift on '%s' – z=%.2f (threshold=%.1f)",
                metric_name,
                z_score,
                threshold,
            )
        return drifted

    # ------------------------------------------------------------------
    # Feedback generation
    # ------------------------------------------------------------------
    def create_feedback(
        self,
        current_metrics: Dict[str, float],
        target_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """Generate a feedback dict comparing current vs. target.

        Parameters
        ----------
        current_metrics : dict
            Latest observed metric values.
        target_metrics : dict
            Desired target values.

        Returns
        -------
        dict
            Keys: ``gaps``, ``recommendations``, ``overall_score``.
        """
        gaps: Dict[str, float] = {}
        recommendations: List[str] = []

        for metric, target in target_metrics.items():
            current = current_metrics.get(metric, 0.0)
            gap = target - current
            gaps[metric] = gap

            if metric == "revenue_gain_pct" and gap > 0:
                recommendations.append(
                    f"Increase revenue gain by {gap:.2f}%."
                )
            elif metric == "congestion_rate" and current > target:
                recommendations.append(
                    f"Reduce congestion rate by {abs(gap):.3f} "
                    f"(current={current:.3f}, target={target:.3f})."
                )
            elif metric == "utilization_rate" and gap > 0.05:
                recommendations.append(
                    f"Improve utilization by {gap:.3f} "
                    f"(current={current:.3f}, target={target:.3f})."
                )

        # Overall score: weighted distance from targets
        weights = {
            "revenue_gain_pct": 0.4,
            "utilization_rate": 0.25,
            "congestion_rate": 0.2,
            "wait_time_proxy": 0.1,
            "pricing_efficiency_score": 0.05,
        }
        total_weighted_gap = sum(
            abs(gaps.get(m, 0.0)) * weights.get(m, 0.1)
            for m in target_metrics
        )
        overall_score = max(0.0, 100.0 - total_weighted_gap)

        feedback: Dict[str, Any] = {
            "gaps": gaps,
            "recommendations": recommendations,
            "overall_score": overall_score,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        logger.info(
            "Feedback generated – overall_score=%.1f, %d recommendations",
            overall_score,
            len(recommendations),
        )
        return feedback

    # ------------------------------------------------------------------
    # Performance history
    # ------------------------------------------------------------------
    def get_performance_history(
        self,
        metric_name: str,
        n_episodes: int = 100,
    ) -> pd.DataFrame:
        """Retrieve recent metric values from the database.

        Parameters
        ----------
        metric_name : str
            Column name in ``episode_metrics``.
        n_episodes : int
            Number of most recent records to return.

        Returns
        -------
        pd.DataFrame
            Columns: ``episode``, ``timestamp``, *metric_name*.
        """
        if metric_name not in TRACKED_METRICS:
            raise ValueError(
                f"Unknown metric '{metric_name}'. "
                f"Choose from {TRACKED_METRICS}."
            )

        with self._connect() as conn:
            query = (
                f"SELECT episode, timestamp, {metric_name} "
                f"FROM episode_metrics "
                f"ORDER BY id DESC LIMIT ?"
            )
            df = pd.read_sql_query(query, conn, params=(n_episodes,))

        # Reverse so oldest is first
        df = df.iloc[::-1].reset_index(drop=True)
        logger.debug(
            "Retrieved %d records for '%s'.", len(df), metric_name,
        )
        return df

    # ------------------------------------------------------------------
    # Retrain decision
    # ------------------------------------------------------------------
    def should_retrain(
        self,
        metrics: Optional[Dict[str, float]] = None,
        drift_window: int = 100,
        drift_threshold: float = 2.0,
    ) -> bool:
        """Decide whether the model should be retrained.

        Returns ``True`` if drift is detected on **any** tracked metric
        or if key metrics fall below configured thresholds.

        Parameters
        ----------
        metrics : dict | None
            Latest metrics to check thresholds.  If ``None``, only
            drift detection is used.
        drift_window : int
            Window size for :meth:`detect_drift`.
        drift_threshold : float
            Z-score threshold for :meth:`detect_drift`.

        Returns
        -------
        bool
        """
        needs_retrain = False

        # Check drift on all tracked metrics
        for metric in TRACKED_METRICS:
            try:
                if self.detect_drift(
                    metric, window=drift_window, threshold=drift_threshold,
                ):
                    logger.warning(
                        "Retrain triggered – drift on '%s'.", metric,
                    )
                    needs_retrain = True
            except Exception as exc:
                logger.debug("Drift check error for '%s': %s", metric, exc)

        # Threshold-based checks
        if metrics is not None:
            rev_gain = metrics.get("revenue_gain_pct", 0.0)
            min_gain = float(self.config.get("min_revenue_gain", -5.0))
            if rev_gain < min_gain:
                logger.warning(
                    "Retrain triggered – revenue_gain=%.2f%% < %.2f%%.",
                    rev_gain,
                    min_gain,
                )
                needs_retrain = True

            congestion = metrics.get("congestion_rate", 0.0)
            max_congestion = float(
                self.config.get("max_congestion_rate", 0.3),
            )
            if congestion > max_congestion:
                logger.warning(
                    "Retrain triggered – congestion=%.3f > %.3f.",
                    congestion,
                    max_congestion,
                )
                needs_retrain = True

        if not needs_retrain:
            logger.info("No retrain needed.")

        return needs_retrain

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def generate_report(self) -> str:
        """Generate a summary report of all tracked metrics.

        Returns
        -------
        str
            Multi-line formatted report.
        """
        lines: List[str] = [
            "=" * 60,
            "  MONITORING AGENT – Performance Report",
            f"  Generated: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "=" * 60,
            "",
        ]

        for metric in TRACKED_METRICS:
            try:
                df = self.get_performance_history(metric, n_episodes=100)
                values = df[metric].dropna()
                if len(values) == 0:
                    lines.append(f"  {metric}: No data")
                    continue

                lines.extend([
                    f"  {metric}:",
                    f"    Count:   {len(values)}",
                    f"    Mean:    {values.mean():.4f}",
                    f"    Std:     {values.std():.4f}",
                    f"    Min:     {values.min():.4f}",
                    f"    Max:     {values.max():.4f}",
                    f"    Latest:  {values.iloc[-1]:.4f}",
                ])

                # Drift status
                drifted = self.detect_drift(metric)
                drift_status = "⚠ DRIFT DETECTED" if drifted else "✓ Stable"
                lines.append(f"    Drift:   {drift_status}")
                lines.append("")
            except Exception as exc:
                lines.append(f"  {metric}: Error – {exc}")
                lines.append("")

        # Overall retrain recommendation
        needs_retrain = self.should_retrain()
        lines.extend([
            "-" * 60,
            f"  RETRAIN RECOMMENDED: {'YES' if needs_retrain else 'NO'}",
            "-" * 60,
        ])

        report = "\n".join(lines)
        logger.info("Monitoring report generated.")
        return report

    def __repr__(self) -> str:
        return f"MonitoringAgent(db={self.db_path})"
