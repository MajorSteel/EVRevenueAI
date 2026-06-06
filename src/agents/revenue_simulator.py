"""
Revenue simulator for comparing fixed vs. dynamic EV charging tariffs.

Provides side-by-side simulation of the baseline (₹15/kWh fixed) and
the RL-optimised dynamic pricing strategy, computing key KPIs:
Revenue Gain %, Utilization Rate, Congestion Reduction %,
Demand Shift, and Off-Peak Uplift %.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class RevenueSimulator:
    """Simulate and compare fixed vs. dynamic pricing strategies.

    Parameters
    ----------
    config : dict | None
        Configuration dict.  Relevant keys:

        * ``baseline_price`` (float, default 15.0)
        * ``elasticity`` (float, default 0.3)
        * ``min_price`` / ``max_price``
        * ``surge_threshold`` (float, default 0.8)
        * ``off_peak_threshold`` (float, default 0.3)
    tariff_agent : TariffPricingAgent | None
        A trained :class:`~src.agents.tariff_agent.TariffPricingAgent`.
        Required for ``simulate_dynamic_pricing``.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        tariff_agent: Optional[Any] = None,
    ) -> None:
        self.config: Dict[str, Any] = config or {}
        self.tariff_agent = tariff_agent

        self._baseline_price: float = float(
            self.config.get("baseline_price", 15.0),
        )
        self._elasticity: float = float(
            self.config.get("elasticity", 0.3),
        )
        self._min_price: float = float(self.config.get("min_price", 5.0))
        self._max_price: float = float(self.config.get("max_price", 30.0))
        self._surge_threshold: float = float(
            self.config.get("surge_threshold", 0.8),
        )
        self._off_peak_threshold: float = float(
            self.config.get("off_peak_threshold", 0.3),
        )

        self._fixed_results: Optional[Dict[str, Any]] = None
        self._dynamic_results: Optional[Dict[str, Any]] = None

        logger.info(
            "RevenueSimulator initialised – baseline=%.1f, elasticity=%.2f",
            self._baseline_price,
            self._elasticity,
        )

    # ------------------------------------------------------------------
    # Fixed-price simulation
    # ------------------------------------------------------------------
    def simulate_fixed_pricing(
        self,
        data_df: pd.DataFrame,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Simulate revenue under a fixed price.

        Parameters
        ----------
        data_df : pd.DataFrame
            Must contain ``volume``, ``occupancy``, ``capacity``
            columns.
        price : float | None
            Fixed price per kWh (defaults to ``baseline_price``).

        Returns
        -------
        dict
            ``total_revenue``, ``mean_revenue_per_step``,
            ``mean_utilization``, ``congestion_rate``,
            ``off_peak_rate``, ``step_revenues``.
        """
        price = price if price is not None else self._baseline_price

        volume = data_df["volume"].values.astype(np.float64)
        occupancy = data_df["occupancy"].values.astype(np.float64)
        capacity = data_df["capacity"].values.astype(np.float64)

        # Revenue
        step_revenues = price * volume
        total_revenue = float(np.sum(step_revenues))

        # Utilization
        utilization = np.where(
            capacity > 0, occupancy / capacity, 0.0,
        )
        utilization = np.clip(utilization, 0.0, 1.0)
        mean_util = float(np.mean(utilization))

        # Congestion (utilization > surge threshold)
        congestion_mask = utilization > self._surge_threshold
        congestion_rate = float(np.mean(congestion_mask))

        # Off-peak (utilization < off_peak threshold)
        off_peak_mask = utilization < self._off_peak_threshold
        off_peak_rate = float(np.mean(off_peak_mask))

        results: Dict[str, Any] = {
            "total_revenue": total_revenue,
            "mean_revenue_per_step": float(np.mean(step_revenues)),
            "mean_utilization": mean_util,
            "congestion_rate": congestion_rate,
            "off_peak_rate": off_peak_rate,
            "step_revenues": step_revenues.tolist(),
            "price_used": price,
            "n_steps": len(data_df),
        }
        self._fixed_results = results

        logger.info(
            "Fixed pricing simulation – revenue=%.2f, util=%.3f, "
            "congestion=%.3f",
            total_revenue,
            mean_util,
            congestion_rate,
        )
        return results

    # ------------------------------------------------------------------
    # Dynamic-price simulation
    # ------------------------------------------------------------------
    def simulate_dynamic_pricing(
        self,
        data_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Simulate revenue under the RL-optimised dynamic strategy.

        Parameters
        ----------
        data_df : pd.DataFrame
            Must satisfy :attr:`EVChargingEnv._REQUIRED_COLS`.

        Returns
        -------
        dict
            Same structure as :meth:`simulate_fixed_pricing`, plus
            ``actions``, ``prices_applied``.

        Raises
        ------
        RuntimeError
            If ``tariff_agent`` has not been set.
        """
        if self.tariff_agent is None:
            raise RuntimeError(
                "tariff_agent is required for dynamic pricing simulation."
            )

        from src.agents.tariff_agent import (
            ACTION_MULTIPLIERS,
            ACTION_NAMES,
            EVChargingEnv,
        )

        env = EVChargingEnv(data_df=data_df, config=self.config)
        obs, _ = env.reset()

        step_revenues: List[float] = []
        utilizations: List[float] = []
        actions: List[int] = []
        prices_applied: List[float] = []

        done = False
        while not done:
            action, _ = self.tariff_agent._model.predict(
                obs, deterministic=True,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            step_revenues.append(info.get("revenue_new", 0.0))
            utilizations.append(info.get("utilization", 0.0))
            actions.append(int(action))
            prices_applied.append(info.get("new_price", self._baseline_price))

        total_revenue = float(np.sum(step_revenues))
        mean_util = float(np.mean(utilizations))

        util_arr = np.array(utilizations)
        congestion_rate = float(np.mean(util_arr > self._surge_threshold))
        off_peak_rate = float(np.mean(util_arr < self._off_peak_threshold))

        # Action distribution
        action_dist: Dict[str, int] = {}
        for a in actions:
            name = ACTION_NAMES.get(a, str(a))
            action_dist[name] = action_dist.get(name, 0) + 1

        results: Dict[str, Any] = {
            "total_revenue": total_revenue,
            "mean_revenue_per_step": float(np.mean(step_revenues)),
            "mean_utilization": mean_util,
            "congestion_rate": congestion_rate,
            "off_peak_rate": off_peak_rate,
            "step_revenues": step_revenues,
            "actions": actions,
            "prices_applied": prices_applied,
            "action_distribution": action_dist,
            "mean_price": float(np.mean(prices_applied)),
            "n_steps": len(step_revenues),
        }
        self._dynamic_results = results

        logger.info(
            "Dynamic pricing simulation – revenue=%.2f, util=%.3f, "
            "congestion=%.3f, mean_price=%.2f",
            total_revenue,
            mean_util,
            congestion_rate,
            results["mean_price"],
        )
        return results

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    def compare(self) -> pd.DataFrame:
        """Compare fixed vs. dynamic pricing results.

        Returns
        -------
        pd.DataFrame
            Comparison table with columns: ``Metric``, ``Fixed``,
            ``Dynamic``, ``Change``.

        Raises
        ------
        RuntimeError
            If either simulation has not been run.
        """
        if self._fixed_results is None:
            raise RuntimeError(
                "Run simulate_fixed_pricing() before comparing."
            )
        if self._dynamic_results is None:
            raise RuntimeError(
                "Run simulate_dynamic_pricing() before comparing."
            )

        fixed = self._fixed_results
        dynamic = self._dynamic_results

        # Revenue Gain %
        revenue_gain_pct = (
            (dynamic["total_revenue"] - fixed["total_revenue"])
            / (fixed["total_revenue"] + 1e-8)
            * 100.0
        )

        # Utilization Rate Change
        util_change = (
            (dynamic["mean_utilization"] - fixed["mean_utilization"])
            * 100.0  # percentage points
        )

        # Demand Shift = change in off-peak volume proportion
        demand_shift = (
            (fixed["off_peak_rate"] - dynamic["off_peak_rate"]) * 100.0
        )

        # Congestion Reduction %
        congestion_reduction_pct = (
            (fixed["congestion_rate"] - dynamic["congestion_rate"])
            / (fixed["congestion_rate"] + 1e-8)
            * 100.0
        )

        # Off-Peak Uplift %
        off_peak_uplift_pct = (
            (fixed["off_peak_rate"] - dynamic["off_peak_rate"])
            / (fixed["off_peak_rate"] + 1e-8)
            * 100.0
        )

        rows = [
            {
                "Metric": "Total Revenue (₹)",
                "Fixed": f"{fixed['total_revenue']:,.2f}",
                "Dynamic": f"{dynamic['total_revenue']:,.2f}",
                "Change": f"{revenue_gain_pct:+.2f}%",
            },
            {
                "Metric": "Revenue Gain %",
                "Fixed": "—",
                "Dynamic": "—",
                "Change": f"{revenue_gain_pct:+.2f}%",
            },
            {
                "Metric": "Mean Utilization",
                "Fixed": f"{fixed['mean_utilization']:.3f}",
                "Dynamic": f"{dynamic['mean_utilization']:.3f}",
                "Change": f"{util_change:+.2f} pp",
            },
            {
                "Metric": "Congestion Rate",
                "Fixed": f"{fixed['congestion_rate']:.3f}",
                "Dynamic": f"{dynamic['congestion_rate']:.3f}",
                "Change": f"{congestion_reduction_pct:+.2f}%",
            },
            {
                "Metric": "Congestion Reduction %",
                "Fixed": "—",
                "Dynamic": "—",
                "Change": f"{congestion_reduction_pct:+.2f}%",
            },
            {
                "Metric": "Off-Peak Rate",
                "Fixed": f"{fixed['off_peak_rate']:.3f}",
                "Dynamic": f"{dynamic['off_peak_rate']:.3f}",
                "Change": f"{off_peak_uplift_pct:+.2f}%",
            },
            {
                "Metric": "Off-Peak Uplift %",
                "Fixed": "—",
                "Dynamic": "—",
                "Change": f"{off_peak_uplift_pct:+.2f}%",
            },
            {
                "Metric": "Demand Shift (pp)",
                "Fixed": "—",
                "Dynamic": "—",
                "Change": f"{demand_shift:+.2f} pp",
            },
        ]

        comparison_df = pd.DataFrame(rows)
        logger.info(
            "Comparison complete – Revenue Gain=%.2f%%, "
            "Congestion Reduction=%.2f%%",
            revenue_gain_pct,
            congestion_reduction_pct,
        )
        return comparison_df

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def generate_report(self) -> str:
        """Generate a formatted summary report.

        Returns
        -------
        str
            Multi-line report string.
        """
        lines: List[str] = [
            "=" * 60,
            "  EV CHARGING TARIFF OPTIMIZATION – Revenue Report",
            "=" * 60,
            "",
        ]

        if self._fixed_results is not None:
            f = self._fixed_results
            lines.extend([
                "FIXED PRICING (Baseline)",
                f"  Price:             ₹{f['price_used']:.2f} / kWh",
                f"  Total Revenue:     ₹{f['total_revenue']:,.2f}",
                f"  Mean Utilization:  {f['mean_utilization']:.3f}",
                f"  Congestion Rate:   {f['congestion_rate']:.3f}",
                f"  Off-Peak Rate:     {f['off_peak_rate']:.3f}",
                f"  Steps:             {f['n_steps']}",
                "",
            ])

        if self._dynamic_results is not None:
            d = self._dynamic_results
            lines.extend([
                "DYNAMIC PRICING (RL-Optimised)",
                f"  Mean Price:        ₹{d['mean_price']:.2f} / kWh",
                f"  Total Revenue:     ₹{d['total_revenue']:,.2f}",
                f"  Mean Utilization:  {d['mean_utilization']:.3f}",
                f"  Congestion Rate:   {d['congestion_rate']:.3f}",
                f"  Off-Peak Rate:     {d['off_peak_rate']:.3f}",
                f"  Steps:             {d['n_steps']}",
                f"  Action Dist:       {d.get('action_distribution', {})}",
                "",
            ])

        if self._fixed_results and self._dynamic_results:
            gain = (
                (self._dynamic_results["total_revenue"]
                 - self._fixed_results["total_revenue"])
                / (self._fixed_results["total_revenue"] + 1e-8)
                * 100.0
            )
            lines.extend([
                "-" * 60,
                f"  REVENUE GAIN:  {gain:+.2f}%",
                "-" * 60,
            ])

        report = "\n".join(lines)
        logger.info("Revenue report generated.")
        return report

    def __repr__(self) -> str:
        has_fixed = self._fixed_results is not None
        has_dynamic = self._dynamic_results is not None
        return (
            f"RevenueSimulator(fixed_run={has_fixed}, "
            f"dynamic_run={has_dynamic})"
        )
