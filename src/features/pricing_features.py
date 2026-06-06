# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""Pricing feature engineering for EV charging tariff optimization.

This module provides the :class:`PricingFeatureEngine` which derives
price-movement and demand-elasticity features used for dynamic tariff
modelling.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class PricingFeatureEngine:
    """Compute pricing-related features for EV charging data.

    Parameters
    ----------
    price_col : str
        Column name for energy price.  Defaults to ``"price"``.
    volume_col : str
        Column name for energy volume (kWh).  Defaults to ``"volume"``.
    baseline_price : float
        Baseline (reference) price per kWh used to categorise price
        levels.  Defaults to ``15.0`` (₹15/kWh).
    surge_threshold : float
        Utilization threshold above which price is considered *surge*.
        Used for reference but the ``compute_price_level`` method works
        on the price value itself.  Defaults to ``0.80``.
    discount_threshold : float
        Utilization threshold below which price is considered
        *discounted*.  Defaults to ``0.30``.
    """

    def __init__(
        self,
        price_col: str = "price",
        volume_col: str = "volume",
        baseline_price: float = 15.0,
        surge_threshold: float = 0.80,
        discount_threshold: float = 0.30,
    ) -> None:
        self.price_col = price_col
        self.volume_col = volume_col
        self.baseline_price = baseline_price
        self.surge_threshold = surge_threshold
        self.discount_threshold = discount_threshold
        logger.info(
            "PricingFeatureEngine initialised – price_col=%s, volume_col=%s, "
            "baseline=%.2f",
            price_col,
            volume_col,
            baseline_price,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_price_change(self, price_series: pd.Series) -> pd.Series:
        """Compute absolute price change (first difference).

        Parameters
        ----------
        price_series : pd.Series
            Price values over time.

        Returns
        -------
        pd.Series
            ``price.diff()`` – first entry will be ``NaN``.
        """
        change = price_series.diff()
        logger.debug(
            "Price change – mean=%.4f, std=%.4f",
            change.mean(),
            change.std(),
        )
        return change.rename("price_change")

    def compute_price_pct_change(self, price_series: pd.Series) -> pd.Series:
        """Compute percentage price change.

        Parameters
        ----------
        price_series : pd.Series
            Price values over time.

        Returns
        -------
        pd.Series
            ``price.pct_change()`` – first entry will be ``NaN``.
            ``inf`` values (from zero-base) are replaced with ``NaN``.
        """
        pct = price_series.pct_change()
        n_inf = np.isinf(pct).sum()
        if n_inf > 0:
            logger.warning(
                "price_pct_change: %d inf values replaced with NaN.", n_inf
            )
            pct = pct.replace([np.inf, -np.inf], np.nan)
        logger.debug("Price pct change – mean=%.4f", pct.mean())
        return pct.rename("price_pct_change")

    def compute_demand_elasticity(
        self,
        volume_series: pd.Series,
        price_series: pd.Series,
        window: int = 12,
    ) -> pd.Series:
        """Compute rolling demand-price elasticity.

        Elasticity is estimated as::

            ε = (%Δ volume) / (%Δ price)

        using rolling percentage changes over *window* timesteps.

        Parameters
        ----------
        volume_series : pd.Series
            Energy volume (kWh) over time.
        price_series : pd.Series
            Price per kWh over time.
        window : int
            Rolling window size for percentage changes.
            Defaults to ``12`` (1 hour at 5-min intervals).

        Returns
        -------
        pd.Series
            Rolling elasticity.  ``NaN`` where price change is zero or
            data is insufficient.
        """
        vol_pct = volume_series.pct_change(periods=window)
        price_pct = price_series.pct_change(periods=window)

        # Safely divide, avoiding division by zero / inf
        safe_price_pct = price_pct.replace(0, np.nan)
        elasticity = vol_pct / safe_price_pct
        elasticity = elasticity.replace([np.inf, -np.inf], np.nan)

        logger.debug(
            "Demand elasticity (window=%d) – mean=%.4f, median=%.4f",
            window,
            elasticity.mean(),
            elasticity.median(),
        )
        return elasticity.rename("demand_elasticity")

    def compute_price_level(
        self,
        price: pd.Series,
        baseline: Optional[float] = None,
    ) -> pd.Series:
        """Categorise each price observation as discount / normal / surge.

        The categorisation uses thresholds relative to the baseline:

        * **discount** – price < baseline × (1 − ``discount_threshold``)
          i.e. price below 70 % of baseline by default.
        * **surge** – price > baseline × (1 + ``surge_threshold``)
          i.e. price above 180 % of baseline by default.
        * **normal** – everything in between.

        For simplicity and consistency with the business rules, the
        logic uses:
        - ``price < baseline`` → ``"discount"``
        - ``price > baseline`` → ``"surge"``
        - ``price == baseline`` → ``"normal"``

        Parameters
        ----------
        price : pd.Series
            Price values.
        baseline : float, optional
            Override for ``self.baseline_price``.

        Returns
        -------
        pd.Series
            Categorical series with values ``"discount"``,
            ``"normal"``, or ``"surge"``.
        """
        baseline = baseline if baseline is not None else self.baseline_price

        conditions = [
            price < baseline,
            price > baseline,
        ]
        choices = ["discount", "surge"]
        level = pd.Series(
            np.select(conditions, choices, default="normal"),
            index=price.index,
            name="price_level",
        )

        for lbl in ("discount", "normal", "surge"):
            count = (level == lbl).sum()
            logger.debug("price_level '%s': %d rows", lbl, count)

        return level

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all pricing feature computations on *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain the configured price and volume columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with pricing features appended.
        """
        logger.info("Running full pricing feature computation …")
        df = df.copy()

        if self.price_col not in df.columns:
            logger.error(
                "Price column '%s' not found – aborting pricing features.",
                self.price_col,
            )
            return df

        price = df[self.price_col]

        df["price_change"] = self.compute_price_change(price)
        df["price_pct_change"] = self.compute_price_pct_change(price)
        df["price_level"] = self.compute_price_level(price)

        if self.volume_col in df.columns:
            df["demand_elasticity"] = self.compute_demand_elasticity(
                df[self.volume_col], price
            )
        else:
            logger.info(
                "Skipping demand_elasticity – '%s' column not found.",
                self.volume_col,
            )

        logger.info(
            "Pricing feature computation complete – DataFrame now has %d "
            "columns.",
            len(df.columns),
        )
        return df
