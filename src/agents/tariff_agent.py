# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""
Tariff pricing agent and custom Gymnasium environment for EV charging.

Classes
-------
EVChargingEnv
    A Gymnasium environment that simulates an EV charging station under
    dynamic pricing.  The agent observes utilization, demand, and
    temporal features and chooses a price multiplier.

TariffPricingAgent
    High-level wrapper that creates environments, trains a PPO policy,
    and exposes ``recommend_tariff`` for inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

# ======================================================================
# Price-multiplier action mapping
# ======================================================================
ACTION_MULTIPLIERS: Dict[int, float] = {
    0: -0.20,   # −20 %
    1: -0.10,   # −10 %
    2:  0.00,   #   0 %
    3:  0.10,   # +10 %
    4:  0.20,   # +20 %
    5:  0.30,   # +30 %
}

ACTION_NAMES: Dict[int, str] = {
    0: "-20%",
    1: "-10%",
    2: "0%",
    3: "+10%",
    4: "+20%",
    5: "+30%",
}


# ======================================================================
# EVChargingEnv
# ======================================================================
class EVChargingEnv(gym.Env):
    """Custom Gymnasium environment for EV charging tariff optimisation.

    The observation is an 8-dimensional vector:

    +---------+-------------------------------------------+
    | Index   | Feature                                   |
    +=========+===========================================+
    | 0       | occupancy                                 |
    | 1       | predicted_demand                          |
    | 2       | current_price (normalised)                |
    | 3       | hour_sin  (sin(2π·hour/24))               |
    | 4       | hour_cos  (cos(2π·hour/24))               |
    | 5       | utilization (occupancy / capacity)         |
    | 6       | fast_ratio                                |
    | 7       | cbd_flag                                  |
    +---------+-------------------------------------------+

    The action space is ``Discrete(6)`` mapping to price multipliers
    from −20 % to +30 %.

    Parameters
    ----------
    data_df : pd.DataFrame
        Must contain columns: ``occupancy``, ``capacity``,
        ``predicted_demand``, ``current_price``, ``hour``,
        ``fast_ratio``, ``cbd_flag``, ``volume``.
    config : dict
        Configuration dictionary.  Relevant keys:

        * ``baseline_price`` (float, default 15.0)
        * ``elasticity`` (float, default 0.3)
        * ``alpha`` (float, default 1.0) – revenue weight
        * ``beta`` (float, default 0.5)  – utilization-balance weight
        * ``gamma_penalty`` (float, default 2.0) – congestion weight
        * ``min_price`` (float, default 5.0)
        * ``max_price`` (float, default 30.0)
    """

    metadata: dict = {"render_modes": ["human"]}

    # Required columns in *data_df*
    _REQUIRED_COLS: List[str] = [
        "occupancy",
        "capacity",
        "predicted_demand",
        "current_price",
        "hour",
        "fast_ratio",
        "cbd_flag",
        "volume",
    ]

    def __init__(
        self,
        data_df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        config = config or {}

        # ---- validate data ----
        missing = [c for c in self._REQUIRED_COLS if c not in data_df.columns]
        if missing:
            raise ValueError(
                f"data_df is missing required columns: {missing}"
            )

        self._data: np.ndarray = data_df[self._REQUIRED_COLS].values.astype(
            np.float32,
        )
        self._n_steps: int = len(data_df)

        # ---- hyperparameters ----
        self._baseline_price: float = float(config.get("baseline_price", 15.0))
        self._elasticity: float = float(config.get("elasticity", 0.3))
        self._alpha: float = float(config.get("alpha", 1.0))
        self._beta: float = float(config.get("beta", 0.5))
        self._gamma_penalty: float = float(config.get("gamma_penalty", 2.0))
        self._min_price: float = float(config.get("min_price", 5.0))
        self._max_price: float = float(config.get("max_price", 30.0))

        # Column index mapping within the stored numpy array
        self._col_idx = {c: i for i, c in enumerate(self._REQUIRED_COLS)}

        # ---- spaces ----
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(ACTION_MULTIPLIERS))

        # ---- episode bookkeeping ----
        self._current_step: int = 0
        self._episode_rewards: List[float] = []
        self._episode_revenues: List[float] = []
        self._episode_utilizations: List[float] = []
        self._episode_count: int = 0

        logger.info(
            "EVChargingEnv created – %d timesteps, baseline_price=%.1f, "
            "elasticity=%.2f",
            self._n_steps,
            self._baseline_price,
            self._elasticity,
        )

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def _get_row(self, idx: int) -> np.ndarray:
        """Return the raw row at *idx*."""
        return self._data[idx]

    def _build_observation(self, row: np.ndarray) -> np.ndarray:
        """Transform a raw data row into the 8-D observation vector."""
        occupancy = row[self._col_idx["occupancy"]]
        capacity = row[self._col_idx["capacity"]]
        predicted_demand = row[self._col_idx["predicted_demand"]]
        current_price = row[self._col_idx["current_price"]]
        hour = row[self._col_idx["hour"]]
        fast_ratio = row[self._col_idx["fast_ratio"]]
        cbd_flag = row[self._col_idx["cbd_flag"]]

        utilization = (
            occupancy / capacity if capacity > 0 else 0.0
        )
        utilization = float(np.clip(utilization, 0.0, 1.0))

        hour_rad = 2.0 * np.pi * hour / 24.0
        hour_sin = float(np.sin(hour_rad))
        hour_cos = float(np.cos(hour_rad))

        # Normalise price to [0, 1] using min/max bounds
        price_norm = (current_price - self._min_price) / (
            self._max_price - self._min_price + 1e-8
        )

        obs = np.array(
            [
                occupancy,
                predicted_demand,
                price_norm,
                hour_sin,
                hour_cos,
                utilization,
                fast_ratio,
                cbd_flag,
            ],
            dtype=np.float32,
        )
        return obs

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to the beginning of the dataset.

        Returns
        -------
        tuple[np.ndarray, dict]
            Initial observation and info dict.
        """
        super().reset(seed=seed)
        self._current_step = 0
        self._episode_rewards = []
        self._episode_revenues = []
        self._episode_utilizations = []
        self._episode_count += 1

        row = self._get_row(self._current_step)
        obs = self._build_observation(row)

        info: Dict[str, Any] = {
            "step": self._current_step,
            "episode": self._episode_count,
        }
        logger.debug("Environment reset – episode %d", self._episode_count)
        return obs, info

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one timestep.

        Parameters
        ----------
        action : int
            Index into :data:`ACTION_MULTIPLIERS`.

        Returns
        -------
        tuple
            ``(observation, reward, terminated, truncated, info)``
        """
        if action not in ACTION_MULTIPLIERS:
            raise ValueError(
                f"Invalid action {action}; expected one of "
                f"{list(ACTION_MULTIPLIERS.keys())}."
            )

        row = self._get_row(self._current_step)
        current_price = float(row[self._col_idx["current_price"]])
        volume = float(row[self._col_idx["volume"]])
        occupancy = float(row[self._col_idx["occupancy"]])
        capacity = float(row[self._col_idx["capacity"]])

        # ---- price adjustment ----
        multiplier = ACTION_MULTIPLIERS[action]
        new_price = current_price * (1.0 + multiplier)
        new_price = float(np.clip(new_price, self._min_price, self._max_price))

        price_change_pct = (
            (new_price - current_price) / (current_price + 1e-8)
        )

        # ---- demand response ----
        volume_new = volume * (1.0 - self._elasticity * price_change_pct)
        volume_new = max(volume_new, 0.0)

        # ---- revenue ----
        new_revenue = new_price * volume_new
        baseline_revenue = self._baseline_price * volume

        revenue_gain = (new_revenue - baseline_revenue) / (
            baseline_revenue + 1e-8
        )

        # ---- utilization ----
        utilization = (
            occupancy / capacity if capacity > 0 else 0.0
        )
        utilization = float(np.clip(utilization, 0.0, 1.0))
        utilization_balance = -abs(utilization - 0.7)

        # ---- congestion penalty ----
        congestion_penalty = max(0.0, utilization - 0.9) ** 2

        # ---- composite reward ----
        reward = (
            self._alpha * revenue_gain
            + self._beta * utilization_balance
            - self._gamma_penalty * congestion_penalty
        )

        # ---- bookkeeping ----
        self._episode_rewards.append(reward)
        self._episode_revenues.append(new_revenue)
        self._episode_utilizations.append(utilization)

        # ---- advance ----
        self._current_step += 1
        terminated = self._current_step >= self._n_steps
        truncated = False

        if terminated:
            next_obs = np.zeros(8, dtype=np.float32)
        else:
            next_row = self._get_row(self._current_step)
            next_obs = self._build_observation(next_row)

        info: Dict[str, Any] = {
            "step": self._current_step,
            "action_name": ACTION_NAMES[action],
            "current_price": current_price,
            "new_price": new_price,
            "volume_original": volume,
            "volume_adjusted": volume_new,
            "revenue_baseline": baseline_revenue,
            "revenue_new": new_revenue,
            "revenue_gain": revenue_gain,
            "utilization": utilization,
            "congestion_penalty": congestion_penalty,
        }

        if terminated:
            ep_stats = {
                "episode_reward": sum(self._episode_rewards),
                "episode_mean_reward": float(np.mean(self._episode_rewards)),
                "episode_total_revenue": sum(self._episode_revenues),
                "episode_mean_utilization": float(
                    np.mean(self._episode_utilizations),
                ),
                "episode_length": len(self._episode_rewards),
            }
            info["episode_stats"] = ep_stats
            logger.info(
                "Episode %d finished – total_reward=%.3f, "
                "total_revenue=%.2f, mean_util=%.3f",
                self._episode_count,
                ep_stats["episode_reward"],
                ep_stats["episode_total_revenue"],
                ep_stats["episode_mean_utilization"],
            )

        return next_obs, float(reward), terminated, truncated, info


# ======================================================================
# TariffPricingAgent
# ======================================================================
class TariffPricingAgent:
    """High-level tariff pricing agent backed by PPO.

    Parameters
    ----------
    config : dict | None
        Configuration dict.  Relevant keys are forwarded to
        :class:`EVChargingEnv` and :class:`PPOTariffModel`.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self._model: Optional[Any] = None  # PPOTariffModel
        self._env: Optional[EVChargingEnv] = None
        logger.info("TariffPricingAgent initialised.")

    # ------------------------------------------------------------------
    # Environment factory
    # ------------------------------------------------------------------
    def create_env(self, data_df: pd.DataFrame) -> EVChargingEnv:
        """Create and return a new :class:`EVChargingEnv`.

        Parameters
        ----------
        data_df : pd.DataFrame
            Must satisfy :attr:`EVChargingEnv._REQUIRED_COLS`.

        Returns
        -------
        EVChargingEnv
        """
        env = EVChargingEnv(data_df=data_df, config=self.config)
        self._env = env
        logger.info("Created EVChargingEnv with %d timesteps.", len(data_df))
        return env

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        data_df: pd.DataFrame,
        total_timesteps: int = 100_000,
        callback: Optional[Any] = None,
    ) -> None:
        """Train the PPO agent on the supplied data.

        Parameters
        ----------
        data_df : pd.DataFrame
            Training data.
        total_timesteps : int
            Number of environment steps to train.
        callback : BaseCallback | None
            ``stable_baselines3`` callback.
        """
        from src.models.ppo_model import PPOTariffModel  # noqa: WPS433

        env = self.create_env(data_df)

        ppo_config = self.config.get("ppo", {})
        self._model = PPOTariffModel(
            env=env,
            learning_rate=ppo_config.get("learning_rate", 3e-4),
            n_steps=ppo_config.get("n_steps", 2048),
            batch_size=ppo_config.get("batch_size", 64),
            n_epochs=ppo_config.get("n_epochs", 10),
            gamma=ppo_config.get("gamma", 0.99),
            clip_range=ppo_config.get("clip_range", 0.2),
            gae_lambda=ppo_config.get("gae_lambda", 0.95),
            ent_coef=ppo_config.get("ent_coef", 0.01),
            vf_coef=ppo_config.get("vf_coef", 0.5),
            max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
            policy=ppo_config.get("policy", "MlpPolicy"),
        )

        self._model.train(
            total_timesteps=total_timesteps,
            callback=callback,
        )
        logger.info("TariffPricingAgent training complete.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def recommend_tariff(
        self,
        observation: np.ndarray,
    ) -> float:
        """Recommend an optimal tariff price.

        Parameters
        ----------
        observation : np.ndarray
            8-D observation vector.

        Returns
        -------
        float
            Recommended tariff price (₹/kWh).

        Raises
        ------
        RuntimeError
            If no model has been trained or loaded.
        """
        if self._model is None:
            raise RuntimeError(
                "No model available.  Call train() or load() first."
            )

        action, info = self._model.predict(observation, deterministic=True)
        multiplier = ACTION_MULTIPLIERS[action]

        # Recover current_price from normalised value in observation
        min_price = self.config.get("min_price", 5.0)
        max_price = self.config.get("max_price", 30.0)
        price_norm = observation[2] if observation.ndim == 1 else observation[0, 2]
        current_price = float(
            price_norm * (max_price - min_price) + min_price,
        )
        optimal_tariff = current_price * (1.0 + multiplier)
        optimal_tariff = float(np.clip(optimal_tariff, min_price, max_price))

        logger.debug(
            "recommend_tariff – action=%d (%s), current=%.2f, "
            "recommended=%.2f",
            action,
            ACTION_NAMES[action],
            current_price,
            optimal_tariff,
        )
        return optimal_tariff

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def get_action_name(action_idx: int) -> str:
        """Return a human-readable label for an action index.

        Parameters
        ----------
        action_idx : int
            Action index (0–5).

        Returns
        -------
        str
            Label such as ``"+10%"``.
        """
        return ACTION_NAMES.get(action_idx, f"unknown({action_idx})")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        test_data: pd.DataFrame,
        n_episodes: int = 1,
    ) -> Dict[str, Any]:
        """Evaluate the trained agent on *test_data*.

        Parameters
        ----------
        test_data : pd.DataFrame
            Held-out test dataset.
        n_episodes : int
            Number of evaluation episodes.

        Returns
        -------
        dict
            Keys: ``mean_reward``, ``total_revenue``,
            ``mean_utilization``, ``revenue_gain_pct``,
            ``action_distribution``.
        """
        if self._model is None:
            raise RuntimeError(
                "No model available.  Call train() or load() first."
            )

        env = self.create_env(test_data)
        all_rewards: List[float] = []
        all_revenues: List[float] = []
        all_utilizations: List[float] = []
        all_actions: List[int] = []
        baseline_revenue_total = 0.0

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = self._model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                all_rewards.append(reward)
                all_revenues.append(info.get("revenue_new", 0.0))
                all_utilizations.append(info.get("utilization", 0.0))
                all_actions.append(action)
                baseline_revenue_total += info.get("revenue_baseline", 0.0)

        total_revenue = sum(all_revenues)
        revenue_gain_pct = (
            ((total_revenue - baseline_revenue_total) / (baseline_revenue_total + 1e-8))
            * 100.0
        )

        # Action distribution
        action_counts: Dict[str, int] = {}
        for a in all_actions:
            name = ACTION_NAMES.get(a, str(a))
            action_counts[name] = action_counts.get(name, 0) + 1

        metrics: Dict[str, Any] = {
            "mean_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "total_revenue": total_revenue,
            "baseline_revenue": baseline_revenue_total,
            "revenue_gain_pct": revenue_gain_pct,
            "mean_utilization": float(np.mean(all_utilizations)),
            "action_distribution": action_counts,
            "n_steps_evaluated": len(all_rewards),
        }
        logger.info(
            "Evaluation – mean_reward=%.4f, revenue_gain=%.2f%%, "
            "mean_util=%.3f",
            metrics["mean_reward"],
            metrics["revenue_gain_pct"],
            metrics["mean_utilization"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Save the underlying PPO model.

        Parameters
        ----------
        path : str | Path
            File path.
        """
        if self._model is None:
            raise RuntimeError("No model to save.")
        self._model.save(path)
        logger.info("TariffPricingAgent saved to %s", path)

    def load(self, path: Union[str, Path], env: Optional[Any] = None) -> None:
        """Load a previously saved PPO model.

        Parameters
        ----------
        path : str | Path
            Path to saved model.
        env : gymnasium.Env | None
            Optional environment to attach.
        """
        from src.models.ppo_model import PPOTariffModel  # noqa: WPS433

        self._model = PPOTariffModel.load(path, env=env)
        logger.info("TariffPricingAgent loaded from %s", path)

    def __repr__(self) -> str:
        status = "trained" if self._model is not None else "untrained"
        return f"TariffPricingAgent(status={status}, config_keys={list(self.config.keys())})"
