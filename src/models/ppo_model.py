"""
PPO Reinforcement Learning model for EV charging tariff optimization.

Wraps :class:`stable_baselines3.PPO` with project-specific defaults,
convenience methods for tariff prediction, and action-distribution
introspection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class PPOTariffModel:
    """PPO-based tariff pricing model.

    Thin wrapper around ``stable_baselines3.PPO`` that adds
    project-specific defaults, serialisation helpers and an
    ``get_action_distribution`` method useful for explainability.

    Parameters
    ----------
    env : gymnasium.Env
        The Gymnasium environment to train on (typically
        :class:`src.agents.tariff_agent.EVChargingEnv`).
    learning_rate : float
        Adam optimiser learning rate.
    n_steps : int
        Number of environment steps per rollout buffer update.
    batch_size : int
        Mini-batch size for PPO updates.
    n_epochs : int
        Number of epochs to run through the rollout buffer each update.
    gamma : float
        Discount factor.
    clip_range : float
        PPO clipping parameter.
    gae_lambda : float
        Generalised Advantage Estimation lambda.
    ent_coef : float
        Entropy coefficient for the loss calculation.
    vf_coef : float
        Value-function coefficient for the loss calculation.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    policy : str
        Policy architecture identifier (``stable-baselines3`` key).
    verbose : int
        Verbosity level passed to ``stable_baselines3.PPO``.
    device : str
        PyTorch device string (``"auto"``, ``"cpu"``, ``"cuda"``).
    """

    def __init__(
        self,
        env: Any,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        clip_range: float = 0.2,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy: str = "MlpPolicy",
        verbose: int = 0,
        device: str = "auto",
    ) -> None:
        try:
            from stable_baselines3 import PPO  # noqa: WPS433
        except ImportError as exc:
            logger.error("stable-baselines3 is required: %s", exc)
            raise

        self.env = env
        self._hyperparams: Dict[str, Any] = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "clip_range": clip_range,
            "gae_lambda": gae_lambda,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
        }

        self.model = PPO(
            policy=policy,
            env=env,
            verbose=verbose,
            device=device,
            **self._hyperparams,
        )
        logger.info(
            "PPOTariffModel initialised – policy=%s, lr=%.1e, gamma=%.3f, "
            "clip=%.2f, device=%s",
            policy,
            learning_rate,
            gamma,
            clip_range,
            device,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        total_timesteps: int = 100_000,
        callback: Optional[Any] = None,
        log_interval: int = 10,
        progress_bar: bool = False,
    ) -> "PPOTariffModel":
        """Train the PPO agent.

        Parameters
        ----------
        total_timesteps : int
            Total number of environment steps to train for.
        callback : BaseCallback | list[BaseCallback] | None
            ``stable_baselines3`` callback(s).
        log_interval : int
            Log training stats every *log_interval* updates.
        progress_bar : bool
            Show a ``tqdm`` progress bar.

        Returns
        -------
        PPOTariffModel
            ``self``, for fluent chaining.
        """
        logger.info("Starting PPO training for %d timesteps …", total_timesteps)
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                log_interval=log_interval,
                progress_bar=progress_bar,
            )
            logger.info("PPO training complete.")
        except Exception:
            logger.exception("PPO training failed.")
            raise
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        """Select an action for a single observation.

        Parameters
        ----------
        observation : np.ndarray
            Observation vector (must match ``env.observation_space``).
        deterministic : bool
            If ``True`` pick the mode of the policy; otherwise sample.

        Returns
        -------
        tuple[int, dict]
            ``(action, info)`` where *info* contains the value estimate
            and log-probability.
        """
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        action, _states = self.model.predict(obs, deterministic=deterministic)
        action_int: int = int(action.item()) if hasattr(action, "item") else int(action)

        # Extract value estimate and log-prob for diagnostics
        info: Dict[str, Any] = {"action": action_int}
        try:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(
                self.model.device,
            )
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_tensor)
                action_tensor = torch.tensor(
                    [action_int], device=self.model.device,
                )
                info["log_prob"] = float(
                    dist.log_prob(action_tensor).cpu().item(),
                )
                value = self.model.policy.predict_values(obs_tensor)
                info["value_estimate"] = float(value.cpu().item())
        except Exception as exc:
            logger.debug("Could not extract diagnostics: %s", exc)

        return action_int, info

    # ------------------------------------------------------------------
    # Action distribution
    # ------------------------------------------------------------------
    def get_action_distribution(
        self,
        observation: np.ndarray,
    ) -> np.ndarray:
        """Return the probability vector over discrete actions.

        Parameters
        ----------
        observation : np.ndarray
            A single observation vector.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(n_actions,)`` summing to 1.
        """
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(
            self.model.device,
        )
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
            probs: np.ndarray = dist.distribution.probs.cpu().numpy().flatten()

        logger.debug(
            "Action distribution: %s",
            {i: f"{p:.3f}" for i, p in enumerate(probs)},
        )
        return probs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Save the trained model to *path*.

        Parameters
        ----------
        path : str | Path
            File path (``stable_baselines3`` appends ``.zip`` automatically
            if not present).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info("PPO model saved to %s", path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        env: Optional[Any] = None,
        device: str = "auto",
    ) -> "PPOTariffModel":
        """Load a previously saved PPO model.

        Parameters
        ----------
        path : str | Path
            Path to the saved model file.
        env : gymnasium.Env | None
            Optional environment to attach.
        device : str
            PyTorch device string.

        Returns
        -------
        PPOTariffModel
            A new instance wrapping the loaded model.
        """
        from stable_baselines3 import PPO  # noqa: WPS433

        path = Path(path)
        logger.info("Loading PPO model from %s", path)

        instance = object.__new__(cls)
        instance.env = env
        instance.model = PPO.load(str(path), env=env, device=device)
        instance._hyperparams = {
            "learning_rate": instance.model.learning_rate,
            "n_steps": instance.model.n_steps,
            "batch_size": instance.model.batch_size,
            "n_epochs": instance.model.n_epochs,
            "gamma": instance.model.gamma,
            "clip_range": instance.model.clip_range,
        }
        logger.info("PPO model loaded successfully.")
        return instance

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"PPOTariffModel(lr={self._hyperparams.get('learning_rate')}, "
            f"gamma={self._hyperparams.get('gamma')}, "
            f"clip={self._hyperparams.get('clip_range')})"
        )
