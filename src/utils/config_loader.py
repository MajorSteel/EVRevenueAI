# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""
YAML configuration loader with singleton pattern and dot-notation access.

Usage::

    from src.utils.config_loader import ConfigLoader

    cfg = ConfigLoader()           # loads config/config.yaml once
    price = cfg.pricing.baseline_price
    lr    = cfg.models.xgboost.learning_rate
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


class _DotDict(dict):
    """A dict subclass that supports attribute-style (dot-notation) access.

    Nested dicts are automatically wrapped so that
    ``cfg.models.xgboost.learning_rate`` works seamlessly.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(
                f"Configuration key '{key}' not found. "
                f"Available keys: {list(self.keys())}"
            ) from None
        if isinstance(value, dict) and not isinstance(value, _DotDict):
            value = _DotDict(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Configuration key '{key}' not found.") from None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"


class ConfigLoader:
    """Thread-safe singleton configuration loader.

    On first instantiation the class reads ``config/config.yaml`` from the
    project root and exposes its contents through dot-notation.  Subsequent
    calls to ``ConfigLoader()`` return the **same** instance.

    Parameters
    ----------
    config_path : str | Path | None
        Override the default config file location.  Useful in tests.
    """

    _instance: Optional["ConfigLoader"] = None
    _lock: threading.Lock = threading.Lock()
    _config: _DotDict

    def __new__(cls, config_path: Optional[str | Path] = None) -> "ConfigLoader":
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialise(config_path)
                    cls._instance = instance
        return cls._instance

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _initialise(self, config_path: Optional[str | Path] = None) -> None:
        """Load the YAML file and store it as a ``_DotDict``."""
        if config_path is None:
            project_root = Path(__file__).resolve().parents[2]
            config_path = project_root / "config" / "config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        logger.info("Loading configuration from %s", config_path)
        with open(config_path, "r", encoding="utf-8") as fh:
            raw: dict = yaml.safe_load(fh) or {}

        self._config = _DotDict(raw)
        self._config_path = config_path
        logger.info(
            "Configuration loaded – top-level keys: %s",
            list(self._config.keys()),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        # Avoid infinite recursion during unpickling / before _config exists
        if key.startswith("_"):
            raise AttributeError(key)
        return getattr(self._config, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style ``.get()`` with an optional default."""
        return self._config.get(key, default)

    def as_dict(self) -> dict:
        """Return a plain-dict copy of the entire configuration."""
        return dict(self._config)

    @classmethod
    def reload(cls, config_path: Optional[str | Path] = None) -> "ConfigLoader":
        """Force-reload configuration (drops the cached singleton)."""
        with cls._lock:
            cls._instance = None
        return cls(config_path)

    def __repr__(self) -> str:
        return f"ConfigLoader(path={getattr(self, '_config_path', '?')})"
