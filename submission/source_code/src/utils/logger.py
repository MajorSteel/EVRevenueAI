# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""
Logging setup for EV Charging Tariff Optimization.

Provides a centralized ``get_logger`` factory that reads configuration
from ``config/config.yaml`` and returns a pre-configured ``logging.Logger``
with both console and rotating-file handlers.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Module-level defaults (used if config cannot be loaded)
# ---------------------------------------------------------------------------
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_FORMAT = "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s"
_DEFAULT_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_LOG_DIR = "logs"
_DEFAULT_LOG_FILE = "ev_tariff.log"
_DEFAULT_MAX_BYTES = 10_485_760  # 10 MB
_DEFAULT_BACKUP_COUNT = 5

# Cache to avoid re-reading config on every call
_config_cache: Optional[dict] = None


def _load_log_config() -> dict:
    """Load the logging section from *config/config.yaml*.

    Returns a plain dict with sane defaults if the file is missing or
    the ``logging`` key is absent.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    defaults: dict = {
        "level": _DEFAULT_LOG_LEVEL,
        "format": _DEFAULT_FORMAT,
        "date_format": _DEFAULT_DATE_FMT,
        "log_dir": _DEFAULT_LOG_DIR,
        "log_file": _DEFAULT_LOG_FILE,
        "max_bytes": _DEFAULT_MAX_BYTES,
        "backup_count": _DEFAULT_BACKUP_COUNT,
        "console_level": _DEFAULT_LOG_LEVEL,
        "file_level": "DEBUG",
    }

    try:
        import yaml  # deferred so the module works even without PyYAML

        # Resolve config path relative to the project root
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "config" / "config.yaml"

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as fh:
                full_cfg = yaml.safe_load(fh) or {}
            log_cfg = full_cfg.get("logging", {})
            defaults.update({k: v for k, v in log_cfg.items() if v is not None})
    except Exception:  # noqa: BLE001
        pass  # fall back to hard-coded defaults

    _config_cache = defaults
    return defaults


def _ensure_log_dir(log_dir: str) -> Path:
    """Create the log directory (relative to the project root) if needed."""
    project_root = Path(__file__).resolve().parents[2]
    log_path = project_root / log_dir
    log_path.mkdir(parents=True, exist_ok=True)
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Return a configured :class:`logging.Logger` for *name*.

    The logger will have:
    * A **console** handler at the level specified by ``logging.console_level``
      in *config.yaml*.
    * A **rotating file** handler at ``logging.file_level``, writing to
      ``<log_dir>/<log_file>``.

    Calling this function multiple times with the same *name* returns the
    **same** logger instance (standard ``logging`` behaviour).

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    cfg = _load_log_config()
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # handlers do the actual filtering

    formatter = logging.Formatter(
        fmt=cfg["format"],
        datefmt=cfg["date_format"],
    )

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, cfg["console_level"].upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- Rotating file handler ---
    try:
        log_dir = _ensure_log_dir(cfg["log_dir"])
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_dir / cfg["log_file"]),
            maxBytes=int(cfg["max_bytes"]),
            backupCount=int(cfg["backup_count"]),
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, cfg["file_level"].upper(), logging.DEBUG))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not set up file logging: %s", exc)

    return logger
