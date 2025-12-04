# -------------------------------------------------------------------------
# config_loader.py
#
# Production‑ready configuration loader for Citadel Quantum Trader (CQT).
#
#   * Loads the default config (config.yaml) or the optimised variant
#     (config_opt.yaml) depending on the presence of a flag file.
#   * Provides a thread‑safe singleton instance.
#   * Supports hot‑reloading at runtime.
#   * Typed property accessors make the rest of the codebase clean.
#
# Usage:
#   from config_loader import Config
#
#   cfg = Config.instance()          # get the singleton
#   if cfg.use_shadow_mode:
#       ...                         # do something
#
#   # To reload (e.g. from a watcher):
#   Config.instance().reload()
# -------------------------------------------------------------------------

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Mapping

import yaml


class _Config:
    """
    Internal implementation of the configuration object.

    The public API is exposed via the ``Config`` wrapper class below.
    """

    # -----------------------------------------------------------------
    # Paths – change only if you relocate the config directory
    # -----------------------------------------------------------------
    _DEFAULT_PATH = Path("/opt/citadel/config/config.yaml")
    _OPTIMISED_PATH = Path("/opt/citadel/config/config_opt.yaml")
    _FLAG_PATH = Path("/opt/citadel/config/use_optimised_cfg.flag")

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------
    def __init__(self) -> None:
        """
        Load the configuration at instantiation time.
        """
        self._lock = threading.RLock()
        self._settings: Mapping[str, Any] = {}
        self._load_from_disk()

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------
    def _choose_path(self) -> Path:
        """
        Decide which YAML file to read:

        * If the flag file exists → use the optimised config.
        * Otherwise → use the default config.
        """
        if self._FLAG_PATH.is_file():
            return self._OPTIMISED_PATH
        return self._DEFAULT_PATH

    def _load_from_disk(self) -> None:
        """
        Actually read the YAML file and store the resulting dict in
        ``self._settings``.  This method is always called under the
        instance lock.
        """
        config_path = self._choose_path()

        if not config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with config_path.open("r", encoding="utf-8") as fp:
            try:
                loaded = yaml.safe_load(fp) or {}
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"Failed to parse YAML config at {config_path}: {exc}"
                ) from exc

        if not isinstance(loaded, dict):
            raise TypeError(
                f"Config file {config_path} must contain a mapping at top level."
            )

        # Store an immutable copy (shallow – values themselves may be mutable)
        self._settings = dict(loaded)

    # -----------------------------------------------------------------
    # Public API – read‑only accessors
    # -----------------------------------------------------------------
    @property
    def raw(self) -> Mapping[str, Any]:
        """
        Return the entire configuration mapping (read‑only).  Use with
        caution – modifying the returned dict will not affect the loader.
        """
        # No lock needed – the dict itself is never mutated after load.
        return self._settings

    # ----- Example typed accessors (add more as needed) -----------------
    @property
    def use_shadow_mode(self) -> bool:
        """Whether the engine should run in shadow (paper‑only) mode."""
        return bool(self._settings.get("use_shadow_mode", False))

    @property
    def scorer_mode(self) -> str:
        """Which scoring backend to use – e.g. ``lightgbm`` or ``linear``."""
        return str(self._settings.get("scorer_mode", "lightgbm")).lower()

    @property
    def guards(self) -> Mapping[str, Any]:
        """Dictionary containing all guard configurations."""
        return self._settings.get("guards", {})

    @property
    def venues(self) -> list[Mapping[str, Any]]:
        """Ordered list of broker venues."""
        return list(self._settings.get("venues", []))

    @property
    def optimiser_weights(self) -> Mapping[str, float]:
        """Weights used by the confluence optimiser."""
        return self._settings.get("optimiser", {}).get("weights", {})

    # -----------------------------------------------------------------
    # Hot‑reload support
    # -----------------------------------------------------------------
    def reload(self) -> None:
        """
        Re‑read the configuration from disk.  This is safe to call from
        any thread – the internal lock guarantees atomic replacement.
        """
        with self._lock:
            self._load_from_disk()

    # -----------------------------------------------------------------
    # Helper for internal use – fetch a nested key safely
    # -----------------------------------------------------------------
    def _get(self, *keys: str, default: Any = None) -> Any:
        """
        Walk the nested mapping using ``keys`` and return the value or
        ``default`` if any intermediate key is missing.
        """
        cur: Any = self._settings
        for k in keys:
            if not isinstance(cur, Mapping):
                return default
            cur = cur.get(k, default)
        return cur


# -------------------------------------------------------------------------
# Public wrapper – singleton access point
# -------------------------------------------------------------------------
class Config:
    """
    Public façade that provides a **singleton** instance of the internal
    ``_Config`` class.  All code should import ``Config`` and call
    ``Config.instance()`` to obtain the shared configuration object.

    Example:
        >>> from config_loader import Config
        >>> cfg = Config.instance()
        >>> if cfg.use_shadow_mode:
        ...     print("Running in shadow mode")
    """

    _instance: _Config | None = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> _Config:
        """
        Return the global ``_Config`` instance, creating it lazily on the
        first call.  Thread‑safe.
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:   # double‑checked locking
                    cls._instance = _Config()
        return cls._instance

    # -----------------------------------------------------------------
    # Convenience shortcuts – delegate attribute access to the singleton
    # -----------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """
        Forward any unknown attribute access to the underlying singleton.
        This lets you write ``Config().use_shadow_mode`` instead of
        ``Config.instance().use_shadow_mode`` if you prefer the shorter
        syntax.
        """
        return getattr(self.instance(), name)


# -------------------------------------------------------------------------
# Example of a simple file‑watcher that could be run from a systemd timer
# -------------------------------------------------------------------------
if __name__ == "__main__":
    """
    When executed directly (e.g. via a systemd timer) this script will
    reload the configuration if the flag file or any of the YAML files
    have changed.  It is deliberately minimal – you can expand it to
    emit a log line, send a notification, etc.
    """
    cfg = Config.instance()
    cfg.reload()
    print("Configuration reloaded – use_shadow_mode =", cfg.use_shadow_mode) 
