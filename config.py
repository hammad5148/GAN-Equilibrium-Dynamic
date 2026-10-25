"""Backward-compatible access to the package configuration.

New code should import from ``gan_game_theory.config``. This shim keeps the
original notebook-era ``config.py`` entry point useful for existing scripts.
"""

from gan_game_theory.config import (  # noqa: F401
    DiagnosticConfig,
    ExperimentConfig,
    ModelConfig,
    PathConfig,
    TrainingConfig,
    default_config,
)

CONFIG = default_config()
CONFIG.ensure_dirs()
