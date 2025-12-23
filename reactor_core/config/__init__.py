"""
Configuration module for Night Shift Training Engine.

Provides dataclass-based configuration with:
- YAML file loading
- Environment variable interpolation
- Type validation
- Sensible defaults
"""

from reactor_core.config.base_config import (
    BaseConfig,
    IngestionConfig,
    TrainingConfig,
    DistillationConfig,
    OrchestrationConfig,
    QuantizationConfig,
    EvalConfig,
    NightShiftConfig,
    load_config,
    get_config,
)

__all__ = [
    "BaseConfig",
    "IngestionConfig",
    "TrainingConfig",
    "DistillationConfig",
    "OrchestrationConfig",
    "QuantizationConfig",
    "EvalConfig",
    "NightShiftConfig",
    "load_config",
    "get_config",
]
