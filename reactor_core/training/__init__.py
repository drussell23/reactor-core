"""
Training modules for Night Shift Training Engine.

Provides:
- AsyncTrainer with full HuggingFace/TRL integration
- LoRA/QLoRA configuration and application
- Training callbacks for progress tracking
- Gradient checkpointing and memory optimization
"""

# Core trainer
from reactor_core.training.trainer import (
    Trainer,
    AsyncTrainer,
    TrainingConfig,
    TrainingProgress,
    TrainingResult,
    TrainingState,
)

# Callbacks
from reactor_core.training.callbacks import (
    TrainingEvent,
    BaseCallback,
    CallbackMixin,
    ProgressCallback,
    EarlyStoppingCallback,
    CheckpointCallback,
    GPUMemoryCallback,
    RichProgressCallback,
    WebhookCallback,
    CompositeCallback,
    create_default_callbacks,
)

# LoRA/QLoRA
from reactor_core.training.lora import (
    # Enums
    QuantizationType,
    TaskType,
    # Configs
    LoRAConfig,
    QLoRAConfig,
    # Core functions
    apply_lora,
    apply_qlora,
    prepare_model_for_qlora,
    load_model_for_qlora,
    # Merge/save/load
    merge_and_unload,
    save_adapter,
    load_adapter,
    # Adapter management
    set_adapter,
    disable_adapter,
    enable_adapter,
    get_adapter_info,
    # Utilities
    detect_model_architecture,
    get_trainable_parameters,
    format_params,
    create_optimal_lora_config,
    # Constants
    TARGET_MODULES_MAP,
)

__all__ = [
    # Trainer
    "Trainer",
    "AsyncTrainer",
    "TrainingConfig",
    "TrainingProgress",
    "TrainingResult",
    "TrainingState",
    # Callbacks
    "TrainingEvent",
    "BaseCallback",
    "CallbackMixin",
    "ProgressCallback",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "GPUMemoryCallback",
    "RichProgressCallback",
    "WebhookCallback",
    "CompositeCallback",
    "create_default_callbacks",
    # LoRA Enums
    "QuantizationType",
    "TaskType",
    # LoRA Configs
    "LoRAConfig",
    "QLoRAConfig",
    # LoRA Core
    "apply_lora",
    "apply_qlora",
    "prepare_model_for_qlora",
    "load_model_for_qlora",
    # LoRA Save/Load
    "merge_and_unload",
    "save_adapter",
    "load_adapter",
    # LoRA Adapter Management
    "set_adapter",
    "disable_adapter",
    "enable_adapter",
    "get_adapter_info",
    # LoRA Utilities
    "detect_model_architecture",
    "get_trainable_parameters",
    "format_params",
    "create_optimal_lora_config",
    "TARGET_MODULES_MAP",
]
