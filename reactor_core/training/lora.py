"""
LoRA (Low-Rank Adaptation) and QLoRA utilities for Night Shift Training Engine.

Provides:
- LoRA configuration with comprehensive hyperparameters
- QLoRA with bitsandbytes 4-bit quantization
- Automatic target module detection for various architectures
- Adapter merging and saving utilities
- Environment-aware quantization defaults
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Quantization types for QLoRA."""
    NONE = "none"
    INT8 = "int8"
    FP4 = "fp4"
    NF4 = "nf4"  # Normalized float 4-bit (recommended)


class TaskType(Enum):
    """Task types for LoRA configuration."""
    CAUSAL_LM = "CAUSAL_LM"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    SEQ_CLS = "SEQ_CLS"
    TOKEN_CLS = "TOKEN_CLS"
    QUESTION_ANS = "QUESTION_ANS"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"


# Architecture-specific target modules
TARGET_MODULES_MAP: Dict[str, List[str]] = {
    # Llama family
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama-attention-only": ["q_proj", "k_proj", "v_proj", "o_proj"],

    # Mistral family
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    # Falcon family
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],

    # GPT-NeoX / Pythia
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],

    # GPT-2 / GPT-J
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "gptj": ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out"],

    # Phi family
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],

    # Qwen family
    "qwen": ["c_attn", "c_proj", "w1", "w2"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    # Default (attention only)
    "default": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


@dataclass
class LoRAConfig:
    """
    Comprehensive LoRA configuration.

    Supports environment variable overrides for dynamic configuration.
    """
    # Core LoRA hyperparameters
    rank: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_LORA_RANK", "64"))
    )
    alpha: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_LORA_ALPHA", "128"))
    )
    dropout: float = field(
        default_factory=lambda: float(os.getenv("NIGHTSHIFT_LORA_DROPOUT", "0.05"))
    )

    # Target modules (None = auto-detect based on architecture)
    target_modules: Optional[List[str]] = None

    # Module configuration
    fan_in_fan_out: bool = False  # For Conv1D layers (GPT-2 style)
    bias: str = "none"  # "none", "all", "lora_only"
    modules_to_save: Optional[List[str]] = None  # Modules to fully fine-tune

    # Task configuration
    task_type: TaskType = TaskType.CAUSAL_LM
    inference_mode: bool = False

    # Advanced options
    init_lora_weights: Union[bool, str] = True  # True, False, or "gaussian", "pissa"
    use_rslora: bool = False  # Rank-stabilized LoRA
    use_dora: bool = False  # Weight-decomposed LoRA

    # Layers configuration
    layers_to_transform: Optional[List[int]] = None
    layers_pattern: Optional[str] = None

    @property
    def scaling(self) -> float:
        """LoRA scaling factor (alpha / rank)."""
        return self.alpha / self.rank

    def get_target_modules(self, model_type: str = "default") -> List[str]:
        """Get target modules for a specific model architecture."""
        if self.target_modules is not None:
            return self.target_modules

        # Normalize model type
        model_type_lower = model_type.lower()

        # Check for known architectures
        for arch_name, modules in TARGET_MODULES_MAP.items():
            if arch_name in model_type_lower:
                return modules

        return TARGET_MODULES_MAP["default"]

    def to_peft_config(self, model_type: str = "default") -> "LoraConfig":
        """Convert to PEFT LoraConfig object."""
        try:
            from peft import LoraConfig as PeftLoraConfig
            from peft import TaskType as PeftTaskType
        except ImportError:
            raise ImportError(
                "PEFT library required. Install with: pip install peft"
            )

        # Map task type
        task_type_map = {
            TaskType.CAUSAL_LM: PeftTaskType.CAUSAL_LM,
            TaskType.SEQ_2_SEQ_LM: PeftTaskType.SEQ_2_SEQ_LM,
            TaskType.SEQ_CLS: PeftTaskType.SEQ_CLS,
            TaskType.TOKEN_CLS: PeftTaskType.TOKEN_CLS,
            TaskType.QUESTION_ANS: PeftTaskType.QUESTION_ANS,
            TaskType.FEATURE_EXTRACTION: PeftTaskType.FEATURE_EXTRACTION,
        }

        return PeftLoraConfig(
            task_type=task_type_map[self.task_type],
            inference_mode=self.inference_mode,
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.get_target_modules(model_type),
            fan_in_fan_out=self.fan_in_fan_out,
            bias=self.bias,
            modules_to_save=self.modules_to_save,
            init_lora_weights=self.init_lora_weights,
            use_rslora=self.use_rslora,
            use_dora=self.use_dora,
            layers_to_transform=self.layers_to_transform,
            layers_pattern=self.layers_pattern,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "fan_in_fan_out": self.fan_in_fan_out,
            "bias": self.bias,
            "modules_to_save": self.modules_to_save,
            "task_type": self.task_type.value,
            "inference_mode": self.inference_mode,
            "init_lora_weights": self.init_lora_weights,
            "use_rslora": self.use_rslora,
            "use_dora": self.use_dora,
            "layers_to_transform": self.layers_to_transform,
            "layers_pattern": self.layers_pattern,
            "scaling": self.scaling,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoRAConfig":
        """Create from dictionary."""
        # Handle task_type conversion
        if "task_type" in data and isinstance(data["task_type"], str):
            data["task_type"] = TaskType(data["task_type"])

        # Remove computed fields
        data.pop("scaling", None)

        return cls(**data)


@dataclass
class QLoRAConfig(LoRAConfig):
    """
    QLoRA configuration with 4-bit quantization.

    Extends LoRAConfig with bitsandbytes quantization settings.
    """
    # Quantization settings
    use_qlora: bool = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_USE_QLORA", "true").lower() == "true"
    )
    quantization_type: QuantizationType = QuantizationType.NF4

    # BitsAndBytes 4-bit config
    bnb_4bit_compute_dtype: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_BNB_COMPUTE_DTYPE", "bfloat16")
    )
    bnb_4bit_use_double_quant: bool = True  # Nested quantization
    bnb_4bit_quant_storage: Optional[str] = None  # Storage dtype

    # Memory optimization
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # Gradient checkpointing
    use_gradient_checkpointing: bool = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_GRADIENT_CHECKPOINTING", "true").lower() == "true"
    )
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None

    @property
    def compute_dtype(self) -> torch.dtype:
        """Get compute dtype as torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.bnb_4bit_compute_dtype, torch.bfloat16)

    def get_bnb_config(self) -> "BitsAndBytesConfig":
        """Create BitsAndBytesConfig for model loading."""
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )

        if not self.use_qlora:
            return None

        quant_type_map = {
            QuantizationType.FP4: "fp4",
            QuantizationType.NF4: "nf4",
        }

        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            bnb_4bit_compute_dtype=self.compute_dtype,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=quant_type_map.get(
                self.quantization_type, "nf4"
            ),
            bnb_4bit_quant_storage=self.bnb_4bit_quant_storage,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            "use_qlora": self.use_qlora,
            "quantization_type": self.quantization_type.value,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "bnb_4bit_quant_storage": self.bnb_4bit_quant_storage,
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
        })
        return base


def detect_model_architecture(model: nn.Module) -> str:
    """
    Detect model architecture from model class name.

    Args:
        model: PyTorch model

    Returns:
        Architecture name string
    """
    class_name = model.__class__.__name__.lower()

    # Check known architectures
    architectures = [
        "llama", "mistral", "falcon", "gpt_neox", "pythia",
        "gpt2", "gptj", "phi", "qwen", "opt", "bloom",
    ]

    for arch in architectures:
        if arch in class_name:
            return arch

    # Try to get from config
    if hasattr(model, "config"):
        model_type = getattr(model.config, "model_type", "")
        if model_type:
            return model_type.lower()

    return "default"


def get_trainable_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Get trainable parameter counts.

    Args:
        model: PyTorch model (with or without LoRA)

    Returns:
        Tuple of (trainable_params, total_params, percentage)
    """
    trainable_params = 0
    total_params = 0

    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    percentage = 100 * trainable_params / total_params if total_params > 0 else 0

    return trainable_params, total_params, percentage


def format_params(num_params: int) -> str:
    """Format parameter count with SI prefix."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def apply_lora(
    model: nn.Module,
    config: LoRAConfig,
    model_type: Optional[str] = None,
) -> nn.Module:
    """
    Apply LoRA to a model.

    Args:
        model: Base model to adapt
        config: LoRA configuration
        model_type: Model architecture type (auto-detected if None)

    Returns:
        Model with LoRA adapters applied
    """
    try:
        from peft import get_peft_model
    except ImportError:
        raise ImportError(
            "PEFT library required. Install with: pip install peft"
        )

    # Auto-detect architecture if not provided
    if model_type is None:
        model_type = detect_model_architecture(model)

    logger.info(f"Applying LoRA to {model_type} architecture")
    logger.info(f"LoRA config: rank={config.rank}, alpha={config.alpha}, "
                f"dropout={config.dropout}, scaling={config.scaling:.2f}")

    # Get PEFT config
    peft_config = config.to_peft_config(model_type)

    # Apply LoRA
    model = get_peft_model(model, peft_config)

    # Log trainable parameters
    trainable, total, pct = get_trainable_parameters(model)
    logger.info(
        f"Trainable parameters: {format_params(trainable)} / {format_params(total)} "
        f"({pct:.2f}%)"
    )

    return model


def prepare_model_for_qlora(
    model: nn.Module,
    config: QLoRAConfig,
) -> nn.Module:
    """
    Prepare a model for QLoRA training.

    This handles:
    - Gradient checkpointing
    - Layer norm casting to float32
    - Enabling input gradients

    Args:
        model: Base model (should be loaded with quantization)
        config: QLoRA configuration

    Returns:
        Prepared model
    """
    try:
        from peft import prepare_model_for_kbit_training
    except ImportError:
        raise ImportError(
            "PEFT library required. Install with: pip install peft"
        )

    # Prepare for k-bit training
    kwargs = {}
    if config.use_gradient_checkpointing:
        kwargs["use_gradient_checkpointing"] = True
        if config.gradient_checkpointing_kwargs:
            kwargs["gradient_checkpointing_kwargs"] = config.gradient_checkpointing_kwargs

    model = prepare_model_for_kbit_training(model, **kwargs)

    logger.info("Model prepared for QLoRA training")
    if config.use_gradient_checkpointing:
        logger.info("Gradient checkpointing enabled")

    return model


def apply_qlora(
    model: nn.Module,
    config: QLoRAConfig,
    model_type: Optional[str] = None,
) -> nn.Module:
    """
    Apply QLoRA to a quantized model.

    Args:
        model: Base model (should be loaded with quantization config)
        config: QLoRA configuration
        model_type: Model architecture type (auto-detected if None)

    Returns:
        Model with QLoRA adapters applied
    """
    # Prepare model for k-bit training
    model = prepare_model_for_qlora(model, config)

    # Apply LoRA adapters
    model = apply_lora(model, config, model_type)

    return model


def load_model_for_qlora(
    model_name_or_path: str,
    config: QLoRAConfig,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
    attn_implementation: Optional[str] = None,
) -> Tuple[nn.Module, "PreTrainedTokenizer"]:
    """
    Load a model with QLoRA quantization applied.

    Args:
        model_name_or_path: HuggingFace model ID or local path
        config: QLoRA configuration
        device_map: Device mapping strategy
        torch_dtype: Model dtype (defaults to config.compute_dtype)
        trust_remote_code: Whether to trust remote code
        attn_implementation: Attention implementation ("flash_attention_2", "sdpa", etc.)

    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers library required. Install with: pip install transformers"
        )

    logger.info(f"Loading model: {model_name_or_path}")

    # Model loading kwargs
    model_kwargs = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }

    # Set dtype
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    elif config.use_qlora:
        model_kwargs["torch_dtype"] = config.compute_dtype

    # Set quantization config
    if config.use_qlora:
        bnb_config = config.get_bnb_config()
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            logger.info(
                f"Quantization: {config.quantization_type.value}, "
                f"compute_dtype={config.bnb_4bit_compute_dtype}, "
                f"double_quant={config.bnb_4bit_use_double_quant}"
            )

    # Set attention implementation
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Apply QLoRA
    if config.use_qlora:
        model = apply_qlora(model, config)
    else:
        model = apply_lora(model, config)

    return model, tokenizer


def merge_and_unload(
    model: nn.Module,
    progressbar: bool = False,
    safe_merge: bool = True,
    adapter_names: Optional[List[str]] = None,
) -> nn.Module:
    """
    Merge LoRA weights into the base model and unload adapter.

    This creates a standalone model without LoRA overhead.

    Args:
        model: Model with LoRA adapters
        progressbar: Show merge progress
        safe_merge: Check for NaN weights during merge
        adapter_names: Specific adapters to merge (None = all)

    Returns:
        Merged model without LoRA adapters
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "PEFT library required. Install with: pip install peft"
        )

    if not isinstance(model, PeftModel):
        logger.warning("Model is not a PeftModel, returning as-is")
        return model

    logger.info("Merging LoRA weights into base model...")

    # Merge and unload
    merged_model = model.merge_and_unload(
        progressbar=progressbar,
        safe_merge=safe_merge,
        adapter_names=adapter_names,
    )

    # Log parameter info
    trainable, total, pct = get_trainable_parameters(merged_model)
    logger.info(
        f"Merged model parameters: {format_params(total)} "
        f"(all trainable after merge)"
    )

    return merged_model


def save_adapter(
    model: nn.Module,
    output_dir: Union[str, Path],
    adapter_name: str = "default",
    save_embedding_layers: Union[bool, str] = "auto",
    is_main_process: bool = True,
    safe_serialization: bool = True,
) -> Path:
    """
    Save LoRA adapter weights.

    Args:
        model: Model with LoRA adapters
        output_dir: Output directory
        adapter_name: Name of adapter to save
        save_embedding_layers: Whether to save embedding layers
        is_main_process: Whether this is the main process (for distributed)
        safe_serialization: Use safetensors format

    Returns:
        Path to saved adapter directory
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "PEFT library required. Install with: pip install peft"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(model, PeftModel):
        raise ValueError("Model is not a PeftModel, cannot save adapter")

    logger.info(f"Saving adapter '{adapter_name}' to {output_dir}")

    model.save_pretrained(
        output_dir,
        save_embedding_layers=save_embedding_layers,
        is_main_process=is_main_process,
        safe_serialization=safe_serialization,
        selected_adapters=[adapter_name] if adapter_name != "default" else None,
    )

    return output_dir


def load_adapter(
    model: nn.Module,
    adapter_path: Union[str, Path],
    adapter_name: str = "default",
    is_trainable: bool = True,
) -> nn.Module:
    """
    Load a LoRA adapter onto a base model.

    Args:
        model: Base model
        adapter_path: Path to adapter directory
        adapter_name: Name to assign to loaded adapter
        is_trainable: Whether adapter should be trainable

    Returns:
        Model with adapter loaded
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "PEFT library required. Install with: pip install peft"
        )

    adapter_path = Path(adapter_path)

    logger.info(f"Loading adapter from {adapter_path}")

    # Load adapter
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        adapter_name=adapter_name,
        is_trainable=is_trainable,
    )

    # Log parameter info
    trainable, total, pct = get_trainable_parameters(model)
    logger.info(
        f"Loaded adapter '{adapter_name}': {format_params(trainable)} trainable / "
        f"{format_params(total)} total ({pct:.2f}%)"
    )

    return model


def set_adapter(
    model: nn.Module,
    adapter_name: str,
) -> None:
    """
    Switch to a specific adapter.

    Args:
        model: Model with LoRA adapters
        adapter_name: Name of adapter to activate
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "PEFT library required. Install with: pip install peft"
        )

    if not isinstance(model, PeftModel):
        raise ValueError("Model is not a PeftModel")

    model.set_adapter(adapter_name)
    logger.info(f"Switched to adapter: {adapter_name}")


def disable_adapter(model: nn.Module) -> None:
    """
    Disable all adapters (use base model only).

    Args:
        model: Model with LoRA adapters
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "PEFT library required. Install with: pip install peft"
        )

    if not isinstance(model, PeftModel):
        return

    model.disable_adapter()
    logger.info("Disabled all adapters")


def enable_adapter(model: nn.Module) -> None:
    """
    Re-enable adapters after disabling.

    Args:
        model: Model with LoRA adapters
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "PEFT library required. Install with: pip install peft"
        )

    if not isinstance(model, PeftModel):
        return

    model.enable_adapter()
    logger.info("Enabled adapters")


def get_adapter_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about loaded adapters.

    Args:
        model: Model with LoRA adapters

    Returns:
        Dictionary with adapter information
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "PEFT library required. Install with: pip install peft"
        )

    if not isinstance(model, PeftModel):
        return {"has_adapters": False}

    # Get adapter names
    adapter_names = list(model.peft_config.keys())
    active_adapter = model.active_adapter

    # Get configs
    configs = {}
    for name in adapter_names:
        peft_cfg = model.peft_config[name]
        configs[name] = {
            "r": peft_cfg.r,
            "lora_alpha": peft_cfg.lora_alpha,
            "lora_dropout": peft_cfg.lora_dropout,
            "target_modules": list(peft_cfg.target_modules) if peft_cfg.target_modules else None,
            "bias": peft_cfg.bias,
        }

    # Get parameter counts
    trainable, total, pct = get_trainable_parameters(model)

    return {
        "has_adapters": True,
        "adapter_names": adapter_names,
        "active_adapter": active_adapter,
        "configs": configs,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percent": pct,
    }


def create_optimal_lora_config(
    model_name: str,
    training_size: str = "medium",
    memory_constrained: bool = False,
) -> Union[LoRAConfig, QLoRAConfig]:
    """
    Create an optimal LoRA/QLoRA config based on model and constraints.

    Args:
        model_name: Model name/path to optimize for
        training_size: "small", "medium", "large" dataset size
        memory_constrained: Whether memory is limited

    Returns:
        Optimized LoRA or QLoRA config
    """
    model_name_lower = model_name.lower()

    # Determine model size category
    if any(x in model_name_lower for x in ["7b", "8b"]):
        model_size = "7b"
    elif any(x in model_name_lower for x in ["13b", "14b"]):
        model_size = "13b"
    elif any(x in model_name_lower for x in ["70b", "72b"]):
        model_size = "70b"
    elif any(x in model_name_lower for x in ["1b", "1.1b", "1.5b", "2b", "3b"]):
        model_size = "3b"
    else:
        model_size = "7b"  # Default assumption

    # Determine optimal rank
    rank_map = {
        "small": {"3b": 32, "7b": 32, "13b": 16, "70b": 8},
        "medium": {"3b": 64, "7b": 64, "13b": 32, "70b": 16},
        "large": {"3b": 128, "7b": 128, "13b": 64, "70b": 32},
    }
    rank = rank_map.get(training_size, rank_map["medium"]).get(model_size, 64)

    # Alpha is typically 2x rank
    alpha = rank * 2

    # Determine if QLoRA is needed
    use_qlora = memory_constrained or model_size in ["13b", "70b"]

    # Detect target modules
    for arch in TARGET_MODULES_MAP:
        if arch in model_name_lower and arch != "default":
            target_modules = TARGET_MODULES_MAP[arch]
            break
    else:
        target_modules = TARGET_MODULES_MAP["llama"]  # Default to Llama-style

    if use_qlora:
        return QLoRAConfig(
            rank=rank,
            alpha=alpha,
            dropout=0.05,
            target_modules=target_modules,
            use_qlora=True,
            quantization_type=QuantizationType.NF4,
            use_gradient_checkpointing=True,
        )
    else:
        return LoRAConfig(
            rank=rank,
            alpha=alpha,
            dropout=0.05,
            target_modules=target_modules,
        )


# Convenience exports
__all__ = [
    # Enums
    "QuantizationType",
    "TaskType",
    # Configs
    "LoRAConfig",
    "QLoRAConfig",
    # Core functions
    "apply_lora",
    "apply_qlora",
    "prepare_model_for_qlora",
    "load_model_for_qlora",
    # Merge/save/load
    "merge_and_unload",
    "save_adapter",
    "load_adapter",
    # Adapter management
    "set_adapter",
    "disable_adapter",
    "enable_adapter",
    "get_adapter_info",
    # Utilities
    "detect_model_architecture",
    "get_trainable_parameters",
    "format_params",
    "create_optimal_lora_config",
    # Constants
    "TARGET_MODULES_MAP",
]
