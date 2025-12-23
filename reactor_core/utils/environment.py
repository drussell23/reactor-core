"""
Environment detection for M1 local vs GCP remote.

Enhanced for Night Shift Training Engine with:
- Quantization capability detection
- llama.cpp availability check
- VRAM estimation
- Optimal settings per environment
"""

import os
import platform
import shutil
import subprocess
import psutil
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class EnvironmentType(Enum):
    """Environment types"""
    M1_LOCAL = "m1_local"
    GCP_VM = "gcp_vm"
    UNKNOWN = "unknown"


class QuantizationType(Enum):
    """Supported quantization types"""
    NONE = "none"
    GGUF_Q4_K_M = "Q4_K_M"      # 4-bit, optimal for M1
    GGUF_Q5_K_M = "Q5_K_M"      # 5-bit, better quality
    GGUF_Q8_0 = "Q8_0"          # 8-bit, best quality
    GGUF_F16 = "F16"            # Float16, for inference
    BNB_4BIT = "bnb_4bit"       # bitsandbytes 4-bit for training
    BNB_8BIT = "bnb_8bit"       # bitsandbytes 8-bit for training


@dataclass
class EnvironmentInfo:
    """Enhanced environment information"""
    env_type: EnvironmentType
    cpu_arch: str
    total_ram_gb: float
    gpu_available: bool
    gpu_memory_gb: Optional[float] = None
    is_spot_vm: bool = False
    is_m1_mac: bool = False

    # Enhanced fields for Night Shift
    cpu_cores: int = 0
    available_ram_gb: float = 0.0

    # Quantization capabilities
    can_quantize_gguf: bool = False
    llama_cpp_available: bool = False
    llama_cpp_path: Optional[str] = None
    bitsandbytes_available: bool = False

    # GPU details
    gpu_name: Optional[str] = None
    cuda_version: Optional[str] = None
    mps_available: bool = False

    # Recommended quantization
    recommended_quant: QuantizationType = QuantizationType.NONE


def _check_llama_cpp() -> tuple[bool, Optional[str]]:
    """Check if llama.cpp quantize tool is available."""
    # Check common locations
    possible_paths = [
        os.getenv("LLAMA_CPP_PATH"),
        shutil.which("llama-quantize"),
        shutil.which("quantize"),
        os.path.expanduser("~/llama.cpp/quantize"),
        "/usr/local/bin/llama-quantize",
        "/opt/homebrew/bin/llama-quantize",
    ]

    for path in possible_paths:
        if path and os.path.isfile(path):
            return True, path

    # Check if llama-cpp-python is installed
    try:
        import llama_cpp
        return True, "llama-cpp-python"
    except ImportError:
        pass

    return False, None


def _check_bitsandbytes() -> bool:
    """Check if bitsandbytes is available for QLoRA."""
    try:
        import bitsandbytes
        return True
    except ImportError:
        return False


def _get_gpu_details() -> Dict[str, Any]:
    """Get detailed GPU information."""
    details = {
        "available": False,
        "name": None,
        "memory_gb": None,
        "cuda_version": None,
        "mps_available": False,
    }

    try:
        import torch

        if torch.cuda.is_available():
            details["available"] = True
            details["name"] = torch.cuda.get_device_name(0)
            details["memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3),
                2
            )
            details["cuda_version"] = torch.version.cuda

        elif torch.backends.mps.is_available():
            details["available"] = True
            details["mps_available"] = True
            details["name"] = "Apple Silicon (MPS)"
            # MPS doesn't expose VRAM directly, estimate from system memory
            # Apple Silicon shares memory with CPU

    except ImportError:
        pass

    return details


def detect_environment() -> EnvironmentInfo:
    """
    Detect the current execution environment with enhanced capabilities.

    Returns:
        EnvironmentInfo with comprehensive environment details
    """
    # Get system info
    cpu_arch = platform.machine()
    total_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB
    available_ram = psutil.virtual_memory().available / (1024**3)
    cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1

    # Check for M1 Mac
    is_m1_mac = cpu_arch == "arm64" and platform.system() == "Darwin"

    # Get detailed GPU info
    gpu_details = _get_gpu_details()
    gpu_available = gpu_details["available"]
    gpu_memory_gb = gpu_details["memory_gb"]
    gpu_name = gpu_details["name"]
    cuda_version = gpu_details["cuda_version"]
    mps_available = gpu_details["mps_available"]

    # Check for GCP metadata
    is_gcp_vm = _is_gcp_vm()
    is_spot_vm = _is_spot_vm() if is_gcp_vm else False

    # Check quantization capabilities
    llama_cpp_available, llama_cpp_path = _check_llama_cpp()
    bitsandbytes_available = _check_bitsandbytes()

    # Can quantize if llama-cpp-python or quantize binary is available
    can_quantize_gguf = llama_cpp_available

    # Determine environment type
    if is_m1_mac:
        env_type = EnvironmentType.M1_LOCAL
    elif is_gcp_vm:
        env_type = EnvironmentType.GCP_VM
    else:
        env_type = EnvironmentType.UNKNOWN

    # Determine recommended quantization
    recommended_quant = _get_recommended_quant(
        env_type=env_type,
        total_ram_gb=total_ram,
        gpu_memory_gb=gpu_memory_gb,
        is_m1_mac=is_m1_mac,
        can_quantize_gguf=can_quantize_gguf,
    )

    return EnvironmentInfo(
        env_type=env_type,
        cpu_arch=cpu_arch,
        total_ram_gb=round(total_ram, 2),
        available_ram_gb=round(available_ram, 2),
        cpu_cores=cpu_cores,
        gpu_available=gpu_available,
        gpu_memory_gb=round(gpu_memory_gb, 2) if gpu_memory_gb else None,
        gpu_name=gpu_name,
        cuda_version=cuda_version,
        mps_available=mps_available,
        is_spot_vm=is_spot_vm,
        is_m1_mac=is_m1_mac,
        can_quantize_gguf=can_quantize_gguf,
        llama_cpp_available=llama_cpp_available,
        llama_cpp_path=llama_cpp_path,
        bitsandbytes_available=bitsandbytes_available,
        recommended_quant=recommended_quant,
    )


def _get_recommended_quant(
    env_type: EnvironmentType,
    total_ram_gb: float,
    gpu_memory_gb: Optional[float],
    is_m1_mac: bool,
    can_quantize_gguf: bool,
) -> QuantizationType:
    """Determine recommended quantization based on environment."""
    if not can_quantize_gguf:
        return QuantizationType.NONE

    # M1 Mac: Use 4-bit for memory efficiency
    if is_m1_mac:
        if total_ram_gb >= 32:
            return QuantizationType.GGUF_Q5_K_M  # Better quality if enough RAM
        return QuantizationType.GGUF_Q4_K_M

    # GCP VM with GPU
    if env_type == EnvironmentType.GCP_VM and gpu_memory_gb:
        if gpu_memory_gb >= 24:  # A100/A10G
            return QuantizationType.GGUF_Q8_0  # Best quality
        elif gpu_memory_gb >= 16:
            return QuantizationType.GGUF_Q5_K_M
        else:
            return QuantizationType.GGUF_Q4_K_M

    # Default to 4-bit
    return QuantizationType.GGUF_Q4_K_M


def _is_gcp_vm() -> bool:
    """Check if running on GCP VM"""
    try:
        # GCP metadata server
        import requests
        response = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/id",
            headers={"Metadata-Flavor": "Google"},
            timeout=1
        )
        return response.status_code == 200
    except:
        return False


def _is_spot_vm() -> bool:
    """Check if running on GCP Spot (preemptible) VM"""
    try:
        import requests
        response = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/preempted",
            headers={"Metadata-Flavor": "Google"},
            timeout=1
        )
        return response.status_code == 200
    except:
        return False


def get_recommended_config(env_info: EnvironmentInfo) -> dict:
    """
    Get recommended training configuration based on environment.

    Args:
        env_info: Environment information

    Returns:
        Dictionary with recommended settings
    """
    if env_info.env_type == EnvironmentType.M1_LOCAL:
        return {
            "mode": "inference_only",
            "device": "mps" if env_info.gpu_available else "cpu",
            "batch_size": 1,
            "use_quantization": True,
            "quantization_type": env_info.recommended_quant.value,
            "max_model_size": "7B",
            "enable_training": False,
            "use_qlora": env_info.bitsandbytes_available,
            "message": "M1 Mac detected: Lightweight inference mode enabled"
        }

    elif env_info.env_type == EnvironmentType.GCP_VM:
        # Determine optimal settings based on RAM/GPU
        config = {
            "mode": "full_training",
            "device": "cuda" if env_info.gpu_available else "cpu",
            "enable_training": True,
            "use_qlora": env_info.bitsandbytes_available,
        }

        if env_info.total_ram_gb >= 32:
            config.update({
                "batch_size": 4,
                "use_gradient_checkpointing": True,
                "use_lora": True,
                "lora_rank": 64,
                "lora_alpha": 128,
                "max_model_size": "13B",
                "max_seq_length": 2048,
                "message": "GCP VM detected: Full training enabled"
            })
        else:
            config.update({
                "batch_size": 2,
                "use_gradient_checkpointing": True,
                "use_lora": True,
                "lora_rank": 32,
                "lora_alpha": 64,
                "max_model_size": "7B",
                "max_seq_length": 1024,
                "message": "GCP VM detected: Memory-efficient training enabled"
            })

        if env_info.is_spot_vm:
            config.update({
                "checkpoint_interval": 500,
                "enable_auto_resume": True,
                "message": config["message"] + " (Spot VM: Auto-resume enabled)"
            })

        return config

    else:
        return {
            "mode": "unknown",
            "device": "cpu",
            "message": "Unknown environment: Using conservative settings"
        }


def get_quantization_config(env_info: Optional[EnvironmentInfo] = None) -> Dict[str, Any]:
    """
    Get recommended quantization configuration based on environment.

    Args:
        env_info: Environment information. If None, detects automatically.

    Returns:
        Dictionary with quantization settings
    """
    if env_info is None:
        env_info = detect_environment()

    config = {
        "can_quantize": env_info.can_quantize_gguf,
        "recommended_type": env_info.recommended_quant.value,
        "llama_cpp_available": env_info.llama_cpp_available,
        "llama_cpp_path": env_info.llama_cpp_path,
        "bitsandbytes_available": env_info.bitsandbytes_available,
    }

    # GGUF-specific settings
    if env_info.can_quantize_gguf:
        quant_type = env_info.recommended_quant

        config.update({
            "gguf_type": quant_type.value,
            "use_gpu_offload": env_info.env_type == EnvironmentType.GCP_VM,
            "num_threads": env_info.cpu_cores,
        })

        # Estimate memory requirements for common model sizes
        # Rough estimates based on quantization type
        memory_multipliers = {
            QuantizationType.GGUF_Q4_K_M: 0.5,   # ~50% of FP16
            QuantizationType.GGUF_Q5_K_M: 0.6,   # ~60% of FP16
            QuantizationType.GGUF_Q8_0: 0.75,    # ~75% of FP16
            QuantizationType.GGUF_F16: 1.0,      # 100% of FP16
        }

        multiplier = memory_multipliers.get(quant_type, 0.5)

        # Estimate what model size can fit
        available_ram = env_info.available_ram_gb
        if env_info.is_m1_mac:
            # M1 uses unified memory, can use most of available RAM
            usable_ram = available_ram * 0.8
        else:
            usable_ram = available_ram * 0.7

        # Model size estimates (FP16 base):
        # 3B model ~ 6GB, 7B ~ 14GB, 13B ~ 26GB, 30B ~ 60GB
        base_sizes = {
            "3B": 6,
            "7B": 14,
            "8B": 16,
            "13B": 26,
            "30B": 60,
            "70B": 140,
        }

        max_model = "3B"
        for size, base_gb in base_sizes.items():
            quantized_gb = base_gb * multiplier
            if quantized_gb <= usable_ram:
                max_model = size
            else:
                break

        config["max_model_size"] = max_model
        config["estimated_memory_usage_gb"] = round(usable_ram * 0.9, 2)

    return config


# Auto-detect on import
_ENV_INFO = detect_environment()


def print_environment_info():
    """Print detected environment information"""
    info = _ENV_INFO

    print("=" * 60)
    print("Night Shift Training Engine - Environment Detection")
    print("=" * 60)

    # Basic info
    print(f"Environment Type: {info.env_type.value}")
    print(f"CPU Architecture: {info.cpu_arch}")
    print(f"CPU Cores: {info.cpu_cores}")
    print(f"Total RAM: {info.total_ram_gb} GB")
    print(f"Available RAM: {info.available_ram_gb} GB")

    # GPU info
    print("-" * 60)
    print(f"GPU Available: {info.gpu_available}")
    if info.gpu_name:
        print(f"GPU Name: {info.gpu_name}")
    if info.gpu_memory_gb:
        print(f"GPU Memory: {info.gpu_memory_gb} GB")
    if info.cuda_version:
        print(f"CUDA Version: {info.cuda_version}")
    print(f"MPS Available: {info.mps_available}")

    # Platform info
    print("-" * 60)
    print(f"M1 Mac: {info.is_m1_mac}")
    print(f"GCP Spot VM: {info.is_spot_vm}")

    # Quantization capabilities
    print("-" * 60)
    print(f"Can Quantize GGUF: {info.can_quantize_gguf}")
    print(f"llama.cpp Available: {info.llama_cpp_available}")
    if info.llama_cpp_path:
        print(f"llama.cpp Path: {info.llama_cpp_path}")
    print(f"bitsandbytes Available: {info.bitsandbytes_available}")
    print(f"Recommended Quantization: {info.recommended_quant.value}")

    print("=" * 60)

    # Print recommended config
    config = get_recommended_config(info)
    print(f"\nRecommended Config: {config['message']}")
    print(f"Mode: {config['mode']}")
    print(f"Device: {config['device']}")

    # Print quantization config
    quant_config = get_quantization_config(info)
    if quant_config["can_quantize"]:
        print(f"Quantization Type: {quant_config['recommended_type']}")
        print(f"Max Model Size: {quant_config.get('max_model_size', 'N/A')}")

    print("=" * 60)


if __name__ == "__main__":
    print_environment_info()
