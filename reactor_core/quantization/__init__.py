"""
Quantization module for Night Shift Training Engine.

Provides:
- GGUF format conversion
- 4-bit/8-bit quantization
- llama.cpp integration
- M1 Mac optimized deployment
"""

from reactor_core.quantization.gguf_converter import (
    GGUFConverter,
    GGUFConfig,
    QuantizationMethod,
    ConversionResult,
    convert_to_gguf,
)

from reactor_core.quantization.llama_cpp_bridge import (
    LlamaCppBridge,
    LlamaCppConfig,
    InferenceResult,
    load_gguf_model,
)

__all__ = [
    # GGUF Converter
    "GGUFConverter",
    "GGUFConfig",
    "QuantizationMethod",
    "ConversionResult",
    "convert_to_gguf",
    # Llama.cpp Bridge
    "LlamaCppBridge",
    "LlamaCppConfig",
    "InferenceResult",
    "load_gguf_model",
]
