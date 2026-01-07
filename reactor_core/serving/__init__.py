"""
Model Serving Module - Reactor Core (Nervous System)
====================================================

Production-grade model serving infrastructure for AGI OS.

Components:
- InferenceEngine: Main entry point for model inference
- ModelRouter: Dynamic routing to appropriate models
- ResponseCache: Intelligent caching with LRU eviction
- ModelEnsemble: Multi-model aggregation
- Multiple backends: Transformers, vLLM, llama.cpp
"""

from reactor_core.serving.inference_engine import (
    # Enums
    ModelBackend,
    QuantizationType,
    TaskType,
    RequestPriority,
    EnsembleStrategy,
    # Data structures
    GenerationConfig,
    InferenceRequest,
    InferenceResponse,
    ModelInfo,
    # Caching
    ResponseCache,
    # Backends
    ModelBackendInterface,
    TransformersBackend,
    VLLMBackend,
    LlamaCppBackend,
    # Routing
    RoutingRule,
    ModelRouter,
    # Engine
    InferenceEngineConfig,
    InferenceEngine,
    # Ensemble
    EnsembleConfig,
    ModelEnsemble,
)

__all__ = [
    # Enums
    "ModelBackend",
    "QuantizationType",
    "TaskType",
    "RequestPriority",
    "EnsembleStrategy",
    # Data structures
    "GenerationConfig",
    "InferenceRequest",
    "InferenceResponse",
    "ModelInfo",
    # Caching
    "ResponseCache",
    # Backends
    "ModelBackendInterface",
    "TransformersBackend",
    "VLLMBackend",
    "LlamaCppBackend",
    # Routing
    "RoutingRule",
    "ModelRouter",
    # Engine
    "InferenceEngineConfig",
    "InferenceEngine",
    # Ensemble
    "EnsembleConfig",
    "ModelEnsemble",
]
