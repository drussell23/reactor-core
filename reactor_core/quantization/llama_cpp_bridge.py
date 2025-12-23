"""
llama.cpp bridge for GGUF model inference.

Provides:
- GGUF model loading and inference
- Async inference support
- M1 Mac optimized settings
- Streaming generation
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class LlamaCppConfig:
    """Configuration for llama.cpp inference."""
    # Model settings
    model_path: Optional[Path] = None
    n_ctx: int = 2048  # Context length
    n_batch: int = 512  # Batch size for prompt processing
    n_threads: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_LLAMA_THREADS", "4"))
    )

    # GPU settings
    n_gpu_layers: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_GPU_LAYERS", "-1"))
    )  # -1 = all layers on GPU
    main_gpu: int = 0

    # Memory settings
    use_mmap: bool = True
    use_mlock: bool = False

    # Generation defaults
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1

    # Verbose output
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": str(self.model_path) if self.model_path else None,
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers,
            "main_gpu": self.main_gpu,
            "use_mmap": self.use_mmap,
            "use_mlock": self.use_mlock,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "verbose": self.verbose,
        }


@dataclass
class InferenceResult:
    """Result of inference."""
    text: str
    tokens_generated: int
    prompt_tokens: int
    total_tokens: int
    tokens_per_second: float
    generation_time_ms: float
    finish_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tokens_generated": self.tokens_generated,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_second": self.tokens_per_second,
            "generation_time_ms": self.generation_time_ms,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata,
        }


class LlamaCppBridge:
    """
    Bridge to llama.cpp for GGUF model inference.

    Provides both sync and async inference methods.
    """

    def __init__(
        self,
        config: Optional[LlamaCppConfig] = None,
    ):
        """
        Initialize llama.cpp bridge.

        Args:
            config: Inference configuration
        """
        self.config = config or LlamaCppConfig()
        self._model = None
        self._loaded = False

    def _get_llama_cpp(self):
        """Import llama-cpp-python."""
        try:
            from llama_cpp import Llama
            return Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python required. Install with: "
                "pip install llama-cpp-python"
            )

    def load(
        self,
        model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Load GGUF model.

        Args:
            model_path: Path to GGUF file (uses config if not provided)
        """
        Llama = self._get_llama_cpp()

        model_path = model_path or self.config.model_path
        if model_path is None:
            raise ValueError("Model path required")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading GGUF model: {model_path}")

        self._model = Llama(
            model_path=str(model_path),
            n_ctx=self.config.n_ctx,
            n_batch=self.config.n_batch,
            n_threads=self.config.n_threads,
            n_gpu_layers=self.config.n_gpu_layers,
            main_gpu=self.config.main_gpu,
            use_mmap=self.config.use_mmap,
            use_mlock=self.config.use_mlock,
            verbose=self.config.verbose,
        )

        self._loaded = True
        logger.info("Model loaded successfully")

    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            logger.info("Model unloaded")

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> InferenceResult:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            repeat_penalty: Repetition penalty
            stop: Stop sequences

        Returns:
            InferenceResult with generated text
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import time
        start_time = time.time()

        output = self._model(
            prompt,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
            top_k=top_k or self.config.top_k,
            repeat_penalty=repeat_penalty or self.config.repeat_penalty,
            stop=stop,
        )

        generation_time = (time.time() - start_time) * 1000

        # Extract result
        text = output["choices"][0]["text"]
        finish_reason = output["choices"][0].get("finish_reason", "unknown")
        usage = output.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        tokens_per_second = (
            completion_tokens / (generation_time / 1000)
            if generation_time > 0 else 0
        )

        return InferenceResult(
            text=text,
            tokens_generated=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            tokens_per_second=tokens_per_second,
            generation_time_ms=generation_time,
            finish_reason=finish_reason,
            metadata={"model": str(self.config.model_path)},
        )

    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """
        Stream text generation.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            repeat_penalty: Repetition penalty
            stop: Stop sequences

        Yields:
            Text chunks as they are generated
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        stream = self._model(
            prompt,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
            top_k=top_k or self.config.top_k,
            repeat_penalty=repeat_penalty or self.config.repeat_penalty,
            stop=stop,
            stream=True,
        )

        for output in stream:
            text = output["choices"][0]["text"]
            yield text

    async def generate_async(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> InferenceResult:
        """
        Async text generation.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            InferenceResult
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            ),
        )

    async def generate_stream_async(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Async streaming text generation.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Yields:
            Text chunks as they are generated
        """
        loop = asyncio.get_event_loop()

        # Create sync generator
        def generate():
            yield from self.generate_stream(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

        gen = generate()

        while True:
            try:
                chunk = await loop.run_in_executor(None, next, gen)
                yield chunk
            except StopIteration:
                break

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> InferenceResult:
        """
        Chat completion with message format.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            InferenceResult
        """
        # Format messages as prompt
        prompt = self._format_chat_prompt(messages)
        return self.generate(prompt, max_tokens, temperature, **kwargs)

    def _format_chat_prompt(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """Format chat messages as prompt string."""
        # Default ChatML format
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"

        # Add generation prompt
        formatted += "<|im_start|>assistant\n"
        return formatted

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to token IDs."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        return self._model.tokenize(text.encode())

    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize token IDs to text."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        return self._model.detokenize(tokens).decode()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        if not self._loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_path": str(self.config.model_path),
            "n_ctx": self.config.n_ctx,
            "n_gpu_layers": self.config.n_gpu_layers,
            "vocab_size": len(self._model),
        }

    def __enter__(self) -> "LlamaCppBridge":
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.unload()


def load_gguf_model(
    model_path: Union[str, Path],
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,
    **kwargs,
) -> LlamaCppBridge:
    """
    Convenience function to load a GGUF model.

    Args:
        model_path: Path to GGUF file
        n_ctx: Context length
        n_gpu_layers: GPU layers (-1 = all)
        **kwargs: Additional configuration

    Returns:
        Loaded LlamaCppBridge instance
    """
    config = LlamaCppConfig(
        model_path=Path(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        **kwargs,
    )

    bridge = LlamaCppBridge(config)
    bridge.load()
    return bridge


# Convenience exports
__all__ = [
    "LlamaCppBridge",
    "LlamaCppConfig",
    "InferenceResult",
    "load_gguf_model",
]
