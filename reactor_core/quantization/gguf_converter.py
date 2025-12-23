"""
GGUF format conversion for llama.cpp deployment.

Provides:
- HuggingFace to GGUF conversion
- Multiple quantization methods
- M1 Mac optimized formats
- Async conversion support
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class QuantizationMethod(Enum):
    """Quantization methods for GGUF."""
    # Full precision
    F32 = "f32"
    F16 = "f16"

    # 8-bit
    Q8_0 = "q8_0"

    # 5-bit
    Q5_0 = "q5_0"
    Q5_1 = "q5_1"
    Q5_K_S = "q5_k_s"
    Q5_K_M = "q5_k_m"

    # 4-bit (most common for M1 Mac)
    Q4_0 = "q4_0"
    Q4_1 = "q4_1"
    Q4_K_S = "q4_k_s"
    Q4_K_M = "q4_k_m"  # Recommended for quality/size balance

    # 3-bit
    Q3_K_S = "q3_k_s"
    Q3_K_M = "q3_k_m"
    Q3_K_L = "q3_k_l"

    # 2-bit (experimental)
    Q2_K = "q2_k"

    # IQ (importance quantization)
    IQ4_NL = "iq4_nl"
    IQ3_XXS = "iq3_xxs"
    IQ2_XXS = "iq2_xxs"


# Recommended methods for different use cases
RECOMMENDED_METHODS = {
    "quality": QuantizationMethod.Q5_K_M,
    "balanced": QuantizationMethod.Q4_K_M,
    "size": QuantizationMethod.Q3_K_M,
    "m1_mac": QuantizationMethod.Q4_K_M,
    "server": QuantizationMethod.Q8_0,
}


@dataclass
class GGUFConfig:
    """Configuration for GGUF conversion."""
    # Quantization
    method: QuantizationMethod = QuantizationMethod.Q4_K_M

    # Output settings
    output_dir: Optional[Path] = None
    output_name: Optional[str] = None

    # llama.cpp settings
    llama_cpp_path: Optional[Path] = field(
        default_factory=lambda: Path(os.getenv(
            "LLAMA_CPP_PATH",
            Path.home() / ".local" / "llama.cpp"
        ))
    )

    # Conversion options
    vocab_type: str = "bpe"  # "bpe", "spm", "hfft"
    use_safetensors: bool = True
    keep_intermediate: bool = False

    # Resource limits
    threads: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_QUANTIZE_THREADS", "4"))
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "output_name": self.output_name,
            "llama_cpp_path": str(self.llama_cpp_path),
            "vocab_type": self.vocab_type,
            "use_safetensors": self.use_safetensors,
            "keep_intermediate": self.keep_intermediate,
            "threads": self.threads,
        }


@dataclass
class ConversionResult:
    """Result of GGUF conversion."""
    success: bool
    output_path: Optional[Path]
    method: QuantizationMethod
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    duration_seconds: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output_path": str(self.output_path) if self.output_path else None,
            "method": self.method.value,
            "original_size_mb": self.original_size_mb,
            "quantized_size_mb": self.quantized_size_mb,
            "compression_ratio": self.compression_ratio,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        if not self.success:
            return f"Conversion FAILED: {self.error}"

        return (
            f"Conversion SUCCESS\n"
            f"  Output: {self.output_path}\n"
            f"  Method: {self.method.value}\n"
            f"  Size: {self.original_size_mb:.1f}MB -> {self.quantized_size_mb:.1f}MB "
            f"({self.compression_ratio:.1f}x compression)\n"
            f"  Time: {self.duration_seconds:.1f}s"
        )


class GGUFConverter:
    """
    Convert HuggingFace models to GGUF format.

    Supports multiple quantization methods for optimal
    performance on different hardware.
    """

    def __init__(
        self,
        config: Optional[GGUFConfig] = None,
    ):
        """
        Initialize GGUF converter.

        Args:
            config: Conversion configuration
        """
        self.config = config or GGUFConfig()
        self._llama_cpp_available = None

    def _check_llama_cpp(self) -> bool:
        """Check if llama.cpp is available."""
        if self._llama_cpp_available is not None:
            return self._llama_cpp_available

        # Check for convert script
        convert_script = self.config.llama_cpp_path / "convert_hf_to_gguf.py"
        quantize_bin = self.config.llama_cpp_path / "llama-quantize"

        if not convert_script.exists():
            # Try alternative locations
            alt_paths = [
                Path.home() / "llama.cpp",
                Path("/usr/local/llama.cpp"),
                Path("/opt/llama.cpp"),
            ]
            for alt in alt_paths:
                if (alt / "convert_hf_to_gguf.py").exists():
                    self.config.llama_cpp_path = alt
                    convert_script = alt / "convert_hf_to_gguf.py"
                    quantize_bin = alt / "llama-quantize"
                    break

        self._llama_cpp_available = convert_script.exists() and quantize_bin.exists()

        if not self._llama_cpp_available:
            logger.warning(
                f"llama.cpp not found at {self.config.llama_cpp_path}. "
                "Please install llama.cpp or set LLAMA_CPP_PATH"
            )

        return self._llama_cpp_available

    def _get_model_size(self, path: Path) -> float:
        """Get model size in MB."""
        total = 0
        if path.is_dir():
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        else:
            total = path.stat().st_size
        return total / (1024 * 1024)

    def _run_convert_script(
        self,
        model_path: Path,
        output_path: Path,
    ) -> bool:
        """Run the HF to GGUF conversion script."""
        convert_script = self.config.llama_cpp_path / "convert_hf_to_gguf.py"

        cmd = [
            "python3",
            str(convert_script),
            str(model_path),
            "--outfile", str(output_path),
            "--outtype", "f16",  # Initial conversion to F16
        ]

        if self.config.vocab_type:
            cmd.extend(["--vocab-type", self.config.vocab_type])

        logger.info(f"Running conversion: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode != 0:
                logger.error(f"Conversion failed: {result.stderr}")
                return False

            logger.info("Initial conversion to F16 GGUF complete")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Conversion timed out")
            return False
        except Exception as e:
            logger.error(f"Conversion error: {e}")
            return False

    def _run_quantize(
        self,
        input_path: Path,
        output_path: Path,
        method: QuantizationMethod,
    ) -> bool:
        """Run the llama.cpp quantization tool."""
        quantize_bin = self.config.llama_cpp_path / "llama-quantize"

        cmd = [
            str(quantize_bin),
            str(input_path),
            str(output_path),
            method.value,
        ]

        if self.config.threads > 0:
            cmd.extend(["--threads", str(self.config.threads)])

        logger.info(f"Running quantization: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )

            if result.returncode != 0:
                logger.error(f"Quantization failed: {result.stderr}")
                return False

            logger.info(f"Quantization to {method.value} complete")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Quantization timed out")
            return False
        except Exception as e:
            logger.error(f"Quantization error: {e}")
            return False

    async def convert(
        self,
        model_path: Union[str, Path],
        output_path: Optional[Path] = None,
        method: Optional[QuantizationMethod] = None,
    ) -> ConversionResult:
        """
        Convert HuggingFace model to GGUF format.

        Args:
            model_path: Path to HuggingFace model
            output_path: Output path for GGUF file
            method: Quantization method

        Returns:
            ConversionResult with conversion status
        """
        import time
        start_time = time.time()

        model_path = Path(model_path)
        method = method or self.config.method

        # Determine output path
        if output_path is None:
            output_dir = self.config.output_dir or model_path.parent
            output_name = self.config.output_name or f"{model_path.name}-{method.value}.gguf"
            output_path = output_dir / output_name

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check llama.cpp availability
        if not self._check_llama_cpp():
            return ConversionResult(
                success=False,
                output_path=None,
                method=method,
                original_size_mb=self._get_model_size(model_path),
                quantized_size_mb=0,
                compression_ratio=0,
                duration_seconds=time.time() - start_time,
                error="llama.cpp not available",
            )

        original_size = self._get_model_size(model_path)

        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            f16_path = temp_path / "model-f16.gguf"

            # Step 1: Convert to F16 GGUF
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None,
                self._run_convert_script,
                model_path,
                f16_path,
            )

            if not success:
                return ConversionResult(
                    success=False,
                    output_path=None,
                    method=method,
                    original_size_mb=original_size,
                    quantized_size_mb=0,
                    compression_ratio=0,
                    duration_seconds=time.time() - start_time,
                    error="F16 conversion failed",
                )

            # Step 2: Quantize if not F16
            if method in [QuantizationMethod.F16, QuantizationMethod.F32]:
                # Just copy F16 file
                shutil.copy(f16_path, output_path)
            else:
                success = await loop.run_in_executor(
                    None,
                    self._run_quantize,
                    f16_path,
                    output_path,
                    method,
                )

                if not success:
                    return ConversionResult(
                        success=False,
                        output_path=None,
                        method=method,
                        original_size_mb=original_size,
                        quantized_size_mb=0,
                        compression_ratio=0,
                        duration_seconds=time.time() - start_time,
                        error="Quantization failed",
                    )

        # Calculate final size
        quantized_size = self._get_model_size(output_path)
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
        duration = time.time() - start_time

        logger.info(
            f"Conversion complete: {original_size:.1f}MB -> {quantized_size:.1f}MB "
            f"({compression_ratio:.1f}x) in {duration:.1f}s"
        )

        return ConversionResult(
            success=True,
            output_path=output_path,
            method=method,
            original_size_mb=original_size,
            quantized_size_mb=quantized_size,
            compression_ratio=compression_ratio,
            duration_seconds=duration,
            metadata={
                "model_path": str(model_path),
            },
        )

    async def convert_merged_model(
        self,
        base_model: Union[str, Path],
        adapter_path: Union[str, Path],
        output_path: Optional[Path] = None,
        method: Optional[QuantizationMethod] = None,
    ) -> ConversionResult:
        """
        Merge LoRA adapter with base model and convert to GGUF.

        Args:
            base_model: Base model path or HuggingFace ID
            adapter_path: LoRA adapter path
            output_path: Output path for GGUF file
            method: Quantization method

        Returns:
            ConversionResult
        """
        import time
        start_time = time.time()

        adapter_path = Path(adapter_path)
        method = method or self.config.method

        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            return ConversionResult(
                success=False,
                output_path=None,
                method=method,
                original_size_mb=0,
                quantized_size_mb=0,
                compression_ratio=0,
                duration_seconds=time.time() - start_time,
                error="transformers and peft required for merging",
            )

        # Create temp directory for merged model
        with tempfile.TemporaryDirectory() as temp_dir:
            merged_path = Path(temp_dir) / "merged"

            try:
                # Load base model
                logger.info(f"Loading base model: {base_model}")
                model = AutoModelForCausalLM.from_pretrained(
                    str(base_model),
                    torch_dtype="auto",
                    device_map="auto",
                )
                tokenizer = AutoTokenizer.from_pretrained(str(base_model))

                # Load and merge adapter
                logger.info(f"Loading adapter: {adapter_path}")
                model = PeftModel.from_pretrained(model, str(adapter_path))
                logger.info("Merging adapter with base model...")
                model = model.merge_and_unload()

                # Save merged model
                logger.info(f"Saving merged model to {merged_path}")
                model.save_pretrained(merged_path)
                tokenizer.save_pretrained(merged_path)

            except Exception as e:
                return ConversionResult(
                    success=False,
                    output_path=None,
                    method=method,
                    original_size_mb=0,
                    quantized_size_mb=0,
                    compression_ratio=0,
                    duration_seconds=time.time() - start_time,
                    error=f"Merge failed: {e}",
                )

            # Convert merged model
            return await self.convert(merged_path, output_path, method)


def convert_to_gguf(
    model_path: Union[str, Path],
    output_path: Optional[Path] = None,
    method: Union[str, QuantizationMethod] = "q4_k_m",
    llama_cpp_path: Optional[Path] = None,
) -> ConversionResult:
    """
    Convenience function to convert model to GGUF.

    Args:
        model_path: Path to HuggingFace model
        output_path: Output path for GGUF file
        method: Quantization method name or enum
        llama_cpp_path: Path to llama.cpp installation

    Returns:
        ConversionResult
    """
    if isinstance(method, str):
        method = QuantizationMethod(method.lower())

    config = GGUFConfig(method=method)
    if llama_cpp_path:
        config.llama_cpp_path = llama_cpp_path

    converter = GGUFConverter(config)

    # Run async conversion
    return asyncio.run(converter.convert(model_path, output_path, method))


# Convenience exports
__all__ = [
    "GGUFConverter",
    "GGUFConfig",
    "QuantizationMethod",
    "ConversionResult",
    "convert_to_gguf",
    "RECOMMENDED_METHODS",
]
