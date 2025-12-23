"""
Base configuration system for Night Shift Training Engine.

Features:
- Dataclass-based configuration with type hints
- YAML file loading with environment variable interpolation
- Dynamic config reloading
- Validation and defaults
"""

from __future__ import annotations

import os
import re
import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TypeVar,
    Type,
    Union,
    Callable,
    Awaitable,
)
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseConfig")

# Singleton config instance
_config_instance: Optional["NightShiftConfig"] = None
_config_lock = asyncio.Lock()


def _interpolate_env_vars(value: Any) -> Any:
    """
    Recursively interpolate environment variables in config values.

    Supports formats:
    - ${VAR_NAME} - Required, raises if not set
    - ${VAR_NAME:-default} - Optional with default
    - ${VAR_NAME:?error message} - Required with custom error
    """
    if isinstance(value, str):
        # Pattern: ${VAR_NAME}, ${VAR_NAME:-default}, ${VAR_NAME:?error}
        pattern = r"\$\{([A-Z_][A-Z0-9_]*)(?:(:-)([^}]*))?(?:(:\?)([^}]*))?\}"

        def replace_var(match: re.Match) -> str:
            var_name = match.group(1)
            has_default = match.group(2) is not None
            default_value = match.group(3) or ""
            has_error = match.group(4) is not None
            error_msg = match.group(5) or f"Required environment variable {var_name} is not set"

            env_value = os.environ.get(var_name)

            if env_value is not None:
                return env_value
            elif has_default:
                return default_value
            elif has_error:
                raise ValueError(error_msg)
            else:
                # Check if the entire string is just the variable
                if match.group(0) == value:
                    raise ValueError(f"Environment variable {var_name} is not set")
                return match.group(0)  # Keep original if part of larger string

        result = re.sub(pattern, replace_var, value)

        # Handle ~ for home directory
        if result.startswith("~"):
            result = str(Path(result).expanduser())

        return result

    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]

    return value


def _coerce_type(value: Any, target_type: type) -> Any:
    """Coerce a value to the target type."""
    if value is None:
        return None

    origin = getattr(target_type, "__origin__", None)

    # Handle Optional types
    if origin is Union:
        args = target_type.__args__
        if type(None) in args:
            non_none_types = [t for t in args if t is not type(None)]
            if len(non_none_types) == 1:
                return _coerce_type(value, non_none_types[0])

    # Handle Path
    if target_type is Path or (hasattr(target_type, "__origin__") and target_type.__origin__ is Path):
        return Path(value).expanduser() if value else None

    # Handle List
    if origin is list:
        item_type = target_type.__args__[0] if target_type.__args__ else str
        if isinstance(value, list):
            return [_coerce_type(item, item_type) for item in value]
        return [_coerce_type(value, item_type)]

    # Handle bool (special case because bool("false") is True)
    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    # Handle numeric types
    if target_type is int:
        return int(float(value)) if value else 0
    if target_type is float:
        return float(value) if value else 0.0

    # Default: try direct conversion
    try:
        return target_type(value)
    except (TypeError, ValueError):
        return value


@dataclass
class BaseConfig:
    """
    Base configuration class with YAML loading and env var interpolation.
    """

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create config from dictionary with env var interpolation."""
        interpolated = _interpolate_env_vars(data)

        # Get field types for coercion
        field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}

        # Only include fields that exist in the dataclass
        filtered = {}
        for key, value in interpolated.items():
            if key in field_types:
                filtered[key] = _coerce_type(value, field_types[key])

        return cls(**filtered)

    @classmethod
    def from_yaml(cls: Type[T], path: Union[str, Path]) -> T:
        """Load config from YAML file with env var interpolation."""
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_env(cls: Type[T], prefix: str = "") -> T:
        """Create config entirely from environment variables."""
        data = {}

        for field_info in cls.__dataclass_fields__.values():
            env_key = f"{prefix}{field_info.name}".upper()
            env_value = os.environ.get(env_key)

            if env_value is not None:
                data[field_info.name] = env_value

        return cls.from_dict(data) if data else cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result

    def merge(self: T, other: Dict[str, Any]) -> T:
        """Create new config with overrides merged in."""
        current = self.to_dict()
        interpolated = _interpolate_env_vars(other)
        current.update(interpolated)
        return self.__class__.from_dict(current)


@dataclass
class IngestionConfig(BaseConfig):
    """Configuration for data ingestion from JARVIS logs."""

    # Source paths
    jarvis_logs_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("NIGHTSHIFT_JARVIS_LOGS", "~/.jarvis/logs")
        ).expanduser()
    )
    jarvis_data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("NIGHTSHIFT_JARVIS_DATA", "~/.jarvis/data")
        ).expanduser()
    )

    # Processing settings
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_BATCH_SIZE", "1000"))
    )
    max_workers: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_MAX_WORKERS", "4"))
    )
    streaming_enabled: bool = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_STREAMING", "false").lower() == "true"
    )

    # Quality thresholds
    min_confidence: float = field(
        default_factory=lambda: float(os.getenv("NIGHTSHIFT_MIN_CONFIDENCE", "0.7"))
    )
    deduplicate: bool = True
    dedup_similarity_threshold: float = 0.95

    # Time range
    lookback_days: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_LOOKBACK_DAYS", "7"))
    )

    # Source types to ingest
    ingest_telemetry: bool = True
    ingest_feedback: bool = True
    ingest_auth_records: bool = True
    ingest_raw_logs: bool = True


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for model training."""

    # Model settings
    base_model: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_BASE_MODEL", "meta-llama/Llama-3.2-3B")
    )
    model_revision: Optional[str] = None
    trust_remote_code: bool = False

    # LoRA settings
    use_lora: bool = True
    lora_rank: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_LORA_RANK", "64"))
    )
    lora_alpha: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_LORA_ALPHA", "128"))
    )
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # QLoRA settings
    use_qlora: bool = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_USE_QLORA", "true").lower() == "true"
    )
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Training hyperparameters
    learning_rate: float = field(
        default_factory=lambda: float(os.getenv("NIGHTSHIFT_LR", "2e-5"))
    )
    num_epochs: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_EPOCHS", "3"))
    )
    per_device_batch_size: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_TRAIN_BATCH_SIZE", "4"))
    )
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Sequence settings
    max_seq_length: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_MAX_SEQ_LEN", "2048"))
    )

    # Checkpointing
    checkpoint_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("NIGHTSHIFT_CHECKPOINT_DIR", "~/.jarvis/training/checkpoints")
        ).expanduser()
    )
    save_steps: int = 500
    eval_steps: int = 500
    max_checkpoints: int = 3
    resume_from_checkpoint: bool = True

    # Output
    output_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("NIGHTSHIFT_OUTPUT_DIR", "~/.jarvis/training/output")
        ).expanduser()
    )

    # Distributed training
    use_fsdp: bool = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_USE_FSDP", "false").lower() == "true"
    )

    # Device
    device: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_DEVICE", "auto")
    )


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for knowledge distillation via teacher model."""

    # Teacher model settings
    teacher_provider: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_TEACHER_PROVIDER", "openai")
    )
    teacher_model: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_TEACHER_MODEL", "gpt-4o")
    )

    # API keys (from environment)
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )

    # Rate limiting
    requests_per_minute: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_RPM", "60"))
    )
    max_concurrent_requests: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_CONCURRENT", "10"))
    )
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0

    # Cost controls
    max_daily_cost_usd: float = field(
        default_factory=lambda: float(os.getenv("NIGHTSHIFT_MAX_DAILY_COST", "50.0"))
    )
    max_tokens_per_request: int = 4096

    # Quality thresholds
    min_quality_score: float = 0.6
    rewrite_threshold: float = 0.4

    # Distillation modes
    enable_scoring: bool = True
    enable_rewriting: bool = True
    enable_synthetic_generation: bool = False

    # Synthetic generation settings
    synthetic_examples_per_topic: int = 5
    synthetic_topics: List[str] = field(default_factory=list)


@dataclass
class QuantizationConfig(BaseConfig):
    """Configuration for model quantization."""

    # Output format
    output_format: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_QUANT_FORMAT", "gguf")
    )

    # GGUF settings
    gguf_quantization_type: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_GGUF_QUANT", "Q4_K_M")
    )

    # Output paths
    output_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("NIGHTSHIFT_QUANT_OUTPUT", "~/.jarvis/models/quantized")
        ).expanduser()
    )

    # llama.cpp settings
    llama_cpp_path: Optional[Path] = field(
        default_factory=lambda: Path(os.getenv("LLAMA_CPP_PATH", "")).expanduser()
        if os.getenv("LLAMA_CPP_PATH") else None
    )

    # Performance settings
    num_threads: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_QUANT_THREADS", "4"))
    )


@dataclass
class EvalConfig(BaseConfig):
    """Configuration for model evaluation and gatekeeper."""

    # Benchmarks to run
    run_humaneval: bool = True
    run_jarvis_eval: bool = True
    run_regression: bool = True
    run_perplexity: bool = True
    run_latency: bool = True

    # Thresholds for gatekeeper
    max_perplexity: float = field(
        default_factory=lambda: float(os.getenv("NIGHTSHIFT_MAX_PERPLEXITY", "5.0"))
    )
    min_humaneval_pass_rate: float = field(
        default_factory=lambda: float(os.getenv("NIGHTSHIFT_MIN_HUMANEVAL", "0.30"))
    )
    max_latency_ms: float = field(
        default_factory=lambda: float(os.getenv("NIGHTSHIFT_MAX_LATENCY", "100.0"))
    )
    max_regression_delta: float = 0.05  # 5% regression allowed

    # Test set paths
    jarvis_test_set_path: Optional[Path] = field(
        default_factory=lambda: Path(os.getenv("NIGHTSHIFT_TEST_SET", "")).expanduser()
        if os.getenv("NIGHTSHIFT_TEST_SET") else None
    )

    # Previous model for comparison
    previous_model_path: Optional[Path] = None

    # Output
    eval_output_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("NIGHTSHIFT_EVAL_OUTPUT", "~/.jarvis/training/eval")
        ).expanduser()
    )


@dataclass
class OrchestrationConfig(BaseConfig):
    """Configuration for pipeline orchestration."""

    # Scheduling
    schedule_cron: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_CRON", "0 2 * * 0")
    )
    timezone: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_TIMEZONE", "America/Los_Angeles")
    )

    # State management
    state_file: Path = field(
        default_factory=lambda: Path(
            os.getenv("NIGHTSHIFT_STATE_FILE", "~/.jarvis/training/pipeline_state.json")
        ).expanduser()
    )

    # Recovery settings
    max_retries: int = 3
    retry_delay_seconds: int = 300
    resume_on_failure: bool = True

    # Notifications
    slack_webhook_url: Optional[str] = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_SLACK_WEBHOOK")
    )
    email_recipients: List[str] = field(default_factory=list)
    email_smtp_host: Optional[str] = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_SMTP_HOST")
    )

    # Alerts
    alert_on_success: bool = True
    alert_on_failure: bool = True
    alert_on_gatekeeper_fail: bool = True

    # GCS artifact storage
    gcs_bucket: Optional[str] = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_GCS_BUCKET")
    )
    gcs_prefix: str = "nightshift/models"

    # Model registry
    model_registry_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("NIGHTSHIFT_REGISTRY", "~/.jarvis/models/registry.json")
        ).expanduser()
    )
    max_model_versions: int = 5


@dataclass
class NightShiftConfig(BaseConfig):
    """
    Master configuration combining all Night Shift components.
    """

    # Component configs
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)

    # Global settings
    run_id: Optional[str] = None
    dry_run: bool = False
    verbose: bool = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_VERBOSE", "false").lower() == "true"
    )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "NightShiftConfig":
        """Load full config from YAML file."""
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Parse nested configs
        ingestion = IngestionConfig.from_dict(data.get("ingestion", {}))
        training = TrainingConfig.from_dict(data.get("training", {}))
        distillation = DistillationConfig.from_dict(data.get("distillation", {}))
        quantization = QuantizationConfig.from_dict(data.get("quantization", {}))
        eval_config = EvalConfig.from_dict(data.get("eval", {}))
        orchestration = OrchestrationConfig.from_dict(data.get("orchestration", {}))

        # Global settings
        global_settings = {
            k: v for k, v in data.items()
            if k not in ("ingestion", "training", "distillation", "quantization", "eval", "orchestration")
        }
        global_settings = _interpolate_env_vars(global_settings)

        return cls(
            ingestion=ingestion,
            training=training,
            distillation=distillation,
            quantization=quantization,
            eval=eval_config,
            orchestration=orchestration,
            **global_settings,
        )

    @classmethod
    def from_yaml_dir(cls, config_dir: Union[str, Path]) -> "NightShiftConfig":
        """Load config from directory of YAML files."""
        config_dir = Path(config_dir)

        ingestion = IngestionConfig()
        training = TrainingConfig()
        distillation = DistillationConfig()
        quantization = QuantizationConfig()
        eval_config = EvalConfig()
        orchestration = OrchestrationConfig()

        if (config_dir / "ingestion.yaml").exists():
            ingestion = IngestionConfig.from_yaml(config_dir / "ingestion.yaml")
        if (config_dir / "training.yaml").exists():
            training = TrainingConfig.from_yaml(config_dir / "training.yaml")
        if (config_dir / "distillation.yaml").exists():
            distillation = DistillationConfig.from_yaml(config_dir / "distillation.yaml")
        if (config_dir / "quantization.yaml").exists():
            quantization = QuantizationConfig.from_yaml(config_dir / "quantization.yaml")
        if (config_dir / "eval.yaml").exists():
            eval_config = EvalConfig.from_yaml(config_dir / "eval.yaml")
        if (config_dir / "orchestration.yaml").exists():
            orchestration = OrchestrationConfig.from_yaml(config_dir / "orchestration.yaml")

        return cls(
            ingestion=ingestion,
            training=training,
            distillation=distillation,
            quantization=quantization,
            eval=eval_config,
            orchestration=orchestration,
        )


async def load_config(
    path: Optional[Union[str, Path]] = None,
    reload: bool = False,
) -> NightShiftConfig:
    """
    Load or get cached configuration.

    Args:
        path: Path to config file or directory. If None, uses defaults.
        reload: Force reload even if cached.

    Returns:
        NightShiftConfig instance.
    """
    global _config_instance

    if _config_instance is not None and not reload:
        return _config_instance

    async with _config_lock:
        if _config_instance is not None and not reload:
            return _config_instance

        if path is None:
            # Use defaults with env var overrides
            _config_instance = NightShiftConfig()
        elif Path(path).is_dir():
            _config_instance = NightShiftConfig.from_yaml_dir(path)
        else:
            _config_instance = NightShiftConfig.from_yaml(path)

        logger.info(f"Configuration loaded: {_config_instance.training.base_model}")
        return _config_instance


def get_config() -> Optional[NightShiftConfig]:
    """Get cached config synchronously. Returns None if not loaded."""
    return _config_instance
