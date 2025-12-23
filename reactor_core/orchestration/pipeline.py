"""
Night Shift Pipeline orchestration.

Provides:
- End-to-end training pipeline
- Stage management and recovery
- Checkpoint-based resumption
- Progress tracking
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages."""
    IDLE = "idle"
    INGESTING = "ingesting"
    FORMATTING = "formatting"
    DISTILLING = "distilling"
    TRAINING = "training"
    EVALUATING = "evaluating"
    QUANTIZING = "quantizing"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineState:
    """Current state of the pipeline."""
    run_id: str
    stage: PipelineStage
    started_at: datetime
    last_updated: datetime = field(default_factory=datetime.now)
    last_completed_stage: Optional[PipelineStage] = None

    # Progress counters
    ingestion_count: int = 0
    formatted_count: int = 0
    distilled_count: int = 0

    # Training state
    training_checkpoint: Optional[str] = None
    training_step: int = 0

    # Evaluation state
    eval_metrics: Dict[str, float] = field(default_factory=dict)
    gatekeeper_passed: bool = False

    # Output artifacts
    model_path: Optional[str] = None
    adapter_path: Optional[str] = None
    quantized_path: Optional[str] = None

    # Error tracking
    error: Optional[str] = None
    error_stage: Optional[PipelineStage] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage.value,
            "started_at": self.started_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "last_completed_stage": (
                self.last_completed_stage.value
                if self.last_completed_stage else None
            ),
            "ingestion_count": self.ingestion_count,
            "formatted_count": self.formatted_count,
            "distilled_count": self.distilled_count,
            "training_checkpoint": self.training_checkpoint,
            "training_step": self.training_step,
            "eval_metrics": self.eval_metrics,
            "gatekeeper_passed": self.gatekeeper_passed,
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "quantized_path": self.quantized_path,
            "error": self.error,
            "error_stage": self.error_stage.value if self.error_stage else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineState":
        return cls(
            run_id=data["run_id"],
            stage=PipelineStage(data["stage"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            last_completed_stage=(
                PipelineStage(data["last_completed_stage"])
                if data.get("last_completed_stage") else None
            ),
            ingestion_count=data.get("ingestion_count", 0),
            formatted_count=data.get("formatted_count", 0),
            distilled_count=data.get("distilled_count", 0),
            training_checkpoint=data.get("training_checkpoint"),
            training_step=data.get("training_step", 0),
            eval_metrics=data.get("eval_metrics", {}),
            gatekeeper_passed=data.get("gatekeeper_passed", False),
            model_path=data.get("model_path"),
            adapter_path=data.get("adapter_path"),
            quantized_path=data.get("quantized_path"),
            error=data.get("error"),
            error_stage=(
                PipelineStage(data["error_stage"])
                if data.get("error_stage") else None
            ),
        )

    def update_stage(self, stage: PipelineStage) -> None:
        """Update current stage."""
        if self.stage != PipelineStage.FAILED:
            self.last_completed_stage = self.stage
        self.stage = stage
        self.last_updated = datetime.now()

    def set_error(self, error: str) -> None:
        """Set error state."""
        self.error = error
        self.error_stage = self.stage
        self.stage = PipelineStage.FAILED
        self.last_updated = datetime.now()


@dataclass
class PipelineResult:
    """Final result of pipeline execution."""
    success: bool
    run_id: str
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    final_state: PipelineState
    artifacts: Dict[str, str]
    metrics: Dict[str, Any]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "final_state": self.final_state.to_dict(),
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "error": self.error,
        }

    def summary(self) -> str:
        lines = [
            f"Pipeline Run: {self.run_id}",
            f"Status: {'SUCCESS' if self.success else 'FAILED'}",
            f"Duration: {self.duration_seconds / 60:.1f} minutes",
            "",
            "Stages Completed:",
        ]

        if self.final_state.last_completed_stage:
            stages = list(PipelineStage)
            idx = stages.index(self.final_state.last_completed_stage)
            for s in stages[1:idx+1]:  # Skip IDLE
                lines.append(f"  ✓ {s.value}")

        if self.error:
            lines.append(f"\nError: {self.error}")

        if self.artifacts:
            lines.append("\nArtifacts:")
            for name, path in self.artifacts.items():
                lines.append(f"  {name}: {path}")

        return "\n".join(lines)


@dataclass
class PipelineConfig:
    """Configuration for the Night Shift pipeline."""
    # Directories
    work_dir: Path = field(
        default_factory=lambda: Path(os.getenv(
            "NIGHTSHIFT_WORK_DIR",
            Path.home() / ".jarvis" / "nightshift"
        ))
    )
    log_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    # Stage configuration
    skip_stages: List[PipelineStage] = field(default_factory=list)
    stop_after: Optional[PipelineStage] = None

    # Ingestion
    log_sources: List[Path] = field(default_factory=list)
    min_examples: int = 100

    # Distillation
    enable_distillation: bool = True
    distillation_budget: float = 10.0

    # Training
    base_model: str = field(
        default_factory=lambda: os.getenv(
            "NIGHTSHIFT_BASE_MODEL",
            "meta-llama/Llama-3.2-3B"
        )
    )
    num_epochs: int = 3

    # Evaluation
    eval_threshold: float = 0.7
    require_gatekeeper: bool = True

    # Quantization
    quantization_method: str = "q4_k_m"
    skip_quantization: bool = False

    # Recovery
    state_file: Optional[Path] = None
    resume_on_error: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "work_dir": str(self.work_dir),
            "log_dir": str(self.log_dir) if self.log_dir else None,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "skip_stages": [s.value for s in self.skip_stages],
            "stop_after": self.stop_after.value if self.stop_after else None,
            "log_sources": [str(p) for p in self.log_sources],
            "min_examples": self.min_examples,
            "enable_distillation": self.enable_distillation,
            "distillation_budget": self.distillation_budget,
            "base_model": self.base_model,
            "num_epochs": self.num_epochs,
            "eval_threshold": self.eval_threshold,
            "require_gatekeeper": self.require_gatekeeper,
            "quantization_method": self.quantization_method,
            "skip_quantization": self.skip_quantization,
            "resume_on_error": self.resume_on_error,
        }


class NightShiftPipeline:
    """
    Night Shift autonomous training pipeline.

    Orchestrates the full training cycle:
    1. Ingestion - Parse JARVIS logs
    2. Formatting - Convert to training format
    3. Distillation - Improve examples with teacher model
    4. Training - Fine-tune with LoRA
    5. Evaluation - Run benchmarks
    6. Quantization - Convert to GGUF
    7. Deployment - Update model registry
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()

        # Initialize directories
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        if self.config.log_dir:
            self.config.log_dir.mkdir(parents=True, exist_ok=True)
        if self.config.output_dir:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # State management
        self._state: Optional[PipelineState] = None
        self._state_file = self.config.state_file or (
            self.config.work_dir / "pipeline_state.json"
        )

        # Stage handlers
        self._stage_handlers: Dict[PipelineStage, Callable] = {}

        # Callbacks
        self._progress_callback: Optional[Callable[[PipelineState], None]] = None
        self._error_callback: Optional[Callable[[Exception, PipelineStage], None]] = None

    def set_progress_callback(
        self,
        callback: Callable[[PipelineState], None],
    ) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def set_error_callback(
        self,
        callback: Callable[[Exception, PipelineStage], None],
    ) -> None:
        """Set callback for error handling."""
        self._error_callback = callback

    def register_stage_handler(
        self,
        stage: PipelineStage,
        handler: Callable,
    ) -> None:
        """Register a custom handler for a stage."""
        self._stage_handlers[stage] = handler

    def _save_state(self) -> None:
        """Save pipeline state to file."""
        if self._state:
            with open(self._state_file, "w") as f:
                json.dump(self._state.to_dict(), f, indent=2)

    def _load_state(self) -> Optional[PipelineState]:
        """Load pipeline state from file."""
        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    data = json.load(f)
                return PipelineState.from_dict(data)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return None

    def _update_stage(self, stage: PipelineStage) -> None:
        """Update current stage and notify."""
        if self._state:
            self._state.update_stage(stage)
            self._save_state()

            if self._progress_callback:
                self._progress_callback(self._state)

            logger.info(f"Pipeline stage: {stage.value}")

    async def _run_stage(
        self,
        stage: PipelineStage,
        handler: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Run a pipeline stage with error handling."""
        if stage in self.config.skip_stages:
            logger.info(f"Skipping stage: {stage.value}")
            return None

        self._update_stage(stage)

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, handler, *args)

            return result

        except Exception as e:
            logger.error(f"Stage {stage.value} failed: {e}")

            if self._error_callback:
                self._error_callback(e, stage)

            if self._state:
                self._state.set_error(str(e))
                self._save_state()

            raise

    async def _run_ingestion(self) -> int:
        """Run ingestion stage."""
        logger.info("Starting ingestion stage...")

        # Import ingestion modules
        from reactor_core.ingestion import BatchIngestionProcessor

        # Process log sources
        processor = BatchIngestionProcessor()
        total = 0

        for source in self.config.log_sources:
            if source.exists():
                count = await processor.process_directory(source)
                total += count

        if self._state:
            self._state.ingestion_count = total

        logger.info(f"Ingested {total} raw interactions")
        return total

    async def _run_formatting(self) -> int:
        """Run formatting stage."""
        logger.info("Starting formatting stage...")

        # Placeholder - actual implementation would use formatting module
        formatted_count = self._state.ingestion_count if self._state else 0

        if self._state:
            self._state.formatted_count = formatted_count

        logger.info(f"Formatted {formatted_count} examples")
        return formatted_count

    async def _run_distillation(self) -> int:
        """Run distillation stage."""
        if not self.config.enable_distillation:
            logger.info("Distillation disabled, skipping...")
            return 0

        logger.info("Starting distillation stage...")

        # Placeholder - actual implementation would use distillation module
        distilled_count = 0

        if self._state:
            self._state.distilled_count = distilled_count

        logger.info(f"Distilled {distilled_count} examples")
        return distilled_count

    async def _run_training(self) -> Dict[str, Any]:
        """Run training stage."""
        logger.info("Starting training stage...")

        # Placeholder - actual implementation would use training module
        training_result = {
            "model_path": str(self.config.work_dir / "model"),
            "adapter_path": str(self.config.work_dir / "adapter"),
            "final_loss": 0.5,
        }

        if self._state:
            self._state.model_path = training_result["model_path"]
            self._state.adapter_path = training_result["adapter_path"]

        logger.info(f"Training complete: {training_result}")
        return training_result

    async def _run_evaluation(self) -> Dict[str, float]:
        """Run evaluation stage."""
        logger.info("Starting evaluation stage...")

        # Placeholder - actual implementation would use eval module
        metrics = {
            "overall_score": 0.85,
            "safety": 0.98,
            "instruction_following": 0.82,
        }

        gatekeeper_passed = metrics["overall_score"] >= self.config.eval_threshold

        if self._state:
            self._state.eval_metrics = metrics
            self._state.gatekeeper_passed = gatekeeper_passed

        logger.info(f"Evaluation complete: {metrics}, gatekeeper: {gatekeeper_passed}")
        return metrics

    async def _run_quantization(self) -> str:
        """Run quantization stage."""
        if self.config.skip_quantization:
            logger.info("Quantization disabled, skipping...")
            return ""

        logger.info("Starting quantization stage...")

        # Placeholder - actual implementation would use quantization module
        output_path = str(
            self.config.work_dir / f"model-{self.config.quantization_method}.gguf"
        )

        if self._state:
            self._state.quantized_path = output_path

        logger.info(f"Quantization complete: {output_path}")
        return output_path

    async def _run_deployment(self) -> None:
        """Run deployment stage."""
        logger.info("Starting deployment stage...")

        # Check gatekeeper
        if self.config.require_gatekeeper:
            if not self._state or not self._state.gatekeeper_passed:
                raise RuntimeError("Gatekeeper did not approve deployment")

        # Placeholder - actual deployment logic
        logger.info("Deployment complete")

    async def run(
        self,
        resume: bool = False,
    ) -> PipelineResult:
        """
        Run the complete pipeline.

        Args:
            resume: Resume from previous state if available

        Returns:
            PipelineResult with final status
        """
        import time
        start_time = time.time()

        # Initialize or resume state
        if resume:
            self._state = self._load_state()

        if self._state is None:
            self._state = PipelineState(
                run_id=str(uuid.uuid4())[:8],
                stage=PipelineStage.IDLE,
                started_at=datetime.now(),
            )

        run_id = self._state.run_id
        logger.info(f"Starting pipeline run: {run_id}")

        try:
            # Determine starting stage
            start_stage = self._state.stage
            if start_stage == PipelineStage.FAILED and self.config.resume_on_error:
                start_stage = self._state.error_stage or PipelineStage.IDLE

            stages = [
                (PipelineStage.INGESTING, self._run_ingestion),
                (PipelineStage.FORMATTING, self._run_formatting),
                (PipelineStage.DISTILLING, self._run_distillation),
                (PipelineStage.TRAINING, self._run_training),
                (PipelineStage.EVALUATING, self._run_evaluation),
                (PipelineStage.QUANTIZING, self._run_quantization),
                (PipelineStage.DEPLOYING, self._run_deployment),
            ]

            # Skip completed stages on resume
            if resume and start_stage != PipelineStage.IDLE:
                stage_order = [s for s, _ in stages]
                if start_stage in stage_order:
                    idx = stage_order.index(start_stage)
                    stages = stages[idx:]

            # Run stages
            for stage, handler in stages:
                # Check for custom handler
                if stage in self._stage_handlers:
                    handler = self._stage_handlers[stage]

                await self._run_stage(stage, handler)

                # Check stop condition
                if self.config.stop_after == stage:
                    logger.info(f"Stopping after {stage.value} as configured")
                    break

            # Mark completed
            self._update_stage(PipelineStage.COMPLETED)

            duration = time.time() - start_time

            return PipelineResult(
                success=True,
                run_id=run_id,
                started_at=self._state.started_at,
                completed_at=datetime.now(),
                duration_seconds=duration,
                final_state=self._state,
                artifacts={
                    "model": self._state.model_path or "",
                    "adapter": self._state.adapter_path or "",
                    "quantized": self._state.quantized_path or "",
                },
                metrics=self._state.eval_metrics,
            )

        except Exception as e:
            duration = time.time() - start_time

            return PipelineResult(
                success=False,
                run_id=run_id,
                started_at=self._state.started_at,
                completed_at=datetime.now(),
                duration_seconds=duration,
                final_state=self._state,
                artifacts={},
                metrics=self._state.eval_metrics if self._state else {},
                error=str(e),
            )

    def get_state(self) -> Optional[PipelineState]:
        """Get current pipeline state."""
        return self._state

    def reset(self) -> None:
        """Reset pipeline state."""
        self._state = None
        if self._state_file.exists():
            self._state_file.unlink()


# Convenience exports
__all__ = [
    "NightShiftPipeline",
    "PipelineConfig",
    "PipelineStage",
    "PipelineState",
    "PipelineResult",
]
