"""
Base evaluator interfaces and common utilities.

Provides:
- Abstract evaluator protocol
- Common metrics computation
- Evaluation result structures
- Async evaluation support
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class EvaluationStatus(Enum):
    """Status of an evaluation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class MetricResult:
    """Result for a single metric."""
    name: str
    value: float
    threshold: Optional[float] = None
    passed: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.threshold is not None and self.passed is None:
            self.passed = self.value >= self.threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    evaluator_name: str
    status: EvaluationStatus
    metrics: List[MetricResult]
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if all metrics passed."""
        if self.status != EvaluationStatus.COMPLETED:
            return False
        return all(m.passed for m in self.metrics if m.passed is not None)

    @property
    def overall_score(self) -> float:
        """Calculate overall score as mean of metrics."""
        if not self.metrics:
            return 0.0
        return sum(m.value for m in self.metrics) / len(self.metrics)

    def get_metric(self, name: str) -> Optional[MetricResult]:
        """Get a specific metric by name."""
        for m in self.metrics:
            if m.name == name:
                return m
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "status": self.status.value,
            "metrics": [m.to_dict() for m in self.metrics],
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "details": self.details,
            "passed": self.passed,
            "overall_score": self.overall_score,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Evaluation: {self.evaluator_name}",
            f"Status: {self.status.value}",
            f"Overall: {'PASSED' if self.passed else 'FAILED'} ({self.overall_score:.2%})",
            "",
            "Metrics:",
        ]
        for m in self.metrics:
            status = "✓" if m.passed else "✗" if m.passed is not None else "-"
            threshold = f" (>={m.threshold:.2%})" if m.threshold else ""
            lines.append(f"  {status} {m.name}: {m.value:.4f}{threshold}")

        if self.error:
            lines.append(f"\nError: {self.error}")

        return "\n".join(lines)


class BaseEvaluator(ABC):
    """
    Abstract base class for model evaluators.

    All evaluators should implement this interface for consistent
    evaluation across different benchmarks.
    """

    def __init__(
        self,
        name: str,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize evaluator.

        Args:
            name: Evaluator name
            thresholds: Metric name to threshold mapping
        """
        self.name = name
        self.thresholds = thresholds or {}

    @abstractmethod
    async def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        **kwargs,
    ) -> EvaluationResult:
        """
        Evaluate the model.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult with metrics
        """
        pass

    @abstractmethod
    def get_metric_names(self) -> List[str]:
        """Get list of metric names this evaluator produces."""
        pass

    def set_threshold(self, metric: str, threshold: float) -> None:
        """Set threshold for a metric."""
        self.thresholds[metric] = threshold

    def _create_metric(
        self,
        name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """Create a metric result with optional threshold."""
        return MetricResult(
            name=name,
            value=value,
            threshold=self.thresholds.get(name),
            metadata=metadata or {},
        )


class CompositeEvaluator(BaseEvaluator):
    """
    Composite evaluator that runs multiple evaluators.

    Combines results from multiple evaluators into a single result.
    """

    def __init__(
        self,
        name: str = "composite",
        evaluators: Optional[List[BaseEvaluator]] = None,
    ):
        """
        Initialize composite evaluator.

        Args:
            name: Evaluator name
            evaluators: List of evaluators to run
        """
        super().__init__(name)
        self.evaluators = evaluators or []

    def add(self, evaluator: BaseEvaluator) -> None:
        """Add an evaluator."""
        self.evaluators.append(evaluator)

    async def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        parallel: bool = False,
        **kwargs,
    ) -> EvaluationResult:
        """
        Run all evaluators.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            parallel: Run evaluators in parallel
            **kwargs: Additional parameters

        Returns:
            Combined EvaluationResult
        """
        import time
        start_time = time.time()

        all_metrics = []
        all_details = {}
        errors = []

        if parallel:
            # Run in parallel
            tasks = [
                e.evaluate(model, tokenizer, **kwargs)
                for e in self.evaluators
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Run sequentially
            results = []
            for e in self.evaluators:
                try:
                    result = await e.evaluate(model, tokenizer, **kwargs)
                    results.append(result)
                except Exception as ex:
                    results.append(ex)

        # Aggregate results
        for i, result in enumerate(results):
            evaluator_name = self.evaluators[i].name

            if isinstance(result, Exception):
                errors.append(f"{evaluator_name}: {result}")
                continue

            if result.status == EvaluationStatus.COMPLETED:
                # Prefix metrics with evaluator name
                for metric in result.metrics:
                    prefixed_metric = MetricResult(
                        name=f"{evaluator_name}/{metric.name}",
                        value=metric.value,
                        threshold=metric.threshold,
                        metadata=metric.metadata,
                    )
                    all_metrics.append(prefixed_metric)

                all_details[evaluator_name] = result.to_dict()
            else:
                errors.append(f"{evaluator_name}: {result.error}")

        duration = time.time() - start_time

        return EvaluationResult(
            evaluator_name=self.name,
            status=(
                EvaluationStatus.COMPLETED if not errors
                else EvaluationStatus.FAILED if len(errors) == len(self.evaluators)
                else EvaluationStatus.COMPLETED  # Partial success
            ),
            metrics=all_metrics,
            duration_seconds=duration,
            error="; ".join(errors) if errors else None,
            details=all_details,
        )

    def get_metric_names(self) -> List[str]:
        """Get all metric names from child evaluators."""
        names = []
        for e in self.evaluators:
            for name in e.get_metric_names():
                names.append(f"{e.name}/{name}")
        return names


def compare_evaluations(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    regression_threshold: float = 0.02,
) -> Dict[str, Any]:
    """
    Compare two evaluation results for regression.

    Args:
        baseline: Baseline evaluation
        candidate: Candidate evaluation
        regression_threshold: Maximum allowed regression

    Returns:
        Comparison results
    """
    comparisons = []
    has_regression = False

    # Match metrics by name
    baseline_metrics = {m.name: m.value for m in baseline.metrics}
    candidate_metrics = {m.name: m.value for m in candidate.metrics}

    for name in set(baseline_metrics.keys()) | set(candidate_metrics.keys()):
        base_val = baseline_metrics.get(name)
        cand_val = candidate_metrics.get(name)

        if base_val is not None and cand_val is not None:
            diff = cand_val - base_val
            pct_change = diff / base_val if base_val != 0 else 0

            is_regression = diff < -regression_threshold
            if is_regression:
                has_regression = True

            comparisons.append({
                "metric": name,
                "baseline": base_val,
                "candidate": cand_val,
                "diff": diff,
                "pct_change": pct_change,
                "is_regression": is_regression,
            })
        elif cand_val is None:
            comparisons.append({
                "metric": name,
                "baseline": base_val,
                "candidate": None,
                "diff": None,
                "pct_change": None,
                "is_regression": False,
                "note": "Metric missing in candidate",
            })

    return {
        "baseline_score": baseline.overall_score,
        "candidate_score": candidate.overall_score,
        "has_regression": has_regression,
        "comparisons": comparisons,
    }


# Convenience exports
__all__ = [
    "EvaluationStatus",
    "MetricResult",
    "EvaluationResult",
    "BaseEvaluator",
    "CompositeEvaluator",
    "compare_evaluations",
]
