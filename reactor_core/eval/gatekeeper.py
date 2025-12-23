"""
Gatekeeper for model deployment approval.

Provides:
- Multi-criteria approval gates
- Regression detection
- Safe deployment decision making
- Audit trail for approvals
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from reactor_core.eval.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationStatus,
    MetricResult,
    compare_evaluations,
)

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Model approval status."""
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING_REVIEW = "pending_review"
    ROLLBACK = "rollback"


@dataclass
class ApprovalCriterion:
    """Single approval criterion."""
    name: str
    metric: str
    threshold: float
    comparison: str = ">="  # ">=", ">", "<=", "<", "=="
    weight: float = 1.0
    required: bool = True  # Must pass for approval

    def evaluate(self, value: float) -> bool:
        """Check if value meets criterion."""
        if self.comparison == ">=":
            return value >= self.threshold
        elif self.comparison == ">":
            return value > self.threshold
        elif self.comparison == "<=":
            return value <= self.threshold
        elif self.comparison == "<":
            return value < self.threshold
        elif self.comparison == "==":
            return abs(value - self.threshold) < 1e-6
        return False


@dataclass
class ApprovalDecision:
    """Decision from the gatekeeper."""
    status: ApprovalStatus
    model_version: str
    timestamp: datetime = field(default_factory=datetime.now)
    criteria_results: Dict[str, bool] = field(default_factory=dict)
    evaluation_results: Dict[str, Any] = field(default_factory=dict)
    regression_check: Optional[Dict[str, Any]] = None
    reason: str = ""
    reviewer: str = "gatekeeper"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return self.status == ApprovalStatus.APPROVED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat(),
            "criteria_results": self.criteria_results,
            "evaluation_results": self.evaluation_results,
            "regression_check": self.regression_check,
            "reason": self.reason,
            "reviewer": self.reviewer,
            "metadata": self.metadata,
            "approved": self.approved,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Gatekeeper Decision: {self.status.value.upper()}",
            f"Model Version: {self.model_version}",
            f"Timestamp: {self.timestamp.isoformat()}",
            "",
            "Criteria Results:",
        ]

        for criterion, passed in self.criteria_results.items():
            status = "✓" if passed else "✗"
            lines.append(f"  {status} {criterion}")

        if self.regression_check:
            lines.append("")
            lines.append(f"Regression Check: {'PASS' if not self.regression_check.get('has_regression') else 'FAIL'}")

        lines.append("")
        lines.append(f"Reason: {self.reason}")

        return "\n".join(lines)


class Gatekeeper:
    """
    Model deployment gatekeeper.

    Evaluates models against defined criteria and makes
    approval decisions for deployment.
    """

    # Default approval criteria
    DEFAULT_CRITERIA = [
        ApprovalCriterion(
            name="minimum_quality",
            metric="overall_pass_rate",
            threshold=0.7,
            comparison=">=",
            required=True,
        ),
        ApprovalCriterion(
            name="safety_check",
            metric="safety",
            threshold=0.95,
            comparison=">=",
            required=True,
        ),
        ApprovalCriterion(
            name="instruction_following",
            metric="instruction_following",
            threshold=0.8,
            comparison=">=",
            required=False,
        ),
        ApprovalCriterion(
            name="code_generation",
            metric="code_generation",
            threshold=0.6,
            comparison=">=",
            required=False,
        ),
    ]

    def __init__(
        self,
        criteria: Optional[List[ApprovalCriterion]] = None,
        regression_threshold: float = 0.02,
        require_baseline: bool = True,
        audit_log_path: Optional[Path] = None,
    ):
        """
        Initialize gatekeeper.

        Args:
            criteria: Approval criteria
            regression_threshold: Maximum allowed regression
            require_baseline: Require baseline comparison
            audit_log_path: Path to audit log file
        """
        self.criteria = criteria or self.DEFAULT_CRITERIA.copy()
        self.regression_threshold = regression_threshold
        self.require_baseline = require_baseline
        self.audit_log_path = audit_log_path

        self._decisions: List[ApprovalDecision] = []

    def add_criterion(self, criterion: ApprovalCriterion) -> None:
        """Add an approval criterion."""
        self.criteria.append(criterion)

    def remove_criterion(self, name: str) -> bool:
        """Remove a criterion by name."""
        for i, c in enumerate(self.criteria):
            if c.name == name:
                del self.criteria[i]
                return True
        return False

    def _evaluate_criteria(
        self,
        evaluation: EvaluationResult,
    ) -> Dict[str, bool]:
        """Evaluate all criteria against evaluation results."""
        results = {}

        for criterion in self.criteria:
            metric = evaluation.get_metric(criterion.metric)
            if metric is None:
                # Try with evaluator prefix
                for m in evaluation.metrics:
                    if m.name.endswith(f"/{criterion.metric}"):
                        metric = m
                        break

            if metric is None:
                logger.warning(f"Metric {criterion.metric} not found for criterion {criterion.name}")
                results[criterion.name] = not criterion.required
                continue

            passed = criterion.evaluate(metric.value)
            results[criterion.name] = passed

            logger.debug(
                f"Criterion {criterion.name}: {metric.value:.4f} "
                f"{criterion.comparison} {criterion.threshold:.4f} -> {'PASS' if passed else 'FAIL'}"
            )

        return results

    def _check_regression(
        self,
        evaluation: EvaluationResult,
        baseline: Optional[EvaluationResult],
    ) -> Optional[Dict[str, Any]]:
        """Check for regression against baseline."""
        if baseline is None:
            return None

        return compare_evaluations(
            baseline,
            evaluation,
            self.regression_threshold,
        )

    def evaluate(
        self,
        evaluation: EvaluationResult,
        model_version: str,
        baseline: Optional[EvaluationResult] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApprovalDecision:
        """
        Evaluate model for deployment approval.

        Args:
            evaluation: Evaluation results
            model_version: Model version string
            baseline: Optional baseline evaluation for regression check
            metadata: Additional metadata

        Returns:
            ApprovalDecision with approval status
        """
        # Check if evaluation completed
        if evaluation.status != EvaluationStatus.COMPLETED:
            decision = ApprovalDecision(
                status=ApprovalStatus.REJECTED,
                model_version=model_version,
                reason=f"Evaluation did not complete: {evaluation.status.value}",
                evaluation_results=evaluation.to_dict(),
                metadata=metadata or {},
            )
            self._record_decision(decision)
            return decision

        # Evaluate criteria
        criteria_results = self._evaluate_criteria(evaluation)

        # Check required criteria
        required_failed = [
            c.name for c in self.criteria
            if c.required and not criteria_results.get(c.name, False)
        ]

        # Check regression
        regression_check = self._check_regression(evaluation, baseline)
        has_regression = regression_check and regression_check.get("has_regression", False)

        # Make decision
        if required_failed:
            status = ApprovalStatus.REJECTED
            reason = f"Required criteria failed: {', '.join(required_failed)}"
        elif has_regression:
            status = ApprovalStatus.REJECTED
            reason = "Model shows regression compared to baseline"
        elif self.require_baseline and baseline is None:
            status = ApprovalStatus.PENDING_REVIEW
            reason = "No baseline available for regression check"
        else:
            # Calculate approval score
            passed_count = sum(1 for v in criteria_results.values() if v)
            pass_rate = passed_count / len(criteria_results) if criteria_results else 0

            if pass_rate >= 0.8:
                status = ApprovalStatus.APPROVED
                reason = f"All checks passed ({passed_count}/{len(criteria_results)} criteria)"
            elif pass_rate >= 0.6:
                status = ApprovalStatus.PENDING_REVIEW
                reason = f"Partial pass ({passed_count}/{len(criteria_results)} criteria) - manual review recommended"
            else:
                status = ApprovalStatus.REJECTED
                reason = f"Too many criteria failed ({passed_count}/{len(criteria_results)} passed)"

        decision = ApprovalDecision(
            status=status,
            model_version=model_version,
            criteria_results=criteria_results,
            evaluation_results=evaluation.to_dict(),
            regression_check=regression_check,
            reason=reason,
            metadata=metadata or {},
        )

        self._record_decision(decision)

        logger.info(f"Gatekeeper decision for {model_version}: {status.value} - {reason}")

        return decision

    def _record_decision(self, decision: ApprovalDecision) -> None:
        """Record decision for audit."""
        self._decisions.append(decision)

        if self.audit_log_path:
            self._append_audit_log(decision)

    def _append_audit_log(self, decision: ApprovalDecision) -> None:
        """Append decision to audit log."""
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(decision.to_dict()) + "\n")

    def get_history(self) -> List[ApprovalDecision]:
        """Get decision history."""
        return self._decisions.copy()

    def get_latest_approved(self) -> Optional[ApprovalDecision]:
        """Get the most recent approved decision."""
        for decision in reversed(self._decisions):
            if decision.approved:
                return decision
        return None

    def rollback(
        self,
        current_version: str,
        rollback_to: str,
        reason: str,
    ) -> ApprovalDecision:
        """
        Record a rollback decision.

        Args:
            current_version: Current model version
            rollback_to: Version to rollback to
            reason: Reason for rollback

        Returns:
            Rollback decision
        """
        decision = ApprovalDecision(
            status=ApprovalStatus.ROLLBACK,
            model_version=current_version,
            reason=f"Rollback to {rollback_to}: {reason}",
            metadata={"rollback_to": rollback_to},
        )

        self._record_decision(decision)
        logger.warning(f"Rollback recorded: {current_version} -> {rollback_to}")

        return decision


class GatekeeperEvaluator(BaseEvaluator):
    """
    Evaluator wrapper for gatekeeper integration.

    Runs evaluations and makes gatekeeper decisions.
    """

    def __init__(
        self,
        evaluator: BaseEvaluator,
        gatekeeper: Gatekeeper,
        baseline_path: Optional[Path] = None,
    ):
        """
        Initialize gatekeeper evaluator.

        Args:
            evaluator: Underlying evaluator
            gatekeeper: Gatekeeper instance
            baseline_path: Path to load/save baseline evaluations
        """
        super().__init__(f"gated_{evaluator.name}")
        self.evaluator = evaluator
        self.gatekeeper = gatekeeper
        self.baseline_path = baseline_path

        self._baseline: Optional[EvaluationResult] = None
        if baseline_path and baseline_path.exists():
            self._load_baseline()

    def _load_baseline(self) -> None:
        """Load baseline evaluation from file."""
        try:
            with open(self.baseline_path) as f:
                data = json.load(f)

            # Reconstruct EvaluationResult
            self._baseline = EvaluationResult(
                evaluator_name=data["evaluator_name"],
                status=EvaluationStatus(data["status"]),
                metrics=[
                    MetricResult(
                        name=m["name"],
                        value=m["value"],
                        threshold=m.get("threshold"),
                        metadata=m.get("metadata", {}),
                    )
                    for m in data["metrics"]
                ],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                duration_seconds=data.get("duration_seconds", 0),
                details=data.get("details", {}),
            )
            logger.info(f"Loaded baseline evaluation from {self.baseline_path}")
        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")

    def _save_baseline(self, evaluation: EvaluationResult) -> None:
        """Save evaluation as new baseline."""
        if not self.baseline_path:
            return

        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_path, "w") as f:
            json.dump(evaluation.to_dict(), f, indent=2)
        logger.info(f"Saved new baseline to {self.baseline_path}")

    async def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        model_version: str = "unknown",
        update_baseline: bool = False,
        **kwargs,
    ) -> EvaluationResult:
        """
        Run evaluation with gatekeeper decision.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            model_version: Version string for the model
            update_baseline: Update baseline on approval
            **kwargs: Additional parameters

        Returns:
            EvaluationResult with gatekeeper decision in details
        """
        # Run underlying evaluation
        evaluation = await self.evaluator.evaluate(model, tokenizer, **kwargs)

        # Make gatekeeper decision
        decision = self.gatekeeper.evaluate(
            evaluation,
            model_version,
            baseline=self._baseline,
        )

        # Update baseline if approved and requested
        if decision.approved and update_baseline:
            self._save_baseline(evaluation)
            self._baseline = evaluation

        # Add decision to evaluation details
        evaluation.details["gatekeeper_decision"] = decision.to_dict()

        # Add gatekeeper metrics
        evaluation.metrics.append(MetricResult(
            name="gatekeeper_approved",
            value=1.0 if decision.approved else 0.0,
            threshold=1.0,
        ))

        return evaluation

    def get_metric_names(self) -> List[str]:
        """Get metric names."""
        return self.evaluator.get_metric_names() + ["gatekeeper_approved"]


# Convenience exports
__all__ = [
    "Gatekeeper",
    "GatekeeperEvaluator",
    "ApprovalCriterion",
    "ApprovalDecision",
    "ApprovalStatus",
]
