"""
API cost tracking for distillation operations.

Provides:
- Per-model pricing configuration
- Usage recording and aggregation
- Cost estimation and reporting
- Budget enforcement
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelPricing:
    """Pricing information for a model."""
    model_id: str
    input_price_per_1k: float  # $ per 1K input tokens
    output_price_per_1k: float  # $ per 1K output tokens
    provider: str = "openai"

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost for given token usage."""
        input_cost = (input_tokens / 1000) * self.input_price_per_1k
        output_cost = (output_tokens / 1000) * self.output_price_per_1k
        return input_cost + output_cost


# Default pricing (as of late 2024)
DEFAULT_PRICING: Dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o": ModelPricing(
        model_id="gpt-4o",
        input_price_per_1k=0.0025,
        output_price_per_1k=0.01,
        provider="openai",
    ),
    "gpt-4o-mini": ModelPricing(
        model_id="gpt-4o-mini",
        input_price_per_1k=0.00015,
        output_price_per_1k=0.0006,
        provider="openai",
    ),
    "gpt-4-turbo": ModelPricing(
        model_id="gpt-4-turbo",
        input_price_per_1k=0.01,
        output_price_per_1k=0.03,
        provider="openai",
    ),
    "gpt-4": ModelPricing(
        model_id="gpt-4",
        input_price_per_1k=0.03,
        output_price_per_1k=0.06,
        provider="openai",
    ),
    "gpt-3.5-turbo": ModelPricing(
        model_id="gpt-3.5-turbo",
        input_price_per_1k=0.0005,
        output_price_per_1k=0.0015,
        provider="openai",
    ),
    # Anthropic
    "claude-3-opus-20240229": ModelPricing(
        model_id="claude-3-opus-20240229",
        input_price_per_1k=0.015,
        output_price_per_1k=0.075,
        provider="anthropic",
    ),
    "claude-3-5-sonnet-20241022": ModelPricing(
        model_id="claude-3-5-sonnet-20241022",
        input_price_per_1k=0.003,
        output_price_per_1k=0.015,
        provider="anthropic",
    ),
    "claude-3-haiku-20240307": ModelPricing(
        model_id="claude-3-haiku-20240307",
        input_price_per_1k=0.00025,
        output_price_per_1k=0.00125,
        provider="anthropic",
    ),
}


@dataclass
class UsageRecord:
    """Record of API usage."""
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    operation: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "operation": self.operation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UsageRecord":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            model=data["model"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            cost=data["cost"],
            operation=data.get("operation", "unknown"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CostReport:
    """Report of costs over a period."""
    start_time: datetime
    end_time: datetime
    total_cost: float
    total_input_tokens: int
    total_output_tokens: int
    total_requests: int
    by_model: Dict[str, Dict[str, Any]]
    by_operation: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_cost": self.total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_requests": self.total_requests,
            "by_model": self.by_model,
            "by_operation": self.by_operation,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Cost Report ({self.start_time.date()} to {self.end_time.date()})",
            "=" * 50,
            f"Total Cost: ${self.total_cost:.4f}",
            f"Total Requests: {self.total_requests}",
            f"Total Tokens: {self.total_input_tokens + self.total_output_tokens:,}",
            f"  - Input: {self.total_input_tokens:,}",
            f"  - Output: {self.total_output_tokens:,}",
            "",
            "By Model:",
        ]
        for model, stats in self.by_model.items():
            lines.append(f"  {model}: ${stats['cost']:.4f} ({stats['requests']} requests)")

        lines.append("")
        lines.append("By Operation:")
        for op, stats in self.by_operation.items():
            lines.append(f"  {op}: ${stats['cost']:.4f} ({stats['requests']} requests)")

        return "\n".join(lines)


class CostTracker:
    """
    Track and manage API costs for distillation operations.

    Features:
    - Real-time cost tracking
    - Budget enforcement
    - Usage reporting
    - Persistence to disk
    """

    def __init__(
        self,
        pricing: Optional[Dict[str, ModelPricing]] = None,
        budget_limit: Optional[float] = None,
        persistence_path: Optional[Path] = None,
    ):
        """
        Initialize cost tracker.

        Args:
            pricing: Model pricing dictionary (uses defaults if not provided)
            budget_limit: Optional budget limit in dollars
            persistence_path: Optional path to persist usage records
        """
        self.pricing = pricing or DEFAULT_PRICING.copy()
        self.budget_limit = budget_limit or float(
            os.getenv("NIGHTSHIFT_BUDGET_LIMIT", "100.0")
        )
        self.persistence_path = persistence_path

        self._records: List[UsageRecord] = []
        self._total_cost = 0.0

        # Load existing records if available
        if self.persistence_path and self.persistence_path.exists():
            self._load_records()

    def add_pricing(self, pricing: ModelPricing) -> None:
        """Add or update model pricing."""
        self.pricing[pricing.model_id] = pricing

    def get_pricing(self, model: str) -> Optional[ModelPricing]:
        """Get pricing for a model."""
        # Direct match
        if model in self.pricing:
            return self.pricing[model]

        # Partial match
        for model_id, pricing in self.pricing.items():
            if model_id in model or model in model_id:
                return pricing

        return None

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """
        Record API usage.

        Args:
            model: Model ID
            input_tokens: Input tokens used
            output_tokens: Output tokens used
            operation: Type of operation
            metadata: Additional metadata

        Returns:
            UsageRecord for this usage
        """
        pricing = self.get_pricing(model)
        if pricing:
            cost = pricing.calculate_cost(input_tokens, output_tokens)
        else:
            # Default estimate
            logger.warning(f"No pricing for model {model}, using estimate")
            cost = (input_tokens + output_tokens) * 0.00001

        record = UsageRecord(
            timestamp=datetime.now(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            operation=operation,
            metadata=metadata or {},
        )

        self._records.append(record)
        self._total_cost += cost

        # Persist
        if self.persistence_path:
            self._save_records()

        logger.debug(
            f"Recorded usage: {model} - {input_tokens}in/{output_tokens}out "
            f"- ${cost:.4f}"
        )

        return record

    def check_budget(self, estimated_cost: float = 0.0) -> bool:
        """
        Check if budget allows operation.

        Args:
            estimated_cost: Estimated cost of upcoming operation

        Returns:
            True if within budget
        """
        if self.budget_limit is None:
            return True

        return (self._total_cost + estimated_cost) <= self.budget_limit

    def get_remaining_budget(self) -> float:
        """Get remaining budget."""
        if self.budget_limit is None:
            return float("inf")
        return max(0, self.budget_limit - self._total_cost)

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Estimate cost before making request.

        Args:
            model: Model ID
            input_tokens: Estimated input tokens
            output_tokens: Estimated output tokens

        Returns:
            Estimated cost
        """
        pricing = self.get_pricing(model)
        if pricing:
            return pricing.calculate_cost(input_tokens, output_tokens)
        return (input_tokens + output_tokens) * 0.00001

    def get_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> CostReport:
        """
        Generate cost report for a period.

        Args:
            start_time: Start of period (default: all time)
            end_time: End of period (default: now)

        Returns:
            CostReport with aggregated statistics
        """
        end_time = end_time or datetime.now()
        if start_time is None:
            if self._records:
                start_time = min(r.timestamp for r in self._records)
            else:
                start_time = end_time

        # Filter records
        filtered = [
            r for r in self._records
            if start_time <= r.timestamp <= end_time
        ]

        # Aggregate
        total_cost = sum(r.cost for r in filtered)
        total_input = sum(r.input_tokens for r in filtered)
        total_output = sum(r.output_tokens for r in filtered)

        # By model
        by_model: Dict[str, Dict[str, Any]] = {}
        for r in filtered:
            if r.model not in by_model:
                by_model[r.model] = {
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "requests": 0,
                }
            by_model[r.model]["cost"] += r.cost
            by_model[r.model]["input_tokens"] += r.input_tokens
            by_model[r.model]["output_tokens"] += r.output_tokens
            by_model[r.model]["requests"] += 1

        # By operation
        by_operation: Dict[str, Dict[str, Any]] = {}
        for r in filtered:
            if r.operation not in by_operation:
                by_operation[r.operation] = {
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "requests": 0,
                }
            by_operation[r.operation]["cost"] += r.cost
            by_operation[r.operation]["input_tokens"] += r.input_tokens
            by_operation[r.operation]["output_tokens"] += r.output_tokens
            by_operation[r.operation]["requests"] += 1

        return CostReport(
            start_time=start_time,
            end_time=end_time,
            total_cost=total_cost,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_requests=len(filtered),
            by_model=by_model,
            by_operation=by_operation,
        )

    def get_daily_report(self, date: Optional[datetime] = None) -> CostReport:
        """Get report for a specific day."""
        date = date or datetime.now()
        start = datetime(date.year, date.month, date.day)
        end = start + timedelta(days=1)
        return self.get_report(start, end)

    def get_current_cost(self) -> float:
        """Get current total cost."""
        return self._total_cost

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        if not self._records:
            return {
                "total_cost": 0.0,
                "total_requests": 0,
                "total_tokens": 0,
                "budget_remaining": self.get_remaining_budget(),
            }

        return {
            "total_cost": self._total_cost,
            "total_requests": len(self._records),
            "total_input_tokens": sum(r.input_tokens for r in self._records),
            "total_output_tokens": sum(r.output_tokens for r in self._records),
            "avg_cost_per_request": self._total_cost / len(self._records),
            "budget_remaining": self.get_remaining_budget(),
            "budget_used_percent": (
                (self._total_cost / self.budget_limit * 100)
                if self.budget_limit else 0
            ),
        }

    def _save_records(self) -> None:
        """Save records to disk."""
        if not self.persistence_path:
            return

        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "records": [r.to_dict() for r in self._records],
            "total_cost": self._total_cost,
        }
        with open(self.persistence_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_records(self) -> None:
        """Load records from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            with open(self.persistence_path) as f:
                data = json.load(f)

            self._records = [
                UsageRecord.from_dict(r) for r in data.get("records", [])
            ]
            self._total_cost = data.get("total_cost", 0.0)

            logger.info(
                f"Loaded {len(self._records)} usage records, "
                f"total cost: ${self._total_cost:.4f}"
            )
        except Exception as e:
            logger.error(f"Failed to load usage records: {e}")

    def reset(self) -> None:
        """Reset all tracking data."""
        self._records.clear()
        self._total_cost = 0.0
        if self.persistence_path and self.persistence_path.exists():
            self.persistence_path.unlink()


class BudgetEnforcer:
    """
    Enforce budget limits during distillation.

    Raises exceptions when budget is exceeded.
    """

    def __init__(
        self,
        cost_tracker: CostTracker,
        soft_limit_percent: float = 80.0,
        hard_limit_percent: float = 100.0,
    ):
        """
        Initialize budget enforcer.

        Args:
            cost_tracker: Cost tracker instance
            soft_limit_percent: Warn at this percentage of budget
            hard_limit_percent: Error at this percentage of budget
        """
        self.cost_tracker = cost_tracker
        self.soft_limit_percent = soft_limit_percent
        self.hard_limit_percent = hard_limit_percent

        self._warned = False

    def check(self, estimated_cost: float = 0.0) -> None:
        """
        Check budget and enforce limits.

        Args:
            estimated_cost: Estimated cost of next operation

        Raises:
            BudgetExceeded: If hard limit would be exceeded
        """
        if self.cost_tracker.budget_limit is None:
            return

        current = self.cost_tracker.get_current_cost()
        budget = self.cost_tracker.budget_limit

        current_percent = (current / budget) * 100
        projected_percent = ((current + estimated_cost) / budget) * 100

        # Hard limit check
        if projected_percent >= self.hard_limit_percent:
            raise BudgetExceeded(
                f"Budget limit exceeded: ${current:.4f} / ${budget:.4f} "
                f"({current_percent:.1f}%)"
            )

        # Soft limit warning
        if current_percent >= self.soft_limit_percent and not self._warned:
            logger.warning(
                f"Approaching budget limit: ${current:.4f} / ${budget:.4f} "
                f"({current_percent:.1f}%)"
            )
            self._warned = True


class BudgetExceeded(Exception):
    """Raised when budget limit is exceeded."""
    pass


# Convenience exports
__all__ = [
    "CostTracker",
    "UsageRecord",
    "CostReport",
    "ModelPricing",
    "BudgetEnforcer",
    "BudgetExceeded",
    "DEFAULT_PRICING",
]
