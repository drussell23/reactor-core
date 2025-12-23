"""
Model evaluation modules for Night Shift Training Engine.

Provides:
- Base evaluation framework
- JARVIS-specific test suite
- Gatekeeper approval system
- Regression detection
"""

from reactor_core.eval.base_evaluator import (
    EvaluationStatus,
    MetricResult,
    EvaluationResult,
    BaseEvaluator,
    CompositeEvaluator,
    compare_evaluations,
)

from reactor_core.eval.jarvis_eval import (
    JARVISEvaluator,
    TestCase,
    TestResult,
)

from reactor_core.eval.gatekeeper import (
    Gatekeeper,
    GatekeeperEvaluator,
    ApprovalCriterion,
    ApprovalDecision,
    ApprovalStatus,
)

__all__ = [
    # Base
    "EvaluationStatus",
    "MetricResult",
    "EvaluationResult",
    "BaseEvaluator",
    "CompositeEvaluator",
    "compare_evaluations",
    # JARVIS Eval
    "JARVISEvaluator",
    "TestCase",
    "TestResult",
    # Gatekeeper
    "Gatekeeper",
    "GatekeeperEvaluator",
    "ApprovalCriterion",
    "ApprovalDecision",
    "ApprovalStatus",
]
