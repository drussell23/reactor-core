"""
Model Evaluation Modules for Night Shift Training Engine - Reactor Core
=========================================================================

Provides:
- Base evaluation framework
- JARVIS-specific test suite
- Gatekeeper approval system
- Regression detection

ADVANCED EVALUATION (v76.0):
- AGI-specific benchmarks (MMLU, HumanEval, GSM8K, etc.)
- Real-time performance monitoring
- Model drift detection with statistical testing
- A/B testing framework
- Safety and alignment metrics
- Active learning for efficient data selection
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

# Advanced evaluation (v76.0)
from reactor_core.eval.advanced_evaluation import (
    # Enums
    BenchmarkType,
    MetricType,
    DriftSeverity,
    SamplingStrategy,
    # Data structures
    BenchmarkResult,
    EvaluationSuite,
    DriftAlert,
    ABTestResult,
    ABTestConfig,
    ActiveLearningSample,
    # Evaluators
    BenchmarkEvaluator,
    MMLUEvaluator,
    HumanEvalEvaluator,
    SafetyEvaluator,
    JARVISTaskEvaluator,
    # Monitoring
    DriftDetector,
    ABTester,
    # Active Learning
    ActiveLearner,
    # Main
    ComprehensiveEvaluator,
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
    # === ADVANCED EVALUATION (v76.0) ===
    # Benchmark Enums
    "BenchmarkType",
    "MetricType",
    "DriftSeverity",
    "SamplingStrategy",
    # Data Structures
    "BenchmarkResult",
    "EvaluationSuite",
    "DriftAlert",
    "ABTestResult",
    "ABTestConfig",
    "ActiveLearningSample",
    # Evaluators
    "BenchmarkEvaluator",
    "MMLUEvaluator",
    "HumanEvalEvaluator",
    "SafetyEvaluator",
    "JARVISTaskEvaluator",
    # Monitoring
    "DriftDetector",
    "ABTester",
    # Active Learning
    "ActiveLearner",
    # Comprehensive Evaluator
    "ComprehensiveEvaluator",
]
