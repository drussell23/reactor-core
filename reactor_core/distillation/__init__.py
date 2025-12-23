"""
Distillation module for Night Shift Training Engine.

Provides:
- Async teacher model clients (OpenAI, Anthropic, Gemini)
- Quality scoring of training examples
- Example rewriting and improvement
- Synthetic data generation
- Rate limiting and cost tracking
"""

from reactor_core.distillation.teacher_client import (
    TeacherClient,
    OpenAIClient,
    AnthropicClient,
    GeminiClient,
    TeacherResponse,
    ModelType,
    create_teacher_client,
)

from reactor_core.distillation.scoring_engine import (
    ScoringEngine,
    QualityScore,
    ScoringCriteria,
    ScoringResult,
)

from reactor_core.distillation.rewriter_engine import (
    RewriterEngine,
    RewriteResult,
    RewriteStrategy,
)

from reactor_core.distillation.rate_limiter import (
    TokenBucketRateLimiter,
    MultiTierRateLimiter,
    RateLimitConfig,
    RateLimitExceeded,
)

from reactor_core.distillation.cost_tracker import (
    CostTracker,
    BudgetEnforcer,
    UsageRecord,
    CostReport,
    ModelPricing,
)

__all__ = [
    # Teacher Client
    "TeacherClient",
    "OpenAIClient",
    "AnthropicClient",
    "GeminiClient",
    "TeacherResponse",
    "ModelType",
    "create_teacher_client",
    # Scoring
    "ScoringEngine",
    "QualityScore",
    "ScoringCriteria",
    "ScoringResult",
    # Rewriting
    "RewriterEngine",
    "RewriteResult",
    "RewriteStrategy",
    # Rate Limiting
    "TokenBucketRateLimiter",
    "MultiTierRateLimiter",
    "RateLimitConfig",
    "RateLimitExceeded",
    # Cost Tracking
    "CostTracker",
    "BudgetEnforcer",
    "UsageRecord",
    "CostReport",
    "ModelPricing",
]
