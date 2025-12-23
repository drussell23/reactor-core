"""
Quality scoring engine for training examples.

Provides:
- Multi-criteria quality assessment
- Teacher model-based scoring
- Configurable scoring rubrics
- Batch scoring with rate limiting
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from reactor_core.distillation.teacher_client import TeacherClient, TeacherResponse

logger = logging.getLogger(__name__)


class ScoringCriteria(Enum):
    """Criteria for quality scoring."""
    HELPFULNESS = "helpfulness"
    ACCURACY = "accuracy"
    CLARITY = "clarity"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    SAFETY = "safety"
    INSTRUCTION_FOLLOWING = "instruction_following"
    COHERENCE = "coherence"


@dataclass
class QualityScore:
    """Quality score for a single criterion."""
    criterion: ScoringCriteria
    score: float  # 0.0 - 1.0
    reasoning: str
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion": self.criterion.value,
            "score": self.score,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
        }


@dataclass
class ScoringResult:
    """Complete scoring result for an example."""
    example_id: str
    overall_score: float
    criteria_scores: List[QualityScore]
    recommendation: str  # "keep", "improve", "reject"
    feedback: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if example passed quality threshold."""
        return self.recommendation == "keep"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "overall_score": self.overall_score,
            "criteria_scores": [s.to_dict() for s in self.criteria_scores],
            "recommendation": self.recommendation,
            "feedback": self.feedback,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


class ScoringEngine:
    """
    Quality scoring engine using teacher models.

    Evaluates training examples across multiple criteria
    and provides recommendations for improvement.
    """

    # Default scoring prompt template
    DEFAULT_SCORING_PROMPT = """You are an expert evaluator assessing the quality of AI assistant training examples.

Evaluate the following conversation example on these criteria (score 0-10):

1. **Helpfulness**: Does the response directly address the user's needs?
2. **Accuracy**: Is the information factually correct?
3. **Clarity**: Is the response clear and well-structured?
4. **Completeness**: Does it fully answer the question without unnecessary padding?
5. **Safety**: Is the response appropriate and free from harmful content?

## Example to Evaluate:

{example}

## Output Format (JSON):
```json
{{
    "scores": {{
        "helpfulness": <0-10>,
        "accuracy": <0-10>,
        "clarity": <0-10>,
        "completeness": <0-10>,
        "safety": <0-10>
    }},
    "overall_score": <0-10>,
    "recommendation": "<keep|improve|reject>",
    "feedback": "<brief explanation of scores and suggestions for improvement>"
}}
```

Provide ONLY the JSON output, no additional text."""

    def __init__(
        self,
        teacher_client: TeacherClient,
        criteria: Optional[List[ScoringCriteria]] = None,
        min_quality_threshold: float = 0.6,
        scoring_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):
        """
        Initialize scoring engine.

        Args:
            teacher_client: Teacher model client
            criteria: Criteria to evaluate (default: all)
            min_quality_threshold: Minimum score to "keep"
            scoring_prompt: Custom scoring prompt template
            temperature: Sampling temperature for scoring
            max_tokens: Maximum tokens for scoring response
        """
        self.teacher_client = teacher_client
        self.criteria = criteria or [
            ScoringCriteria.HELPFULNESS,
            ScoringCriteria.ACCURACY,
            ScoringCriteria.CLARITY,
            ScoringCriteria.COMPLETENESS,
            ScoringCriteria.SAFETY,
        ]
        self.min_quality_threshold = min_quality_threshold
        self.scoring_prompt = scoring_prompt or self.DEFAULT_SCORING_PROMPT
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _format_example(self, example: Dict[str, Any]) -> str:
        """Format example for scoring prompt."""
        # Handle ChatML format
        if "messages" in example:
            formatted = ""
            for msg in example["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                formatted += f"**{role.capitalize()}**: {content}\n\n"
            return formatted.strip()

        # Handle Alpaca format
        if "instruction" in example:
            formatted = f"**Instruction**: {example['instruction']}\n\n"
            if example.get("input"):
                formatted += f"**Input**: {example['input']}\n\n"
            formatted += f"**Response**: {example.get('output', '')}"
            return formatted

        # Handle preference format
        if "prompt" in example and "chosen" in example:
            formatted = f"**Prompt**: {example['prompt']}\n\n"
            formatted += f"**Response**: {example['chosen']}"
            return formatted

        # Fallback: JSON dump
        return json.dumps(example, indent=2)

    def _parse_scoring_response(
        self,
        response: str,
        example_id: str,
    ) -> ScoringResult:
        """Parse teacher model's scoring response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                raise ValueError("No JSON found in response")

            data = json.loads(json_match.group())

            # Parse scores
            scores_data = data.get("scores", {})
            criteria_scores = []

            for criterion in self.criteria:
                criterion_name = criterion.value
                score = scores_data.get(criterion_name, 5) / 10.0  # Normalize to 0-1
                criteria_scores.append(QualityScore(
                    criterion=criterion,
                    score=score,
                    reasoning="",  # Not included in compact response
                ))

            # Parse overall score
            overall_score = data.get("overall_score", 5) / 10.0

            # Determine recommendation
            recommendation = data.get("recommendation", "improve").lower()
            if recommendation not in ("keep", "improve", "reject"):
                if overall_score >= self.min_quality_threshold:
                    recommendation = "keep"
                elif overall_score >= self.min_quality_threshold * 0.5:
                    recommendation = "improve"
                else:
                    recommendation = "reject"

            return ScoringResult(
                example_id=example_id,
                overall_score=overall_score,
                criteria_scores=criteria_scores,
                recommendation=recommendation,
                feedback=data.get("feedback", ""),
            )

        except Exception as e:
            logger.error(f"Failed to parse scoring response: {e}")
            # Return default low score on parse failure
            return ScoringResult(
                example_id=example_id,
                overall_score=0.0,
                criteria_scores=[],
                recommendation="reject",
                feedback=f"Failed to parse scoring response: {e}",
                metadata={"error": str(e), "raw_response": response},
            )

    async def score_example(
        self,
        example: Dict[str, Any],
        example_id: Optional[str] = None,
    ) -> ScoringResult:
        """
        Score a single training example.

        Args:
            example: Training example to score
            example_id: Optional identifier for the example

        Returns:
            ScoringResult with scores and recommendation
        """
        example_id = example_id or str(hash(json.dumps(example, sort_keys=True)))

        # Format prompt
        formatted_example = self._format_example(example)
        prompt = self.scoring_prompt.format(example=formatted_example)

        # Get teacher response
        response = await self.teacher_client.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Parse response
        result = self._parse_scoring_response(response.content, example_id)
        result.tokens_used = response.total_tokens
        result.latency_ms = response.latency_ms

        logger.debug(
            f"Scored example {example_id}: {result.overall_score:.2f} "
            f"({result.recommendation})"
        )

        return result

    async def score_batch(
        self,
        examples: List[Dict[str, Any]],
        example_ids: Optional[List[str]] = None,
        max_concurrent: int = 5,
        rate_limit_delay: float = 0.1,
    ) -> List[ScoringResult]:
        """
        Score a batch of examples with rate limiting.

        Args:
            examples: List of examples to score
            example_ids: Optional list of identifiers
            max_concurrent: Maximum concurrent requests
            rate_limit_delay: Delay between requests

        Returns:
            List of ScoringResults
        """
        if example_ids is None:
            example_ids = [str(i) for i in range(len(examples))]

        semaphore = asyncio.Semaphore(max_concurrent)

        async def score_with_limit(
            example: Dict[str, Any],
            example_id: str,
        ) -> ScoringResult:
            async with semaphore:
                await asyncio.sleep(rate_limit_delay)
                return await self.score_example(example, example_id)

        tasks = [
            score_with_limit(example, example_id)
            for example, example_id in zip(examples, example_ids)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Scoring failed for example {example_ids[i]}: {result}")
                final_results.append(ScoringResult(
                    example_id=example_ids[i],
                    overall_score=0.0,
                    criteria_scores=[],
                    recommendation="reject",
                    feedback=f"Scoring failed: {result}",
                    metadata={"error": str(result)},
                ))
            else:
                final_results.append(result)

        return final_results

    def filter_by_quality(
        self,
        examples: List[Dict[str, Any]],
        results: List[ScoringResult],
        min_score: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter examples by quality scores.

        Args:
            examples: Original examples
            results: Scoring results
            min_score: Override minimum score threshold

        Returns:
            Tuple of (keep, improve, reject) example lists
        """
        min_score = min_score or self.min_quality_threshold

        keep = []
        improve = []
        reject = []

        for example, result in zip(examples, results):
            if result.recommendation == "keep" or result.overall_score >= min_score:
                keep.append(example)
            elif result.recommendation == "improve":
                improve.append(example)
            else:
                reject.append(example)

        logger.info(
            f"Quality filter: {len(keep)} keep, {len(improve)} improve, "
            f"{len(reject)} reject"
        )

        return keep, improve, reject

    def get_statistics(
        self,
        results: List[ScoringResult],
    ) -> Dict[str, Any]:
        """
        Get statistics from scoring results.

        Args:
            results: List of scoring results

        Returns:
            Dictionary with statistics
        """
        if not results:
            return {"count": 0}

        scores = [r.overall_score for r in results]
        recommendations = [r.recommendation for r in results]

        return {
            "count": len(results),
            "mean_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "keep_count": recommendations.count("keep"),
            "improve_count": recommendations.count("improve"),
            "reject_count": recommendations.count("reject"),
            "keep_rate": recommendations.count("keep") / len(results),
            "total_tokens": sum(r.tokens_used for r in results),
            "total_latency_ms": sum(r.latency_ms for r in results),
        }


# Convenience exports
__all__ = [
    "ScoringEngine",
    "QualityScore",
    "ScoringCriteria",
    "ScoringResult",
]
