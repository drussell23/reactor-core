"""
Example rewriter engine for improving training data quality.

Provides:
- Response improvement and expansion
- Correction-based rewriting
- Multi-strategy rewriting approaches
- Batch processing with rate limiting
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from reactor_core.distillation.teacher_client import TeacherClient

logger = logging.getLogger(__name__)


class RewriteStrategy(Enum):
    """Strategies for rewriting examples."""
    IMPROVE = "improve"  # General quality improvement
    EXPAND = "expand"  # Add more detail/depth
    SIMPLIFY = "simplify"  # Make more concise
    CORRECT = "correct"  # Fix factual errors
    CLARIFY = "clarify"  # Improve clarity
    STYLE = "style"  # Adjust tone/style
    COMPLETE = "complete"  # Add missing information


@dataclass
class RewriteResult:
    """Result of rewriting an example."""
    original: Dict[str, Any]
    rewritten: Dict[str, Any]
    strategy: RewriteStrategy
    changes_made: str
    quality_improvement: float  # Estimated improvement
    tokens_used: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "rewritten": self.rewritten,
            "strategy": self.strategy.value,
            "changes_made": self.changes_made,
            "quality_improvement": self.quality_improvement,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error": self.error,
        }


class RewriterEngine:
    """
    Engine for rewriting and improving training examples.

    Uses teacher models to enhance low-quality examples
    while preserving the original intent.
    """

    # Rewrite prompt templates
    REWRITE_PROMPTS = {
        RewriteStrategy.IMPROVE: """You are an expert AI trainer improving conversation examples.

Rewrite the assistant's response to be more helpful, accurate, and well-structured.
Keep the same general approach but improve quality significantly.

## Original Example:
{example}

## Feedback (if available):
{feedback}

## Output Format (JSON):
```json
{{
    "rewritten_response": "<improved response>",
    "changes_made": "<brief description of improvements>",
    "quality_improvement": <0.0-1.0 estimated improvement>
}}
```

Provide ONLY the JSON output.""",

        RewriteStrategy.EXPAND: """You are an expert AI trainer expanding conversation examples.

Expand the assistant's response with more detail, examples, and thorough explanations.
The response should be more comprehensive while remaining focused.

## Original Example:
{example}

## Output Format (JSON):
```json
{{
    "rewritten_response": "<expanded response>",
    "changes_made": "<what was added>",
    "quality_improvement": <0.0-1.0 estimated improvement>
}}
```

Provide ONLY the JSON output.""",

        RewriteStrategy.SIMPLIFY: """You are an expert AI trainer simplifying conversation examples.

Make the assistant's response more concise and to-the-point.
Remove unnecessary verbosity while keeping key information.

## Original Example:
{example}

## Output Format (JSON):
```json
{{
    "rewritten_response": "<simplified response>",
    "changes_made": "<what was removed/simplified>",
    "quality_improvement": <0.0-1.0 estimated improvement>
}}
```

Provide ONLY the JSON output.""",

        RewriteStrategy.CORRECT: """You are an expert AI trainer correcting errors in conversation examples.

Fix any factual errors, logical inconsistencies, or incorrect information.
Ensure the response is accurate and reliable.

## Original Example:
{example}

## Known Issues (if available):
{feedback}

## Output Format (JSON):
```json
{{
    "rewritten_response": "<corrected response>",
    "changes_made": "<corrections made>",
    "quality_improvement": <0.0-1.0 estimated improvement>
}}
```

Provide ONLY the JSON output.""",

        RewriteStrategy.CLARIFY: """You are an expert AI trainer improving clarity in conversation examples.

Rewrite the assistant's response to be clearer, better organized, and easier to understand.
Improve structure and flow without changing the content.

## Original Example:
{example}

## Output Format (JSON):
```json
{{
    "rewritten_response": "<clarified response>",
    "changes_made": "<clarity improvements>",
    "quality_improvement": <0.0-1.0 estimated improvement>
}}
```

Provide ONLY the JSON output.""",

        RewriteStrategy.STYLE: """You are an expert AI trainer adjusting style in conversation examples.

Adjust the assistant's tone to be more professional, friendly, and appropriate.
Make it sound like a helpful AI assistant: clear, warm, and knowledgeable.

## Original Example:
{example}

## Target Style:
- Professional yet approachable
- Clear and direct
- Helpful and patient
- Confident but not arrogant

## Output Format (JSON):
```json
{{
    "rewritten_response": "<style-adjusted response>",
    "changes_made": "<style adjustments>",
    "quality_improvement": <0.0-1.0 estimated improvement>
}}
```

Provide ONLY the JSON output.""",

        RewriteStrategy.COMPLETE: """You are an expert AI trainer completing conversation examples.

Complete any missing information or unfinished thoughts in the assistant's response.
Add any relevant details that would make the response fully helpful.

## Original Example:
{example}

## Output Format (JSON):
```json
{{
    "rewritten_response": "<completed response>",
    "changes_made": "<what was added to complete>",
    "quality_improvement": <0.0-1.0 estimated improvement>
}}
```

Provide ONLY the JSON output.""",
    }

    def __init__(
        self,
        teacher_client: TeacherClient,
        default_strategy: RewriteStrategy = RewriteStrategy.IMPROVE,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        preserve_format: bool = True,
    ):
        """
        Initialize rewriter engine.

        Args:
            teacher_client: Teacher model client
            default_strategy: Default rewriting strategy
            temperature: Sampling temperature
            max_tokens: Maximum tokens for rewritten response
            preserve_format: Preserve original example format
        """
        self.teacher_client = teacher_client
        self.default_strategy = default_strategy
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.preserve_format = preserve_format

    def _format_example(self, example: Dict[str, Any]) -> str:
        """Format example for rewriting prompt."""
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

        # Fallback
        return json.dumps(example, indent=2)

    def _extract_assistant_response(self, example: Dict[str, Any]) -> str:
        """Extract the assistant's response from an example."""
        if "messages" in example:
            for msg in reversed(example["messages"]):
                if msg.get("role") == "assistant":
                    return msg.get("content", "")
            return ""

        if "output" in example:
            return example["output"]

        return ""

    def _apply_rewritten_response(
        self,
        original: Dict[str, Any],
        new_response: str,
    ) -> Dict[str, Any]:
        """Apply rewritten response to original example format."""
        import copy
        result = copy.deepcopy(original)

        if "messages" in result:
            # Find and replace assistant message
            for i in range(len(result["messages"]) - 1, -1, -1):
                if result["messages"][i].get("role") == "assistant":
                    result["messages"][i]["content"] = new_response
                    break
        elif "output" in result:
            result["output"] = new_response
        elif "chosen" in result:
            result["chosen"] = new_response

        # Mark as rewritten
        result["is_rewritten"] = True

        return result

    def _parse_rewrite_response(
        self,
        response: str,
        original: Dict[str, Any],
        strategy: RewriteStrategy,
    ) -> RewriteResult:
        """Parse teacher model's rewrite response."""
        try:
            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                raise ValueError("No JSON found in response")

            data = json.loads(json_match.group())

            new_response = data.get("rewritten_response", "")
            changes_made = data.get("changes_made", "")
            quality_improvement = data.get("quality_improvement", 0.5)

            # Apply to original format
            rewritten = self._apply_rewritten_response(original, new_response)

            return RewriteResult(
                original=original,
                rewritten=rewritten,
                strategy=strategy,
                changes_made=changes_made,
                quality_improvement=quality_improvement,
                success=True,
            )

        except Exception as e:
            logger.error(f"Failed to parse rewrite response: {e}")
            return RewriteResult(
                original=original,
                rewritten=original,
                strategy=strategy,
                changes_made="",
                quality_improvement=0.0,
                success=False,
                error=str(e),
            )

    async def rewrite_example(
        self,
        example: Dict[str, Any],
        strategy: Optional[RewriteStrategy] = None,
        feedback: str = "",
    ) -> RewriteResult:
        """
        Rewrite a single example using the teacher model.

        Args:
            example: Example to rewrite
            strategy: Rewriting strategy (default: improve)
            feedback: Optional feedback to guide rewriting

        Returns:
            RewriteResult with original and rewritten examples
        """
        strategy = strategy or self.default_strategy

        # Get prompt template
        prompt_template = self.REWRITE_PROMPTS.get(
            strategy,
            self.REWRITE_PROMPTS[RewriteStrategy.IMPROVE],
        )

        # Format example
        formatted_example = self._format_example(example)

        # Build prompt
        prompt = prompt_template.format(
            example=formatted_example,
            feedback=feedback or "No specific feedback provided.",
        )

        # Get teacher response
        response = await self.teacher_client.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Parse response
        result = self._parse_rewrite_response(response.content, example, strategy)
        result.tokens_used = response.total_tokens
        result.latency_ms = response.latency_ms

        if result.success:
            logger.debug(
                f"Rewrote example with {strategy.value}: "
                f"+{result.quality_improvement:.1%} improvement"
            )
        else:
            logger.warning(f"Rewrite failed: {result.error}")

        return result

    async def rewrite_batch(
        self,
        examples: List[Dict[str, Any]],
        strategy: Optional[RewriteStrategy] = None,
        feedbacks: Optional[List[str]] = None,
        max_concurrent: int = 5,
        rate_limit_delay: float = 0.1,
    ) -> List[RewriteResult]:
        """
        Rewrite a batch of examples with rate limiting.

        Args:
            examples: Examples to rewrite
            strategy: Rewriting strategy
            feedbacks: Optional feedback for each example
            max_concurrent: Maximum concurrent requests
            rate_limit_delay: Delay between requests

        Returns:
            List of RewriteResults
        """
        if feedbacks is None:
            feedbacks = [""] * len(examples)

        semaphore = asyncio.Semaphore(max_concurrent)

        async def rewrite_with_limit(
            example: Dict[str, Any],
            feedback: str,
        ) -> RewriteResult:
            async with semaphore:
                await asyncio.sleep(rate_limit_delay)
                return await self.rewrite_example(example, strategy, feedback)

        tasks = [
            rewrite_with_limit(example, feedback)
            for example, feedback in zip(examples, feedbacks)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Rewrite failed for example {i}: {result}")
                final_results.append(RewriteResult(
                    original=examples[i],
                    rewritten=examples[i],
                    strategy=strategy or self.default_strategy,
                    changes_made="",
                    quality_improvement=0.0,
                    success=False,
                    error=str(result),
                ))
            else:
                final_results.append(result)

        return final_results

    async def iterative_rewrite(
        self,
        example: Dict[str, Any],
        strategies: List[RewriteStrategy],
        feedback: str = "",
    ) -> RewriteResult:
        """
        Apply multiple rewriting strategies sequentially.

        Args:
            example: Example to rewrite
            strategies: Ordered list of strategies to apply
            feedback: Initial feedback

        Returns:
            Final RewriteResult after all strategies
        """
        current = example
        total_improvement = 0.0
        all_changes = []
        total_tokens = 0
        total_latency = 0.0

        for strategy in strategies:
            result = await self.rewrite_example(current, strategy, feedback)

            if result.success:
                current = result.rewritten
                total_improvement += result.quality_improvement
                all_changes.append(f"{strategy.value}: {result.changes_made}")
                total_tokens += result.tokens_used
                total_latency += result.latency_ms
            else:
                logger.warning(f"Strategy {strategy.value} failed, skipping")

        return RewriteResult(
            original=example,
            rewritten=current,
            strategy=strategies[-1] if strategies else RewriteStrategy.IMPROVE,
            changes_made=" | ".join(all_changes),
            quality_improvement=min(total_improvement, 1.0),
            tokens_used=total_tokens,
            latency_ms=total_latency,
            success=True,
        )

    def get_statistics(
        self,
        results: List[RewriteResult],
    ) -> Dict[str, Any]:
        """
        Get statistics from rewriting results.

        Args:
            results: List of rewrite results

        Returns:
            Dictionary with statistics
        """
        if not results:
            return {"count": 0}

        successful = [r for r in results if r.success]
        improvements = [r.quality_improvement for r in successful]

        return {
            "count": len(results),
            "success_count": len(successful),
            "success_rate": len(successful) / len(results),
            "mean_improvement": sum(improvements) / len(improvements) if improvements else 0,
            "max_improvement": max(improvements) if improvements else 0,
            "total_tokens": sum(r.tokens_used for r in results),
            "total_latency_ms": sum(r.latency_ms for r in results),
        }


# Convenience exports
__all__ = [
    "RewriterEngine",
    "RewriteResult",
    "RewriteStrategy",
]
