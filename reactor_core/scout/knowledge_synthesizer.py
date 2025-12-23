"""
Knowledge Synthesizer for Safe Scout.

Provides:
- Synthetic Q&A pair generation from documentation
- Multiple synthesis strategies
- Quality filtering of generated pairs
- ChatML format output
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from reactor_core.scout.content_extractor import ExtractedContent, CodeBlock, Section

logger = logging.getLogger(__name__)


class SynthesisStrategy(Enum):
    """Strategy for generating Q&A pairs."""
    FACTUAL = "factual"              # "What is X?" / "X is..."
    HOWTO = "howto"                  # "How do I X?" / "To X, you..."
    CODE_EXAMPLE = "code_example"    # "Show me code for X" / ```code```
    COMPARISON = "comparison"        # "What's the difference between X and Y?"
    TROUBLESHOOTING = "troubleshooting"  # "Why is X happening?" / "X happens because..."
    BEST_PRACTICE = "best_practice"  # "What's the best way to X?"
    CONCEPTUAL = "conceptual"        # "Explain X" / Deep explanation


class SynthesisQuality(Enum):
    """Quality level of synthesized content."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    REJECTED = "rejected"


@dataclass
class SynthesizedPair:
    """A synthesized Q&A pair."""
    pair_id: str
    question: str
    answer: str
    strategy: SynthesisStrategy
    quality: SynthesisQuality

    # Source attribution
    source_url: str
    source_section: str = ""
    source_code_block: bool = False

    # ChatML format
    chatml_messages: List[Dict[str, str]] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.pair_id:
            content = f"{self.question}:{self.answer}"
            self.pair_id = hashlib.md5(content.encode()).hexdigest()[:12]

        if not self.chatml_messages:
            self.chatml_messages = [
                {"role": "user", "content": self.question},
                {"role": "assistant", "content": self.answer},
            ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "question": self.question,
            "answer": self.answer,
            "strategy": self.strategy.value,
            "quality": self.quality.value,
            "source_url": self.source_url,
            "source_section": self.source_section,
            "source_code_block": self.source_code_block,
            "chatml_messages": self.chatml_messages,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    def to_chatml(self) -> Dict[str, Any]:
        """Convert to ChatML training format."""
        return {
            "messages": self.chatml_messages,
            "metadata": {
                "source": self.source_url,
                "strategy": self.strategy.value,
                "quality": self.quality.value,
            }
        }


@dataclass
class SynthesisResult:
    """Result of synthesis operation."""
    source_url: str
    pairs: List[SynthesizedPair] = field(default_factory=list)
    total_generated: int = 0
    high_quality_count: int = 0
    rejected_count: int = 0
    error_message: str = ""
    synthesized_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_url": self.source_url,
            "pair_count": len(self.pairs),
            "total_generated": self.total_generated,
            "high_quality_count": self.high_quality_count,
            "rejected_count": self.rejected_count,
            "error_message": self.error_message,
            "synthesized_at": self.synthesized_at.isoformat(),
        }


@dataclass
class SynthesizerConfig:
    """Configuration for knowledge synthesizer."""
    # Generation limits
    max_pairs_per_page: int = 20
    max_pairs_per_section: int = 5
    max_pairs_per_code_block: int = 3

    # Quality thresholds
    min_question_length: int = 10
    max_question_length: int = 200
    min_answer_length: int = 50
    max_answer_length: int = 2000

    # Strategy weights (probability of each strategy)
    strategy_weights: Dict[SynthesisStrategy, float] = field(
        default_factory=lambda: {
            SynthesisStrategy.HOWTO: 0.3,
            SynthesisStrategy.CODE_EXAMPLE: 0.25,
            SynthesisStrategy.FACTUAL: 0.2,
            SynthesisStrategy.BEST_PRACTICE: 0.1,
            SynthesisStrategy.TROUBLESHOOTING: 0.1,
            SynthesisStrategy.CONCEPTUAL: 0.05,
        }
    )

    # Teacher model
    teacher_model: str = field(
        default_factory=lambda: os.getenv(
            "NIGHTSHIFT_TEACHER_MODEL",
            "gemini-1.5-flash"
        )
    )

    # Concurrency
    max_concurrent_synthesis: int = 5


# Prompt templates for each strategy
SYNTHESIS_PROMPTS = {
    SynthesisStrategy.FACTUAL: """
Based on the following documentation excerpt, generate a factual Q&A pair.
The question should ask "What is X?" or "What does X do?" style questions.
The answer should be direct and informative.

Documentation:
{content}

Generate a single Q&A pair in JSON format:
{{"question": "...", "answer": "..."}}
""",

    SynthesisStrategy.HOWTO: """
Based on the following documentation excerpt, generate a how-to Q&A pair.
The question should start with "How do I..." or "How can I..." or "How to..."
The answer should provide step-by-step instructions or clear guidance.

Documentation:
{content}

Generate a single Q&A pair in JSON format:
{{"question": "...", "answer": "..."}}
""",

    SynthesisStrategy.CODE_EXAMPLE: """
Based on the following documentation with code, generate a code-focused Q&A pair.
The question should ask for a code example or implementation.
The answer MUST include working code wrapped in triple backticks with the language specified.

Documentation and Code:
{content}

Code:
```{language}
{code}
```

Generate a single Q&A pair in JSON format:
{{"question": "...", "answer": "..."}}

The answer MUST include the code example formatted properly.
""",

    SynthesisStrategy.BEST_PRACTICE: """
Based on the following documentation, generate a best practices Q&A pair.
The question should ask "What's the best way to..." or "What are best practices for..."
The answer should explain recommended approaches and why they're preferred.

Documentation:
{content}

Generate a single Q&A pair in JSON format:
{{"question": "...", "answer": "..."}}
""",

    SynthesisStrategy.TROUBLESHOOTING: """
Based on the following documentation, generate a troubleshooting Q&A pair.
The question should describe a problem or ask "Why is X happening?"
The answer should explain the cause and provide a solution.

Documentation:
{content}

Generate a single Q&A pair in JSON format:
{{"question": "...", "answer": "..."}}
""",

    SynthesisStrategy.CONCEPTUAL: """
Based on the following documentation, generate a conceptual explanation Q&A pair.
The question should ask "Explain..." or "What is the concept of..."
The answer should provide a thorough, educational explanation.

Documentation:
{content}

Generate a single Q&A pair in JSON format:
{{"question": "...", "answer": "..."}}
""",
}


class KnowledgeSynthesizer:
    """
    Synthesizes Q&A training pairs from extracted documentation.

    Uses teacher model (Gemini Flash by default) to generate
    high-quality question-answer pairs from raw documentation.
    """

    def __init__(
        self,
        config: Optional[SynthesizerConfig] = None,
        teacher_client: Optional[Any] = None,
    ):
        self.config = config or SynthesizerConfig()
        self._teacher_client = teacher_client
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_synthesis)

    async def get_teacher_client(self):
        """Get or create teacher client."""
        if self._teacher_client is None:
            # Lazy import to avoid circular dependency
            from reactor_core.distillation.teacher_client import create_teacher_client
            self._teacher_client = create_teacher_client(
                model_name=self.config.teacher_model
            )
        return self._teacher_client

    async def synthesize(
        self,
        content: ExtractedContent,
    ) -> SynthesisResult:
        """
        Synthesize Q&A pairs from extracted content.

        Args:
            content: Extracted content from a page

        Returns:
            SynthesisResult with generated pairs
        """
        pairs: List[SynthesizedPair] = []
        total_generated = 0
        rejected_count = 0

        try:
            # Synthesize from sections
            for section in content.sections[:10]:  # Limit sections
                section_pairs = await self._synthesize_from_section(
                    section, content.url
                )
                for pair in section_pairs:
                    total_generated += 1
                    if pair.quality != SynthesisQuality.REJECTED:
                        pairs.append(pair)
                    else:
                        rejected_count += 1

                    if len(pairs) >= self.config.max_pairs_per_page:
                        break

                if len(pairs) >= self.config.max_pairs_per_page:
                    break

            # Synthesize from standalone code blocks
            if len(pairs) < self.config.max_pairs_per_page:
                for code_block in content.code_blocks[:5]:
                    code_pairs = await self._synthesize_from_code(
                        code_block, content.url, content.title
                    )
                    for pair in code_pairs:
                        total_generated += 1
                        if pair.quality != SynthesisQuality.REJECTED:
                            pairs.append(pair)
                        else:
                            rejected_count += 1

                        if len(pairs) >= self.config.max_pairs_per_page:
                            break

                    if len(pairs) >= self.config.max_pairs_per_page:
                        break

            # If no structured content, synthesize from raw text
            if not pairs and content.text_content:
                text_pairs = await self._synthesize_from_text(
                    content.text_content, content.url, content.title
                )
                for pair in text_pairs:
                    total_generated += 1
                    if pair.quality != SynthesisQuality.REJECTED:
                        pairs.append(pair)
                    else:
                        rejected_count += 1

            high_quality = sum(
                1 for p in pairs if p.quality == SynthesisQuality.HIGH
            )

            return SynthesisResult(
                source_url=content.url,
                pairs=pairs,
                total_generated=total_generated,
                high_quality_count=high_quality,
                rejected_count=rejected_count,
            )

        except Exception as e:
            logger.error(f"Synthesis error for {content.url}: {e}")
            return SynthesisResult(
                source_url=content.url,
                error_message=str(e),
            )

    async def _synthesize_from_section(
        self,
        section: Section,
        source_url: str,
    ) -> List[SynthesizedPair]:
        """Synthesize pairs from a document section."""
        pairs = []

        # Skip sections with little content
        if len(section.content) < 100:
            return pairs

        # Determine strategies based on content
        strategies = self._select_strategies_for_content(
            section.content, section.code_blocks
        )

        for strategy in strategies[:self.config.max_pairs_per_section]:
            async with self._semaphore:
                pair = await self._generate_pair(
                    strategy=strategy,
                    content=section.content,
                    code_blocks=section.code_blocks,
                    source_url=source_url,
                    source_section=section.heading,
                )
                if pair:
                    pairs.append(pair)

        return pairs

    async def _synthesize_from_code(
        self,
        code_block: CodeBlock,
        source_url: str,
        page_title: str,
    ) -> List[SynthesizedPair]:
        """Synthesize pairs from a code block."""
        pairs = []

        # Skip very short code
        if len(code_block.code) < 30:
            return pairs

        async with self._semaphore:
            pair = await self._generate_pair(
                strategy=SynthesisStrategy.CODE_EXAMPLE,
                content=code_block.context or page_title,
                code_blocks=[code_block],
                source_url=source_url,
                source_section="",
            )
            if pair:
                pair.source_code_block = True
                pairs.append(pair)

        return pairs

    async def _synthesize_from_text(
        self,
        text: str,
        source_url: str,
        page_title: str,
    ) -> List[SynthesizedPair]:
        """Synthesize pairs from raw text content."""
        pairs = []

        # Split into chunks
        chunks = self._split_into_chunks(text, max_chars=1500)

        for chunk in chunks[:5]:  # Limit chunks
            strategies = self._select_strategies_for_content(chunk, [])

            for strategy in strategies[:2]:
                async with self._semaphore:
                    pair = await self._generate_pair(
                        strategy=strategy,
                        content=chunk,
                        code_blocks=[],
                        source_url=source_url,
                        source_section=page_title,
                    )
                    if pair:
                        pairs.append(pair)

                    if len(pairs) >= self.config.max_pairs_per_page:
                        break

            if len(pairs) >= self.config.max_pairs_per_page:
                break

        return pairs

    async def _generate_pair(
        self,
        strategy: SynthesisStrategy,
        content: str,
        code_blocks: List[CodeBlock],
        source_url: str,
        source_section: str,
    ) -> Optional[SynthesizedPair]:
        """Generate a single Q&A pair using teacher model."""
        try:
            # Get prompt template
            prompt_template = SYNTHESIS_PROMPTS.get(strategy)
            if not prompt_template:
                return None

            # Build prompt
            if strategy == SynthesisStrategy.CODE_EXAMPLE and code_blocks:
                code_block = code_blocks[0]
                prompt = prompt_template.format(
                    content=content[:1000],
                    code=code_block.code[:1000],
                    language=code_block.language or "python",
                )
            else:
                prompt = prompt_template.format(content=content[:1500])

            # Call teacher model
            client = await self.get_teacher_client()
            response = await client.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
            )

            # Parse response
            result_text = response.content.strip()

            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if not json_match:
                logger.warning(f"No JSON found in synthesis response")
                return None

            try:
                qa_data = json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse synthesis JSON")
                return None

            question = qa_data.get("question", "").strip()
            answer = qa_data.get("answer", "").strip()

            # Validate
            quality = self._assess_quality(question, answer, strategy)

            if quality == SynthesisQuality.REJECTED:
                return SynthesizedPair(
                    pair_id="",
                    question=question,
                    answer=answer,
                    strategy=strategy,
                    quality=quality,
                    source_url=source_url,
                    source_section=source_section,
                )

            return SynthesizedPair(
                pair_id="",
                question=question,
                answer=answer,
                strategy=strategy,
                quality=quality,
                source_url=source_url,
                source_section=source_section,
            )

        except Exception as e:
            logger.error(f"Pair generation error: {e}")
            return None

    def _select_strategies_for_content(
        self,
        content: str,
        code_blocks: List[CodeBlock],
    ) -> List[SynthesisStrategy]:
        """Select appropriate synthesis strategies based on content."""
        strategies = []

        # Code blocks -> CODE_EXAMPLE
        if code_blocks:
            strategies.append(SynthesisStrategy.CODE_EXAMPLE)

        # How-to indicators
        howto_patterns = [
            r'\bhow to\b', r'\bsteps?\b', r'\bfirst\b.*\bthen\b',
            r'\binstall\b', r'\bconfigure\b', r'\bsetup\b',
        ]
        if any(re.search(p, content, re.IGNORECASE) for p in howto_patterns):
            strategies.append(SynthesisStrategy.HOWTO)

        # Best practice indicators
        bp_patterns = [
            r'\bbest practice\b', r'\brecommend\b', r'\bshould\b',
            r'\bavoid\b', r'\bprefer\b',
        ]
        if any(re.search(p, content, re.IGNORECASE) for p in bp_patterns):
            strategies.append(SynthesisStrategy.BEST_PRACTICE)

        # Error/troubleshooting indicators
        error_patterns = [
            r'\berror\b', r'\bfail\b', r'\bissue\b', r'\bproblem\b',
            r'\bfix\b', r'\bresolve\b', r'\btroubleshoot\b',
        ]
        if any(re.search(p, content, re.IGNORECASE) for p in error_patterns):
            strategies.append(SynthesisStrategy.TROUBLESHOOTING)

        # Default to factual/howto
        if not strategies:
            strategies.extend([
                SynthesisStrategy.FACTUAL,
                SynthesisStrategy.HOWTO,
            ])

        return strategies[:3]  # Max 3 strategies per content

    def _assess_quality(
        self,
        question: str,
        answer: str,
        strategy: SynthesisStrategy,
    ) -> SynthesisQuality:
        """Assess quality of generated Q&A pair."""
        # Length checks
        if len(question) < self.config.min_question_length:
            return SynthesisQuality.REJECTED
        if len(question) > self.config.max_question_length:
            return SynthesisQuality.REJECTED
        if len(answer) < self.config.min_answer_length:
            return SynthesisQuality.REJECTED
        if len(answer) > self.config.max_answer_length:
            return SynthesisQuality.LOW

        # Question quality
        if not question.endswith("?"):
            return SynthesisQuality.LOW

        # Code example should have code
        if strategy == SynthesisStrategy.CODE_EXAMPLE:
            if "```" not in answer and not re.search(r'`[^`]+`', answer):
                return SynthesisQuality.LOW

        # Answer should be informative
        word_count = len(answer.split())
        if word_count < 20:
            return SynthesisQuality.LOW

        # High quality indicators
        high_quality_indicators = 0

        # Has examples
        if "```" in answer or "example" in answer.lower():
            high_quality_indicators += 1

        # Has structure (lists, steps)
        if re.search(r'^\d+\.|\*\s|^-\s', answer, re.MULTILINE):
            high_quality_indicators += 1

        # Good length
        if word_count > 50:
            high_quality_indicators += 1

        # Technical terms
        tech_terms = [
            r'\bfunction\b', r'\bmethod\b', r'\bclass\b',
            r'\bparameter\b', r'\breturn\b', r'\bAPI\b',
        ]
        if sum(1 for t in tech_terms if re.search(t, answer)) >= 2:
            high_quality_indicators += 1

        if high_quality_indicators >= 3:
            return SynthesisQuality.HIGH
        elif high_quality_indicators >= 1:
            return SynthesisQuality.MEDIUM
        else:
            return SynthesisQuality.LOW

    def _split_into_chunks(
        self,
        text: str,
        max_chars: int = 1500,
    ) -> List[str]:
        """Split text into manageable chunks."""
        # Split by paragraphs
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para)

            if current_length + para_length > max_chars and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(para)
            current_length += para_length

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks


# Convenience exports
__all__ = [
    "KnowledgeSynthesizer",
    "SynthesizerConfig",
    "SynthesizedPair",
    "SynthesisResult",
    "SynthesisStrategy",
    "SynthesisQuality",
]
