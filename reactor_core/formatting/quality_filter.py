"""
Quality filtering and deduplication for training data.

Provides:
- Confidence-based filtering
- Content-based deduplication
- Length filtering
- Embedding-based similarity dedup
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from reactor_core.formatting.base_formatter import FormattedExample
from reactor_core.ingestion.base_ingestor import RawInteraction

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of filtering operation."""
    passed: List[FormattedExample]
    filtered: int
    duplicates: int
    below_quality: int
    too_short: int
    too_long: int

    @property
    def total_processed(self) -> int:
        return len(self.passed) + self.filtered

    @property
    def pass_rate(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return len(self.passed) / self.total_processed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": len(self.passed),
            "filtered": self.filtered,
            "duplicates": self.duplicates,
            "below_quality": self.below_quality,
            "too_short": self.too_short,
            "too_long": self.too_long,
            "pass_rate": self.pass_rate,
        }


class QualityFilter:
    """
    Filter and deduplicate training examples based on quality.

    Filtering criteria:
    - Minimum quality score
    - Minimum/maximum token length
    - Content deduplication (exact and fuzzy)
    - Empty content filtering
    """

    def __init__(
        self,
        min_quality: float = 0.5,
        min_tokens: int = 10,
        max_tokens: int = 4096,
        deduplicate: bool = True,
        similarity_threshold: float = 0.95,
        prioritize_corrections: bool = True,
    ):
        """
        Initialize quality filter.

        Args:
            min_quality: Minimum quality score (0-1)
            min_tokens: Minimum token count
            max_tokens: Maximum token count
            deduplicate: Enable content deduplication
            similarity_threshold: Threshold for fuzzy dedup (unused if no embeddings)
            prioritize_corrections: Keep corrections even if below threshold
        """
        self.min_quality = min_quality
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.deduplicate = deduplicate
        self.similarity_threshold = similarity_threshold
        self.prioritize_corrections = prioritize_corrections

        # Deduplication state
        self._seen_hashes: Set[str] = set()
        self._stats = {
            "duplicates": 0,
            "below_quality": 0,
            "too_short": 0,
            "too_long": 0,
        }

    def filter(
        self,
        examples: List[FormattedExample],
    ) -> FilterResult:
        """
        Filter a list of examples.

        Args:
            examples: List of examples to filter

        Returns:
            FilterResult with passed examples and stats
        """
        passed = []
        filtered = 0

        for example in examples:
            if self._should_pass(example):
                passed.append(example)

                # Track hash for dedup
                if self.deduplicate:
                    self._seen_hashes.add(example.content_hash())
            else:
                filtered += 1

        return FilterResult(
            passed=passed,
            filtered=filtered,
            duplicates=self._stats["duplicates"],
            below_quality=self._stats["below_quality"],
            too_short=self._stats["too_short"],
            too_long=self._stats["too_long"],
        )

    def filter_single(
        self,
        example: FormattedExample,
    ) -> bool:
        """
        Check if a single example passes the filter.

        Args:
            example: Example to check

        Returns:
            True if passes, False if filtered
        """
        return self._should_pass(example)

    def _should_pass(self, example: FormattedExample) -> bool:
        """Check if example passes all filter criteria."""
        # Check for empty content
        if not self._has_content(example):
            return False

        # Token length check
        if example.estimated_tokens < self.min_tokens:
            self._stats["too_short"] += 1
            return False

        if example.estimated_tokens > self.max_tokens:
            self._stats["too_long"] += 1
            return False

        # Quality check (with correction priority)
        if example.quality_score < self.min_quality:
            if not (self.prioritize_corrections and example.is_correction):
                self._stats["below_quality"] += 1
                return False

        # Deduplication check
        if self.deduplicate:
            content_hash = example.content_hash()
            if content_hash in self._seen_hashes:
                self._stats["duplicates"] += 1
                return False

        return True

    def _has_content(self, example: FormattedExample) -> bool:
        """Check if example has meaningful content."""
        if example.messages:
            # ChatML: check for non-empty messages
            has_user = any(
                m.get("role") == "user" and m.get("content", "").strip()
                for m in example.messages
            )
            has_assistant = any(
                m.get("role") == "assistant" and m.get("content", "").strip()
                for m in example.messages
            )
            return has_user and has_assistant

        if example.instruction is not None:
            # Alpaca: check for instruction and output
            return bool(example.output_text and example.output_text.strip())

        if example.prompt is not None:
            # Preference: check for prompt and chosen
            return bool(
                example.prompt.strip()
                and example.chosen
                and example.chosen.strip()
            )

        return False

    def reset(self) -> None:
        """Reset filter state (clears dedup cache)."""
        self._seen_hashes.clear()
        self._stats = {
            "duplicates": 0,
            "below_quality": 0,
            "too_short": 0,
            "too_long": 0,
        }

    def get_stats(self) -> Dict[str, int]:
        """Get filter statistics."""
        return self._stats.copy()


class EmbeddingDeduplicator:
    """
    Fuzzy deduplication using sentence embeddings.

    Uses cosine similarity to find near-duplicate examples.
    Requires sentence-transformers to be installed.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.95,
        batch_size: int = 32,
    ):
        """
        Initialize embedding-based deduplicator.

        Args:
            model_name: Sentence transformer model name
            similarity_threshold: Similarity threshold for dedup
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size

        self._model = None
        self._embeddings: List[Any] = []
        self._examples: List[FormattedExample] = []

    def _get_model(self):
        """Lazy load sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Falling back to exact dedup."
                )
                self._model = None
        return self._model

    def _get_text(self, example: FormattedExample) -> str:
        """Extract text content from example."""
        if example.messages:
            return " ".join(
                m.get("content", "") for m in example.messages
            )
        if example.instruction:
            return f"{example.instruction} {example.input_text or ''} {example.output_text or ''}"
        if example.prompt:
            return f"{example.prompt} {example.chosen or ''}"
        return ""

    def deduplicate(
        self,
        examples: List[FormattedExample],
    ) -> List[FormattedExample]:
        """
        Deduplicate examples using embedding similarity.

        Args:
            examples: Examples to deduplicate

        Returns:
            Deduplicated list of examples
        """
        model = self._get_model()
        if model is None:
            # Fallback to hash-based dedup
            seen = set()
            result = []
            for ex in examples:
                h = ex.content_hash()
                if h not in seen:
                    seen.add(h)
                    result.append(ex)
            return result

        import numpy as np

        # Extract texts
        texts = [self._get_text(ex) for ex in examples]

        # Encode in batches
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        embeddings = embeddings / norms

        # Find unique examples
        keep_indices = []
        for i in range(len(embeddings)):
            is_duplicate = False
            for j in keep_indices:
                similarity = np.dot(embeddings[i], embeddings[j])
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep_indices.append(i)

        return [examples[i] for i in keep_indices]
