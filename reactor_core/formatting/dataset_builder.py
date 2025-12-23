"""
Dataset builder for HuggingFace datasets output.

Converts FormattedExamples to HuggingFace Dataset format
and saves in various formats (JSONL, Parquet, Arrow).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from reactor_core.formatting.base_formatter import FormattedExample, OutputFormat
from reactor_core.formatting.quality_filter import FilterResult, QualityFilter

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Statistics for built dataset."""
    total_examples: int = 0
    chatml_count: int = 0
    alpaca_count: int = 0
    preference_count: int = 0
    correction_count: int = 0
    synthetic_count: int = 0
    avg_quality: float = 0.0
    avg_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_examples": self.total_examples,
            "chatml_count": self.chatml_count,
            "alpaca_count": self.alpaca_count,
            "preference_count": self.preference_count,
            "correction_count": self.correction_count,
            "synthetic_count": self.synthetic_count,
            "avg_quality": round(self.avg_quality, 4),
            "avg_tokens": self.avg_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class DatasetMetadata:
    """Metadata for saved dataset."""
    name: str
    version: str
    created_at: datetime
    format_type: str
    stats: DatasetStats
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "format_type": self.format_type,
            "stats": self.stats.to_dict(),
            "config": self.config,
        }


class DatasetBuilder:
    """
    Build HuggingFace-compatible datasets from FormattedExamples.

    Supports:
    - JSONL output (for easy inspection)
    - Parquet output (for efficient loading)
    - HuggingFace Dataset objects
    - Train/validation splits
    """

    def __init__(
        self,
        output_format: OutputFormat = OutputFormat.CHATML,
        train_split: float = 0.9,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize dataset builder.

        Args:
            output_format: Primary output format
            train_split: Fraction for training set (rest for validation)
            shuffle: Whether to shuffle before splitting
            seed: Random seed for reproducibility
        """
        self.output_format = output_format
        self.train_split = train_split
        self.shuffle = shuffle
        self.seed = seed

        self._examples: List[FormattedExample] = []
        self._stats = DatasetStats()

    def add(self, example: FormattedExample) -> None:
        """Add a single example."""
        self._examples.append(example)
        self._update_stats(example)

    def add_batch(self, examples: List[FormattedExample]) -> None:
        """Add a batch of examples."""
        for example in examples:
            self.add(example)

    def add_from_filter_result(self, result: FilterResult) -> None:
        """Add examples from a filter result."""
        self.add_batch(result.passed)

    def _update_stats(self, example: FormattedExample) -> None:
        """Update running statistics."""
        self._stats.total_examples += 1
        self._stats.total_tokens += example.estimated_tokens

        # Format counts
        if example.format_type == OutputFormat.CHATML:
            self._stats.chatml_count += 1
        elif example.format_type == OutputFormat.ALPACA:
            self._stats.alpaca_count += 1
        elif example.format_type == OutputFormat.PREFERENCE:
            self._stats.preference_count += 1

        # Special flags
        if example.is_correction:
            self._stats.correction_count += 1
        if example.is_synthetic:
            self._stats.synthetic_count += 1

        # Update averages
        n = self._stats.total_examples
        self._stats.avg_quality = (
            (self._stats.avg_quality * (n - 1) + example.quality_score) / n
        )
        self._stats.avg_tokens = self._stats.total_tokens // n

    def _convert_to_output_format(
        self,
        example: FormattedExample,
    ) -> Dict[str, Any]:
        """Convert example to target output format."""
        if self.output_format == OutputFormat.CHATML:
            return example.to_chatml_dict()
        elif self.output_format == OutputFormat.ALPACA:
            return example.to_alpaca_dict()
        elif self.output_format == OutputFormat.PREFERENCE:
            return example.to_preference_dict()
        else:
            return example.to_dict()

    def _split_data(
        self,
        data: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split data into train and validation sets."""
        if self.shuffle:
            import random
            random.seed(self.seed)
            data = data.copy()
            random.shuffle(data)

        split_idx = int(len(data) * self.train_split)
        return data[:split_idx], data[split_idx:]

    def build(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build the dataset as dictionaries.

        Returns:
            Dict with 'train' and 'validation' splits
        """
        converted = [
            self._convert_to_output_format(ex)
            for ex in self._examples
        ]

        train, val = self._split_data(converted)

        return {
            "train": train,
            "validation": val,
        }

    def save_jsonl(
        self,
        output_dir: Union[str, Path],
        name: str = "dataset",
    ) -> Path:
        """
        Save dataset as JSONL files.

        Args:
            output_dir: Output directory
            name: Dataset name

        Returns:
            Path to output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        splits = self.build()

        # Save train split
        train_path = output_dir / f"{name}_train.jsonl"
        with open(train_path, "w") as f:
            for item in splits["train"]:
                f.write(json.dumps(item) + "\n")

        # Save validation split
        val_path = output_dir / f"{name}_validation.jsonl"
        with open(val_path, "w") as f:
            for item in splits["validation"]:
                f.write(json.dumps(item) + "\n")

        # Save metadata
        metadata = DatasetMetadata(
            name=name,
            version="1.0.0",
            created_at=datetime.now(),
            format_type=self.output_format.value,
            stats=self._stats,
            config={
                "train_split": self.train_split,
                "shuffle": self.shuffle,
                "seed": self.seed,
            },
        )

        meta_path = output_dir / f"{name}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(
            f"Saved dataset to {output_dir}: "
            f"{len(splits['train'])} train, {len(splits['validation'])} validation"
        )

        return output_dir

    def to_huggingface(self) -> "Dataset":
        """
        Convert to HuggingFace Dataset object.

        Returns:
            HuggingFace Dataset with train and validation splits
        """
        try:
            from datasets import Dataset, DatasetDict
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library required. "
                "Install with: pip install datasets"
            )

        splits = self.build()

        return DatasetDict({
            "train": Dataset.from_list(splits["train"]),
            "validation": Dataset.from_list(splits["validation"]),
        })

    def save_huggingface(
        self,
        output_dir: Union[str, Path],
        name: str = "dataset",
    ) -> Path:
        """
        Save as HuggingFace dataset (Arrow format).

        Args:
            output_dir: Output directory
            name: Dataset name

        Returns:
            Path to saved dataset
        """
        output_dir = Path(output_dir) / name
        dataset = self.to_huggingface()
        dataset.save_to_disk(str(output_dir))

        # Save metadata
        metadata = DatasetMetadata(
            name=name,
            version="1.0.0",
            created_at=datetime.now(),
            format_type=self.output_format.value,
            stats=self._stats,
            config={
                "train_split": self.train_split,
                "shuffle": self.shuffle,
                "seed": self.seed,
            },
        )

        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(f"Saved HuggingFace dataset to {output_dir}")
        return output_dir

    def save_parquet(
        self,
        output_dir: Union[str, Path],
        name: str = "dataset",
    ) -> Path:
        """
        Save dataset as Parquet files.

        Args:
            output_dir: Output directory
            name: Dataset name

        Returns:
            Path to output directory
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "PyArrow required for Parquet output. "
                "Install with: pip install pyarrow"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        splits = self.build()

        for split_name, data in splits.items():
            if not data:
                continue

            table = pa.Table.from_pylist(data)
            path = output_dir / f"{name}_{split_name}.parquet"
            pq.write_table(table, path)

        logger.info(f"Saved Parquet dataset to {output_dir}")
        return output_dir

    def get_stats(self) -> DatasetStats:
        """Get current dataset statistics."""
        return self._stats

    def __len__(self) -> int:
        return len(self._examples)

    def __iter__(self) -> Iterator[FormattedExample]:
        return iter(self._examples)

    def clear(self) -> None:
        """Clear all examples and reset stats."""
        self._examples.clear()
        self._stats = DatasetStats()
