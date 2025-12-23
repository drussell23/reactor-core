"""
Batch ingestion processor for Night Shift Training Engine.

Provides parallel processing of multiple data sources with:
- Concurrent ingestion from multiple sources
- Progress tracking and metrics
- Error handling and recovery
- Deduplication support
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    Union,
)

from reactor_core.ingestion.base_ingestor import (
    AbstractIngestor,
    BaseIngestor,
    RawInteraction,
    SourceType,
)
from reactor_core.utils.async_helpers import (
    AsyncSemaphore,
    ProgressTracker,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Statistics for an ingestion run."""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_interactions: int = 0
    filtered_interactions: int = 0
    duplicate_interactions: int = 0

    # By source type
    by_source: Dict[str, int] = field(default_factory=dict)

    # By outcome
    by_outcome: Dict[str, int] = field(default_factory=dict)

    # Timing
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        if not self.started_at or not self.finished_at:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def interactions_per_second(self) -> float:
        if self.duration_seconds == 0:
            return 0.0
        return self.total_interactions / self.duration_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "total_interactions": self.total_interactions,
            "filtered_interactions": self.filtered_interactions,
            "duplicate_interactions": self.duplicate_interactions,
            "by_source": self.by_source,
            "by_outcome": self.by_outcome,
            "duration_seconds": self.duration_seconds,
            "interactions_per_second": self.interactions_per_second,
        }


@dataclass
class IngestionResult:
    """Result from batch ingestion."""
    success: bool
    interactions: List[RawInteraction]
    stats: IngestionStats
    errors: List[str] = field(default_factory=list)

    @property
    def trainable_count(self) -> int:
        """Count of interactions suitable for training."""
        return sum(1 for i in self.interactions if i.is_trainable())


class BatchIngestionProcessor:
    """
    Process multiple data sources in parallel with deduplication.

    Example:
        processor = BatchIngestionProcessor(
            ingestors={
                SourceType.TELEMETRY: TelemetryIngestor(),
                SourceType.FEEDBACK: FeedbackIngestor(),
            },
            max_concurrent=4,
        )

        result = await processor.ingest_all(
            sources=[
                (SourceType.TELEMETRY, "/path/to/events.jsonl"),
                (SourceType.FEEDBACK, "/path/to/feedback.json"),
            ],
            since=datetime.now() - timedelta(days=7),
        )

        for interaction in result.interactions:
            print(interaction.user_input)
    """

    def __init__(
        self,
        ingestors: Dict[SourceType, AbstractIngestor],
        max_concurrent: int = 4,
        deduplicate: bool = True,
        min_quality: float = 0.0,
        progress_callback: Optional[
            Callable[[int, int, str], None]
        ] = None,
    ):
        """
        Initialize batch processor.

        Args:
            ingestors: Map of source type to ingestor instance
            max_concurrent: Max concurrent ingestion tasks
            deduplicate: Enable content-based deduplication
            min_quality: Minimum quality score threshold
            progress_callback: Callback for progress updates (current, total, message)
        """
        self.ingestors = ingestors
        self.max_concurrent = max_concurrent
        self.deduplicate = deduplicate
        self.min_quality = min_quality
        self.progress_callback = progress_callback

        self._semaphore = AsyncSemaphore(max_concurrent)
        self._seen_hashes: Set[str] = set()
        self._lock = asyncio.Lock()

    async def ingest_source(
        self,
        source_type: SourceType,
        source_path: Union[str, Path],
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> AsyncIterator[RawInteraction]:
        """
        Ingest a single source.

        Args:
            source_type: Type of data source
            source_path: Path to source file/directory
            since: Only include data after this timestamp
            until: Only include data before this timestamp

        Yields:
            RawInteraction for each ingested item
        """
        ingestor = self.ingestors.get(source_type)
        if not ingestor:
            logger.warning(f"No ingestor registered for {source_type}")
            return

        source_path = Path(source_path)

        async for interaction in ingestor.ingest(source_path, since, until):
            # Quality filter
            if interaction.quality_score < self.min_quality:
                continue

            # Deduplication
            if self.deduplicate:
                content_hash = interaction.content_hash()
                async with self._lock:
                    if content_hash in self._seen_hashes:
                        continue
                    self._seen_hashes.add(content_hash)

            yield interaction

    async def _ingest_file(
        self,
        source_type: SourceType,
        source_path: Path,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> tuple[List[RawInteraction], Optional[str]]:
        """Ingest a single file with semaphore control."""
        interactions: List[RawInteraction] = []
        error: Optional[str] = None

        try:
            async with self._semaphore:
                async for interaction in self.ingest_source(
                    source_type, source_path, since, until
                ):
                    interactions.append(interaction)

        except Exception as e:
            error = f"Error ingesting {source_path}: {e}"
            logger.error(error)

        return interactions, error

    async def ingest_all(
        self,
        sources: List[tuple[SourceType, Union[str, Path]]],
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> IngestionResult:
        """
        Ingest all sources in parallel.

        Args:
            sources: List of (source_type, path) tuples
            since: Only include data after this timestamp
            until: Only include data before this timestamp

        Returns:
            IngestionResult with all interactions and stats
        """
        stats = IngestionStats(
            total_files=len(sources),
            started_at=datetime.now(),
        )

        all_interactions: List[RawInteraction] = []
        errors: List[str] = []

        # Reset dedup set
        self._seen_hashes.clear()

        # Create progress tracker
        tracker = ProgressTracker(
            total=len(sources),
            callback=self._report_progress if self.progress_callback else None,
        )

        # Process all sources in parallel
        tasks = [
            self._ingest_file(source_type, Path(path), since, until)
            for source_type, path in sources
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            source_type, path = sources[i]

            if isinstance(result, Exception):
                errors.append(f"Error processing {path}: {result}")
                stats.failed_files += 1
            else:
                interactions, error = result
                if error:
                    errors.append(error)
                    stats.failed_files += 1
                else:
                    stats.processed_files += 1

                # Collect interactions
                for interaction in interactions:
                    all_interactions.append(interaction)
                    stats.total_interactions += 1

                    # Track by source
                    source_key = source_type.value
                    stats.by_source[source_key] = stats.by_source.get(source_key, 0) + 1

                    # Track by outcome
                    outcome_key = interaction.outcome.value
                    stats.by_outcome[outcome_key] = stats.by_outcome.get(outcome_key, 0) + 1

            await tracker.update()

        stats.finished_at = datetime.now()
        stats.duplicate_interactions = len(self._seen_hashes) - stats.total_interactions

        return IngestionResult(
            success=stats.failed_files == 0,
            interactions=all_interactions,
            stats=stats,
            errors=errors,
        )

    async def ingest_directory(
        self,
        directory: Union[str, Path],
        source_type: SourceType,
        pattern: str = "*.jsonl",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> IngestionResult:
        """
        Ingest all matching files in a directory.

        Args:
            directory: Directory path
            source_type: Type of data in files
            pattern: Glob pattern for files
            since: Only include data after this timestamp
            until: Only include data before this timestamp

        Returns:
            IngestionResult
        """
        directory = Path(directory)

        if not directory.exists():
            return IngestionResult(
                success=False,
                interactions=[],
                stats=IngestionStats(),
                errors=[f"Directory not found: {directory}"],
            )

        # Find all matching files
        files = list(directory.glob(pattern))

        if not files:
            return IngestionResult(
                success=True,
                interactions=[],
                stats=IngestionStats(total_files=0),
                errors=[],
            )

        # Build source list
        sources = [(source_type, f) for f in files]

        return await self.ingest_all(sources, since, until)

    async def ingest_jarvis_logs(
        self,
        logs_dir: Union[str, Path],
        data_dir: Union[str, Path],
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> IngestionResult:
        """
        Convenience method to ingest all JARVIS data sources.

        Args:
            logs_dir: Path to JARVIS logs directory
            data_dir: Path to JARVIS data directory
            since: Only include data after this timestamp
            until: Only include data before this timestamp

        Returns:
            IngestionResult
        """
        logs_dir = Path(logs_dir)
        data_dir = Path(data_dir)

        sources: List[tuple[SourceType, Path]] = []

        # Find log files
        if logs_dir.exists():
            for log_file in logs_dir.glob("*.log"):
                sources.append((SourceType.RAW_LOG, log_file))

            for jsonl_file in logs_dir.glob("*.jsonl"):
                sources.append((SourceType.TELEMETRY, jsonl_file))

        # Find data files
        if data_dir.exists():
            # State files
            for json_file in data_dir.glob("*_state.json"):
                sources.append((SourceType.STATE_FILE, json_file))

            # Feedback files
            for feedback_file in data_dir.glob("*feedback*.json"):
                sources.append((SourceType.FEEDBACK, feedback_file))

            # Auth records
            for auth_file in data_dir.glob("*auth*.json"):
                sources.append((SourceType.AUTH_RECORD, auth_file))

        logger.info(f"Found {len(sources)} data sources to ingest")

        return await self.ingest_all(sources, since, until)

    async def _report_progress(
        self,
        current: int,
        total: int,
        percent: float,
    ) -> None:
        """Report progress via callback."""
        if self.progress_callback:
            self.progress_callback(
                current,
                total,
                f"Processing {current}/{total} sources ({percent:.1f}%)",
            )

    def get_registered_types(self) -> List[SourceType]:
        """Get list of registered source types."""
        return list(self.ingestors.keys())

    def register_ingestor(
        self,
        source_type: SourceType,
        ingestor: AbstractIngestor,
    ) -> None:
        """Register a new ingestor for a source type."""
        self.ingestors[source_type] = ingestor
