"""
Learning Topic Queue for Safe Scout.

Provides:
- Priority-based topic queue management
- Topic state tracking (pending, scouting, completed, failed)
- Persistent queue storage with SQLite/JSON backends
- Async queue operations
- Topic deduplication and rate limiting
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class TopicStatus(Enum):
    """Status of a learning topic."""
    PENDING = "pending"
    QUEUED = "queued"
    SCOUTING = "scouting"
    EXTRACTING = "extracting"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Failed compliance check
    RATE_LIMITED = "rate_limited"


class TopicPriority(Enum):
    """Priority levels for learning topics."""
    CRITICAL = 1  # Urgent learning needs
    HIGH = 2      # Important documentation
    NORMAL = 3    # Standard topics
    LOW = 4       # Background learning
    BACKGROUND = 5  # Fill-in when idle


class TopicCategory(Enum):
    """Categories of learning topics."""
    DOCUMENTATION = "documentation"  # Official docs (Python, React, etc.)
    TUTORIAL = "tutorial"            # Learning tutorials
    REFERENCE = "reference"          # API references
    RELEASE_NOTES = "release_notes"  # Version updates
    BEST_PRACTICES = "best_practices"
    SECURITY = "security"            # Security advisories
    RESEARCH = "research"            # Research papers (arXiv, etc.)
    COMMUNITY = "community"          # Stack Overflow, GitHub discussions


@dataclass
class LearningTopic:
    """A topic for the Scout to learn about."""
    topic_id: str
    title: str
    description: str

    # Source URLs to scout (can be multiple)
    seed_urls: List[str] = field(default_factory=list)

    # Search queries to find relevant content
    search_queries: List[str] = field(default_factory=list)

    # Categorization
    category: TopicCategory = TopicCategory.DOCUMENTATION
    priority: TopicPriority = TopicPriority.NORMAL

    # Constraints
    max_pages: int = 10  # Maximum pages to scout per topic
    max_depth: int = 2   # Maximum link depth to follow
    domain_constraints: List[str] = field(default_factory=list)  # Allowed domains

    # State
    status: TopicStatus = TopicStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    pages_scouted: int = 0
    examples_generated: int = 0
    error_message: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.topic_id:
            self.topic_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique topic ID."""
        content = f"{self.title}:{self.description}:{','.join(self.seed_urls)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "topic_id": self.topic_id,
            "title": self.title,
            "description": self.description,
            "seed_urls": self.seed_urls,
            "search_queries": self.search_queries,
            "category": self.category.value,
            "priority": self.priority.value,
            "max_pages": self.max_pages,
            "max_depth": self.max_depth,
            "domain_constraints": self.domain_constraints,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "pages_scouted": self.pages_scouted,
            "examples_generated": self.examples_generated,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningTopic":
        """Create from dictionary."""
        return cls(
            topic_id=data["topic_id"],
            title=data["title"],
            description=data["description"],
            seed_urls=data.get("seed_urls", []),
            search_queries=data.get("search_queries", []),
            category=TopicCategory(data.get("category", "documentation")),
            priority=TopicPriority(data.get("priority", 3)),
            max_pages=data.get("max_pages", 10),
            max_depth=data.get("max_depth", 2),
            domain_constraints=data.get("domain_constraints", []),
            status=TopicStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            pages_scouted=data.get("pages_scouted", 0),
            examples_generated=data.get("examples_generated", 0),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TopicQueueConfig:
    """Configuration for the topic queue."""
    # Storage
    storage_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("NIGHTSHIFT_QUEUE_PATH", "~/.jarvis/scout/queue")
        ).expanduser()
    )
    use_sqlite: bool = True  # Use SQLite for persistence

    # Queue behavior
    max_concurrent_topics: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_MAX_CONCURRENT_TOPICS", "3"))
    )
    max_queue_size: int = 1000

    # Rate limiting per domain
    domain_rate_limit_seconds: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_DOMAIN_RATE_LIMIT", "5"))
    )

    # Retry configuration
    max_retries: int = 3
    retry_delay_minutes: int = 30

    # Cleanup
    completed_retention_days: int = 30
    failed_retention_days: int = 7


class TopicQueueBackend(ABC):
    """Abstract backend for topic queue storage."""

    @abstractmethod
    async def save_topic(self, topic: LearningTopic) -> None:
        """Save a topic to storage."""
        pass

    @abstractmethod
    async def load_topic(self, topic_id: str) -> Optional[LearningTopic]:
        """Load a topic by ID."""
        pass

    @abstractmethod
    async def delete_topic(self, topic_id: str) -> bool:
        """Delete a topic."""
        pass

    @abstractmethod
    async def list_topics(
        self,
        status: Optional[TopicStatus] = None,
        priority: Optional[TopicPriority] = None,
        limit: int = 100,
    ) -> List[LearningTopic]:
        """List topics with optional filters."""
        pass

    @abstractmethod
    async def get_next_pending(self) -> Optional[LearningTopic]:
        """Get next pending topic by priority."""
        pass


class SQLiteQueueBackend(TopicQueueBackend):
    """SQLite-based queue storage."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS topics (
                    topic_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    data TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    category TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status_priority
                ON topics(status, priority, created_at)
            """)
            conn.commit()
        finally:
            conn.close()

    async def save_topic(self, topic: LearningTopic) -> None:
        """Save topic to SQLite."""
        topic.updated_at = datetime.now()

        def _save():
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO topics
                    (topic_id, title, description, data, status, priority, category, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    topic.topic_id,
                    topic.title,
                    topic.description,
                    json.dumps(topic.to_dict()),
                    topic.status.value,
                    topic.priority.value,
                    topic.category.value,
                    topic.created_at.isoformat(),
                    topic.updated_at.isoformat(),
                ))
                conn.commit()
            finally:
                conn.close()

        await asyncio.get_event_loop().run_in_executor(None, _save)

    async def load_topic(self, topic_id: str) -> Optional[LearningTopic]:
        """Load topic from SQLite."""
        def _load():
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    "SELECT data FROM topics WHERE topic_id = ?",
                    (topic_id,)
                )
                row = cursor.fetchone()
                if row:
                    return LearningTopic.from_dict(json.loads(row[0]))
                return None
            finally:
                conn.close()

        return await asyncio.get_event_loop().run_in_executor(None, _load)

    async def delete_topic(self, topic_id: str) -> bool:
        """Delete topic from SQLite."""
        def _delete():
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    "DELETE FROM topics WHERE topic_id = ?",
                    (topic_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        return await asyncio.get_event_loop().run_in_executor(None, _delete)

    async def list_topics(
        self,
        status: Optional[TopicStatus] = None,
        priority: Optional[TopicPriority] = None,
        limit: int = 100,
    ) -> List[LearningTopic]:
        """List topics with filters."""
        def _list():
            conn = sqlite3.connect(self.db_path)
            try:
                query = "SELECT data FROM topics WHERE 1=1"
                params = []

                if status:
                    query += " AND status = ?"
                    params.append(status.value)

                if priority:
                    query += " AND priority = ?"
                    params.append(priority.value)

                query += " ORDER BY priority ASC, created_at ASC LIMIT ?"
                params.append(limit)

                cursor = conn.execute(query, params)
                return [
                    LearningTopic.from_dict(json.loads(row[0]))
                    for row in cursor.fetchall()
                ]
            finally:
                conn.close()

        return await asyncio.get_event_loop().run_in_executor(None, _list)

    async def get_next_pending(self) -> Optional[LearningTopic]:
        """Get next pending topic by priority."""
        def _get_next():
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute("""
                    SELECT data FROM topics
                    WHERE status = 'pending'
                    ORDER BY priority ASC, created_at ASC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                if row:
                    return LearningTopic.from_dict(json.loads(row[0]))
                return None
            finally:
                conn.close()

        return await asyncio.get_event_loop().run_in_executor(None, _get_next)


class TopicQueue:
    """
    Priority-based learning topic queue.

    Manages topics for the Safe Scout to process, with:
    - Priority ordering
    - Domain rate limiting
    - Deduplication
    - Persistent storage
    """

    def __init__(
        self,
        config: Optional[TopicQueueConfig] = None,
        backend: Optional[TopicQueueBackend] = None,
    ):
        self.config = config or TopicQueueConfig()

        # Initialize backend
        if backend:
            self._backend = backend
        elif self.config.use_sqlite:
            db_path = self.config.storage_path / "topics.db"
            self._backend = SQLiteQueueBackend(db_path)
        else:
            raise ValueError("No backend configured")

        # In-memory tracking
        self._active_topics: Set[str] = set()
        self._domain_last_access: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def enqueue(
        self,
        topic: LearningTopic,
        deduplicate: bool = True,
    ) -> bool:
        """
        Add a topic to the queue.

        Args:
            topic: The learning topic to enqueue
            deduplicate: Skip if similar topic exists

        Returns:
            True if topic was enqueued
        """
        async with self._lock:
            # Check for duplicates
            if deduplicate:
                existing = await self._backend.load_topic(topic.topic_id)
                if existing:
                    logger.debug(f"Topic already exists: {topic.topic_id}")
                    return False

            # Check queue size
            pending = await self._backend.list_topics(
                status=TopicStatus.PENDING,
                limit=self.config.max_queue_size + 1,
            )
            if len(pending) >= self.config.max_queue_size:
                logger.warning("Queue is full, rejecting topic")
                return False

            # Save topic
            topic.status = TopicStatus.PENDING
            await self._backend.save_topic(topic)

            logger.info(f"Enqueued topic: {topic.title} (priority: {topic.priority.name})")
            return True

    async def dequeue(self) -> Optional[LearningTopic]:
        """
        Get the next topic to process.

        Returns:
            Next pending topic or None if queue is empty
        """
        async with self._lock:
            # Check concurrent limit
            if len(self._active_topics) >= self.config.max_concurrent_topics:
                logger.debug("Max concurrent topics reached")
                return None

            # Get next pending topic
            topic = await self._backend.get_next_pending()
            if not topic:
                return None

            # Check domain rate limits
            for url in topic.seed_urls:
                domain = urlparse(url).netloc
                if domain in self._domain_last_access:
                    elapsed = (datetime.now() - self._domain_last_access[domain]).total_seconds()
                    if elapsed < self.config.domain_rate_limit_seconds:
                        topic.status = TopicStatus.RATE_LIMITED
                        await self._backend.save_topic(topic)
                        continue

            # Mark as scouting
            topic.status = TopicStatus.SCOUTING
            topic.started_at = datetime.now()
            await self._backend.save_topic(topic)

            self._active_topics.add(topic.topic_id)

            # Update domain access times
            for url in topic.seed_urls:
                domain = urlparse(url).netloc
                self._domain_last_access[domain] = datetime.now()

            logger.info(f"Dequeued topic: {topic.title}")
            return topic

    async def complete(
        self,
        topic_id: str,
        pages_scouted: int = 0,
        examples_generated: int = 0,
    ) -> None:
        """Mark a topic as completed."""
        async with self._lock:
            topic = await self._backend.load_topic(topic_id)
            if not topic:
                return

            topic.status = TopicStatus.COMPLETED
            topic.completed_at = datetime.now()
            topic.pages_scouted = pages_scouted
            topic.examples_generated = examples_generated

            await self._backend.save_topic(topic)
            self._active_topics.discard(topic_id)

            logger.info(
                f"Completed topic: {topic.title} "
                f"(pages: {pages_scouted}, examples: {examples_generated})"
            )

    async def fail(
        self,
        topic_id: str,
        error: str,
        blocked: bool = False,
    ) -> None:
        """Mark a topic as failed."""
        async with self._lock:
            topic = await self._backend.load_topic(topic_id)
            if not topic:
                return

            topic.status = TopicStatus.BLOCKED if blocked else TopicStatus.FAILED
            topic.error_message = error
            topic.completed_at = datetime.now()

            await self._backend.save_topic(topic)
            self._active_topics.discard(topic_id)

            logger.warning(f"Failed topic: {topic.title} - {error}")

    async def retry(self, topic_id: str) -> bool:
        """Retry a failed topic."""
        topic = await self._backend.load_topic(topic_id)
        if not topic:
            return False

        retry_count = topic.metadata.get("retry_count", 0)
        if retry_count >= self.config.max_retries:
            logger.warning(f"Max retries reached for topic: {topic.title}")
            return False

        topic.status = TopicStatus.PENDING
        topic.error_message = None
        topic.metadata["retry_count"] = retry_count + 1

        await self._backend.save_topic(topic)
        return True

    async def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        pending = await self._backend.list_topics(status=TopicStatus.PENDING)
        scouting = await self._backend.list_topics(status=TopicStatus.SCOUTING)
        completed = await self._backend.list_topics(status=TopicStatus.COMPLETED)
        failed = await self._backend.list_topics(status=TopicStatus.FAILED)
        blocked = await self._backend.list_topics(status=TopicStatus.BLOCKED)

        return {
            "pending": len(pending),
            "scouting": len(scouting),
            "completed": len(completed),
            "failed": len(failed),
            "blocked": len(blocked),
            "active_topics": len(self._active_topics),
            "total_examples": sum(t.examples_generated for t in completed),
        }

    async def cleanup_old_topics(self) -> int:
        """Remove old completed/failed topics."""
        now = datetime.now()
        removed = 0

        # Cleanup completed
        completed = await self._backend.list_topics(status=TopicStatus.COMPLETED)
        for topic in completed:
            if topic.completed_at:
                age = (now - topic.completed_at).days
                if age > self.config.completed_retention_days:
                    await self._backend.delete_topic(topic.topic_id)
                    removed += 1

        # Cleanup failed
        failed = await self._backend.list_topics(status=TopicStatus.FAILED)
        for topic in failed:
            if topic.completed_at:
                age = (now - topic.completed_at).days
                if age > self.config.failed_retention_days:
                    await self._backend.delete_topic(topic.topic_id)
                    removed += 1

        if removed:
            logger.info(f"Cleaned up {removed} old topics")

        return removed


# Factory functions for common topic types
def create_documentation_topic(
    title: str,
    docs_url: str,
    description: Optional[str] = None,
    priority: TopicPriority = TopicPriority.NORMAL,
) -> LearningTopic:
    """Create a documentation learning topic."""
    domain = urlparse(docs_url).netloc

    return LearningTopic(
        topic_id="",  # Will be auto-generated
        title=title,
        description=description or f"Learn {title} from official documentation",
        seed_urls=[docs_url],
        category=TopicCategory.DOCUMENTATION,
        priority=priority,
        domain_constraints=[domain],
        max_pages=20,
        max_depth=3,
    )


def create_release_notes_topic(
    project: str,
    version: str,
    url: str,
) -> LearningTopic:
    """Create a release notes learning topic."""
    return LearningTopic(
        topic_id="",
        title=f"{project} {version} Release Notes",
        description=f"Learn about new features in {project} {version}",
        seed_urls=[url],
        category=TopicCategory.RELEASE_NOTES,
        priority=TopicPriority.HIGH,
        max_pages=5,
        max_depth=1,
    )


def create_tutorial_topic(
    title: str,
    urls: List[str],
    search_queries: Optional[List[str]] = None,
) -> LearningTopic:
    """Create a tutorial learning topic."""
    return LearningTopic(
        topic_id="",
        title=title,
        description=f"Learn {title} through tutorials",
        seed_urls=urls,
        search_queries=search_queries or [],
        category=TopicCategory.TUTORIAL,
        priority=TopicPriority.NORMAL,
        max_pages=15,
        max_depth=2,
    )


# Convenience exports
__all__ = [
    "TopicQueue",
    "TopicQueueConfig",
    "LearningTopic",
    "TopicStatus",
    "TopicPriority",
    "TopicCategory",
    "SQLiteQueueBackend",
    "create_documentation_topic",
    "create_release_notes_topic",
    "create_tutorial_topic",
]
