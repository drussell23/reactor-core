"""
Base formatter protocol and data structures for training data formatting.

Defines:
- FormattedExample: Training-ready example in various formats
- OutputFormat: Enum of supported output formats
- BaseFormatter: Abstract protocol for formatters
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable

from reactor_core.ingestion.base_ingestor import RawInteraction


class OutputFormat(Enum):
    """Supported training data output formats."""
    CHATML = "chatml"           # OpenAI-style messages
    ALPACA = "alpaca"           # Instruction-tuning format
    PREFERENCE = "preference"   # DPO pairs (chosen/rejected)
    RAW = "raw"                 # Raw text format


@dataclass
class FormattedExample:
    """
    Training-ready example in a standardized format.

    Supports multiple output formats:
    - ChatML: List of messages with roles
    - Alpaca: Instruction/input/output tuple
    - Preference: DPO pairs with chosen/rejected responses
    """

    # Format type
    format_type: OutputFormat

    # ChatML format: [{"role": "system/user/assistant", "content": "..."}]
    messages: Optional[List[Dict[str, str]]] = None

    # Alpaca format
    instruction: Optional[str] = None
    input_text: Optional[str] = None
    output_text: Optional[str] = None

    # Preference format (for DPO)
    prompt: Optional[str] = None
    chosen: Optional[str] = None
    rejected: Optional[str] = None

    # Quality metadata
    quality_score: float = 1.0
    source_id: Optional[str] = None
    source_type: Optional[str] = None

    # Token count estimates
    estimated_tokens: int = 0

    # Flags
    is_correction: bool = False
    is_synthetic: bool = False

    def __post_init__(self):
        """Compute derived fields."""
        if self.estimated_tokens == 0:
            self.estimated_tokens = self._estimate_tokens()

    def _estimate_tokens(self) -> int:
        """Rough token count estimate (4 chars per token)."""
        total_chars = 0

        if self.messages:
            for msg in self.messages:
                total_chars += len(msg.get("content", ""))

        if self.instruction:
            total_chars += len(self.instruction)
        if self.input_text:
            total_chars += len(self.input_text)
        if self.output_text:
            total_chars += len(self.output_text)
        if self.prompt:
            total_chars += len(self.prompt)
        if self.chosen:
            total_chars += len(self.chosen)
        if self.rejected:
            total_chars += len(self.rejected)

        return total_chars // 4

    def content_hash(self) -> str:
        """Generate hash for deduplication."""
        content = ""

        if self.messages:
            content = str(self.messages)
        elif self.instruction:
            content = f"{self.instruction}{self.input_text or ''}{self.output_text or ''}"
        elif self.prompt:
            content = f"{self.prompt}{self.chosen or ''}"

        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_chatml_dict(self) -> Dict[str, Any]:
        """Convert to ChatML format dictionary."""
        if self.format_type == OutputFormat.CHATML and self.messages:
            return {"messages": self.messages}

        # Convert other formats to ChatML
        messages = []

        if self.instruction:
            messages.append({"role": "system", "content": self.instruction})
        if self.input_text:
            messages.append({"role": "user", "content": self.input_text})
        if self.output_text:
            messages.append({"role": "assistant", "content": self.output_text})

        return {"messages": messages}

    def to_alpaca_dict(self) -> Dict[str, Any]:
        """Convert to Alpaca format dictionary."""
        if self.format_type == OutputFormat.ALPACA:
            return {
                "instruction": self.instruction or "",
                "input": self.input_text or "",
                "output": self.output_text or "",
            }

        # Convert from ChatML
        if self.messages:
            instruction = ""
            input_text = ""
            output_text = ""

            for msg in self.messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "system":
                    instruction = content
                elif role == "user":
                    input_text = content
                elif role == "assistant":
                    output_text = content

            return {
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
            }

        return {
            "instruction": "",
            "input": "",
            "output": "",
        }

    def to_preference_dict(self) -> Dict[str, Any]:
        """Convert to DPO preference format dictionary."""
        if self.format_type == OutputFormat.PREFERENCE:
            return {
                "prompt": self.prompt or "",
                "chosen": self.chosen or "",
                "rejected": self.rejected or "",
            }

        return {
            "prompt": "",
            "chosen": "",
            "rejected": "",
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary based on format type."""
        base = {
            "format_type": self.format_type.value,
            "quality_score": self.quality_score,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "estimated_tokens": self.estimated_tokens,
            "is_correction": self.is_correction,
            "is_synthetic": self.is_synthetic,
        }

        if self.format_type == OutputFormat.CHATML:
            base.update(self.to_chatml_dict())
        elif self.format_type == OutputFormat.ALPACA:
            base.update(self.to_alpaca_dict())
        elif self.format_type == OutputFormat.PREFERENCE:
            base.update(self.to_preference_dict())

        return base


@runtime_checkable
class BaseFormatter(Protocol):
    """Protocol for formatters."""

    @property
    def output_format(self) -> OutputFormat:
        """The format this formatter produces."""
        ...

    async def format(
        self,
        interaction: RawInteraction,
    ) -> Optional[FormattedExample]:
        """
        Format a single interaction.

        Args:
            interaction: Raw interaction to format

        Returns:
            FormattedExample or None if formatting fails
        """
        ...

    async def format_batch(
        self,
        interactions: List[RawInteraction],
    ) -> AsyncIterator[FormattedExample]:
        """
        Format a batch of interactions.

        Args:
            interactions: List of interactions to format

        Yields:
            FormattedExample for each successfully formatted interaction
        """
        ...


class AbstractFormatter(ABC):
    """Abstract base for formatters with common functionality."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_length: int = 4096,
        include_system_context: bool = True,
    ):
        """
        Initialize formatter.

        Args:
            system_prompt: Default system prompt to use
            max_length: Maximum sequence length (in estimated tokens)
            include_system_context: Whether to include system context
        """
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.include_system_context = include_system_context

    @property
    @abstractmethod
    def output_format(self) -> OutputFormat:
        """The format this formatter produces."""
        ...

    @abstractmethod
    async def format(
        self,
        interaction: RawInteraction,
    ) -> Optional[FormattedExample]:
        """Format a single interaction."""
        ...

    async def format_batch(
        self,
        interactions: List[RawInteraction],
    ) -> AsyncIterator[FormattedExample]:
        """Format a batch of interactions."""
        for interaction in interactions:
            example = await self.format(interaction)
            if example is not None:
                yield example

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximate token limit."""
        # Rough estimate: 4 chars per token
        max_chars = max_tokens * 4

        if len(text) <= max_chars:
            return text

        # Truncate and add ellipsis
        return text[: max_chars - 3] + "..."

    def _build_system_context(self, interaction: RawInteraction) -> Optional[str]:
        """Build system context from interaction."""
        if not self.include_system_context:
            return self.system_prompt

        parts = []

        if self.system_prompt:
            parts.append(self.system_prompt)

        if interaction.system_context:
            parts.append(f"Context: {interaction.system_context}")

        return "\n\n".join(parts) if parts else None
