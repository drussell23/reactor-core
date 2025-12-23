"""
Alpaca formatter for instruction-tuning format.

Produces format:
{
    "instruction": "Task description or system prompt",
    "input": "User's specific request or context",
    "output": "Expected response"
}
"""

from __future__ import annotations

from typing import Optional

from reactor_core.formatting.base_formatter import (
    AbstractFormatter,
    FormattedExample,
    OutputFormat,
)
from reactor_core.ingestion.base_ingestor import RawInteraction


class AlpacaFormatter(AbstractFormatter):
    """
    Formatter for Alpaca instruction-tuning format.

    The Alpaca format is:
    - instruction: High-level task description
    - input: User's specific request (can be empty)
    - output: Model's expected response

    This is commonly used for fine-tuning Llama models.
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_length: int = 4096,
        include_system_context: bool = True,
        empty_input_if_no_context: bool = True,
    ):
        """
        Initialize Alpaca formatter.

        Args:
            system_prompt: Default instruction/system prompt
            max_length: Maximum sequence length (tokens)
            include_system_context: Include system context in instruction
            empty_input_if_no_context: Use empty input if no context
        """
        super().__init__(
            system_prompt=system_prompt,
            max_length=max_length,
            include_system_context=include_system_context,
        )
        self.empty_input_if_no_context = empty_input_if_no_context

    @property
    def output_format(self) -> OutputFormat:
        return OutputFormat.ALPACA

    async def format(
        self,
        interaction: RawInteraction,
    ) -> Optional[FormattedExample]:
        """Format interaction to Alpaca format."""
        # Must have both input and output
        if not interaction.user_input or not interaction.assistant_output:
            return None

        # Build instruction
        instruction = self._build_instruction(interaction)

        # Build input (user's request)
        input_text = interaction.user_input

        # Build output
        if interaction.is_correction and interaction.correction_improved:
            output_text = interaction.correction_improved
        else:
            output_text = interaction.assistant_output

        # Check if we need to add context to input
        if interaction.system_context and self.include_system_context:
            input_text = f"{interaction.system_context}\n\n{input_text}"

        # Truncate if necessary
        example = FormattedExample(
            format_type=OutputFormat.ALPACA,
            instruction=instruction,
            input_text=input_text,
            output_text=output_text,
            quality_score=interaction.quality_score,
            source_id=interaction.id,
            source_type=interaction.source_type.value,
            is_correction=interaction.is_correction,
        )

        if example.estimated_tokens > self.max_length:
            # Truncate output first, then input
            remaining = self.max_length - (len(instruction) // 4)
            input_budget = remaining // 2
            output_budget = remaining - input_budget

            if example.input_text:
                example.input_text = self._truncate_text(
                    example.input_text, input_budget
                )
            if example.output_text:
                example.output_text = self._truncate_text(
                    example.output_text, output_budget
                )

            example.estimated_tokens = example._estimate_tokens()

        return example

    def _build_instruction(self, interaction: RawInteraction) -> str:
        """Build instruction field from interaction context."""
        parts = []

        # Start with system prompt
        if self.system_prompt:
            parts.append(self.system_prompt)

        # Add task-specific instruction based on source type
        source_type = interaction.source_type.value

        if source_type == "feedback" and interaction.is_correction:
            parts.append(
                "The following is a corrected interaction. "
                "Learn from the improved response."
            )
        elif source_type == "auth_record":
            parts.append(
                "Respond appropriately to this authentication-related request."
            )
        elif source_type == "telemetry":
            parts.append(
                "Respond to the user's request based on the given context."
            )

        # Add tags as hints
        if interaction.tags:
            relevant_tags = [
                t for t in interaction.tags
                if not t.startswith("event:") and not t.startswith("response:")
            ]
            if relevant_tags:
                parts.append(f"Context tags: {', '.join(relevant_tags)}")

        return "\n".join(parts) if parts else "Respond to the user's request."


class RawTextFormatter(AbstractFormatter):
    """
    Simple raw text formatter for basic fine-tuning.

    Produces plain text in format:
    <|user|>
    User message here
    <|assistant|>
    Assistant response here
    """

    def __init__(
        self,
        user_token: str = "<|user|>",
        assistant_token: str = "<|assistant|>",
        system_token: str = "<|system|>",
        max_length: int = 4096,
        include_system: bool = True,
    ):
        super().__init__(max_length=max_length)
        self.user_token = user_token
        self.assistant_token = assistant_token
        self.system_token = system_token
        self.include_system = include_system

    @property
    def output_format(self) -> OutputFormat:
        return OutputFormat.RAW

    async def format(
        self,
        interaction: RawInteraction,
    ) -> Optional[FormattedExample]:
        """Format interaction to raw text."""
        if not interaction.user_input or not interaction.assistant_output:
            return None

        parts = []

        # Add system if present
        if self.include_system and interaction.system_context:
            parts.append(f"{self.system_token}\n{interaction.system_context}")

        # Add user message
        parts.append(f"{self.user_token}\n{interaction.user_input}")

        # Add assistant response
        output = (
            interaction.correction_improved
            if interaction.is_correction and interaction.correction_improved
            else interaction.assistant_output
        )
        parts.append(f"{self.assistant_token}\n{output}")

        raw_text = "\n".join(parts)

        return FormattedExample(
            format_type=OutputFormat.RAW,
            output_text=raw_text,
            quality_score=interaction.quality_score,
            source_id=interaction.id,
            source_type=interaction.source_type.value,
            is_correction=interaction.is_correction,
        )
