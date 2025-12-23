"""
ChatML formatter for OpenAI-style conversation format.

Produces format:
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from reactor_core.formatting.base_formatter import (
    AbstractFormatter,
    FormattedExample,
    OutputFormat,
)
from reactor_core.ingestion.base_ingestor import InteractionOutcome, RawInteraction


class ChatMLFormatter(AbstractFormatter):
    """
    Formatter for ChatML (OpenAI) conversation format.

    Supports:
    - Single-turn conversations
    - Multi-turn with history
    - System prompts
    - Correction handling
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_length: int = 4096,
        include_system_context: bool = True,
        include_previous_turns: bool = True,
        max_previous_turns: int = 5,
    ):
        """
        Initialize ChatML formatter.

        Args:
            system_prompt: Default system prompt
            max_length: Maximum sequence length (tokens)
            include_system_context: Include interaction system context
            include_previous_turns: Include conversation history
            max_previous_turns: Maximum turns of history to include
        """
        super().__init__(
            system_prompt=system_prompt,
            max_length=max_length,
            include_system_context=include_system_context,
        )
        self.include_previous_turns = include_previous_turns
        self.max_previous_turns = max_previous_turns

    @property
    def output_format(self) -> OutputFormat:
        return OutputFormat.CHATML

    async def format(
        self,
        interaction: RawInteraction,
    ) -> Optional[FormattedExample]:
        """Format interaction to ChatML."""
        # Must have user input and output
        if not interaction.user_input or not interaction.assistant_output:
            return None

        messages: List[Dict[str, str]] = []

        # Add system message
        system_content = self._build_system_context(interaction)
        if system_content:
            messages.append({
                "role": "system",
                "content": system_content,
            })

        # Add previous turns if available
        if self.include_previous_turns and interaction.previous_turns:
            for turn in interaction.previous_turns[-self.max_previous_turns:]:
                if "user" in turn:
                    messages.append({
                        "role": "user",
                        "content": turn["user"],
                    })
                if "assistant" in turn:
                    messages.append({
                        "role": "assistant",
                        "content": turn["assistant"],
                    })

        # Add current turn
        messages.append({
            "role": "user",
            "content": interaction.user_input,
        })

        # Handle corrections specially
        if interaction.is_correction and interaction.correction_improved:
            # Use the corrected response as the training target
            messages.append({
                "role": "assistant",
                "content": interaction.correction_improved,
            })
        else:
            messages.append({
                "role": "assistant",
                "content": interaction.assistant_output,
            })

        # Check length
        example = FormattedExample(
            format_type=OutputFormat.CHATML,
            messages=messages,
            quality_score=interaction.quality_score,
            source_id=interaction.id,
            source_type=interaction.source_type.value,
            is_correction=interaction.is_correction,
        )

        if example.estimated_tokens > self.max_length:
            # Truncate messages if too long
            messages = self._truncate_messages(messages, self.max_length)
            example.messages = messages
            example.estimated_tokens = example._estimate_tokens()

        return example

    def _truncate_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
    ) -> List[Dict[str, str]]:
        """Truncate messages to fit within token limit."""
        # Calculate current length
        total = sum(len(m.get("content", "")) // 4 for m in messages)

        if total <= max_tokens:
            return messages

        # Prioritize: system, last user, last assistant
        # Truncate earlier messages first
        truncated = []

        # Keep system message if present
        if messages and messages[0].get("role") == "system":
            truncated.append(messages[0])
            messages = messages[1:]

        # Keep last user-assistant pair
        if len(messages) >= 2:
            last_two = messages[-2:]
            earlier = messages[:-2]

            # Include only as many earlier messages as fit
            remaining_budget = max_tokens - sum(
                len(m.get("content", "")) // 4 for m in truncated + last_two
            )

            for msg in reversed(earlier):
                msg_tokens = len(msg.get("content", "")) // 4
                if remaining_budget >= msg_tokens:
                    truncated.insert(len(truncated), msg)
                    remaining_budget -= msg_tokens

            truncated.extend(last_two)
        else:
            truncated.extend(messages)

        return truncated


class PreferenceFormatter(AbstractFormatter):
    """
    Formatter for DPO preference pairs.

    Creates (prompt, chosen, rejected) tuples from:
    - Corrections: original is rejected, corrected is chosen
    - Success/failure pairs with same input

    This is especially valuable for training models to:
    - Avoid mistakes that led to corrections
    - Prefer responses that led to positive feedback
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_length: int = 4096,
    ):
        super().__init__(
            system_prompt=system_prompt,
            max_length=max_length,
        )
        self._pending_pairs: Dict[str, RawInteraction] = {}

    @property
    def output_format(self) -> OutputFormat:
        return OutputFormat.PREFERENCE

    async def format(
        self,
        interaction: RawInteraction,
    ) -> Optional[FormattedExample]:
        """Format correction as preference pair."""
        # Corrections directly provide chosen/rejected
        if interaction.is_correction:
            if not interaction.correction_original or not interaction.correction_improved:
                return None

            # Build prompt from user input
            prompt_parts = []

            if self.system_prompt:
                prompt_parts.append(f"<|system|>\n{self.system_prompt}")

            prompt_parts.append(f"<|user|>\n{interaction.user_input or ''}")
            prompt_parts.append("<|assistant|>\n")

            prompt = "\n".join(prompt_parts)

            return FormattedExample(
                format_type=OutputFormat.PREFERENCE,
                prompt=prompt,
                chosen=interaction.correction_improved,
                rejected=interaction.correction_original,
                quality_score=interaction.quality_score,
                source_id=interaction.id,
                source_type=interaction.source_type.value,
                is_correction=True,
            )

        # Non-corrections: try to find matching pair
        # This requires maintaining state across interactions
        if interaction.user_input:
            input_hash = hash(interaction.user_input)

            if input_hash in self._pending_pairs:
                other = self._pending_pairs.pop(input_hash)

                # Determine which is chosen/rejected based on outcome
                if self._is_better_outcome(interaction, other):
                    chosen = interaction.assistant_output
                    rejected = other.assistant_output
                else:
                    chosen = other.assistant_output
                    rejected = interaction.assistant_output

                if chosen and rejected and chosen != rejected:
                    prompt = f"<|user|>\n{interaction.user_input}\n<|assistant|>\n"

                    return FormattedExample(
                        format_type=OutputFormat.PREFERENCE,
                        prompt=prompt,
                        chosen=chosen,
                        rejected=rejected,
                        quality_score=(interaction.quality_score + other.quality_score) / 2,
                        source_id=interaction.id,
                        source_type=interaction.source_type.value,
                    )
            else:
                # Store for potential future pairing
                self._pending_pairs[input_hash] = interaction

        return None

    def _is_better_outcome(
        self,
        a: RawInteraction,
        b: RawInteraction,
    ) -> bool:
        """Determine if interaction 'a' has a better outcome than 'b'."""
        outcome_order = {
            InteractionOutcome.POSITIVE_FEEDBACK: 5,
            InteractionOutcome.SUCCESS: 4,
            InteractionOutcome.ENGAGED: 3,
            InteractionOutcome.PARTIAL: 2,
            InteractionOutcome.UNKNOWN: 1,
            InteractionOutcome.DEFERRED: 0,
            InteractionOutcome.DISMISSED: -1,
            InteractionOutcome.FAILURE: -2,
            InteractionOutcome.NEGATIVE_FEEDBACK: -3,
        }

        a_score = outcome_order.get(a.outcome, 0)
        b_score = outcome_order.get(b.outcome, 0)

        if a_score != b_score:
            return a_score > b_score

        # Tie-break by confidence
        return a.confidence > b.confidence
