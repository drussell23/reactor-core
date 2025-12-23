"""
Feedback event ingestor for JARVIS user feedback data.

Parses FeedbackEvent data which captures user corrections and engagement.
This is CRITICAL for training as corrections represent high-quality
training signals.

Expected format (from JARVIS backend/core/learning/feedback_loop.py):
{
    "pattern": "TERMINAL_ERROR|TERMINAL_COMPLETION|...",
    "response": "ENGAGED|DISMISSED|DEFERRED|NEGATIVE_FEEDBACK",
    "timestamp": "2024-12-22T02:15:30.123456Z",
    "notification_text": "...",
    "context": {...},
    "time_to_respond": 2.5
}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import aiofiles

from reactor_core.ingestion.base_ingestor import (
    AbstractIngestor,
    InteractionOutcome,
    RawInteraction,
    SourceType,
)

logger = logging.getLogger(__name__)


# Map JARVIS UserResponse to outcomes
RESPONSE_OUTCOME_MAP = {
    "ENGAGED": InteractionOutcome.ENGAGED,
    "DISMISSED": InteractionOutcome.DISMISSED,
    "DEFERRED": InteractionOutcome.DEFERRED,
    "NEGATIVE_FEEDBACK": InteractionOutcome.NEGATIVE_FEEDBACK,
}

# Map NotificationPattern to tags
PATTERN_TAGS = {
    "TERMINAL_ERROR": ["error", "terminal"],
    "TERMINAL_COMPLETION": ["success", "terminal"],
    "TERMINAL_WARNING": ["warning", "terminal"],
    "BROWSER_UPDATE": ["browser", "update"],
    "CODE_DIAGNOSTIC": ["code", "diagnostic"],
    "WORKFLOW_SUGGESTION": ["workflow", "suggestion"],
    "RESOURCE_WARNING": ["resource", "warning"],
    "SECURITY_ALERT": ["security", "alert"],
    "OTHER": ["other"],
}


class FeedbackIngestor(AbstractIngestor):
    """
    Ingestor for JARVIS user feedback events.

    This is the most valuable source for training as it captures:
    - User corrections (NEGATIVE_FEEDBACK)
    - Successful interactions (ENGAGED)
    - Dismissed suggestions (potential negative examples)

    Corrections are especially valuable because they show:
    - What JARVIS said (original response)
    - What the user wanted (correction)
    - Why it was wrong (implicit from context)
    """

    def __init__(
        self,
        min_confidence: float = 0.0,
        include_dismissed: bool = True,
        include_deferred: bool = False,
        patterns: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Initialize feedback ingestor.

        Args:
            min_confidence: Minimum confidence threshold
            include_dismissed: Include dismissed notifications
            include_deferred: Include deferred notifications
            patterns: Filter to specific patterns (None = all)
            tags: Tags to add to all interactions
        """
        super().__init__(
            min_confidence=min_confidence,
            include_failures=True,  # We want corrections!
            tags=tags,
        )
        self.include_dismissed = include_dismissed
        self.include_deferred = include_deferred
        self.patterns = set(patterns) if patterns else None

    @property
    def source_type(self) -> SourceType:
        return SourceType.FEEDBACK

    async def _iter_items(
        self,
        source_path: Path,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Iterate over feedback events from JSON file."""
        if not source_path.exists():
            logger.warning(f"Feedback file not found: {source_path}")
            return

        try:
            async with aiofiles.open(source_path, "r") as f:
                content = await f.read()

            data = json.loads(content)

            # Handle both single object and array formats
            events = data if isinstance(data, list) else [data]

            # Also handle nested structures
            if isinstance(data, dict):
                if "events" in data:
                    events = data["events"]
                elif "feedback" in data:
                    events = data["feedback"]
                elif "history" in data:
                    events = data["history"]

            for event in events:
                if not isinstance(event, dict):
                    continue

                # Parse timestamp
                timestamp_str = event.get("timestamp")
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(
                            timestamp_str.replace("Z", "+00:00")
                        )

                        # Time range filter
                        if since and timestamp < since:
                            continue
                        if until and timestamp > until:
                            continue

                        event["_parsed_timestamp"] = timestamp
                    except ValueError:
                        pass

                # Pattern filter
                if self.patterns:
                    pattern = event.get("pattern", "")
                    if pattern not in self.patterns:
                        continue

                yield event

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in feedback file {source_path}: {e}")
        except Exception as e:
            logger.error(f"Error reading feedback file {source_path}: {e}")

    async def _parse_item(
        self,
        item: Dict[str, Any],
        source_file: Optional[str] = None,
    ) -> Optional[RawInteraction]:
        """Parse a feedback event into RawInteraction."""
        response = item.get("response", "")
        pattern = item.get("pattern", "OTHER")
        context = item.get("context", {})

        # Determine outcome
        outcome = RESPONSE_OUTCOME_MAP.get(response, InteractionOutcome.UNKNOWN)

        # Filter dismissed/deferred if not wanted
        if outcome == InteractionOutcome.DISMISSED and not self.include_dismissed:
            return None
        if outcome == InteractionOutcome.DEFERRED and not self.include_deferred:
            return None

        # Extract user input and assistant output
        notification_text = item.get("notification_text", "")

        # For corrections, try to find what the user wanted instead
        user_input = self._extract_user_input(context)
        assistant_output = notification_text or self._extract_assistant_output(context)

        # Check if this is a correction
        is_correction = outcome == InteractionOutcome.NEGATIVE_FEEDBACK
        correction_original = None
        correction_improved = None
        correction_reason = None

        if is_correction:
            correction_original = assistant_output
            correction_improved = self._extract_correction(context)
            correction_reason = self._extract_correction_reason(context)

            # If we found a correction, use it as the "correct" output
            if correction_improved:
                # Keep original output as what was wrong
                pass

        # Calculate confidence based on response time and engagement
        confidence = self._calculate_confidence(item, outcome)

        # Parse timestamp
        timestamp = item.get("_parsed_timestamp")
        if not timestamp:
            timestamp_str = item.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(
                    timestamp_str.replace("Z", "+00:00")
                )
            except ValueError:
                timestamp = datetime.now()

        # Build tags
        tags = PATTERN_TAGS.get(pattern, []).copy()
        tags.append(f"response:{response}")

        return RawInteraction(
            timestamp=timestamp,
            source_type=SourceType.FEEDBACK,
            source_id=item.get("id", item.get("event_id", "")),
            source_file=source_file,
            user_input=user_input,
            assistant_output=assistant_output,
            system_context=self._extract_system_context(context),
            outcome=outcome,
            confidence=confidence,
            is_correction=is_correction,
            correction_original=correction_original,
            correction_improved=correction_improved,
            correction_reason=correction_reason,
            properties=context,
            tags=tags,
        )

    def _extract_user_input(self, context: Dict[str, Any]) -> Optional[str]:
        """Extract user input from context."""
        for key in ("user_input", "input", "query", "command", "original_request"):
            if key in context:
                value = context[key]
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return None

    def _extract_assistant_output(self, context: Dict[str, Any]) -> Optional[str]:
        """Extract assistant output from context."""
        for key in ("output", "response", "assistant_response", "jarvis_response"):
            if key in context:
                value = context[key]
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return None

    def _extract_correction(self, context: Dict[str, Any]) -> Optional[str]:
        """Extract what the user wanted instead (correction)."""
        for key in (
            "correction",
            "expected",
            "wanted",
            "correct_response",
            "user_correction",
            "feedback_text",
            "improved_response",
        ):
            if key in context:
                value = context[key]
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return None

    def _extract_correction_reason(self, context: Dict[str, Any]) -> Optional[str]:
        """Extract why the original was wrong."""
        for key in (
            "reason",
            "correction_reason",
            "why_wrong",
            "feedback_reason",
            "notes",
        ):
            if key in context:
                value = context[key]
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return None

    def _extract_system_context(self, context: Dict[str, Any]) -> Optional[str]:
        """Extract system context for the interaction."""
        parts = []

        for key in ("system_state", "environment", "active_context"):
            if key in context:
                value = context[key]
                if isinstance(value, str):
                    parts.append(f"{key}: {value}")
                elif isinstance(value, dict):
                    parts.append(f"{key}: {json.dumps(value)}")

        return "\n".join(parts) if parts else None

    def _calculate_confidence(
        self,
        item: Dict[str, Any],
        outcome: InteractionOutcome,
    ) -> float:
        """Calculate confidence based on engagement signals."""
        base_confidence = 0.5

        # Response time affects confidence
        # Quick responses = more confident feedback
        time_to_respond = item.get("time_to_respond", 10.0)
        if time_to_respond < 2:
            base_confidence += 0.2  # Quick, confident response
        elif time_to_respond < 5:
            base_confidence += 0.1  # Moderate response time
        elif time_to_respond > 30:
            base_confidence -= 0.1  # Slow, possibly unsure

        # Outcome affects confidence
        if outcome == InteractionOutcome.ENGAGED:
            base_confidence += 0.2
        elif outcome == InteractionOutcome.NEGATIVE_FEEDBACK:
            base_confidence += 0.3  # Corrections are high-value signals!
        elif outcome == InteractionOutcome.DISMISSED:
            base_confidence += 0.1

        return min(1.0, max(0.0, base_confidence))

    def supports_streaming(self) -> bool:
        return False  # JSON files need full parsing


class AuthRecordIngestor(AbstractIngestor):
    """
    Ingestor for authentication records.

    Captures voice authentication attempts with rich context:
    - Voice confidence scores
    - Multi-factor fusion results
    - Decision reasoning chains
    """

    def __init__(
        self,
        min_confidence: float = 0.0,
        include_failures: bool = True,
        tags: Optional[List[str]] = None,
    ):
        super().__init__(
            min_confidence=min_confidence,
            include_failures=include_failures,
            tags=tags,
        )

    @property
    def source_type(self) -> SourceType:
        return SourceType.AUTH_RECORD

    async def _iter_items(
        self,
        source_path: Path,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Iterate over authentication records."""
        if not source_path.exists():
            logger.warning(f"Auth record file not found: {source_path}")
            return

        try:
            async with aiofiles.open(source_path, "r") as f:
                content = await f.read()

            data = json.loads(content)

            # Handle various formats
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                if "records" in data:
                    records = data["records"]
                elif "auth_history" in data:
                    records = data["auth_history"]
                else:
                    records = [data]
            else:
                records = []

            for record in records:
                if not isinstance(record, dict):
                    continue

                # Parse and filter by timestamp
                timestamp_str = record.get("timestamp")
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(
                            timestamp_str.replace("Z", "+00:00")
                        )

                        if since and timestamp < since:
                            continue
                        if until and timestamp > until:
                            continue

                        record["_parsed_timestamp"] = timestamp
                    except ValueError:
                        pass

                yield record

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in auth file {source_path}: {e}")
        except Exception as e:
            logger.error(f"Error reading auth file {source_path}: {e}")

    async def _parse_item(
        self,
        item: Dict[str, Any],
        source_file: Optional[str] = None,
    ) -> Optional[RawInteraction]:
        """Parse an authentication record into RawInteraction."""
        outcome_str = item.get("outcome", "")
        decision = item.get("decision", "")

        # Determine outcome
        if outcome_str == "SUCCESS" or decision == "authenticate":
            outcome = InteractionOutcome.SUCCESS
        elif outcome_str == "FAILURE" or decision == "deny":
            outcome = InteractionOutcome.FAILURE
        elif outcome_str == "CHALLENGED" or decision == "challenge":
            outcome = InteractionOutcome.CHALLENGED
        else:
            outcome = InteractionOutcome.UNKNOWN

        # Extract voice confidence as main confidence
        confidence = item.get("voice_confidence", item.get("final_confidence", 0.5))

        # Build user input from command/intent
        user_input = item.get("command", item.get("intent"))
        if not user_input:
            user_input = f"Authentication attempt: {outcome_str}"

        # Build assistant output from decision and reasoning
        reasoning = item.get("reasoning", [])
        if reasoning:
            assistant_output = "\n".join(reasoning)
        else:
            assistant_output = f"Decision: {decision}"

        # Parse timestamp
        timestamp = item.get("_parsed_timestamp", datetime.now())

        return RawInteraction(
            timestamp=timestamp,
            source_type=SourceType.AUTH_RECORD,
            source_id=item.get("id", ""),
            source_file=source_file,
            user_input=user_input,
            assistant_output=assistant_output,
            user_id=item.get("user_id"),
            outcome=outcome,
            confidence=confidence,
            auth_confidence=item.get("voice_confidence"),
            properties={
                "voice_confidence": item.get("voice_confidence"),
                "network_trust": item.get("network_trust"),
                "temporal_confidence": item.get("temporal_confidence"),
                "device_trust": item.get("device_trust"),
                "risk_score": item.get("risk_score"),
                "reasoning": reasoning,
            },
            tags=["auth", f"decision:{decision}"],
            environment={
                "network": item.get("network_ssid_hash"),
                "device_state": item.get("device_state"),
            },
        )

    def supports_streaming(self) -> bool:
        return False
