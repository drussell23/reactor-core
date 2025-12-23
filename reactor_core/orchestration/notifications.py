"""
Notification system for pipeline events.

Provides:
- Slack notifications
- Webhook notifications
- Email notifications (via webhook)
- Unified notification interface
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Types of notifications."""
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"
    STAGE_COMPLETED = "stage_completed"
    GATEKEEPER_PASSED = "gatekeeper_passed"
    GATEKEEPER_FAILED = "gatekeeper_failed"
    DEPLOYMENT_COMPLETE = "deployment_complete"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class NotificationConfig:
    """Configuration for notifications."""
    # Slack
    slack_webhook_url: Optional[str] = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_SLACK_WEBHOOK")
    )
    slack_channel: Optional[str] = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_SLACK_CHANNEL")
    )

    # Generic webhook
    webhook_url: Optional[str] = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_WEBHOOK_URL")
    )

    # Filtering
    enabled_types: List[NotificationType] = field(
        default_factory=lambda: list(NotificationType)
    )

    # Throttling
    min_interval_seconds: int = 60  # Minimum time between notifications
    batch_notifications: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slack_webhook_url": bool(self.slack_webhook_url),  # Don't expose URLs
            "slack_channel": self.slack_channel,
            "webhook_url": bool(self.webhook_url),
            "enabled_types": [t.value for t in self.enabled_types],
            "min_interval_seconds": self.min_interval_seconds,
            "batch_notifications": self.batch_notifications,
        }


@dataclass
class Notification:
    """A notification message."""
    type: NotificationType
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, error, success

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "severity": self.severity,
        }


class BaseNotifier(ABC):
    """Abstract base class for notifiers."""

    @abstractmethod
    async def send(self, notification: Notification) -> bool:
        """Send a notification."""
        pass

    @abstractmethod
    async def send_batch(self, notifications: List[Notification]) -> bool:
        """Send multiple notifications."""
        pass


class SlackNotifier(BaseNotifier):
    """Slack webhook notifier."""

    # Emoji mapping for notification types
    EMOJI_MAP = {
        NotificationType.PIPELINE_STARTED: ":rocket:",
        NotificationType.PIPELINE_COMPLETED: ":white_check_mark:",
        NotificationType.PIPELINE_FAILED: ":x:",
        NotificationType.STAGE_COMPLETED: ":heavy_check_mark:",
        NotificationType.GATEKEEPER_PASSED: ":unlock:",
        NotificationType.GATEKEEPER_FAILED: ":lock:",
        NotificationType.DEPLOYMENT_COMPLETE: ":tada:",
        NotificationType.WARNING: ":warning:",
        NotificationType.ERROR: ":rotating_light:",
    }

    def __init__(
        self,
        webhook_url: str,
        channel: Optional[str] = None,
    ):
        """
        Initialize Slack notifier.

        Args:
            webhook_url: Slack incoming webhook URL
            channel: Optional channel override
        """
        self.webhook_url = webhook_url
        self.channel = channel
        self._session = None

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session

    def _format_slack_message(
        self,
        notification: Notification,
    ) -> Dict[str, Any]:
        """Format notification as Slack message."""
        emoji = self.EMOJI_MAP.get(notification.type, ":bell:")

        # Build blocks for rich formatting
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {notification.title}",
                    "emoji": True,
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": notification.message,
                }
            },
        ]

        # Add details if present
        if notification.details:
            fields = []
            for key, value in notification.details.items():
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*{key}:*\n{value}",
                })
            if fields:
                blocks.append({
                    "type": "section",
                    "fields": fields[:10],  # Slack limit
                })

        # Add timestamp
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Night Shift | {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                }
            ]
        })

        message = {"blocks": blocks}
        if self.channel:
            message["channel"] = self.channel

        return message

    async def send(self, notification: Notification) -> bool:
        """Send a Slack notification."""
        try:
            session = await self._get_session()
            message = self._format_slack_message(notification)

            async with session.post(
                self.webhook_url,
                json=message,
                timeout=10,
            ) as response:
                if response.status == 200:
                    logger.debug(f"Slack notification sent: {notification.title}")
                    return True
                else:
                    logger.error(
                        f"Slack notification failed: {response.status}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Slack notification error: {e}")
            return False

    async def send_batch(self, notifications: List[Notification]) -> bool:
        """Send batch of notifications as a thread."""
        if not notifications:
            return True

        # Send first as main message
        success = await self.send(notifications[0])

        # Send rest as individual messages (simplified)
        for notification in notifications[1:]:
            await self.send(notification)

        return success

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()


class WebhookNotifier(BaseNotifier):
    """Generic webhook notifier."""

    def __init__(
        self,
        webhook_url: str,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize webhook notifier.

        Args:
            webhook_url: Webhook URL
            headers: Optional custom headers
        """
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self._session = None

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session

    async def send(self, notification: Notification) -> bool:
        """Send a webhook notification."""
        try:
            session = await self._get_session()

            payload = {
                "event": notification.type.value,
                "notification": notification.to_dict(),
            }

            async with session.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10,
            ) as response:
                if response.status in (200, 201, 202, 204):
                    logger.debug(f"Webhook notification sent: {notification.title}")
                    return True
                else:
                    logger.error(
                        f"Webhook notification failed: {response.status}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Webhook notification error: {e}")
            return False

    async def send_batch(self, notifications: List[Notification]) -> bool:
        """Send batch of notifications."""
        try:
            session = await self._get_session()

            payload = {
                "event": "batch_notification",
                "notifications": [n.to_dict() for n in notifications],
            }

            async with session.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10,
            ) as response:
                return response.status in (200, 201, 202, 204)

        except Exception as e:
            logger.error(f"Webhook batch notification error: {e}")
            return False

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()


class NotificationManager:
    """
    Unified notification manager.

    Handles routing notifications to appropriate channels.
    """

    def __init__(
        self,
        config: Optional[NotificationConfig] = None,
    ):
        """
        Initialize notification manager.

        Args:
            config: Notification configuration
        """
        self.config = config or NotificationConfig()
        self._notifiers: List[BaseNotifier] = []
        self._pending: List[Notification] = []
        self._last_send_time: Optional[datetime] = None

        # Initialize notifiers
        self._setup_notifiers()

    def _setup_notifiers(self) -> None:
        """Setup notifiers based on configuration."""
        if self.config.slack_webhook_url:
            self._notifiers.append(SlackNotifier(
                self.config.slack_webhook_url,
                self.config.slack_channel,
            ))

        if self.config.webhook_url:
            self._notifiers.append(WebhookNotifier(
                self.config.webhook_url,
            ))

    def add_notifier(self, notifier: BaseNotifier) -> None:
        """Add a custom notifier."""
        self._notifiers.append(notifier)

    async def notify(
        self,
        notification_type: NotificationType,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "info",
    ) -> bool:
        """
        Send a notification.

        Args:
            notification_type: Type of notification
            title: Notification title
            message: Notification message
            details: Optional additional details
            severity: Severity level

        Returns:
            True if sent successfully
        """
        # Check if type is enabled
        if notification_type not in self.config.enabled_types:
            logger.debug(f"Notification type {notification_type.value} is disabled")
            return True

        notification = Notification(
            type=notification_type,
            title=title,
            message=message,
            details=details or {},
            severity=severity,
        )

        # Check throttling
        if self.config.batch_notifications:
            self._pending.append(notification)

            if self._last_send_time:
                elapsed = (datetime.now() - self._last_send_time).total_seconds()
                if elapsed < self.config.min_interval_seconds:
                    logger.debug("Notification batched due to throttling")
                    return True

        return await self._send_notification(notification)

    async def _send_notification(self, notification: Notification) -> bool:
        """Send notification to all notifiers."""
        if not self._notifiers:
            logger.debug("No notifiers configured")
            return True

        success = True
        for notifier in self._notifiers:
            try:
                result = await notifier.send(notification)
                if not result:
                    success = False
            except Exception as e:
                logger.error(f"Notifier error: {e}")
                success = False

        self._last_send_time = datetime.now()
        return success

    async def flush(self) -> bool:
        """Flush pending notifications."""
        if not self._pending:
            return True

        notifications = self._pending.copy()
        self._pending.clear()

        success = True
        for notifier in self._notifiers:
            try:
                result = await notifier.send_batch(notifications)
                if not result:
                    success = False
            except Exception as e:
                logger.error(f"Batch notifier error: {e}")
                success = False

        self._last_send_time = datetime.now()
        return success

    async def notify_pipeline_started(
        self,
        run_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send pipeline started notification."""
        return await self.notify(
            NotificationType.PIPELINE_STARTED,
            "Night Shift Pipeline Started",
            f"Training run `{run_id}` has started.",
            details=details,
            severity="info",
        )

    async def notify_pipeline_completed(
        self,
        run_id: str,
        duration_minutes: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send pipeline completed notification."""
        return await self.notify(
            NotificationType.PIPELINE_COMPLETED,
            "Night Shift Pipeline Completed",
            f"Training run `{run_id}` completed successfully in {duration_minutes:.1f} minutes.",
            details=details,
            severity="success",
        )

    async def notify_pipeline_failed(
        self,
        run_id: str,
        error: str,
        stage: Optional[str] = None,
    ) -> bool:
        """Send pipeline failed notification."""
        message = f"Training run `{run_id}` failed"
        if stage:
            message += f" at stage `{stage}`"
        message += f": {error}"

        return await self.notify(
            NotificationType.PIPELINE_FAILED,
            "Night Shift Pipeline Failed",
            message,
            details={"error": error, "stage": stage},
            severity="error",
        )

    async def notify_gatekeeper_result(
        self,
        passed: bool,
        model_version: str,
        metrics: Dict[str, float],
    ) -> bool:
        """Send gatekeeper result notification."""
        if passed:
            return await self.notify(
                NotificationType.GATEKEEPER_PASSED,
                "Gatekeeper Approved",
                f"Model `{model_version}` passed all quality checks and is approved for deployment.",
                details=metrics,
                severity="success",
            )
        else:
            return await self.notify(
                NotificationType.GATEKEEPER_FAILED,
                "Gatekeeper Rejected",
                f"Model `{model_version}` did not pass quality checks.",
                details=metrics,
                severity="warning",
            )

    async def close(self) -> None:
        """Close all notifiers."""
        await self.flush()
        for notifier in self._notifiers:
            if hasattr(notifier, "close"):
                await notifier.close()


# Convenience exports
__all__ = [
    "NotificationManager",
    "NotificationConfig",
    "NotificationType",
    "Notification",
    "BaseNotifier",
    "SlackNotifier",
    "WebhookNotifier",
]
