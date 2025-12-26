"""
Reactor Core VBIA (Voice Biometric Intelligent Authentication) Event Connector
==============================================================================

Ingests and analyzes VBIA authentication events from JARVIS for monitoring,
analytics, and intelligence.

Features:
- Real-time VBIA event ingestion
- Multi-factor confidence analytics
- Visual security threat tracking
- LangGraph reasoning chain analysis
- Cost optimization metrics
- Attack pattern detection

Author: Reactor Core Team
Version: 6.2.0 - VBIA Event Integration
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("reactor-core.vbia")


# ============================================================================
# Constants
# ============================================================================

VBIA_EVENTS_FILE = Path.home() / ".jarvis" / "cross_repo" / "vbia_events.json"
VBIA_RESULTS_FILE = Path.home() / ".jarvis" / "cross_repo" / "vbia_results.json"
VBIA_STATE_FILE = Path.home() / ".jarvis" / "cross_repo" / "vbia_state.json"

# Analytics configuration
DEFAULT_ANALYSIS_WINDOW_HOURS = 24
MIN_EVENTS_FOR_ANALYSIS = 5


# ============================================================================
# Enums
# ============================================================================

class VBIAEventType(Enum):
    """VBIA event types."""
    AUTHENTICATION_REQUEST = "vbia_authentication_request"
    AUTHENTICATION_RESULT = "vbia_authentication_result"
    VISUAL_SECURITY = "vbia_visual_security"
    SPOOFING_DETECTION = "vbia_spoofing_detection"
    HYPOTHESIS_GENERATION = "vbia_hypothesis_generation"
    REASONING_CHAIN = "vbia_reasoning_chain"
    PATTERN_LEARNING = "vbia_pattern_learning"
    COST_TRACKING = "vbia_cost_tracking"


class SecurityRiskLevel(Enum):
    """Security risk assessment levels."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class VBIAEventMetrics:
    """Aggregated VBIA event metrics."""

    # Time window
    start_time: str
    end_time: str
    total_events: int

    # Authentication metrics
    total_authentications: int = 0
    successful_authentications: int = 0
    failed_authentications: int = 0
    success_rate: float = 0.0

    # Confidence breakdown
    avg_ml_confidence: float = 0.0
    avg_physics_confidence: float = 0.0
    avg_behavioral_confidence: float = 0.0
    avg_visual_confidence: float = 0.0
    avg_final_confidence: float = 0.0

    # Security metrics
    spoofing_attempts: int = 0
    visual_threats_detected: int = 0
    liveness_failures: int = 0

    # Performance metrics
    avg_execution_time_ms: float = 0.0
    fast_path_usage_rate: float = 0.0
    langgraph_usage_rate: float = 0.0

    # Cost metrics (if available)
    total_cost_usd: float = 0.0
    avg_cost_per_auth: float = 0.0
    cache_hit_rate: float = 0.0

    # Reasoning insights
    most_common_hypotheses: List[str] = None
    reasoning_chain_avg_length: float = 0.0

    # Risk assessment
    risk_level: SecurityRiskLevel = SecurityRiskLevel.MINIMAL
    risk_factors: List[str] = None

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.most_common_hypotheses is None:
            self.most_common_hypotheses = []
        if self.risk_factors is None:
            self.risk_factors = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["risk_level"] = self.risk_level.value
        return data


@dataclass
class VBIAThreatAlert:
    """VBIA security threat alert."""
    timestamp: str
    alert_id: str
    threat_type: str
    severity: SecurityRiskLevel
    description: str
    affected_sessions: List[str] = None
    recommended_actions: List[str] = None

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.affected_sessions is None:
            self.affected_sessions = []
        if self.recommended_actions is None:
            self.recommended_actions = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["severity"] = self.severity.value
        return data


# ============================================================================
# VBIA Connector
# ============================================================================

class VBIAConnector:
    """
    Connects to JARVIS VBIA system for event ingestion and analysis.

    Features:
    - Real-time event monitoring
    - Multi-factor confidence analytics
    - Threat detection and alerting
    - Cost optimization tracking
    - Reasoning chain analysis
    """

    def __init__(self):
        """Initialize VBIA connector."""
        self._last_processed_event_idx = 0
        self._threat_alerts: List[VBIAThreatAlert] = []
        self._metrics_cache: Dict[str, VBIAEventMetrics] = {}

        logger.info("[VBIA CONNECTOR] Initialized")

    async def get_recent_events(
        self,
        since: Optional[datetime] = None,
        event_types: Optional[List[VBIAEventType]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get recent VBIA events.

        Args:
            since: Only return events after this time
            event_types: Filter by event types
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries
        """
        try:
            if not VBIA_EVENTS_FILE.exists():
                return []

            content = VBIA_EVENTS_FILE.read_text()
            events = json.loads(content)

            # Filter by timestamp
            if since:
                since_iso = since.isoformat()
                events = [e for e in events if e.get("timestamp", "") >= since_iso]

            # Filter by event type
            if event_types:
                type_values = [t.value for t in event_types]
                events = [e for e in events if e.get("event_type") in type_values]

            # Return latest N events
            return events[-limit:]

        except Exception as e:
            logger.error(f"Failed to get VBIA events: {e}")
            return []

    async def get_authentication_results(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get recent authentication results.

        Args:
            since: Only return results after this time
            limit: Maximum number of results to return

        Returns:
            List of authentication result dictionaries
        """
        try:
            if not VBIA_RESULTS_FILE.exists():
                return []

            content = VBIA_RESULTS_FILE.read_text()
            results = json.loads(content)

            # Filter by timestamp
            if since:
                since_iso = since.isoformat()
                results = [r for r in results if r.get("timestamp", "") >= since_iso]

            # Return latest N results
            return results[-limit:]

        except Exception as e:
            logger.error(f"Failed to get authentication results: {e}")
            return []

    async def analyze_metrics(
        self,
        window_hours: int = DEFAULT_ANALYSIS_WINDOW_HOURS,
    ) -> VBIAEventMetrics:
        """
        Analyze VBIA metrics over a time window.

        Args:
            window_hours: Analysis time window in hours

        Returns:
            VBIAEventMetrics with aggregated analytics
        """
        # Check cache
        cache_key = f"{window_hours}h"
        if cache_key in self._metrics_cache:
            cached = self._metrics_cache[cache_key]
            # Return cached if less than 5 minutes old
            cache_age = datetime.now() - datetime.fromisoformat(cached.end_time)
            if cache_age < timedelta(minutes=5):
                return cached

        start_time = datetime.now() - timedelta(hours=window_hours)
        end_time = datetime.now()

        # Get events and results
        events = await self.get_recent_events(since=start_time, limit=10000)
        results = await self.get_authentication_results(since=start_time, limit=10000)

        if len(results) < MIN_EVENTS_FOR_ANALYSIS:
            logger.info(f"[VBIA CONNECTOR] Insufficient data for analysis: {len(results)} results")

        # Create metrics
        metrics = VBIAEventMetrics(
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_events=len(events),
        )

        # Analyze authentication results
        if results:
            metrics.total_authentications = len(results)
            metrics.successful_authentications = sum(
                1 for r in results if r.get("authenticated", False)
            )
            metrics.failed_authentications = len(results) - metrics.successful_authentications
            metrics.success_rate = (
                metrics.successful_authentications / metrics.total_authentications * 100
            )

            # Confidence metrics
            ml_confs = [r.get("ml_confidence", 0.0) for r in results]
            physics_confs = [r.get("physics_confidence", 0.0) for r in results]
            behavioral_confs = [r.get("behavioral_confidence", 0.0) for r in results]
            visual_confs = [r.get("visual_confidence", 0.0) for r in results]
            final_confs = [r.get("final_confidence", 0.0) for r in results]

            metrics.avg_ml_confidence = sum(ml_confs) / len(ml_confs) if ml_confs else 0.0
            metrics.avg_physics_confidence = sum(physics_confs) / len(physics_confs) if physics_confs else 0.0
            metrics.avg_behavioral_confidence = sum(behavioral_confs) / len(behavioral_confs) if behavioral_confs else 0.0
            metrics.avg_visual_confidence = sum(visual_confs) / len(visual_confs) if visual_confs else 0.0
            metrics.avg_final_confidence = sum(final_confs) / len(final_confs) if final_confs else 0.0

            # Security metrics
            metrics.spoofing_attempts = sum(
                1 for r in results if r.get("spoofing_detected", False)
            )
            metrics.visual_threats_detected = sum(
                1 for r in results if r.get("visual_threat_detected", False)
            )
            metrics.liveness_failures = sum(
                1 for r in results if not r.get("liveness_passed", True)
            )

            # Performance metrics
            exec_times = [r.get("execution_time_ms", 0.0) for r in results]
            metrics.avg_execution_time_ms = sum(exec_times) / len(exec_times) if exec_times else 0.0

            fast_path_count = sum(
                1 for r in results if r.get("analysis_mode_used") == "fast_path"
            )
            metrics.fast_path_usage_rate = fast_path_count / len(results) * 100

            langgraph_count = sum(1 for r in results if r.get("used_langgraph", False))
            metrics.langgraph_usage_rate = langgraph_count / len(results) * 100

            # Reasoning insights
            all_hypotheses = []
            reasoning_lengths = []
            for r in results:
                hypotheses = r.get("hypotheses_evaluated", [])
                all_hypotheses.extend(hypotheses)

                reasoning_chain = r.get("reasoning_chain", [])
                if reasoning_chain:
                    reasoning_lengths.append(len(reasoning_chain))

            # Count hypothesis frequency
            hypothesis_counts = defaultdict(int)
            for h in all_hypotheses:
                hypothesis_counts[h] += 1

            # Get top 5 hypotheses
            sorted_hypotheses = sorted(
                hypothesis_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            metrics.most_common_hypotheses = [h[0] for h in sorted_hypotheses[:5]]

            if reasoning_lengths:
                metrics.reasoning_chain_avg_length = sum(reasoning_lengths) / len(reasoning_lengths)

        # Assess risk level
        metrics.risk_level, metrics.risk_factors = self._assess_risk_level(metrics)

        # Cache metrics
        self._metrics_cache[cache_key] = metrics

        logger.info(
            f"[VBIA CONNECTOR] Metrics analyzed: "
            f"{metrics.total_authentications} auths, "
            f"{metrics.success_rate:.1f}% success, "
            f"risk={metrics.risk_level.value}"
        )

        return metrics

    def _assess_risk_level(
        self,
        metrics: VBIAEventMetrics,
    ) -> Tuple[SecurityRiskLevel, List[str]]:
        """Assess security risk level from metrics."""
        risk_factors = []
        risk_score = 0

        # Check success rate
        if metrics.success_rate < 50:
            risk_score += 3
            risk_factors.append(f"Low success rate: {metrics.success_rate:.1f}%")
        elif metrics.success_rate < 70:
            risk_score += 2
            risk_factors.append(f"Moderate success rate: {metrics.success_rate:.1f}%")

        # Check spoofing attempts
        if metrics.spoofing_attempts > 0:
            risk_score += 2
            risk_factors.append(f"{metrics.spoofing_attempts} spoofing attempts detected")

        # Check visual threats
        if metrics.visual_threats_detected > 0:
            risk_score += 3
            risk_factors.append(f"{metrics.visual_threats_detected} visual threats detected")

        # Check liveness failures
        if metrics.liveness_failures > 0:
            risk_score += 2
            risk_factors.append(f"{metrics.liveness_failures} liveness check failures")

        # Check confidence levels
        if metrics.avg_final_confidence < 0.70:
            risk_score += 1
            risk_factors.append(f"Low average confidence: {metrics.avg_final_confidence:.1%}")

        # Determine risk level
        if risk_score == 0:
            return SecurityRiskLevel.MINIMAL, risk_factors
        elif risk_score <= 2:
            return SecurityRiskLevel.LOW, risk_factors
        elif risk_score <= 4:
            return SecurityRiskLevel.MODERATE, risk_factors
        elif risk_score <= 6:
            return SecurityRiskLevel.HIGH, risk_factors
        else:
            return SecurityRiskLevel.CRITICAL, risk_factors

    async def detect_threats(
        self,
        window_hours: int = 1,
    ) -> List[VBIAThreatAlert]:
        """
        Detect security threats in recent VBIA activity.

        Args:
            window_hours: Time window to analyze

        Returns:
            List of threat alerts
        """
        alerts = []

        start_time = datetime.now() - timedelta(hours=window_hours)
        results = await self.get_authentication_results(since=start_time)

        # Detect patterns

        # 1. Multiple spoofing attempts
        spoofing_sessions = [
            r.get("request_id", "") for r in results if r.get("spoofing_detected", False)
        ]
        if len(spoofing_sessions) >= 3:
            alerts.append(VBIAThreatAlert(
                timestamp=datetime.now().isoformat(),
                alert_id=f"spoofing-{int(datetime.now().timestamp())}",
                threat_type="multiple_spoofing_attempts",
                severity=SecurityRiskLevel.HIGH,
                description=f"{len(spoofing_sessions)} spoofing attempts detected in last {window_hours}h",
                affected_sessions=spoofing_sessions,
                recommended_actions=[
                    "Review spoofing detection logs",
                    "Consider increasing security level",
                    "Analyze attack patterns in ChromaDB",
                ],
            ))

        # 2. Visual threats
        visual_threat_sessions = [
            r.get("request_id", "") for r in results if r.get("visual_threat_detected", False)
        ]
        if len(visual_threat_sessions) > 0:
            alerts.append(VBIAThreatAlert(
                timestamp=datetime.now().isoformat(),
                alert_id=f"visual-threat-{int(datetime.now().timestamp())}",
                threat_type="visual_security_threats",
                severity=SecurityRiskLevel.CRITICAL,
                description=f"{len(visual_threat_sessions)} visual security threats detected",
                affected_sessions=visual_threat_sessions,
                recommended_actions=[
                    "Investigate screen state at time of threat",
                    "Check for ransomware or fake lock screens",
                    "Review visual security logs",
                ],
            ))

        # 3. Unusual failure rate
        if len(results) >= 10:
            failed = sum(1 for r in results if not r.get("authenticated", False))
            failure_rate = failed / len(results) * 100

            if failure_rate > 50:
                alerts.append(VBIAThreatAlert(
                    timestamp=datetime.now().isoformat(),
                    alert_id=f"high-failure-{int(datetime.now().timestamp())}",
                    threat_type="high_authentication_failure_rate",
                    severity=SecurityRiskLevel.MODERATE,
                    description=f"High failure rate: {failure_rate:.1f}% ({failed}/{len(results)})",
                    recommended_actions=[
                        "Check for system issues",
                        "Review recent voice enrollment quality",
                        "Analyze failure patterns",
                    ],
                ))

        self._threat_alerts.extend(alerts)

        if alerts:
            logger.warning(f"[VBIA CONNECTOR] {len(alerts)} threat alerts generated")

        return alerts

    async def get_confidence_breakdown(
        self,
        since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get confidence breakdown across all factors.

        Returns:
            Dictionary with confidence statistics per factor
        """
        results = await self.get_authentication_results(since=since)

        if not results:
            return {}

        breakdown = {
            "ml": {
                "avg": 0.0,
                "min": 1.0,
                "max": 0.0,
                "std": 0.0,
            },
            "physics": {
                "avg": 0.0,
                "min": 1.0,
                "max": 0.0,
                "std": 0.0,
            },
            "behavioral": {
                "avg": 0.0,
                "min": 1.0,
                "max": 0.0,
                "std": 0.0,
            },
            "visual": {
                "avg": 0.0,
                "min": 1.0,
                "max": 0.0,
                "std": 0.0,
            },
            "final": {
                "avg": 0.0,
                "min": 1.0,
                "max": 0.0,
                "std": 0.0,
            },
        }

        # Collect confidence values
        for factor in ["ml", "physics", "behavioral", "visual", "final"]:
            key = f"{factor}_confidence"
            values = [r.get(key, 0.0) for r in results]

            if values:
                import statistics
                breakdown[factor]["avg"] = sum(values) / len(values)
                breakdown[factor]["min"] = min(values)
                breakdown[factor]["max"] = max(values)
                if len(values) > 1:
                    breakdown[factor]["std"] = statistics.stdev(values)

        return breakdown


# ============================================================================
# Global Instance
# ============================================================================

_connector_instance: Optional[VBIAConnector] = None


def get_vbia_connector() -> VBIAConnector:
    """Get or create global VBIA connector."""
    global _connector_instance

    if _connector_instance is None:
        _connector_instance = VBIAConnector()

    return _connector_instance
