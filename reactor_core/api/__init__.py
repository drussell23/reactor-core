"""
Reactor-Core API Module v3.0

Advanced REST API for the AGI OS ecosystem providing:
- Training pipeline management
- Real-time telemetry ingestion and streaming
- Night Shift automated training scheduler
- Model versioning, A/B testing, and deployment
- Cross-repo health aggregation dashboard

Architecture:
    Server (FastAPI) ─────┬─── Telemetry Collector
                          ├─── Night Shift Scheduler
                          ├─── Model Registry
                          └─── Health Aggregator
"""

# Server
from reactor_core.api.server import app, main

# Telemetry System
from reactor_core.api.telemetry import (
    TelemetryCollector,
    TelemetryEvent,
    EventType,
    MetricType,
    MetricValue,
    AggregatedMetric,
    AnomalyAlert,
    RingBuffer,
    AdaptiveSampler,
    MetricsAggregator,
    AnomalyDetector,
    WebSocketBroadcaster,
    EventCorrelator,
    get_telemetry,
    get_correlator,
    telemetry_context,
    TelemetryConfig,
)

# Night Shift Scheduler
from reactor_core.api.scheduler import (
    NightShiftScheduler,
    ScheduleRule,
    ScheduleType,
    JobPriority,
    JobStatus,
    ScheduledJob,
    ResourceSnapshot,
    ResourceMonitor,
    ExperienceThresholdTrigger,
    ScheduleArbiter,
    DistributedLock,
    CronParser,
    ScheduleTemplates,
    get_scheduler,
    init_scheduler,
    SchedulerConfig,
)

# Model Registry
from reactor_core.api.model_registry import (
    ModelRegistry,
    VersionManager,
    ABTestManager,
    CheckpointManager,
    DeploymentManager,
    ModelVersion,
    ModelStatus,
    ModelMetrics,
    SemanticVersion,
    DeploymentTarget,
    DeploymentRecord,
    ABTestConfig,
    ABTestResult,
    Checkpoint,
    get_registry,
    RegistryConfig,
)

# Health Aggregator
from reactor_core.api.health_aggregator import (
    HealthAggregator,
    HealthStatus,
    HealthCheck,
    HealthAlert,
    AlertSeverity,
    ComponentHealth,
    ComponentChecker,
    HealthHistoryManager,
    AlertManager,
    SLAReport,
    DashboardData,
    WebhookNotifier,
    get_health_aggregator,
    init_health_aggregator,
    HealthConfig,
)

__all__ = [
    # Server
    "app",
    "main",
    # Telemetry
    "TelemetryCollector",
    "TelemetryEvent",
    "EventType",
    "MetricType",
    "MetricValue",
    "AggregatedMetric",
    "AnomalyAlert",
    "RingBuffer",
    "AdaptiveSampler",
    "MetricsAggregator",
    "AnomalyDetector",
    "WebSocketBroadcaster",
    "EventCorrelator",
    "get_telemetry",
    "get_correlator",
    "telemetry_context",
    "TelemetryConfig",
    # Scheduler
    "NightShiftScheduler",
    "ScheduleRule",
    "ScheduleType",
    "JobPriority",
    "JobStatus",
    "ScheduledJob",
    "ResourceSnapshot",
    "ResourceMonitor",
    "ExperienceThresholdTrigger",
    "ScheduleArbiter",
    "DistributedLock",
    "CronParser",
    "ScheduleTemplates",
    "get_scheduler",
    "init_scheduler",
    "SchedulerConfig",
    # Model Registry
    "ModelRegistry",
    "VersionManager",
    "ABTestManager",
    "CheckpointManager",
    "DeploymentManager",
    "ModelVersion",
    "ModelStatus",
    "ModelMetrics",
    "SemanticVersion",
    "DeploymentTarget",
    "DeploymentRecord",
    "ABTestConfig",
    "ABTestResult",
    "Checkpoint",
    "get_registry",
    "RegistryConfig",
    # Health Aggregator
    "HealthAggregator",
    "HealthStatus",
    "HealthCheck",
    "HealthAlert",
    "AlertSeverity",
    "ComponentHealth",
    "ComponentChecker",
    "HealthHistoryManager",
    "AlertManager",
    "SLAReport",
    "DashboardData",
    "WebhookNotifier",
    "get_health_aggregator",
    "init_health_aggregator",
    "HealthConfig",
]
