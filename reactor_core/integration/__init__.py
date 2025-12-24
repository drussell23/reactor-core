"""
Integration module for Night Shift Training Engine.

Provides:
- JARVIS-AI-Agent log ingestion
- JARVIS Prime WebSocket/REST integration
- Cross-repo event bridge for real-time sync
- Experience streaming and transformation
- Unified cost tracking across all repos (v10.0)
"""

from reactor_core.integration.jarvis_connector import (
    JARVISConnector,
    JARVISConnectorConfig,
    JARVISEvent,
    EventType,
    CorrectionType,
)

from reactor_core.integration.prime_connector import (
    PrimeConnector,
    PrimeConnectorConfig,
    PrimeEvent,
    PrimeEventType,
    ConnectionState,
)

from reactor_core.integration.event_bridge import (
    EventBridge,
    EventTransport,
    FileTransport,
    WebSocketTransport,
    CrossRepoEvent,
    EventSource,
    EventType as BridgeEventType,
    create_event_bridge,
)

from reactor_core.integration.cost_bridge import (
    CostBridge,
    CostSummary,
    get_cost_bridge,
    initialize_cost_bridge,
    shutdown_cost_bridge,
    get_aggregated_costs,
    emit_cost_event,
    record_distillation_cost,
)

__all__ = [
    # JARVIS-AI-Agent
    "JARVISConnector",
    "JARVISConnectorConfig",
    "JARVISEvent",
    "EventType",
    "CorrectionType",
    # JARVIS Prime
    "PrimeConnector",
    "PrimeConnectorConfig",
    "PrimeEvent",
    "PrimeEventType",
    "ConnectionState",
    # Event Bridge
    "EventBridge",
    "EventTransport",
    "FileTransport",
    "WebSocketTransport",
    "CrossRepoEvent",
    "EventSource",
    "BridgeEventType",
    "create_event_bridge",
    # Cost Bridge (v10.0)
    "CostBridge",
    "CostSummary",
    "get_cost_bridge",
    "initialize_cost_bridge",
    "shutdown_cost_bridge",
    "get_aggregated_costs",
    "emit_cost_event",
    "record_distillation_cost",
]
