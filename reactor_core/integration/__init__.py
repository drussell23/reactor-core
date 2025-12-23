"""
Integration module for Night Shift Training Engine.

Provides:
- JARVIS-AI-Agent log ingestion
- Cross-repo experience streaming
- Event transformation for training
"""

from reactor_core.integration.jarvis_connector import (
    JARVISConnector,
    JARVISConnectorConfig,
    JARVISEvent,
    EventType,
    CorrectionType,
)

__all__ = [
    "JARVISConnector",
    "JARVISConnectorConfig",
    "JARVISEvent",
    "EventType",
    "CorrectionType",
]
