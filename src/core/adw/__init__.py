"""Autonomous Development Workflow (ADW) module.

This module provides the core framework for extended autonomous development
sessions with built-in safety mechanisms and continuous improvement.
"""

from .integration_validation import (
    IntegrationValidationEngine,
    IntegrationValidationReport,
    ValidationResult,
)
from .meta_learning import LearningInsight, MetaLearningEngine, MetaLearningReport
from .micro_development import MicroDevelopmentEngine, MicroIterationResult
from .reconnaissance import ReconnaissanceEngine, ReconnaissanceReport
from .session_manager import ADWSession, ADWSessionConfig, SessionPhase
from .session_persistence import (
    SessionCheckpoint,
    SessionStateManager,
    SessionStatePersistence,
)

__all__ = [
    "ADWSession",
    "ADWSessionConfig",
    "SessionPhase",
    "SessionStatePersistence",
    "SessionStateManager",
    "SessionCheckpoint",
    "ReconnaissanceEngine",
    "ReconnaissanceReport",
    "MicroDevelopmentEngine",
    "MicroIterationResult",
    "IntegrationValidationEngine",
    "ValidationResult",
    "IntegrationValidationReport",
    "MetaLearningEngine",
    "LearningInsight",
    "MetaLearningReport",
]
