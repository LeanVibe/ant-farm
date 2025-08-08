"""Autonomous Development Workflow (ADW) module.

This module provides the core framework for extended autonomous development
sessions with built-in safety mechanisms and continuous improvement.
"""

from .session_manager import ADWSession, ADWSessionConfig, SessionPhase
from .session_persistence import (
    SessionStatePersistence,
    SessionStateManager,
    SessionCheckpoint,
)
from .reconnaissance import ReconnaissanceEngine, ReconnaissanceReport
from .micro_development import MicroDevelopmentEngine, MicroIterationResult
from .integration_validation import (
    IntegrationValidationEngine,
    ValidationResult,
    IntegrationValidationReport,
)
from .meta_learning import MetaLearningEngine, LearningInsight, MetaLearningReport

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
