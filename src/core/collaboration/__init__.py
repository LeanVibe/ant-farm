"""Collaboration module for multi-agent workflows."""

from .enhanced_pair_programming import (
    CollaborationMode,
    CollaborationSession,
    ContextShareType,
    EnhancedAIPairProgramming,
    SharedContext,
    get_enhanced_pair_programming,
)
from .large_project_coordination import (
    LargeProjectCoordinator,
    ProjectPhase,
    ProjectScale,
    ProjectWorkspace,
    TaskDependencyGraph,
    get_large_project_coordinator,
)
from .pair_programming import (
    CollaborationResult,
    PairProgrammingSession,
    SessionPhase,
    SessionStatus,
)

__all__ = [
    "PairProgrammingSession",
    "CollaborationResult",
    "SessionPhase",
    "SessionStatus",
    "LargeProjectCoordinator",
    "ProjectScale",
    "ProjectPhase",
    "ProjectWorkspace",
    "TaskDependencyGraph",
    "get_large_project_coordinator",
    "EnhancedAIPairProgramming",
    "CollaborationSession",
    "ContextShareType",
    "CollaborationMode",
    "SharedContext",
    "get_enhanced_pair_programming",
]
