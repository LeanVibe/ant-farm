"""Collaboration module for multi-agent workflows."""

from .pair_programming import (
    PairProgrammingSession,
    CollaborationResult,
    SessionPhase,
    SessionStatus,
)

from .large_project_coordination import (
    LargeProjectCoordinator,
    ProjectScale,
    ProjectPhase,
    ProjectWorkspace,
    TaskDependencyGraph,
    get_large_project_coordinator,
)

from .enhanced_pair_programming import (
    EnhancedAIPairProgramming,
    CollaborationSession,
    ContextShareType,
    CollaborationMode,
    SharedContext,
    get_enhanced_pair_programming,
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
