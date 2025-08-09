"""Collaboration module for multi-agent workflows."""

from .pair_programming import (
    PairProgrammingSession,
    CollaborationResult,
    SessionPhase,
    SessionStatus,
)

__all__ = [
    "PairProgrammingSession",
    "CollaborationResult",
    "SessionPhase",
    "SessionStatus",
]
