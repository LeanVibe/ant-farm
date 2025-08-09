"""Emergency Intervention System for Critical ADW Failures.

This module provides emergency intervention capabilities for autonomous
development sessions, including automatic termination and human escalation.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class CriticalFailureType(Enum):
    """Types of critical failures that trigger emergency intervention."""

    RESOURCE_EXHAUSTION = "resource_exhaustion"
    INFINITE_LOOP = "infinite_loop"
    SECURITY_VIOLATION = "security_violation"
    DATA_CORRUPTION = "data_corruption"
    REPEATED_FAILURES = "repeated_failures"
    SYSTEM_INSTABILITY = "system_instability"
    HUMAN_INTERVENTION_REQUESTED = "human_intervention_requested"


class InterventionLevel(Enum):
    """Levels of emergency intervention."""

    WARNING = "warning"  # Issue alert, continue monitoring
    PAUSE = "pause"  # Pause current operation, wait for clearance
    ROLLBACK = "rollback"  # Emergency rollback to last safe state
    TERMINATE = "terminate"  # Immediate session termination
    ESCALATE = "escalate"  # Escalate to human intervention


@dataclass
class EmergencyTrigger:
    """Configuration for emergency triggers."""

    failure_type: CriticalFailureType
    intervention_level: InterventionLevel
    threshold: float
    window_minutes: int = 5
    description: str = ""


@dataclass
class EmergencyEvent:
    """Record of an emergency event."""

    timestamp: float
    failure_type: CriticalFailureType
    intervention_level: InterventionLevel
    session_id: str
    description: str
    metrics: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: float | None = None


class EmergencyInterventionSystem:
    """Emergency intervention system for critical failures."""

    def __init__(self, project_path: Path, session_id: str):
        self.project_path = project_path
        self.session_id = session_id
        self.active = False

        # Emergency state
        self.emergency_events: list[EmergencyEvent] = []
        self.current_intervention_level = InterventionLevel.WARNING
        self.human_intervention_requested = False
        self.emergency_contacts: list[str] = []

        # Callbacks for emergency actions
        self.session_termination_callback: Callable | None = None
        self.rollback_callback: Callable | None = None
        self.pause_callback: Callable | None = None
        self.alert_callback: Callable | None = None

        # Configure default emergency triggers
        self.triggers = self._configure_default_triggers()

        # Monitoring state
        self.monitoring_active = False
        self.last_check_time = time.time()

    def _configure_default_triggers(self) -> list[EmergencyTrigger]:
        """Configure default emergency triggers."""
        return [
            EmergencyTrigger(
                CriticalFailureType.RESOURCE_EXHAUSTION,
                InterventionLevel.PAUSE,
                0.95,  # 95% resource usage
                window_minutes=2,
                description="Critical resource exhaustion detected",
            ),
            EmergencyTrigger(
                CriticalFailureType.INFINITE_LOOP,
                InterventionLevel.TERMINATE,
                300,  # 5 minutes without progress
                window_minutes=10,
                description="Potential infinite loop detected",
            ),
            EmergencyTrigger(
                CriticalFailureType.REPEATED_FAILURES,
                InterventionLevel.ROLLBACK,
                5,  # 5 consecutive failures
                window_minutes=15,
                description="Repeated failures indicate systemic issue",
            ),
            EmergencyTrigger(
                CriticalFailureType.SECURITY_VIOLATION,
                InterventionLevel.TERMINATE,
                1,  # Any security violation
                window_minutes=1,
                description="Security violation requires immediate termination",
            ),
            EmergencyTrigger(
                CriticalFailureType.DATA_CORRUPTION,
                InterventionLevel.ROLLBACK,
                1,  # Any data corruption
                window_minutes=1,
                description="Data corruption requires immediate rollback",
            ),
            EmergencyTrigger(
                CriticalFailureType.SYSTEM_INSTABILITY,
                InterventionLevel.ESCALATE,
                3,  # 3 system crashes
                window_minutes=30,
                description="System instability requires human intervention",
            ),
        ]

    async def start_monitoring(self):
        """Start emergency monitoring."""
        self.active = True
        self.monitoring_active = True
        self.last_check_time = time.time()

        logger.info(
            "Emergency intervention monitoring started",
            session_id=self.session_id,
            triggers_count=len(self.triggers),
        )

        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """Stop emergency monitoring."""
        self.monitoring_active = False
        self.active = False

        logger.info(
            "Emergency intervention monitoring stopped",
            session_id=self.session_id,
            events_count=len(self.emergency_events),
        )

    async def _monitoring_loop(self):
        """Main monitoring loop for emergency conditions."""
        while self.monitoring_active:
            try:
                await self._check_emergency_conditions()
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(
                    "Error in emergency monitoring loop",
                    session_id=self.session_id,
                    error=str(e),
                )
                await asyncio.sleep(30)  # Back off on error

    async def _check_emergency_conditions(self):
        """Check for emergency conditions."""
        current_time = time.time()

        # Check each trigger
        for trigger in self.triggers:
            if await self._evaluate_trigger(trigger, current_time):
                await self._handle_emergency(trigger, current_time)

        self.last_check_time = current_time

    async def _evaluate_trigger(
        self, trigger: EmergencyTrigger, current_time: float
    ) -> bool:
        """Evaluate if a trigger condition is met."""
        try:
            if trigger.failure_type == CriticalFailureType.RESOURCE_EXHAUSTION:
                return await self._check_resource_exhaustion(trigger.threshold)

            elif trigger.failure_type == CriticalFailureType.INFINITE_LOOP:
                return await self._check_infinite_loop(trigger.threshold)

            elif trigger.failure_type == CriticalFailureType.REPEATED_FAILURES:
                return await self._check_repeated_failures(
                    trigger.threshold, trigger.window_minutes
                )

            elif (
                trigger.failure_type == CriticalFailureType.HUMAN_INTERVENTION_REQUESTED
            ):
                return self.human_intervention_requested

            # Add more trigger evaluations as needed
            return False

        except Exception as e:
            logger.error(
                "Error evaluating emergency trigger",
                trigger_type=trigger.failure_type.value,
                error=str(e),
            )
            return False

    async def _check_resource_exhaustion(self, threshold: float) -> bool:
        """Check for resource exhaustion."""
        try:
            import psutil

            # Check memory usage
            memory_percent = psutil.virtual_memory().percent / 100.0
            if memory_percent > threshold:
                return True

            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1) / 100.0
            if cpu_percent > threshold:
                return True

            # Check disk usage
            disk_percent = psutil.disk_usage("/").percent / 100.0
            if disk_percent > threshold:
                return True

            return False

        except Exception as e:
            logger.error("Error checking resource exhaustion", error=str(e))
            return False

    async def _check_infinite_loop(self, threshold_seconds: float) -> bool:
        """Check for potential infinite loops."""
        # This would need integration with session progress tracking
        # For now, return False as a placeholder
        return False

    async def _check_repeated_failures(
        self, threshold_count: int, window_minutes: int
    ) -> bool:
        """Check for repeated failures within time window."""
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)

        # Count failures in the time window
        failure_count = sum(
            1
            for event in self.emergency_events
            if (
                event.timestamp >= window_start
                and not event.resolved
                and event.failure_type
                != CriticalFailureType.HUMAN_INTERVENTION_REQUESTED
            )
        )

        return failure_count >= threshold_count

    async def _handle_emergency(self, trigger: EmergencyTrigger, current_time: float):
        """Handle an emergency situation."""
        # Create emergency event
        event = EmergencyEvent(
            timestamp=current_time,
            failure_type=trigger.failure_type,
            intervention_level=trigger.intervention_level,
            session_id=self.session_id,
            description=trigger.description,
            metrics=await self._collect_emergency_metrics(),
        )

        self.emergency_events.append(event)
        self.current_intervention_level = trigger.intervention_level

        logger.critical(
            "Emergency intervention triggered",
            session_id=self.session_id,
            failure_type=trigger.failure_type.value,
            intervention_level=trigger.intervention_level.value,
            description=trigger.description,
        )

        # Execute intervention based on level
        if trigger.intervention_level == InterventionLevel.WARNING:
            await self._handle_warning(event)
        elif trigger.intervention_level == InterventionLevel.PAUSE:
            await self._handle_pause(event)
        elif trigger.intervention_level == InterventionLevel.ROLLBACK:
            await self._handle_rollback(event)
        elif trigger.intervention_level == InterventionLevel.TERMINATE:
            await self._handle_termination(event)
        elif trigger.intervention_level == InterventionLevel.ESCALATE:
            await self._handle_escalation(event)

    async def _handle_warning(self, event: EmergencyEvent):
        """Handle warning level intervention."""
        if self.alert_callback:
            await self.alert_callback(event)

        logger.warning(
            "Emergency warning issued",
            session_id=self.session_id,
            event_id=len(self.emergency_events) - 1,
        )

    async def _handle_pause(self, event: EmergencyEvent):
        """Handle pause level intervention."""
        if self.pause_callback:
            await self.pause_callback(event)

        logger.warning(
            "Emergency pause executed",
            session_id=self.session_id,
            event_id=len(self.emergency_events) - 1,
        )

    async def _handle_rollback(self, event: EmergencyEvent):
        """Handle rollback level intervention."""
        if self.rollback_callback:
            await self.rollback_callback(event)

        logger.error(
            "Emergency rollback executed",
            session_id=self.session_id,
            event_id=len(self.emergency_events) - 1,
        )

    async def _handle_termination(self, event: EmergencyEvent):
        """Handle termination level intervention."""
        if self.session_termination_callback:
            await self.session_termination_callback(event)

        # Stop monitoring
        await self.stop_monitoring()

        logger.critical(
            "Emergency session termination executed",
            session_id=self.session_id,
            event_id=len(self.emergency_events) - 1,
        )

    async def _handle_escalation(self, event: EmergencyEvent):
        """Handle escalation level intervention."""
        self.human_intervention_requested = True

        # Send notifications to emergency contacts
        for contact in self.emergency_contacts:
            await self._notify_contact(contact, event)

        logger.critical(
            "Emergency escalated to human intervention",
            session_id=self.session_id,
            event_id=len(self.emergency_events) - 1,
            contacts_notified=len(self.emergency_contacts),
        )

    async def _collect_emergency_metrics(self) -> dict[str, Any]:
        """Collect metrics at time of emergency."""
        metrics = {}

        try:
            import psutil

            metrics.update(
                {
                    "memory_percent": psutil.virtual_memory().percent,
                    "cpu_percent": psutil.cpu_percent(),
                    "disk_percent": psutil.disk_usage("/").percent,
                    "timestamp": time.time(),
                    "session_id": self.session_id,
                }
            )

        except Exception as e:
            metrics["error"] = str(e)

        return metrics

    async def _notify_contact(self, contact: str, event: EmergencyEvent):
        """Notify emergency contact."""
        # Placeholder for notification implementation
        logger.info(
            "Emergency contact notification",
            contact=contact,
            failure_type=event.failure_type.value,
            session_id=self.session_id,
        )

    def request_human_intervention(self, reason: str):
        """Request human intervention."""
        self.human_intervention_requested = True

        logger.warning(
            "Human intervention requested", session_id=self.session_id, reason=reason
        )

    def resolve_emergency(self, event_index: int, resolution_notes: str = ""):
        """Mark an emergency event as resolved."""
        if 0 <= event_index < len(self.emergency_events):
            event = self.emergency_events[event_index]
            event.resolved = True
            event.resolution_timestamp = time.time()

            logger.info(
                "Emergency event resolved",
                session_id=self.session_id,
                event_index=event_index,
                failure_type=event.failure_type.value,
                resolution_notes=resolution_notes,
            )

    def get_emergency_status(self) -> dict[str, Any]:
        """Get current emergency status."""
        unresolved_events = [e for e in self.emergency_events if not e.resolved]

        return {
            "active": self.active,
            "current_intervention_level": self.current_intervention_level.value,
            "total_events": len(self.emergency_events),
            "unresolved_events": len(unresolved_events),
            "human_intervention_requested": self.human_intervention_requested,
            "last_check_time": self.last_check_time,
            "recent_events": [
                {
                    "timestamp": e.timestamp,
                    "failure_type": e.failure_type.value,
                    "intervention_level": e.intervention_level.value,
                    "description": e.description,
                    "resolved": e.resolved,
                }
                for e in self.emergency_events[-10:]  # Last 10 events
            ],
        }

    def register_callbacks(
        self,
        session_termination_callback: Callable | None = None,
        rollback_callback: Callable | None = None,
        pause_callback: Callable | None = None,
        alert_callback: Callable | None = None,
    ):
        """Register callbacks for emergency actions."""
        if session_termination_callback:
            self.session_termination_callback = session_termination_callback
        if rollback_callback:
            self.rollback_callback = rollback_callback
        if pause_callback:
            self.pause_callback = pause_callback
        if alert_callback:
            self.alert_callback = alert_callback

        logger.info(
            "Emergency intervention callbacks registered",
            session_id=self.session_id,
            callbacks_registered=sum(
                [
                    bool(session_termination_callback),
                    bool(rollback_callback),
                    bool(pause_callback),
                    bool(alert_callback),
                ]
            ),
        )
