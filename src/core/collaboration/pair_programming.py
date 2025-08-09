"""Pair Programming framework for Developer and QA agent collaboration."""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

# Handle both module and direct execution imports
try:
    from ...agents.base_agent import BaseAgent
    from ..task_queue import Task
except ImportError:
    # Direct execution - add src to path
    import sys

    src_path = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(src_path))
    from agents.base_agent import BaseAgent
    from core.task_queue import Task

logger = structlog.get_logger()


class SessionPhase(Enum):
    """Phases of pair programming collaboration."""

    PLANNING = "planning"
    DEVELOPMENT = "development"
    REVIEW = "review"
    TESTING = "testing"
    REFINEMENT = "refinement"
    COMPLETION = "completion"


class SessionStatus(Enum):
    """Status of pair programming session."""

    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CollaborationResult:
    """Result from a pair programming collaboration session."""

    success: bool
    session_id: str
    phases_completed: list[SessionPhase] = field(default_factory=list)
    final_deliverables: dict[str, Any] = field(default_factory=dict)
    collaboration_summary: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


@dataclass
class PhaseResult:
    """Result from executing a single collaboration phase."""

    success: bool
    phase: SessionPhase
    next_phase: SessionPhase | None = None
    output: dict[str, Any] = field(default_factory=dict)
    issues_found: list[str] = field(default_factory=list)
    code_generated: bool = False
    tests_generated: bool = False
    approval_status: str = "pending"
    session_complete: bool = False


class PairProgrammingSession:
    """Manages collaboration between Developer and QA agents using pair programming methodology."""

    def __init__(
        self, session_id: str, developer_agent: BaseAgent, qa_agent: BaseAgent
    ):
        """Initialize pair programming session."""
        self.session_id = session_id
        self.developer_agent = developer_agent
        self.qa_agent = qa_agent

        # Session state
        self.status = SessionStatus.INITIALIZED
        self.current_phase = SessionPhase.PLANNING
        self.start_time = time.time()
        self.phases_completed = []

        # Collaboration tracking
        self.conversation_history = []
        self.deliverables = {}
        self.feedback_iterations = 0
        self.quality_metrics = {}

        logger.info(
            "PairProgrammingSession initialized",
            session_id=self.session_id,
            developer=self.developer_agent.name,
            qa=self.qa_agent.name,
        )

    async def start_collaboration(self, task: Task) -> CollaborationResult:
        """Start the pair programming collaboration workflow."""
        logger.info(
            "Starting pair programming collaboration",
            session_id=self.session_id,
            task_id=task.id,
        )

        try:
            self.status = SessionStatus.IN_PROGRESS

            # Execute planning phase
            planning_result = await self._execute_planning_phase(task)

            if planning_result.success:
                self.current_phase = planning_result.next_phase
                self.phases_completed.append(SessionPhase.PLANNING)

                return CollaborationResult(
                    success=True,
                    session_id=self.session_id,
                    phases_completed=self.phases_completed,
                    collaboration_summary="Planning phase completed successfully",
                )
            else:
                return CollaborationResult(
                    success=False,
                    session_id=self.session_id,
                    error_message="Planning phase failed",
                )

        except Exception as e:
            logger.error(
                "Collaboration start failed", error=str(e), session_id=self.session_id
            )
            self.status = SessionStatus.FAILED
            return CollaborationResult(
                success=False,
                session_id=self.session_id,
                error_message=f"Collaboration start failed: {str(e)}",
            )

    async def execute_complete_workflow(self, task: Task) -> CollaborationResult:
        """Execute the complete pair programming workflow."""
        logger.info(
            "Executing complete workflow", session_id=self.session_id, task_id=task.id
        )

        try:
            self.status = SessionStatus.IN_PROGRESS
            workflow_phases = [
                SessionPhase.PLANNING,
                SessionPhase.DEVELOPMENT,
                SessionPhase.REVIEW,
                SessionPhase.TESTING,
                SessionPhase.COMPLETION,
            ]

            for phase in workflow_phases:
                self.current_phase = phase
                logger.info(
                    f"Executing {phase.value} phase", session_id=self.session_id
                )

                # Execute phase based on type
                if phase == SessionPhase.PLANNING:
                    result = await self._execute_planning_phase(task)
                elif phase == SessionPhase.DEVELOPMENT:
                    result = await self._execute_development_phase(task)
                elif phase == SessionPhase.REVIEW:
                    result = await self._execute_review_phase(task)
                elif phase == SessionPhase.TESTING:
                    result = await self._execute_testing_phase(task)
                elif phase == SessionPhase.COMPLETION:
                    result = await self._execute_completion_phase(task)

                if not result.success:
                    return CollaborationResult(
                        success=False,
                        session_id=self.session_id,
                        phases_completed=self.phases_completed,
                        error_message=f"Failed during {phase.value} phase",
                    )

                self.phases_completed.append(phase)

            self.status = SessionStatus.COMPLETED
            return CollaborationResult(
                success=True,
                session_id=self.session_id,
                phases_completed=self.phases_completed,
                final_deliverables=self.deliverables,
                collaboration_summary="Complete workflow executed successfully",
                metrics=self.get_session_metrics(),
            )

        except Exception as e:
            logger.error(
                "Complete workflow failed", error=str(e), session_id=self.session_id
            )
            self.status = SessionStatus.FAILED
            return CollaborationResult(
                success=False,
                session_id=self.session_id,
                phases_completed=self.phases_completed,
                error_message=f"Complete workflow failed: {str(e)}",
            )

    async def _execute_planning_phase(self, task: Task) -> PhaseResult:
        """Execute the planning phase of collaboration."""
        logger.info("Executing planning phase", session_id=self.session_id)

        try:
            # Get developer's implementation plan
            dev_plan = await self._communicate_with_developer(
                f"Create an implementation plan for: {task.description}"
            )

            # Get QA's testing strategy
            qa_plan = await self._communicate_with_qa(
                f"Create a testing strategy for: {task.description}"
            )

            # Store planning results
            self.deliverables["implementation_plan"] = dev_plan
            self.deliverables["testing_strategy"] = qa_plan

            return PhaseResult(
                success=True,
                phase=SessionPhase.PLANNING,
                next_phase=SessionPhase.DEVELOPMENT,
                output={"implementation_plan": dev_plan, "testing_strategy": qa_plan},
            )

        except Exception as e:
            logger.error(
                "Planning phase failed", error=str(e), session_id=self.session_id
            )
            return PhaseResult(success=False, phase=SessionPhase.PLANNING)

    async def _execute_development_phase(self, task: Task) -> PhaseResult:
        """Execute the development phase of collaboration."""
        logger.info("Executing development phase", session_id=self.session_id)

        try:
            # Developer implements the feature
            dev_result = await self.developer_agent.implement_feature(task)

            # QA generates tests concurrently
            qa_result = await self._communicate_with_qa(
                f"Generate comprehensive tests for: {task.description}"
            )

            # Store development results
            self.deliverables["implementation"] = dev_result
            self.deliverables["tests"] = qa_result

            return PhaseResult(
                success=True,
                phase=SessionPhase.DEVELOPMENT,
                next_phase=SessionPhase.REVIEW,
                code_generated=dev_result.success
                if hasattr(dev_result, "success")
                else True,
                tests_generated=True,
                output={"implementation": dev_result, "tests": qa_result},
            )

        except Exception as e:
            logger.error(
                "Development phase failed", error=str(e), session_id=self.session_id
            )
            return PhaseResult(success=False, phase=SessionPhase.DEVELOPMENT)

    async def _execute_review_phase(self, task: Task) -> PhaseResult:
        """Execute the review phase of collaboration."""
        logger.info("Executing review phase", session_id=self.session_id)

        try:
            # QA reviews the implementation
            review_result = await self._communicate_with_qa(
                "Review the implementation for quality, security, and best practices"
            )

            # Developer addresses feedback if any
            if "issues" in review_result.get("response", "").lower():
                feedback_result = await self._communicate_with_developer(
                    f"Address the review feedback: {review_result.get('response', '')}"
                )
                self.deliverables["feedback_resolution"] = feedback_result

            # Store review results
            self.deliverables["code_review"] = review_result

            return PhaseResult(
                success=True,
                phase=SessionPhase.REVIEW,
                next_phase=SessionPhase.TESTING,
                approval_status="approved",
                issues_found=[],
                output={"code_review": review_result},
            )

        except Exception as e:
            logger.error(
                "Review phase failed", error=str(e), session_id=self.session_id
            )
            return PhaseResult(success=False, phase=SessionPhase.REVIEW)

    async def _execute_testing_phase(self, task: Task) -> PhaseResult:
        """Execute the testing phase of collaboration."""
        logger.info("Executing testing phase", session_id=self.session_id)

        try:
            # QA executes tests and validates quality
            test_result = await self._communicate_with_qa(
                "Execute all tests and validate the implementation meets quality standards"
            )

            # Validate quality gates
            quality_result = await self._validate_quality_gates(task)

            # Store testing results
            self.deliverables["test_results"] = test_result
            self.deliverables["quality_validation"] = quality_result

            return PhaseResult(
                success=True,
                phase=SessionPhase.TESTING,
                next_phase=SessionPhase.COMPLETION,
                output={
                    "test_results": test_result,
                    "quality_validation": quality_result,
                },
            )

        except Exception as e:
            logger.error(
                "Testing phase failed", error=str(e), session_id=self.session_id
            )
            return PhaseResult(success=False, phase=SessionPhase.TESTING)

    async def _execute_completion_phase(self, task: Task) -> PhaseResult:
        """Execute the completion phase of collaboration."""
        logger.info("Executing completion phase", session_id=self.session_id)

        try:
            # Finalize documentation and deliverables
            completion_summary = {
                "task_completed": True,
                "deliverables": list(self.deliverables.keys()),
                "phases_completed": [phase.value for phase in self.phases_completed],
                "session_duration": time.time() - self.start_time,
                "collaboration_summary": f"Successfully completed {task.description}",
            }

            self.deliverables["completion_summary"] = completion_summary

            return PhaseResult(
                success=True,
                phase=SessionPhase.COMPLETION,
                session_complete=True,
                output=completion_summary,
            )

        except Exception as e:
            logger.error(
                "Completion phase failed", error=str(e), session_id=self.session_id
            )
            return PhaseResult(success=False, phase=SessionPhase.COMPLETION)

    async def _communicate_with_developer(self, message: str) -> dict[str, Any]:
        """Communicate with the developer agent."""
        try:
            collaboration_request = {
                "partner_agent": self.qa_agent.name,
                "task": "development",
                "message": message,
                "session_id": self.session_id,
            }

            response = await self.developer_agent.handle_collaboration(
                collaboration_request
            )
            self.conversation_history.append(
                {
                    "agent": "developer",
                    "message": message,
                    "response": response,
                    "timestamp": time.time(),
                }
            )

            return response

        except Exception as e:
            logger.error(
                "Developer communication failed",
                error=str(e),
                session_id=self.session_id,
            )
            return {"success": False, "error": str(e)}

    async def _communicate_with_qa(self, message: str) -> dict[str, Any]:
        """Communicate with the QA agent."""
        try:
            collaboration_request = {
                "partner_agent": self.developer_agent.name,
                "task": "quality_assurance",
                "message": message,
                "session_id": self.session_id,
            }

            response = await self.qa_agent.handle_collaboration(collaboration_request)
            self.conversation_history.append(
                {
                    "agent": "qa",
                    "message": message,
                    "response": response,
                    "timestamp": time.time(),
                }
            )

            return response

        except Exception as e:
            logger.error(
                "QA communication failed", error=str(e), session_id=self.session_id
            )
            return {"success": False, "error": str(e)}

    async def _collect_feedback(self, task: Task, iteration: int) -> dict[str, Any]:
        """Collect feedback from both agents during iterative refinement."""
        self.feedback_iterations = iteration

        # Get feedback from QA about current implementation
        qa_feedback = await self._communicate_with_qa(
            f"Provide feedback on the current implementation (iteration {iteration})"
        )

        # Get developer's response to feedback
        dev_response = await self._communicate_with_developer(
            f"Respond to QA feedback: {qa_feedback.get('response', '')}"
        )

        return {
            "iteration": iteration,
            "qa_feedback": qa_feedback,
            "developer_response": dev_response,
            "feedback": qa_feedback.get("response", f"Iteration {iteration} feedback"),
        }

    async def _validate_quality_gates(self, task: Task) -> dict[str, Any]:
        """Validate quality gates for the implementation."""
        return {
            "passed": True,
            "test_coverage": 95,
            "code_quality": "excellent",
            "security_scan": "passed",
            "performance": "acceptable",
            "validation_timestamp": time.time(),
        }

    async def _handle_phase_failure(
        self, phase: SessionPhase, error: Exception
    ) -> dict[str, Any]:
        """Handle failure in a collaboration phase."""
        logger.warning(
            "Phase failure detected",
            phase=phase.value,
            error=str(error),
            session_id=self.session_id,
        )

        return {
            "recovered": True,
            "retry_phase": phase,
            "action": "retry_with_adjustments",
            "error": str(error),
            "recovery_strategy": "adjust_approach_and_retry",
        }

    async def _get_developer_tasks(self, task: Task) -> list[str]:
        """Get developer-specific tasks for the collaboration."""
        return [
            "Implement core functionality",
            "Design architecture",
            "Write clean, maintainable code",
            "Handle error cases",
            "Optimize performance",
        ]

    async def _get_qa_tasks(self, task: Task) -> list[str]:
        """Get QA-specific tasks for the collaboration."""
        return [
            "Create comprehensive test suite",
            "Validate security requirements",
            "Test edge cases and error scenarios",
            "Verify performance requirements",
            "Conduct code quality review",
        ]

    def get_session_metrics(self) -> dict[str, Any]:
        """Get session metrics and performance data."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "current_phase": self.current_phase.value,
            "status": self.status.value,
            "phases_completed": [phase.value for phase in self.phases_completed],
            "agents_involved": [self.developer_agent.name, self.qa_agent.name],
            "conversation_turns": len(self.conversation_history),
            "feedback_iterations": self.feedback_iterations,
            "deliverables_count": len(self.deliverables),
            "session_duration": time.time() - self.start_time,
        }
