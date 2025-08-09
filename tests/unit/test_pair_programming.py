"""Unit tests for PairProgramming framework with TDD approach."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.collaboration.pair_programming import (
    PairProgrammingSession,
    CollaborationResult,
    SessionPhase,
    SessionStatus,
)
from src.agents.developer_agent import DeveloperAgent
from src.agents.qa_agent import QAAgent
from src.core.task_queue import Task


@pytest.fixture
def developer_agent():
    """Create a DeveloperAgent instance for testing."""
    return DeveloperAgent(name="dev-agent-test")


@pytest.fixture
def qa_agent():
    """Create a QAAgent instance for testing."""
    return QAAgent(name="qa-agent-test")


@pytest.fixture
def pair_programming_session(developer_agent, qa_agent):
    """Create a PairProgrammingSession instance for testing."""
    return PairProgrammingSession(
        session_id="test-session-123",
        developer_agent=developer_agent,
        qa_agent=qa_agent,
    )


@pytest.fixture
def sample_task():
    """Create a sample task for pair programming."""
    return Task(
        id="pair-task-456",
        title="User Authentication",
        description="Implement user authentication with JWT tokens",
        priority=1,
        task_type="implementation",
        payload={"framework": "FastAPI", "database": "PostgreSQL", "security": "JWT"},
    )


@pytest.mark.asyncio
class TestPairProgrammingSession:
    """Test cases for PairProgrammingSession functionality."""

    async def test_pair_programming_session_initialization(
        self, pair_programming_session
    ):
        """Test that PairProgrammingSession initializes correctly."""
        assert pair_programming_session.session_id == "test-session-123"
        assert pair_programming_session.developer_agent.name == "dev-agent-test"
        assert pair_programming_session.qa_agent.name == "qa-agent-test"
        assert pair_programming_session.status == SessionStatus.INITIALIZED
        assert pair_programming_session.current_phase == SessionPhase.PLANNING

    async def test_session_has_collaboration_phases(self, pair_programming_session):
        """Test that session supports different collaboration phases."""
        assert hasattr(pair_programming_session, "current_phase")
        # Check all required phases exist
        assert SessionPhase.PLANNING in SessionPhase
        assert SessionPhase.DEVELOPMENT in SessionPhase
        assert SessionPhase.REVIEW in SessionPhase
        assert SessionPhase.TESTING in SessionPhase
        assert SessionPhase.REFINEMENT in SessionPhase
        assert SessionPhase.COMPLETION in SessionPhase

    async def test_session_status_tracking(self, pair_programming_session):
        """Test session status tracking capabilities."""
        assert hasattr(pair_programming_session, "status")
        # Check all required statuses exist
        assert SessionStatus.INITIALIZED in SessionStatus
        assert SessionStatus.IN_PROGRESS in SessionStatus
        assert SessionStatus.COMPLETED in SessionStatus
        assert SessionStatus.FAILED in SessionStatus
        assert SessionStatus.CANCELLED in SessionStatus

    @patch(
        "src.core.collaboration.pair_programming.PairProgrammingSession._execute_planning_phase"
    )
    async def test_start_collaboration(
        self, mock_planning, pair_programming_session, sample_task
    ):
        """Test starting a pair programming collaboration."""
        mock_planning.return_value = AsyncMock()
        mock_planning.return_value.success = True
        mock_planning.return_value.next_phase = SessionPhase.DEVELOPMENT

        result = await pair_programming_session.start_collaboration(sample_task)

        assert isinstance(result, CollaborationResult)
        assert result.success is True
        assert pair_programming_session.status == SessionStatus.IN_PROGRESS
        mock_planning.assert_called_once_with(sample_task)

    @patch(
        "src.core.collaboration.pair_programming.PairProgrammingSession._execute_development_phase"
    )
    async def test_development_phase_execution(
        self, mock_development, pair_programming_session, sample_task
    ):
        """Test development phase execution."""
        mock_development.return_value = AsyncMock()
        mock_development.return_value.success = True
        mock_development.return_value.code_generated = True
        mock_development.return_value.tests_generated = True

        # Set up session in development phase
        pair_programming_session.current_phase = SessionPhase.DEVELOPMENT

        result = await pair_programming_session._execute_development_phase(sample_task)

        assert result.success is True
        assert result.code_generated is True
        assert result.tests_generated is True

    @patch(
        "src.core.collaboration.pair_programming.PairProgrammingSession._execute_review_phase"
    )
    async def test_review_phase_execution(
        self, mock_review, pair_programming_session, sample_task
    ):
        """Test review phase execution."""
        mock_review.return_value = AsyncMock()
        mock_review.return_value.success = True
        mock_review.return_value.issues_found = []
        mock_review.return_value.approval_status = "approved"

        # Set up session in review phase
        pair_programming_session.current_phase = SessionPhase.REVIEW

        result = await pair_programming_session._execute_review_phase(sample_task)

        assert result.success is True
        assert result.issues_found == []
        assert result.approval_status == "approved"

    async def test_developer_qa_communication(
        self, pair_programming_session, sample_task
    ):
        """Test communication between developer and QA agents."""
        # Mock agent communication methods
        with (
            patch.object(
                pair_programming_session.developer_agent, "handle_collaboration"
            ) as mock_dev,
            patch.object(
                pair_programming_session.qa_agent, "handle_collaboration"
            ) as mock_qa,
        ):
            mock_dev.return_value = {
                "success": True,
                "response": "I'll implement the authentication system with JWT",
                "agent": "dev-agent-test",
            }

            mock_qa.return_value = {
                "success": True,
                "response": "I'll create comprehensive tests for security validation",
                "agent": "qa-agent-test",
            }

            dev_response = await pair_programming_session._communicate_with_developer(
                "Please implement JWT authentication"
            )
            qa_response = await pair_programming_session._communicate_with_qa(
                "Please create security tests"
            )

            assert dev_response["success"] is True
            assert "authentication" in dev_response["response"]
            assert qa_response["success"] is True
            assert "tests" in qa_response["response"]

    async def test_feedback_loop_implementation(
        self, pair_programming_session, sample_task
    ):
        """Test iterative feedback loop between agents."""
        feedback_history = []

        # Mock feedback collection
        with patch.object(
            pair_programming_session, "_collect_feedback"
        ) as mock_feedback:
            mock_feedback.side_effect = [
                {"iteration": 1, "feedback": "Add input validation"},
                {"iteration": 2, "feedback": "Improve error handling"},
                {"iteration": 3, "feedback": "Code looks good"},
            ]

            # Test feedback loop
            for i in range(3):
                feedback = await pair_programming_session._collect_feedback(
                    sample_task, i + 1
                )
                feedback_history.append(feedback)

            assert len(feedback_history) == 3
            assert feedback_history[0]["iteration"] == 1
            assert feedback_history[-1]["feedback"] == "Code looks good"

    async def test_quality_gate_validation(self, pair_programming_session, sample_task):
        """Test quality gate validation during collaboration."""
        with patch.object(
            pair_programming_session, "_validate_quality_gates"
        ) as mock_validation:
            mock_validation.return_value = {
                "passed": True,
                "test_coverage": 95,
                "code_quality": "excellent",
                "security_scan": "passed",
                "performance": "acceptable",
            }

            result = await pair_programming_session._validate_quality_gates(sample_task)

            assert result["passed"] is True
            assert result["test_coverage"] >= 90
            assert result["security_scan"] == "passed"

    async def test_session_completion_workflow(
        self, pair_programming_session, sample_task
    ):
        """Test complete session workflow from start to finish."""
        # Mock all phase executions
        with (
            patch.object(
                pair_programming_session, "_execute_planning_phase"
            ) as mock_planning,
            patch.object(
                pair_programming_session, "_execute_development_phase"
            ) as mock_dev,
            patch.object(
                pair_programming_session, "_execute_review_phase"
            ) as mock_review,
            patch.object(
                pair_programming_session, "_execute_testing_phase"
            ) as mock_testing,
            patch.object(
                pair_programming_session, "_execute_completion_phase"
            ) as mock_completion,
        ):
            # Configure mock returns for successful workflow
            mock_planning.return_value = AsyncMock(
                success=True, next_phase=SessionPhase.DEVELOPMENT
            )
            mock_dev.return_value = AsyncMock(
                success=True, next_phase=SessionPhase.REVIEW
            )
            mock_review.return_value = AsyncMock(
                success=True, next_phase=SessionPhase.TESTING
            )
            mock_testing.return_value = AsyncMock(
                success=True, next_phase=SessionPhase.COMPLETION
            )
            mock_completion.return_value = AsyncMock(
                success=True, session_complete=True
            )

            # Execute complete workflow
            result = await pair_programming_session.execute_complete_workflow(
                sample_task
            )

            assert result.success is True
            assert pair_programming_session.status == SessionStatus.COMPLETED

            # Verify all phases were executed
            mock_planning.assert_called_once()
            mock_dev.assert_called_once()
            mock_review.assert_called_once()
            mock_testing.assert_called_once()
            mock_completion.assert_called_once()

    async def test_error_handling_and_recovery(
        self, pair_programming_session, sample_task
    ):
        """Test error handling and recovery mechanisms."""
        # Test development phase failure and recovery
        with patch.object(
            pair_programming_session, "_execute_development_phase"
        ) as mock_dev:
            mock_dev.side_effect = Exception("Development failed")

            with patch.object(
                pair_programming_session, "_handle_phase_failure"
            ) as mock_recovery:
                mock_recovery.return_value = {
                    "recovered": True,
                    "retry_phase": SessionPhase.DEVELOPMENT,
                    "action": "retry_with_adjustments",
                }

                # This should handle the error gracefully
                recovery = await pair_programming_session._handle_phase_failure(
                    SessionPhase.DEVELOPMENT, Exception("Development failed")
                )

                assert recovery["recovered"] is True
                assert recovery["action"] == "retry_with_adjustments"

    async def test_session_metrics_tracking(
        self, pair_programming_session, sample_task
    ):
        """Test session metrics and performance tracking."""
        # Test metrics collection
        metrics = pair_programming_session.get_session_metrics()

        assert isinstance(metrics, dict)
        assert "session_id" in metrics
        assert "start_time" in metrics
        assert "current_phase" in metrics
        assert "status" in metrics
        assert "agents_involved" in metrics

    async def test_collaboration_result_structure(
        self, pair_programming_session, sample_task
    ):
        """Test CollaborationResult data structure."""
        # Create a sample collaboration result
        result = CollaborationResult(
            success=True,
            session_id="test-session-123",
            phases_completed=[SessionPhase.PLANNING, SessionPhase.DEVELOPMENT],
            final_deliverables={
                "code_files": ["auth.py", "test_auth.py"],
                "documentation": ["README.md"],
                "test_coverage": 95,
            },
            collaboration_summary="Successfully implemented JWT authentication with comprehensive tests",
        )

        assert result.success is True
        assert result.session_id == "test-session-123"
        assert len(result.phases_completed) == 2
        assert "auth.py" in result.final_deliverables["code_files"]
        assert result.final_deliverables["test_coverage"] == 95

    async def test_agent_role_specialization(
        self, pair_programming_session, sample_task
    ):
        """Test that agents maintain their specialized roles during collaboration."""
        # Test developer agent focus areas
        dev_tasks = await pair_programming_session._get_developer_tasks(sample_task)
        assert isinstance(dev_tasks, list)

        # Test QA agent focus areas
        qa_tasks = await pair_programming_session._get_qa_tasks(sample_task)
        assert isinstance(qa_tasks, list)

        # Ensure no overlap in core responsibilities
        dev_focus = {"implementation", "coding", "architecture"}
        qa_focus = {"testing", "quality", "validation", "security"}

        # This is a conceptual test - in reality these would be determined by the task analysis
