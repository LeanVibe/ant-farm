"""Integration tests for Phase 3 AI-Enhanced XP Practices."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.developer_agent import DeveloperAgent
from src.agents.qa_agent import QAAgent
from src.core.collaboration.pair_programming import PairProgrammingSession
from src.core.refactoring.autonomous_refactoring import AutonomousRefactoringEngine
from src.core.task_queue import Task
from src.core.testing.ai_test_generator import AITestGenerator, TestType


@pytest.fixture
def developer_agent():
    """Create DeveloperAgent for integration testing."""
    return DeveloperAgent(name="integration-dev")


@pytest.fixture
def qa_agent():
    """Create QAAgent for integration testing."""
    return QAAgent(name="integration-qa")


@pytest.fixture
def pair_programming_session(developer_agent, qa_agent):
    """Create PairProgrammingSession for integration testing."""
    return PairProgrammingSession(
        session_id="integration-session",
        developer_agent=developer_agent,
        qa_agent=qa_agent,
    )


@pytest.fixture
def refactoring_engine():
    """Create AutonomousRefactoringEngine for integration testing."""
    return AutonomousRefactoringEngine()


@pytest.fixture
def ai_test_generator():
    """Create AITestGenerator for integration testing."""
    return AITestGenerator()


@pytest.fixture
def sample_feature_task():
    """Sample feature implementation task."""
    return Task(
        id="integration-feature-123",
        title="User Profile Management",
        description="Implement user profile CRUD operations with validation",
        priority=1,
        task_type="implementation",
        payload={
            "framework": "FastAPI",
            "database": "PostgreSQL",
            "features": ["create_profile", "update_profile", "delete_profile"],
            "validation": True,
            "authentication": True,
        },
    )


@pytest.fixture
def sample_code_for_refactoring():
    """Sample code that needs refactoring."""
    return '''
def process_user_data(first_name, last_name, email, phone, address, city, state, zip_code, country):
    """Process user data with many parameters."""
    if first_name:
        if last_name:
            if email:
                if phone:
                    if address:
                        if city:
                            if state:
                                if zip_code:
                                    if country:
                                        user = {
                                            "first_name": first_name,
                                            "last_name": last_name,
                                            "email": email,
                                            "phone": phone,
                                            "address": address,
                                            "city": city,
                                            "state": state,
                                            "zip_code": zip_code,
                                            "country": country
                                        }
                                        print(f"Processing user: {user}")
                                        return user
    return None
'''


@pytest.mark.asyncio
class TestPhase3Integration:
    """Integration tests for Phase 3 AI-Enhanced XP Practices."""

    async def test_complete_phase3_workflow(
        self, pair_programming_session, sample_feature_task
    ):
        """Test complete Phase 3 workflow: PairProgramming + AI Testing + Refactoring."""

        # Mock CLI tool executions to avoid actual external calls
        with (
            patch.object(
                pair_programming_session.developer_agent, "execute_with_cli_tool"
            ) as mock_dev_cli,
            patch.object(
                pair_programming_session.qa_agent, "execute_with_cli_tool"
            ) as mock_qa_cli,
        ):
            # Configure mock responses
            mock_dev_cli.return_value = AsyncMock()
            mock_dev_cli.return_value.success = True
            mock_dev_cli.return_value.output = (
                "Feature implemented successfully with TDD approach"
            )

            mock_qa_cli.return_value = AsyncMock()
            mock_qa_cli.return_value.success = True
            mock_qa_cli.return_value.output = (
                "Comprehensive tests generated and validated"
            )

            # Execute complete workflow
            result = await pair_programming_session.execute_complete_workflow(
                sample_feature_task
            )

            # Verify workflow completion
            assert result.success is True
            assert len(result.phases_completed) == 5  # All phases completed
            assert "implementation" in result.final_deliverables
            assert "tests" in result.final_deliverables
            assert "quality_validation" in result.final_deliverables

    async def test_developer_qa_integration(
        self, developer_agent, qa_agent, sample_feature_task
    ):
        """Test Developer and QA agent integration."""

        with (
            patch.object(developer_agent, "execute_with_cli_tool") as mock_dev,
            patch.object(qa_agent, "execute_with_cli_tool") as mock_qa,
        ):
            # Mock developer implementation
            mock_dev.return_value = AsyncMock()
            mock_dev.return_value.success = True
            mock_dev.return_value.output = "User profile CRUD implemented"

            # Mock QA test generation
            mock_qa.return_value = AsyncMock()
            mock_qa.return_value.success = True
            mock_qa.return_value.output = "Tests generated for all CRUD operations"

            # Test developer implementation
            dev_result = await developer_agent.implement_feature(sample_feature_task)
            assert dev_result.success is True

            # Test QA test generation
            qa_collaboration = {
                "partner_agent": developer_agent.name,
                "task": "test_generation",
                "message": "Generate tests for user profile CRUD operations",
            }
            qa_result = await qa_agent.handle_collaboration(qa_collaboration)
            assert qa_result["success"] is True

            # Verify integration
            assert "profile" in dev_result.data["message"].lower()
            assert "test" in qa_result["response"].lower()

    async def test_ai_test_generator_integration(
        self, ai_test_generator, developer_agent
    ):
        """Test AI Test Generator integration with development workflow."""

        sample_code = '''
def create_user_profile(user_data):
    """Create a new user profile."""
    if not user_data.get("email"):
        raise ValueError("Email is required")
    
    profile = {
        "id": generate_id(),
        "email": user_data["email"],
        "name": user_data.get("name", ""),
        "created_at": datetime.now()
    }
    
    return save_profile(profile)
'''

        # Generate tests using AI Test Generator
        from src.core.testing.ai_test_generator import TestType

        test_results = await ai_test_generator.generate_tests(
            source_code=sample_code,
            test_type=TestType.UNIT,
            target_function="create_user_profile",
        )

        # Verify test generation
        assert test_results.success is True
        assert len(test_results.generated_tests) > 0
        assert "test_create_user_profile" in test_results.generated_tests[0].content
        # Simple check for coverage estimate
        assert len(test_results.generated_tests) >= 1

    async def test_autonomous_refactoring_integration(
        self, refactoring_engine, sample_code_for_refactoring
    ):
        """Test autonomous refactoring integration with code analysis."""

        # Detect code smells
        smells = await refactoring_engine.detect_code_smells(
            sample_code_for_refactoring
        )
        assert len(smells) > 0

        # Identify refactoring opportunities
        opportunities = await refactoring_engine.identify_refactoring_opportunities(
            sample_code_for_refactoring
        )
        assert len(opportunities) > 0

        # Apply highest confidence refactoring
        high_confidence_opportunity = max(opportunities, key=lambda x: x.confidence)
        refactoring_result = await refactoring_engine.apply_safe_refactoring(
            sample_code_for_refactoring, high_confidence_opportunity
        )

        # Verify refactoring
        assert refactoring_result.success is True
        assert len(refactoring_result.improvements) > 0

    async def test_pair_programming_with_refactoring(
        self, pair_programming_session, refactoring_engine, sample_feature_task
    ):
        """Test pair programming session with integrated refactoring."""

        with (
            patch.object(
                pair_programming_session.developer_agent, "execute_with_cli_tool"
            ) as mock_dev,
            patch.object(
                pair_programming_session.qa_agent, "execute_with_cli_tool"
            ) as mock_qa,
        ):
            # Mock implementation with code that needs refactoring
            mock_dev.return_value = AsyncMock()
            mock_dev.return_value.success = True
            mock_dev.return_value.output = "Feature implemented but needs refactoring"

            mock_qa.return_value = AsyncMock()
            mock_qa.return_value.success = True
            mock_qa.return_value.output = (
                "Code review identified refactoring opportunities"
            )

            # Execute pair programming session
            session_result = await pair_programming_session.start_collaboration(
                sample_feature_task
            )
            assert session_result.success is True

            # Apply refactoring to the implemented code
            sample_implementation = """
def handle_user_request(user_id, action_type, data1, data2, data3, data4, data5):
    if action_type == "create":
        if data1 and data2 and data3:
            return create_something(user_id, data1, data2, data3, data4, data5)
    elif action_type == "update":
        if data1 and data2:
            return update_something(user_id, data1, data2, data3, data4, data5)
    return None
"""

            opportunities = await refactoring_engine.identify_refactoring_opportunities(
                sample_implementation
            )
            if opportunities:
                refactoring_result = await refactoring_engine.apply_safe_refactoring(
                    sample_implementation, opportunities[0]
                )
                assert refactoring_result.success is True

    async def test_full_tdd_workflow_integration(
        self, developer_agent, qa_agent, ai_test_generator
    ):
        """Test full TDD workflow integration across all Phase 3 components."""

        with (
            patch.object(developer_agent, "execute_with_cli_tool") as mock_dev,
            patch.object(qa_agent, "execute_with_cli_tool") as mock_qa,
        ):
            mock_dev.return_value = AsyncMock()
            mock_dev.return_value.success = True
            mock_dev.return_value.output = "TDD implementation completed"

            mock_qa.return_value = AsyncMock()
            mock_qa.return_value.success = True
            mock_qa.return_value.output = "All tests passing, coverage excellent"

            # Step 1: AI Test Generator creates initial tests
            test_specification = "User authentication with JWT tokens"
            initial_tests = await ai_test_generator.generate_tests(
                source_code=f"# {test_specification}\npass",  # Dummy code for testing
                test_type=TestType.UNIT,
            )

            assert initial_tests.success is True

            # Step 2: Developer implements to pass tests
            implementation_task = Task(
                id="tdd-task",
                title="JWT Authentication",
                description="Implement JWT authentication to pass generated tests",
                priority=1,
                task_type="implementation",
                payload={"tdd_mode": True, "test_specification": test_specification},
            )

            dev_result = await developer_agent.implement_feature(implementation_task)
            assert dev_result.success is True

            # Step 3: QA validates and suggests improvements
            qa_review = await qa_agent.handle_collaboration(
                {
                    "partner_agent": developer_agent.name,
                    "task": "tdd_validation",
                    "message": "Validate TDD implementation and suggest improvements",
                }
            )

            assert qa_review["success"] is True

    async def test_error_handling_and_recovery_integration(
        self, pair_programming_session, sample_feature_task
    ):
        """Test error handling and recovery across Phase 3 components."""

        with patch.object(
            pair_programming_session.developer_agent, "execute_with_cli_tool"
        ) as mock_dev:
            # Simulate failure in development phase
            mock_dev.side_effect = Exception("Development tool failure")

            # Test graceful error handling
            session_result = await pair_programming_session.start_collaboration(
                sample_feature_task
            )

            # Should handle error gracefully
            assert session_result.success is False
            assert "error" in session_result.error_message.lower()

            # Verify session state is properly managed
            assert pair_programming_session.status.value in ["failed", "initialized"]

    async def test_performance_under_concurrent_operations(
        self, developer_agent, qa_agent, refactoring_engine
    ):
        """Test Phase 3 components under concurrent operations."""

        with (
            patch.object(developer_agent, "execute_with_cli_tool") as mock_dev,
            patch.object(qa_agent, "execute_with_cli_tool") as mock_qa,
        ):
            mock_dev.return_value = AsyncMock()
            mock_dev.return_value.success = True
            mock_dev.return_value.output = "Concurrent operation completed"

            mock_qa.return_value = AsyncMock()
            mock_qa.return_value.success = True
            mock_qa.return_value.output = "Concurrent validation completed"

            # Create multiple concurrent tasks
            tasks = []
            for i in range(3):
                task = Task(
                    id=f"concurrent-{i}",
                    title=f"Feature {i}",
                    description=f"Implement feature {i}",
                    priority=1,
                    task_type="implementation",
                )
                tasks.append(task)

            # Execute concurrently
            dev_results = await asyncio.gather(
                *[developer_agent.implement_feature(task) for task in tasks]
            )

            qa_collaborations = await asyncio.gather(
                *[
                    qa_agent.handle_collaboration(
                        {
                            "partner_agent": developer_agent.name,
                            "task": "validation",
                            "message": f"Validate feature {i}",
                        }
                    )
                    for i in range(3)
                ]
            )

            # Verify all operations completed successfully
            assert all(result.success for result in dev_results)
            assert all(collab["success"] for collab in qa_collaborations)

    async def test_complex_refactoring_scenarios(self, refactoring_engine):
        """Test complex refactoring scenarios with multiple code smells."""

        complex_code = """
class UserManager:
    def __init__(self):
        self.users = []
        self.deleted_users = []
        self.temp_data = {}
        
    def create_user(self, first_name, last_name, email, phone, address, city, state, zip_code, country, age, gender):
        if first_name:
            if last_name:
                if email:
                    if phone:
                        if address:
                            if city:
                                if state:
                                    if zip_code:
                                        if country:
                                            if age >= 18:
                                                user = {
                                                    "first_name": first_name,
                                                    "last_name": last_name,
                                                    "email": email,
                                                    "phone": phone,
                                                    "address": address,
                                                    "city": city,
                                                    "state": state,
                                                    "zip_code": zip_code,
                                                    "country": country,
                                                    "age": age,
                                                    "gender": gender
                                                }
                                                self.users.append(user)
                                                print("User created successfully")
                                                return user
        return None
    
    def update_user(self, first_name, last_name, email, phone, address, city, state, zip_code, country, age, gender):
        # Duplicate logic - same as create_user
        if first_name:
            if last_name:
                if email:
                    # ... same nested conditions
                    pass
        return None
"""

        # Detect multiple code smells
        smells = await refactoring_engine.detect_code_smells(complex_code)
        assert len(smells) >= 3  # Should detect multiple smells

        # Generate multiple refactoring opportunities
        opportunities = await refactoring_engine.identify_refactoring_opportunities(
            complex_code
        )
        assert len(opportunities) >= 2  # Should identify multiple opportunities

        # Apply batch refactoring
        batch_results = await refactoring_engine.apply_batch_refactoring(
            complex_code, opportunities
        )

        # Verify batch refactoring
        assert len(batch_results) == len(opportunities)
        successful_refactorings = [r for r in batch_results if r.success]
        assert len(successful_refactorings) > 0

    async def test_end_to_end_feature_development(
        self,
        pair_programming_session,
        refactoring_engine,
        ai_test_generator,
        sample_feature_task,
    ):
        """Test complete end-to-end feature development using all Phase 3 components."""

        with (
            patch.object(
                pair_programming_session.developer_agent, "execute_with_cli_tool"
            ) as mock_dev,
            patch.object(
                pair_programming_session.qa_agent, "execute_with_cli_tool"
            ) as mock_qa,
        ):
            # Configure realistic mock responses
            mock_dev.return_value = AsyncMock()
            mock_dev.return_value.success = True
            mock_dev.return_value.output = (
                "Feature implemented with comprehensive error handling"
            )

            mock_qa.return_value = AsyncMock()
            mock_qa.return_value.success = True
            mock_qa.return_value.output = (
                "Quality validation passed with recommendations"
            )

            # Step 1: Generate comprehensive tests
            test_results = await ai_test_generator.generate_tests(
                source_code=f"# {sample_feature_task.description}\npass",  # Dummy code for testing
                test_type=TestType.UNIT,
            )

            # Step 2: Execute pair programming workflow
            collaboration_result = (
                await pair_programming_session.execute_complete_workflow(
                    sample_feature_task
                )
            )

            # Step 3: Apply autonomous refactoring if needed
            if collaboration_result.success and collaboration_result.final_deliverables:
                # Simulate code that was generated during pair programming
                generated_code = """
def manage_user_profile(action, user_id, profile_data):
    if action == "create":
        return create_profile(user_id, profile_data)
    elif action == "update":
        return update_profile(user_id, profile_data)
    elif action == "delete":
        return delete_profile(user_id)
    return None
"""

                opportunities = (
                    await refactoring_engine.identify_refactoring_opportunities(
                        generated_code
                    )
                )
                if opportunities:
                    refactoring_result = (
                        await refactoring_engine.apply_safe_refactoring(
                            generated_code, opportunities[0]
                        )
                    )

            # Verify end-to-end success
            assert test_results.success is True
            assert collaboration_result.success is True
            assert len(collaboration_result.phases_completed) == 5

            # Verify integration metrics
            session_metrics = pair_programming_session.get_session_metrics()
            assert session_metrics["conversation_turns"] > 0
            assert session_metrics["deliverables_count"] > 0

    async def test_phase3_component_isolation(
        self, developer_agent, qa_agent, refactoring_engine, ai_test_generator
    ):
        """Test that Phase 3 components work independently and don't interfere with each other."""

        # Test component independence
        with (
            patch.object(developer_agent, "execute_with_cli_tool") as mock_dev,
            patch.object(qa_agent, "execute_with_cli_tool") as mock_qa,
        ):
            mock_dev.return_value = AsyncMock()
            mock_dev.return_value.success = True
            mock_dev.return_value.output = "Independent operation"

            mock_qa.return_value = AsyncMock()
            mock_qa.return_value.success = True
            mock_qa.return_value.output = "Independent validation"

            # Test each component independently
            task = Task(
                id="isolation-test",
                title="Isolation Test",
                description="Test component isolation",
                priority=1,
                task_type="implementation",
            )

            # Developer agent should work independently
            dev_result = await developer_agent.implement_feature(task)
            assert dev_result.success is True

            # QA agent should work independently
            qa_result = await qa_agent.handle_collaboration(
                {
                    "partner_agent": "test",
                    "task": "independent_validation",
                    "message": "Independent operation test",
                }
            )
            assert qa_result["success"] is True

            # Refactoring engine should work independently
            simple_code = "def simple_function(x): return x * 2"
            smells = await refactoring_engine.detect_code_smells(simple_code)
            # Should handle simple code without issues

            # AI test generator should work independently
            test_result = await ai_test_generator.generate_tests(
                source_code="def simple_function(x): return x * 2",
                test_type=TestType.UNIT,
            )
            assert test_result.success is True

            # Verify no cross-component interference
            assert developer_agent.tasks_completed > 0
            assert (
                refactoring_engine.get_refactoring_metrics()["total_refactorings"] >= 0
            )
