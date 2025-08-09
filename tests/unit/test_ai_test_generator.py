"""
Unit tests for AI Test Generator - TDD implementation.

The AI Test Generator analyzes code and generates comprehensive test suites
that go beyond what humans typically think to test.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from src.core.testing.ai_test_generator import (
    AITestGenerator,
    CodeAnalysis,
    EdgeCase,
    TestGenerationResult,
    TestType,
)


class TestAITestGenerator:
    """Test suite for AI Test Generator using TDD approach."""

    @pytest.fixture
    def mock_code_analyzer(self):
        """Mock code analyzer for testing."""
        analyzer = Mock()
        analyzer.analyze_function = AsyncMock()
        analyzer.analyze_class = AsyncMock()
        analyzer.find_edge_cases = AsyncMock()
        return analyzer

    @pytest.fixture
    def test_generator(self, mock_code_analyzer):
        """Create AI Test Generator instance for testing."""
        return AITestGenerator(code_analyzer=mock_code_analyzer)

    @pytest.mark.asyncio
    async def test_generate_unit_tests_for_simple_function(
        self, test_generator, mock_code_analyzer
    ):
        """Test generation of unit tests for a simple function."""
        # Arrange
        source_code = """
def calculate_discount(price: float, discount_percent: float) -> float:
    if price < 0:
        raise ValueError("Price cannot be negative")
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount must be between 0 and 100")
    return price * (1 - discount_percent / 100)
        """

        mock_analysis = CodeAnalysis(
            function_name="calculate_discount",
            parameters=["price", "discount_percent"],
            return_type="float",
            exceptions=["ValueError"],
            complexity_score=3,
            edge_cases=[
                EdgeCase(condition="price < 0", expected_behavior="raises ValueError"),
                EdgeCase(
                    condition="discount_percent < 0",
                    expected_behavior="raises ValueError",
                ),
                EdgeCase(
                    condition="discount_percent > 100",
                    expected_behavior="raises ValueError",
                ),
                EdgeCase(condition="price = 0", expected_behavior="returns 0"),
                EdgeCase(
                    condition="discount_percent = 0",
                    expected_behavior="returns original price",
                ),
                EdgeCase(
                    condition="discount_percent = 100", expected_behavior="returns 0"
                ),
            ],
        )

        mock_code_analyzer.analyze_function.return_value = mock_analysis

        # Act
        result = await test_generator.generate_tests(
            source_code=source_code, test_type=TestType.UNIT
        )

        # Assert
        assert isinstance(result, TestGenerationResult)
        assert result.success is True
        assert len(result.generated_tests) > 0

        # Verify test covers all edge cases
        all_test_content = "\n".join([test.content for test in result.generated_tests])
        assert "test_negative_price_raises_error" in all_test_content
        assert "test_negative_discount_raises_error" in all_test_content
        assert "test_discount_over_100_raises_error" in all_test_content
        assert "test_zero_price_returns_zero" in all_test_content
        assert "test_zero_discount_returns_original_price" in all_test_content
        assert "test_full_discount_returns_zero" in all_test_content

    @pytest.mark.asyncio
    async def test_generate_integration_tests_for_class(
        self, test_generator, mock_code_analyzer
    ):
        """Test generation of integration tests for a class."""
        # Arrange
        source_code = """
class UserRepository:
    def __init__(self, db_session):
        self.db = db_session

    async def create_user(self, user_data: dict) -> User:
        # Create user in database
        pass

    async def get_user(self, user_id: str) -> Optional[User]:
        # Retrieve user from database
        pass
        """

        mock_analysis = CodeAnalysis(
            class_name="UserRepository",
            methods=["create_user", "get_user"],
            dependencies=["db_session"],
            complexity_score=5,
            edge_cases=[
                EdgeCase(
                    condition="database connection fails",
                    expected_behavior="raises DatabaseError",
                ),
                EdgeCase(
                    condition="user_id not found", expected_behavior="returns None"
                ),
                EdgeCase(
                    condition="duplicate user creation",
                    expected_behavior="raises IntegrityError",
                ),
            ],
        )

        mock_code_analyzer.analyze_class.return_value = mock_analysis

        # Act
        result = await test_generator.generate_tests(
            source_code=source_code, test_type=TestType.INTEGRATION
        )

        # Assert
        assert result.success is True
        assert len(result.generated_tests) > 0

        all_test_content = "\n".join([test.content for test in result.generated_tests])
        assert "test_create_user_success" in all_test_content
        assert "test_get_user_success" in all_test_content
        # Check that the edge case generated a test (the test name will be in the function def)
        assert any("database" in test.name for test in result.generated_tests)
        assert "@pytest.mark.asyncio" in all_test_content

    @pytest.mark.asyncio
    async def test_generate_performance_tests(self, test_generator, mock_code_analyzer):
        """Test generation of performance tests."""
        # Arrange
        source_code = """
def process_large_dataset(data: List[Dict]) -> List[Dict]:
    return [transform_record(record) for record in data]
        """

        mock_analysis = CodeAnalysis(
            function_name="process_large_dataset",
            parameters=["data"],
            return_type="List[Dict]",
            complexity_score=2,
            performance_considerations=["large_dataset_processing", "memory_usage"],
        )

        mock_code_analyzer.analyze_function.return_value = mock_analysis

        # Act
        result = await test_generator.generate_tests(
            source_code=source_code, test_type=TestType.PERFORMANCE
        )

        # Assert
        assert result.success is True
        assert (
            len(result.generated_tests) == 2
        )  # Should generate 2 tests for 2 considerations

        all_test_content = "\n".join([test.content for test in result.generated_tests])
        assert "test_performance_large_dataset" in all_test_content
        assert "time.time()" in all_test_content or "timeit" in all_test_content
        assert "memory_usage" in all_test_content or "tracemalloc" in all_test_content

    @pytest.mark.asyncio
    async def test_generate_security_tests(self, test_generator, mock_code_analyzer):
        """Test generation of security-focused tests."""
        # Arrange
        source_code = """
def execute_query(query: str, params: dict) -> List[Dict]:
    return db.execute(query, params)
        """

        mock_analysis = CodeAnalysis(
            function_name="execute_query",
            parameters=["query", "params"],
            security_risks=["sql_injection", "malicious_input"],
            complexity_score=4,
        )

        mock_code_analyzer.analyze_function.return_value = mock_analysis

        # Act
        result = await test_generator.generate_tests(
            source_code=source_code, test_type=TestType.SECURITY
        )

        # Assert
        assert result.success is True
        test_content = result.generated_tests[0].content
        assert "sql_injection" in test_content.lower()
        assert "malicious" in test_content.lower()

    @pytest.mark.asyncio
    async def test_generation_failure_handling(
        self, test_generator, mock_code_analyzer
    ):
        """Test handling of test generation failures."""
        # Arrange
        mock_code_analyzer.analyze_function.side_effect = Exception("Analysis failed")

        # Act
        result = await test_generator.generate_tests(
            source_code="invalid code", test_type=TestType.UNIT
        )

        # Assert
        assert result.success is False
        assert "Analysis failed" in result.error_message
        assert len(result.generated_tests) == 0

    def test_test_type_enum_values(self):
        """Test that TestType enum has expected values."""
        assert TestType.UNIT.value == "unit"
        assert TestType.INTEGRATION.value == "integration"
        assert TestType.PERFORMANCE.value == "performance"
        assert TestType.SECURITY.value == "security"
        assert TestType.CHAOS.value == "chaos"
