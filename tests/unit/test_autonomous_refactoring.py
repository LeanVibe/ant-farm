"""Unit tests for AutonomousRefactoring engine with TDD approach."""

import asyncio
import ast
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.refactoring.autonomous_refactoring import (
    AutonomousRefactoringEngine,
    RefactoringResult,
    RefactoringType,
    CodeSmell,
    RefactoringOpportunity,
    RefactoringConfidence,
)


@pytest.fixture
def refactoring_engine():
    """Create an AutonomousRefactoringEngine instance for testing."""
    return AutonomousRefactoringEngine()


@pytest.fixture
def sample_code_with_smells():
    """Sample code with various code smells for testing."""
    return '''
def long_function(x, y, z, a, b, c, d, e, f, g):
    """A function with too many parameters."""
    if x > 0:
        if y > 0:
            if z > 0:
                if a > 0:
                    if b > 0:
                        result = x + y + z + a + b
                        print(result)  # Direct print usage
                        return result
                    else:
                        return 0
                else:
                    return 0
            else:
                return 0
        else:
            return 0
    else:
        return 0

class DuplicatedCode:
    def method1(self):
        data = self.get_data()
        processed = self.process_data(data)
        self.save_data(processed)
        
    def method2(self):
        data = self.get_data()
        processed = self.process_data(data)
        self.save_data(processed)
'''


@pytest.fixture
def clean_code():
    """Sample of well-written code for testing."""
    return '''
def calculate_area(radius: float) -> float:
    """Calculate the area of a circle."""
    return 3.14159 * radius * radius

class Calculator:
    """A simple calculator class."""
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
'''


@pytest.mark.asyncio
class TestAutonomousRefactoringEngine:
    """Test cases for AutonomousRefactoringEngine functionality."""

    async def test_refactoring_engine_initialization(self, refactoring_engine):
        """Test that AutonomousRefactoringEngine initializes correctly."""
        assert refactoring_engine.confidence_threshold == 0.8
        assert hasattr(refactoring_engine, "code_smell_detectors")
        assert hasattr(refactoring_engine, "refactoring_strategies")
        assert refactoring_engine.auto_apply_threshold == 0.9

    async def test_refactoring_types_enumeration(self):
        """Test that all required refactoring types are available."""
        assert RefactoringType.EXTRACT_METHOD in RefactoringType
        assert RefactoringType.EXTRACT_CLASS in RefactoringType
        assert RefactoringType.RENAME_VARIABLE in RefactoringType
        assert RefactoringType.REMOVE_DUPLICATES in RefactoringType
        assert RefactoringType.SIMPLIFY_CONDITIONALS in RefactoringType
        assert RefactoringType.REDUCE_PARAMETERS in RefactoringType

    async def test_code_smell_enumeration(self):
        """Test that all code smell types are available."""
        assert CodeSmell.LONG_METHOD in CodeSmell
        assert CodeSmell.LARGE_CLASS in CodeSmell
        assert CodeSmell.DUPLICATE_CODE in CodeSmell
        assert CodeSmell.LONG_PARAMETER_LIST in CodeSmell
        assert CodeSmell.COMPLEX_CONDITIONALS in CodeSmell
        assert CodeSmell.MAGIC_NUMBERS in CodeSmell

    async def test_detect_code_smells(
        self, refactoring_engine, sample_code_with_smells
    ):
        """Test code smell detection capabilities."""
        smells = await refactoring_engine.detect_code_smells(sample_code_with_smells)

        assert isinstance(smells, list)
        assert len(smells) > 0

        # Should detect long parameter list
        parameter_smell = next(
            (s for s in smells if s.smell_type == CodeSmell.LONG_PARAMETER_LIST), None
        )
        assert parameter_smell is not None
        assert "long_function" in parameter_smell.location

        # Should detect complex conditionals
        conditional_smell = next(
            (s for s in smells if s.smell_type == CodeSmell.COMPLEX_CONDITIONALS), None
        )
        assert conditional_smell is not None

    async def test_analyze_complexity(
        self, refactoring_engine, sample_code_with_smells
    ):
        """Test code complexity analysis."""
        complexity = await refactoring_engine.analyze_complexity(
            sample_code_with_smells
        )

        assert isinstance(complexity, dict)
        assert "cyclomatic_complexity" in complexity
        assert "function_length" in complexity
        assert "parameter_count" in complexity
        assert complexity["cyclomatic_complexity"] > 5  # High complexity

    async def test_identify_refactoring_opportunities(
        self, refactoring_engine, sample_code_with_smells
    ):
        """Test identification of refactoring opportunities."""
        opportunities = await refactoring_engine.identify_refactoring_opportunities(
            sample_code_with_smells
        )

        assert isinstance(opportunities, list)
        assert len(opportunities) > 0

        # Should identify extract method opportunity
        extract_method = next(
            (
                o
                for o in opportunities
                if o.refactoring_type == RefactoringType.EXTRACT_METHOD
            ),
            None,
        )
        assert extract_method is not None
        assert extract_method.confidence > 0.5

    async def test_confidence_scoring(self, refactoring_engine):
        """Test refactoring confidence scoring."""
        # High confidence case
        high_confidence = refactoring_engine.calculate_confidence(
            smell_severity=0.9, pattern_match=0.8, risk_assessment=0.1
        )
        assert high_confidence >= RefactoringConfidence.HIGH.value

        # Low confidence case
        low_confidence = refactoring_engine.calculate_confidence(
            smell_severity=0.2, pattern_match=0.3, risk_assessment=0.9
        )
        assert low_confidence <= RefactoringConfidence.MEDIUM.value

    @patch(
        "src.core.refactoring.autonomous_refactoring.AutonomousRefactoringEngine._apply_refactoring"
    )
    async def test_apply_safe_refactoring(
        self, mock_apply, refactoring_engine, sample_code_with_smells
    ):
        """Test safe application of refactoring."""
        mock_apply.return_value = RefactoringResult(
            success=True,
            original_code=sample_code_with_smells,
            refactored_code="# Refactored code",
            refactoring_type=RefactoringType.EXTRACT_METHOD,
            confidence=0.95,
            improvements=["Extracted complex logic", "Reduced cyclomatic complexity"],
        )

        opportunity = RefactoringOpportunity(
            refactoring_type=RefactoringType.EXTRACT_METHOD,
            location="long_function",
            description="Extract nested conditional logic",
            confidence=0.95,
            estimated_benefit=0.8,
        )

        result = await refactoring_engine.apply_safe_refactoring(
            sample_code_with_smells, opportunity
        )

        assert result.success is True
        assert result.confidence == 0.95
        assert "Extracted complex logic" in result.improvements
        mock_apply.assert_called_once()

    async def test_validate_refactoring_safety(
        self, refactoring_engine, sample_code_with_smells
    ):
        """Test refactoring safety validation."""
        refactored_code = sample_code_with_smells.replace(
            "long_function", "shorter_function"
        )

        is_safe = await refactoring_engine.validate_refactoring_safety(
            original_code=sample_code_with_smells, refactored_code=refactored_code
        )

        assert isinstance(is_safe, bool)
        # Should be safe since we only renamed a function

    async def test_extract_method_refactoring(self, refactoring_engine):
        """Test extract method refactoring strategy."""
        code_with_long_method = """
def process_order(order):
    # Validate order
    if not order.customer:
        raise ValueError("No customer")
    if not order.items:
        raise ValueError("No items")
    
    # Calculate total
    total = 0
    for item in order.items:
        total += item.price * item.quantity
    
    # Apply discounts
    if order.customer.is_premium:
        total *= 0.9
    
    return total
"""

        result = await refactoring_engine.extract_method(
            code=code_with_long_method,
            method_name="calculate_total",
            start_line=7,
            end_line=10,
        )

        assert result.success is True
        assert "calculate_total" in result.refactored_code
        assert result.refactoring_type == RefactoringType.EXTRACT_METHOD

    async def test_remove_duplicate_code(
        self, refactoring_engine, sample_code_with_smells
    ):
        """Test duplicate code removal."""
        result = await refactoring_engine.remove_duplicate_code(sample_code_with_smells)

        assert result.success is True
        assert result.refactoring_type == RefactoringType.REMOVE_DUPLICATES
        # Should extract common pattern into a method

    async def test_simplify_conditionals(self, refactoring_engine):
        """Test conditional simplification."""
        complex_conditionals = """
def check_eligibility(age, income, credit_score):
    if age >= 18:
        if income > 30000:
            if credit_score > 600:
                return True
            else:
                return False
        else:
            return False
    else:
        return False
"""

        result = await refactoring_engine.simplify_conditionals(complex_conditionals)

        assert result.success is True
        assert result.refactoring_type == RefactoringType.SIMPLIFY_CONDITIONALS
        # Should combine conditions into single expression

    async def test_reduce_parameter_count(self, refactoring_engine):
        """Test parameter count reduction."""
        many_params = """
def create_user(first_name, last_name, email, phone, address, city, state, zip_code, country):
    return User(first_name, last_name, email, phone, address, city, state, zip_code, country)
"""

        result = await refactoring_engine.reduce_parameter_count(many_params)

        assert result.success is True
        assert result.refactoring_type == RefactoringType.REDUCE_PARAMETERS
        # Should suggest parameter object pattern

    async def test_refactoring_impact_analysis(
        self, refactoring_engine, sample_code_with_smells
    ):
        """Test analysis of refactoring impact."""
        opportunity = RefactoringOpportunity(
            refactoring_type=RefactoringType.EXTRACT_METHOD,
            location="long_function",
            description="Extract method",
            confidence=0.8,
            estimated_benefit=0.7,
        )

        impact = await refactoring_engine.analyze_refactoring_impact(
            code=sample_code_with_smells, opportunity=opportunity
        )

        assert isinstance(impact, dict)
        assert "maintainability_improvement" in impact
        assert "readability_improvement" in impact
        assert "complexity_reduction" in impact
        assert "risk_assessment" in impact

    async def test_batch_refactoring(self, refactoring_engine, sample_code_with_smells):
        """Test batch refactoring of multiple opportunities."""
        opportunities = [
            RefactoringOpportunity(
                refactoring_type=RefactoringType.EXTRACT_METHOD,
                location="long_function",
                description="Extract method",
                confidence=0.9,
                estimated_benefit=0.8,
            ),
            RefactoringOpportunity(
                refactoring_type=RefactoringType.SIMPLIFY_CONDITIONALS,
                location="long_function",
                description="Simplify nested conditionals",
                confidence=0.85,
                estimated_benefit=0.7,
            ),
        ]

        results = await refactoring_engine.apply_batch_refactoring(
            code=sample_code_with_smells, opportunities=opportunities
        )

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, RefactoringResult) for r in results)

    async def test_rollback_capability(
        self, refactoring_engine, sample_code_with_smells
    ):
        """Test rollback capability for failed refactoring."""
        # Simulate a failed refactoring
        with patch.object(refactoring_engine, "_apply_refactoring") as mock_apply:
            mock_apply.side_effect = Exception("Refactoring failed")

            opportunity = RefactoringOpportunity(
                refactoring_type=RefactoringType.EXTRACT_METHOD,
                location="long_function",
                description="Extract method",
                confidence=0.8,
                estimated_benefit=0.7,
            )

            result = await refactoring_engine.apply_safe_refactoring(
                code=sample_code_with_smells, opportunity=opportunity
            )

            assert result.success is False
            assert "failed" in result.error_message.lower()
            # Original code should be preserved
            assert result.original_code == sample_code_with_smells

    async def test_clean_code_detection(self, refactoring_engine, clean_code):
        """Test that clean code produces minimal refactoring suggestions."""
        smells = await refactoring_engine.detect_code_smells(clean_code)
        opportunities = await refactoring_engine.identify_refactoring_opportunities(
            clean_code
        )

        # Clean code should have few or no smells
        assert len(smells) <= 2  # Allow for minor suggestions
        assert len(opportunities) <= 1  # Should have minimal refactoring opportunities

    async def test_refactoring_metrics_tracking(
        self, refactoring_engine, sample_code_with_smells
    ):
        """Test metrics tracking for refactoring operations."""
        initial_metrics = refactoring_engine.get_refactoring_metrics()

        opportunity = RefactoringOpportunity(
            refactoring_type=RefactoringType.EXTRACT_METHOD,
            location="long_function",
            description="Extract method",
            confidence=0.8,
            estimated_benefit=0.7,
        )

        with patch.object(refactoring_engine, "_apply_refactoring") as mock_apply:
            mock_apply.return_value = RefactoringResult(
                success=True,
                original_code=sample_code_with_smells,
                refactored_code="# Refactored",
                refactoring_type=RefactoringType.EXTRACT_METHOD,
                confidence=0.8,
                improvements=["Improved readability"],
            )

            await refactoring_engine.apply_safe_refactoring(
                sample_code_with_smells, opportunity
            )

        final_metrics = refactoring_engine.get_refactoring_metrics()

        assert (
            final_metrics["total_refactorings"]
            == initial_metrics["total_refactorings"] + 1
        )
        assert (
            final_metrics["successful_refactorings"]
            == initial_metrics["successful_refactorings"] + 1
        )

    async def test_confidence_thresholds(self, refactoring_engine):
        """Test confidence threshold enforcement."""
        # Low confidence opportunity should not be auto-applied
        low_confidence_opportunity = RefactoringOpportunity(
            refactoring_type=RefactoringType.EXTRACT_METHOD,
            location="function",
            description="Low confidence refactoring",
            confidence=0.3,  # Below threshold
            estimated_benefit=0.5,
        )

        should_apply = refactoring_engine.should_auto_apply(low_confidence_opportunity)
        assert should_apply is False

        # High confidence opportunity should be auto-applied
        high_confidence_opportunity = RefactoringOpportunity(
            refactoring_type=RefactoringType.EXTRACT_METHOD,
            location="function",
            description="High confidence refactoring",
            confidence=0.95,  # Above threshold
            estimated_benefit=0.8,
        )

        should_apply = refactoring_engine.should_auto_apply(high_confidence_opportunity)
        assert should_apply is True
