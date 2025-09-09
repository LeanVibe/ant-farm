import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.core.ai.model_orchestrator import ModelOrchestrator, ModelResponse, ModelType
from src.core.ai.cost_optimizer import CostOptimizer
from src.core.ai.response_validator import ResponseValidator


@dataclass
class MockModelResponse:
    """Mock response for testing AI models"""
    content: str
    model: str
    tokens_used: int
    cost: float
    quality_score: float = 0.8
    confidence: float = 0.9


class TestModelOrchestrator:
    """Test suite for AI Model Orchestrator using TDD approach"""

    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator instance for testing"""
        cost_optimizer = Mock(spec=CostOptimizer)
        response_validator = Mock(spec=ResponseValidator)
        
        orchestrator = ModelOrchestrator(
            cost_optimizer=cost_optimizer,
            response_validator=response_validator
        )
        return orchestrator

    @pytest.mark.asyncio
    async def test_select_best_model_for_task(self, orchestrator):
        """Test that orchestrator selects optimal model based on task requirements"""
        # Arrange
        task = {
            "type": "code_generation",
            "complexity": "high",
            "context_length": 5000,
            "requirements": ["python", "async"]
        }
        
        # Act
        selected_model = await orchestrator.select_best_model(task)
        
        # Assert
        assert selected_model in [ModelType.GPT4, ModelType.CLAUDE_SONNET]
        assert selected_model is not None

    @pytest.mark.asyncio
    async def test_generate_with_fallback_models(self, orchestrator):
        """Test fallback mechanism when primary model fails"""
        # Arrange
        prompt = "Generate a Python function"
        
        with patch.object(orchestrator, '_call_gpt4', side_effect=Exception("API Error")):
            with patch.object(orchestrator, '_call_claude', return_value=MockModelResponse(
                content="def test(): pass", 
                model="claude", 
                tokens_used=50, 
                cost=0.001
            )):
                # Act
                response = await orchestrator.generate_response(prompt, preferred_model=ModelType.GPT4)
                
                # Assert
                assert response is not None
                assert response.content == "def test(): pass"
                assert response.model == "claude"

    @pytest.mark.asyncio
    async def test_hybrid_reasoning_combines_models(self, orchestrator):
        """Test that hybrid reasoning combines responses from multiple models"""
        # Arrange
        prompt = "Analyze this code for bugs"
        
        gpt4_response = MockModelResponse("Found syntax error", "gpt4", 100, 0.02, 0.9)
        claude_response = MockModelResponse("Logic error in loop", "claude", 80, 0.015, 0.85)
        
        with patch.object(orchestrator, '_call_gpt4', return_value=gpt4_response):
            with patch.object(orchestrator, '_call_claude', return_value=claude_response):
                # Act
                response = await orchestrator.hybrid_reasoning(prompt)
                
                # Assert
                assert response is not None
                assert "syntax error" in response.content.lower()
                assert "logic error" in response.content.lower()

    @pytest.mark.asyncio
    async def test_context_aware_model_selection(self, orchestrator):
        """Test that model selection considers context and capabilities"""
        # Arrange
        multimodal_task = {
            "type": "image_analysis",
            "requires_vision": True,
            "context_length": 1000
        }
        
        text_only_task = {
            "type": "text_generation", 
            "requires_vision": False,
            "context_length": 2000
        }
        
        # Act
        multimodal_model = await orchestrator.select_best_model(multimodal_task)
        text_model = await orchestrator.select_best_model(text_only_task)
        
        # Assert
        assert multimodal_model == ModelType.GPT4_VISION  # GPT-4 has vision capabilities
        assert text_model in [ModelType.GPT4, ModelType.CLAUDE_SONNET]

    @pytest.mark.asyncio 
    async def test_cost_optimization_selects_efficient_model(self, orchestrator):
        """Test that cost optimizer influences model selection for simple tasks"""
        # Arrange
        simple_task = {
            "type": "text_summary",
            "complexity": "low", 
            "max_tokens": 500,
            "budget_constraint": True
        }
        
        orchestrator.cost_optimizer.get_most_cost_effective.return_value = ModelType.GPT4_MINI
        
        # Act
        selected_model = await orchestrator.select_best_model(simple_task)
        
        # Assert
        assert selected_model == ModelType.GPT4_MINI
        orchestrator.cost_optimizer.get_most_cost_effective.assert_called_once()

    @pytest.mark.asyncio
    async def test_response_quality_validation(self, orchestrator):
        """Test that responses are validated for quality"""
        # Arrange
        prompt = "Write a test function"
        low_quality_response = MockModelResponse(
            content="bad code", 
            model="test", 
            tokens_used=10, 
            cost=0.001,
            quality_score=0.3  # Below threshold
        )
        
        high_quality_response = MockModelResponse(
            content="def test_function(): assert True", 
            model="test", 
            tokens_used=20, 
            cost=0.002,
            quality_score=0.9  # Above threshold
        )
        
        orchestrator.response_validator.validate_response.side_effect = [False, True]
        
        with patch.object(orchestrator, '_call_gpt4', side_effect=[low_quality_response, high_quality_response]):
            # Act
            response = await orchestrator.generate_response(prompt, quality_threshold=0.8)
            
            # Assert
            assert response.content == "def test_function(): assert True"
            assert orchestrator.response_validator.validate_response.call_count == 2

    @pytest.mark.asyncio
    async def test_token_usage_tracking(self, orchestrator):
        """Test that token usage is properly tracked across models"""
        # Arrange
        prompt = "Generate code"
        
        response = MockModelResponse("code here", "gpt4", 150, 0.03)
        
        with patch.object(orchestrator, '_call_gpt4', return_value=response):
            # Act
            result = await orchestrator.generate_response(prompt)
            usage_stats = await orchestrator.get_usage_stats()
            
            # Assert
            assert usage_stats["total_tokens"] >= 150
            assert usage_stats["total_cost"] >= 0.03

    @pytest.mark.asyncio
    async def test_concurrent_model_calls_for_comparison(self, orchestrator):
        """Test concurrent calls to multiple models for comparison"""
        # Arrange
        prompt = "Solve this problem"
        
        responses = [
            MockModelResponse("solution A", "gpt4", 100, 0.02, 0.85),
            MockModelResponse("solution B", "claude", 90, 0.015, 0.80),
            MockModelResponse("solution C", "gemini", 110, 0.018, 0.90)
        ]
        
        with patch.object(orchestrator, '_call_all_models', return_value=responses):
            # Act
            best_response = await orchestrator.get_best_response(prompt)
            
            # Assert
            assert best_response.content == "solution C"  # Highest quality score
            assert best_response.model == "gemini"


class TestCostOptimizer:
    """Test suite for AI Cost Optimizer"""

    @pytest.fixture
    def cost_optimizer(self):
        return CostOptimizer()

    def test_calculate_model_cost_per_token(self, cost_optimizer):
        """Test cost calculation for different models"""
        # Act & Assert
        gpt4_cost = cost_optimizer.calculate_cost(ModelType.GPT4, input_tokens=1000, output_tokens=500)
        gpt4_mini_cost = cost_optimizer.calculate_cost(ModelType.GPT4_MINI, input_tokens=1000, output_tokens=500)
        
        assert gpt4_cost > gpt4_mini_cost  # GPT-4 should be more expensive
        assert gpt4_cost > 0
        assert gpt4_mini_cost > 0

    def test_recommend_cost_effective_model(self, cost_optimizer):
        """Test recommendation of most cost-effective model for task"""
        # Arrange
        simple_task = {"complexity": "low", "max_tokens": 500}
        complex_task = {"complexity": "high", "max_tokens": 4000}
        
        # Act
        simple_model = cost_optimizer.get_most_cost_effective(simple_task)
        complex_model = cost_optimizer.get_most_cost_effective(complex_task)
        
        # Assert
        assert simple_model == ModelType.GPT4_MINI  # Cheaper for simple tasks
        assert complex_model == ModelType.GPT4  # Better quality for complex tasks

    def test_budget_constraint_enforcement(self, cost_optimizer):
        """Test that budget constraints are enforced"""
        # Arrange
        task = {"complexity": "medium", "budget_limit": 0.01}
        
        # Act
        recommended_models = cost_optimizer.get_models_within_budget(task)
        
        # Assert
        assert ModelType.GPT4_MINI in recommended_models
        assert len(recommended_models) > 0
        
        for model in recommended_models:
            estimated_cost = cost_optimizer.estimate_task_cost(task, model)
            assert estimated_cost <= 0.01


class TestResponseValidator:
    """Test suite for AI Response Validator"""

    @pytest.fixture
    def response_validator(self):
        return ResponseValidator()

    def test_detect_hallucination_in_code(self, response_validator):
        """Test detection of hallucinated code patterns"""
        # Arrange
        hallucinated_response = "import nonexistent_library\nfrom fake_module import invalid_function"
        valid_response = "import os\nfrom pathlib import Path"
        
        # Act & Assert
        assert response_validator.detect_hallucination(hallucinated_response) == True
        assert response_validator.detect_hallucination(valid_response) == False

    def test_validate_code_syntax(self, response_validator):
        """Test syntax validation for code responses"""
        # Arrange
        invalid_syntax = "def test(: print('hello')"  # Missing parameter
        valid_syntax = "def test(): print('hello')"
        
        # Act & Assert
        assert response_validator.validate_syntax(invalid_syntax, language="python") == False
        assert response_validator.validate_syntax(valid_syntax, language="python") == True

    def test_response_completeness_check(self, response_validator):
        """Test that responses are complete and address the prompt"""
        # Arrange
        prompt = "Write a function that calculates factorial"
        incomplete_response = "def factorial("  # Incomplete
        complete_response = "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
        
        # Act & Assert
        assert response_validator.check_completeness(prompt, incomplete_response) == False
        assert response_validator.check_completeness(prompt, complete_response) == True

    def test_confidence_scoring(self, response_validator):
        """Test confidence score calculation based on response quality"""
        # Arrange
        high_quality_response = "def factorial(n: int) -> int:\n    '''Calculate factorial of n'''\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
        low_quality_response = "factorial function"
        
        # Act
        high_score = response_validator.calculate_confidence(high_quality_response)
        low_score = response_validator.calculate_confidence(low_quality_response)
        
        # Assert
        assert high_score > low_score
        assert 0 <= high_score <= 1
        assert 0 <= low_score <= 1