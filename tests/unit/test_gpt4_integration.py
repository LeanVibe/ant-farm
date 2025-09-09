import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from src.core.ai.gpt4_client import GPT4Client, GPT4Config
from src.core.ai import ModelResponse, ModelType


class TestGPT4Integration:
    """Test suite for GPT-4 integration using TDD approach"""
    
    @pytest.fixture
    def gpt4_config(self):
        """Create GPT-4 configuration for testing"""
        return GPT4Config(
            api_key="test-api-key",
            model_name="gpt-4o-mini",
            max_tokens=4096,
            temperature=0.7,
            timeout=30
        )
    
    @pytest.fixture
    def gpt4_client(self, gpt4_config):
        """Create GPT-4 client instance for testing"""
        return GPT4Client(config=gpt4_config)
    
    @pytest.mark.asyncio
    async def test_gpt4_client_initialization(self, gpt4_client):
        """Test that GPT-4 client initializes properly"""
        assert gpt4_client.config.model_name == "gpt-4o-mini"
        assert gpt4_client.config.api_key == "test-api-key"
        assert gpt4_client.config.max_tokens == 4096
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self, gpt4_client):
        """Test successful response generation from GPT-4"""
        # Arrange
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 35,
                "total_tokens": 55
            },
            "model": "gpt-4o-mini"
        }
        
        with patch.object(gpt4_client, '_make_api_request', return_value=mock_response):
            # Act
            response = await gpt4_client.generate_response("Write a factorial function")
            
            # Assert
            assert isinstance(response, ModelResponse)
            assert "def factorial" in response.content
            assert response.model == "gpt-4o-mini"
            assert response.tokens_used == 55
            assert response.cost > 0
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, gpt4_client):
        """Test handling of API errors from OpenAI"""
        # Arrange
        with patch.object(gpt4_client, '_make_api_request', side_effect=Exception("API rate limit exceeded")):
            # Act & Assert
            with pytest.raises(Exception) as exc_info:
                await gpt4_client.generate_response("Test prompt")
            assert "API rate limit exceeded" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_token_usage_calculation(self, gpt4_client):
        """Test accurate token usage tracking"""
        # Arrange
        mock_response = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 25,
                "total_tokens": 75
            },
            "model": "gpt-4o-mini"
        }
        
        with patch.object(gpt4_client, '_make_api_request', return_value=mock_response):
            # Act
            response = await gpt4_client.generate_response("Test prompt")
            
            # Assert
            assert response.tokens_used == 75
            expected_cost = gpt4_client.calculate_cost(50, 25)
            assert abs(response.cost - expected_cost) < 0.0001
    
    @pytest.mark.asyncio
    async def test_cost_calculation_accuracy(self, gpt4_client):
        """Test that cost calculation matches current pricing"""
        # Act
        cost = gpt4_client.calculate_cost(input_tokens=1000, output_tokens=500)
        
        # Assert - based on GPT-4o-mini pricing
        expected_cost = (1000 * 0.00000015) + (500 * 0.0000006)
        assert abs(cost - expected_cost) < 0.0001
        assert cost > 0
    
    @pytest.mark.asyncio
    async def test_model_capabilities_metadata(self, gpt4_client):
        """Test that model provides accurate capability information"""
        # Act
        capabilities = gpt4_client.get_model_info()
        
        # Assert
        assert capabilities["supports_vision"] == False  # GPT-4o-mini doesn't support vision
        assert capabilities["max_context"] == 128000
        assert capabilities["supports_function_calling"] == True
        assert "reasoning" in capabilities["capabilities"]
        assert "code_generation" in capabilities["capabilities"]
    
    @pytest.mark.asyncio
    async def test_prompt_formatting_for_different_tasks(self, gpt4_client):
        """Test that prompts are formatted appropriately for different task types"""
        # Arrange
        code_task = {"type": "code_generation", "language": "python"}
        analysis_task = {"type": "code_analysis", "focus": "performance"}
        
        # Act
        code_prompt = gpt4_client._format_prompt("Write a function", code_task)
        analysis_prompt = gpt4_client._format_prompt("Analyze this code", analysis_task)
        
        # Assert
        assert "python" in code_prompt.lower()
        assert "performance" in analysis_prompt.lower()
        assert code_prompt != analysis_prompt
    
    @pytest.mark.asyncio
    async def test_response_streaming_support(self, gpt4_client):
        """Test streaming response functionality"""
        # Arrange
        mock_stream_chunks = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [{"delta": {"content": "!"}}]},
        ]
        
        with patch.object(gpt4_client, '_stream_api_request', return_value=mock_stream_chunks):
            # Act
            full_response = ""
            async for chunk in gpt4_client.stream_response("Test prompt"):
                full_response += chunk
            
            # Assert
            assert full_response == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_function_calling_capability(self, gpt4_client):
        """Test GPT-4 function calling feature"""
        # Arrange
        functions = [
            {
                "name": "calculate_sum",
                "description": "Calculate sum of two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["a", "b"]
                }
            }
        ]
        
        mock_response = {
            "choices": [{
                "message": {
                    "content": None,
                    "function_call": {
                        "name": "calculate_sum",
                        "arguments": '{"a": 5, "b": 3}'
                    }
                }
            }],
            "usage": {"total_tokens": 30},
            "model": "gpt-4o-mini"
        }
        
        with patch.object(gpt4_client, '_make_api_request', return_value=mock_response):
            # Act
            response = await gpt4_client.generate_response(
                "Calculate 5 + 3",
                functions=functions
            )
            
            # Assert
            assert response.metadata.get("function_call") is not None
            assert response.metadata["function_call"]["name"] == "calculate_sum"
    
    @pytest.mark.asyncio
    async def test_context_window_management(self, gpt4_client):
        """Test handling of context window limits"""
        # Arrange
        very_long_prompt = "x" * 200000  # Exceed context window
        
        with patch.object(gpt4_client, '_count_tokens', return_value=150000):
            # Act
            truncated_prompt = gpt4_client._manage_context_window(very_long_prompt)
            
            # Assert
            assert len(truncated_prompt) < len(very_long_prompt)
            token_count = gpt4_client._count_tokens(truncated_prompt)
            assert token_count <= gpt4_client.config.max_context_tokens
    
    @pytest.mark.asyncio
    async def test_response_quality_scoring(self, gpt4_client):
        """Test internal quality scoring of responses"""
        # Arrange
        high_quality_response = """
        def factorial(n: int) -> int:
            '''Calculate factorial of n using recursion'''
            if n <= 1:
                return 1
            return n * factorial(n-1)
        """
        
        low_quality_response = "factorial function"
        
        # Act
        high_score = gpt4_client._calculate_response_quality(high_quality_response)
        low_score = gpt4_client._calculate_response_quality(low_quality_response)
        
        # Assert
        assert high_score > low_score
        assert 0 <= high_score <= 1
        assert 0 <= low_score <= 1
    
    @pytest.mark.asyncio
    async def test_retry_mechanism_on_failure(self, gpt4_client):
        """Test retry mechanism for failed requests"""
        # Arrange
        call_count = 0
        
        def mock_request(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary API error")
            return {
                "choices": [{"message": {"content": "Success after retry"}}],
                "usage": {"total_tokens": 20},
                "model": "gpt-4o-mini"
            }
        
        with patch.object(gpt4_client, '_make_api_request', side_effect=mock_request):
            # Act
            response = await gpt4_client.generate_response("Test prompt", max_retries=3)
            
            # Assert
            assert response.content == "Success after retry"
            assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_temperature_affects_response_creativity(self, gpt4_client):
        """Test that temperature setting affects response variability"""
        # Arrange
        mock_creative_response = {
            "choices": [{"message": {"content": "Creative and varied response"}}],
            "usage": {"total_tokens": 25},
            "model": "gpt-4o-mini"
        }
        
        mock_deterministic_response = {
            "choices": [{"message": {"content": "Deterministic response"}}],
            "usage": {"total_tokens": 20},
            "model": "gpt-4o-mini"
        }
        
        with patch.object(gpt4_client, '_make_api_request') as mock_request:
            # Test high temperature (creative)
            mock_request.return_value = mock_creative_response
            creative_response = await gpt4_client.generate_response("Write a story", temperature=1.0)
            
            # Test low temperature (deterministic)  
            mock_request.return_value = mock_deterministic_response
            deterministic_response = await gpt4_client.generate_response("Write a story", temperature=0.1)
            
            # Assert
            assert creative_response.content != deterministic_response.content
            # Verify temperature was passed to API
            mock_request.assert_called()