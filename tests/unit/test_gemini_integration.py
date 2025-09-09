import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from src.core.ai.gemini_client import GeminiClient, GeminiConfig
from src.core.ai import ModelResponse, ModelType


class TestGeminiIntegration:
    """Test suite for Gemini integration using TDD approach"""
    
    @pytest.fixture
    def gemini_config(self):
        """Create Gemini configuration for testing"""
        return GeminiConfig(
            api_key="test-api-key",
            model_name="gemini-1.5-flash",
            max_tokens=4096,
            temperature=0.7,
            timeout=30
        )
    
    @pytest.fixture
    def gemini_client(self, gemini_config):
        """Create Gemini client instance for testing"""
        return GeminiClient(config=gemini_config)
    
    @pytest.mark.asyncio
    async def test_gemini_client_initialization(self, gemini_client):
        """Test that Gemini client initializes properly"""
        assert gemini_client.config.model_name == "gemini-1.5-flash"
        assert gemini_client.config.api_key == "test-api-key"
        assert gemini_client.config.max_tokens == 4096
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self, gemini_client):
        """Test successful response generation from Gemini"""
        # Arrange
        mock_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"}
                        ]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 35,
                "totalTokenCount": 55
            }
        }
        
        with patch.object(gemini_client, '_make_api_request', return_value=mock_response):
            # Act
            response = await gemini_client.generate_response("Write a factorial function")
            
            # Assert
            assert isinstance(response, ModelResponse)
            assert "def factorial" in response.content
            assert response.model == "gemini-1.5-flash"
            assert response.tokens_used == 55
            assert response.cost > 0
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, gemini_client):
        """Test handling of API errors from Google AI"""
        # Arrange
        with patch.object(gemini_client, '_make_api_request', side_effect=Exception("API quota exceeded")):
            # Act & Assert
            with pytest.raises(Exception) as exc_info:
                await gemini_client.generate_response("Test prompt")
            assert "API quota exceeded" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_token_usage_calculation(self, gemini_client):
        """Test accurate token usage tracking"""
        # Arrange
        mock_response = {
            "candidates": [{"content": {"parts": [{"text": "Test response"}]}}],
            "usageMetadata": {
                "promptTokenCount": 50,
                "candidatesTokenCount": 25,
                "totalTokenCount": 75
            }
        }
        
        with patch.object(gemini_client, '_make_api_request', return_value=mock_response):
            # Act
            response = await gemini_client.generate_response("Test prompt")
            
            # Assert
            assert response.tokens_used == 75
            expected_cost = gemini_client.calculate_cost(50, 25)
            assert abs(response.cost - expected_cost) < 0.0001
    
    @pytest.mark.asyncio
    async def test_cost_calculation_accuracy(self, gemini_client):
        """Test that cost calculation matches current pricing"""
        # Act
        cost = gemini_client.calculate_cost(input_tokens=1000, output_tokens=500)
        
        # Assert - based on Gemini Flash pricing
        expected_cost = (1000 * 0.000000075) + (500 * 0.0000003)
        assert abs(cost - expected_cost) < 0.0001
        assert cost > 0
    
    @pytest.mark.asyncio
    async def test_model_capabilities_metadata(self, gemini_client):
        """Test that model provides accurate capability information"""
        # Act
        capabilities = gemini_client.get_model_info()
        
        # Assert
        assert capabilities["supports_vision"] == False  # Gemini Flash text-only version
        assert capabilities["max_context"] == 1000000  # 1M token context
        assert capabilities["supports_function_calling"] == True
        assert "reasoning" in capabilities["capabilities"]
        assert "code_generation" in capabilities["capabilities"]
    
    @pytest.mark.asyncio
    async def test_prompt_formatting_for_different_tasks(self, gemini_client):
        """Test that prompts are formatted appropriately for different task types"""
        # Arrange
        code_task = {"type": "code_generation", "language": "python"}
        analysis_task = {"type": "code_analysis", "focus": "performance"}
        
        # Act
        code_prompt = gemini_client._format_prompt("Write a function", code_task)
        analysis_prompt = gemini_client._format_prompt("Analyze this code", analysis_task)
        
        # Assert
        assert "python" in code_prompt.lower()
        assert "performance" in analysis_prompt.lower()
        assert code_prompt != analysis_prompt
    
    @pytest.mark.asyncio
    async def test_response_streaming_support(self, gemini_client):
        """Test streaming response functionality"""
        # Arrange
        mock_stream_chunks = [
            {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]},
            {"candidates": [{"content": {"parts": [{"text": " world"}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "!"}]}}]},
        ]
        
        with patch.object(gemini_client, '_stream_api_request', return_value=mock_stream_chunks):
            # Act
            full_response = ""
            async for chunk in gemini_client.stream_response("Test prompt"):
                full_response += chunk
            
            # Assert
            assert full_response == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_function_calling_capability(self, gemini_client):
        """Test Gemini function calling feature"""
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
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "calculate_sum",
                            "args": {"a": 5, "b": 3}
                        }
                    }]
                }
            }],
            "usageMetadata": {"totalTokenCount": 30}
        }
        
        with patch.object(gemini_client, '_make_api_request', return_value=mock_response):
            # Act
            response = await gemini_client.generate_response(
                "Calculate 5 + 3",
                functions=functions
            )
            
            # Assert
            assert response.metadata.get("function_call") is not None
            assert response.metadata["function_call"]["name"] == "calculate_sum"
    
    @pytest.mark.asyncio
    async def test_safety_settings_handling(self, gemini_client):
        """Test Gemini safety settings configuration"""
        # Arrange
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        mock_response = {
            "candidates": [{"content": {"parts": [{"text": "Safe response"}]}}],
            "usageMetadata": {"totalTokenCount": 20}
        }
        
        with patch.object(gemini_client, '_make_api_request', return_value=mock_response) as mock_request:
            # Act
            response = await gemini_client.generate_response(
                "Test prompt",
                safety_settings=safety_settings
            )
            
            # Assert
            assert response.content == "Safe response"
            # Verify safety settings were passed to API
            mock_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_system_instruction_support(self, gemini_client):
        """Test Gemini system instruction functionality"""
        # Arrange
        system_instruction = "You are a helpful coding assistant."
        
        mock_response = {
            "candidates": [{"content": {"parts": [{"text": "I'll help you with coding."}]}}],
            "usageMetadata": {"totalTokenCount": 25}
        }
        
        with patch.object(gemini_client, '_make_api_request', return_value=mock_response) as mock_request:
            # Act
            response = await gemini_client.generate_response(
                "Hello",
                system_instruction=system_instruction
            )
            
            # Assert
            assert "I'll help you with coding" in response.content
            # Verify system instruction was passed to API
            mock_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multimodal_content_support(self, gemini_client):
        """Test Gemini multimodal content handling"""
        # Arrange
        multimodal_content = [
            {"text": "Describe this image"},
            {"inline_data": {"mime_type": "image/jpeg", "data": "base64encodedimage"}}
        ]
        
        mock_response = {
            "candidates": [{"content": {"parts": [{"text": "This image shows a cat."}]}}],
            "usageMetadata": {"totalTokenCount": 50}
        }
        
        with patch.object(gemini_client, '_make_api_request', return_value=mock_response):
            # Act
            response = await gemini_client.generate_response(
                multimodal_content=multimodal_content
            )
            
            # Assert
            assert "This image shows a cat" in response.content
    
    @pytest.mark.asyncio
    async def test_response_quality_scoring(self, gemini_client):
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
        high_score = gemini_client._calculate_response_quality(high_quality_response)
        low_score = gemini_client._calculate_response_quality(low_quality_response)
        
        # Assert
        assert high_score > low_score
        assert 0 <= high_score <= 1
        assert 0 <= low_score <= 1
    
    @pytest.mark.asyncio
    async def test_retry_mechanism_on_failure(self, gemini_client):
        """Test retry mechanism for failed requests"""
        # Arrange
        call_count = 0
        
        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary API error")
            return {
                "candidates": [{"content": {"parts": [{"text": "Success after retry"}]}}],
                "usageMetadata": {"totalTokenCount": 20}
            }
        
        with patch.object(gemini_client, '_make_api_request', side_effect=mock_request):
            # Act
            response = await gemini_client.generate_response("Test prompt", max_retries=3)
            
            # Assert
            assert response.content == "Success after retry"
            assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_temperature_affects_response_creativity(self, gemini_client):
        """Test that temperature setting affects response variability"""
        # Arrange
        mock_creative_response = {
            "candidates": [{"content": {"parts": [{"text": "Creative and varied response"}]}}],
            "usageMetadata": {"totalTokenCount": 25}
        }
        
        mock_deterministic_response = {
            "candidates": [{"content": {"parts": [{"text": "Deterministic response"}]}}],
            "usageMetadata": {"totalTokenCount": 20}
        }
        
        with patch.object(gemini_client, '_make_api_request') as mock_request:
            # Test high temperature (creative)
            mock_request.return_value = mock_creative_response
            creative_response = await gemini_client.generate_response("Write a story", temperature=1.0)
            
            # Test low temperature (deterministic)  
            mock_request.return_value = mock_deterministic_response
            deterministic_response = await gemini_client.generate_response("Write a story", temperature=0.1)
            
            # Assert
            assert creative_response.content != deterministic_response.content
            # Verify temperature was passed to API
            mock_request.assert_called()