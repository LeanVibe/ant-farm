"""
Gemini Client Implementation for AI Integration
"""
import asyncio
import json
import logging
import aiohttp
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
from dataclasses import dataclass

from . import ModelResponse, ModelType, AIModelInterface

logger = logging.getLogger(__name__)


@dataclass
class GeminiConfig:
    """Configuration for Gemini client"""
    api_key: str
    model_name: str = "gemini-1.5-flash"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 30
    base_url: str = "https://generativelanguage.googleapis.com/v1"


class GeminiClient(AIModelInterface):
    """Gemini client implementation"""
    
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Model pricing (per token) - current Google AI pricing
        self._pricing = {
            "gemini-1.5-flash": {"input": 0.000000075, "output": 0.0000003},
            "gemini-1.5-pro": {"input": 0.00000125, "output": 0.00000375},
            "gemini-pro": {"input": 0.00000125, "output": 0.00000375},
            "gemini-pro-vision": {"input": 0.00000125, "output": 0.00000375}
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def generate_response(
        self, 
        prompt: Optional[str] = None,
        task_context: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
        functions: Optional[List[Dict[str, Any]]] = None,
        safety_settings: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None,
        multimodal_content: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response using Gemini"""
        
        # Handle different content types
        if multimodal_content:
            contents = [{"parts": multimodal_content}]
        else:
            # Format prompt based on task context
            formatted_prompt = self._format_prompt(prompt or "", task_context)
            contents = [{"parts": [{"text": formatted_prompt}]}]
        
        # Prepare API request
        request_data = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature or self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            }
        }
        
        # Add system instruction if provided
        if system_instruction:
            request_data["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        
        # Add function calling if provided
        if functions:
            request_data["tools"] = [{"functionDeclarations": functions}]
        
        # Add safety settings if provided
        if safety_settings:
            request_data["safetySettings"] = safety_settings
        
        # Retry logic
        last_exception = None
        for attempt in range(max_retries):
            try:
                response_data = await self._make_api_request(request_data)
                return self._parse_response(response_data)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 1.0  # Exponential backoff
                    await asyncio.sleep(wait_time)
                    logger.warning(f"Gemini request failed, retrying in {wait_time}s: {e}")
                else:
                    logger.error(f"Gemini request failed after {max_retries} attempts: {e}")
        
        if last_exception:
            raise last_exception
        else:
            raise Exception("Gemini request failed after maximum retries")
    
    async def stream_response(
        self, 
        prompt: str, 
        task_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response from Gemini"""
        
        formatted_prompt = self._format_prompt(prompt, task_context)
        
        request_data = {
            "contents": [{"parts": [{"text": formatted_prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            }
        }
        
        stream_chunks = await self._stream_api_request(request_data)
        for chunk in stream_chunks:
            if chunk.get("candidates") and len(chunk["candidates"]) > 0:
                candidate = chunk["candidates"][0]
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    if "text" in part:
                        yield part["text"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities"""
        model_capabilities = {
            "gemini-1.5-flash": {
                "supports_vision": False,  # Text-only version
                "max_context": 1000000,  # 1M token context
                "supports_function_calling": True,
                "capabilities": ["reasoning", "code_generation", "analysis", "writing"],
                "cost_per_1k_tokens": {"input": 0.075, "output": 0.3}
            },
            "gemini-1.5-pro": {
                "supports_vision": True,
                "max_context": 2000000,  # 2M token context
                "supports_function_calling": True,
                "capabilities": ["vision", "reasoning", "code_generation", "analysis", "writing"],
                "cost_per_1k_tokens": {"input": 1.25, "output": 3.75}
            },
            "gemini-pro": {
                "supports_vision": False,
                "max_context": 32000,
                "supports_function_calling": True,
                "capabilities": ["reasoning", "code_generation", "analysis", "writing"],
                "cost_per_1k_tokens": {"input": 1.25, "output": 3.75}
            }
        }
        
        return model_capabilities.get(self.config.model_name, model_capabilities["gemini-1.5-flash"])
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token usage"""
        pricing = self._pricing.get(self.config.model_name, self._pricing["gemini-1.5-flash"])
        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
        return input_cost + output_cost
    
    def _format_prompt(self, prompt: str, task_context: Optional[Dict[str, Any]] = None) -> str:
        """Format prompt based on task context"""
        if not task_context:
            return prompt
        
        task_type = task_context.get("type", "")
        language = task_context.get("language", "")
        focus = task_context.get("focus", "")
        
        formatted_prompt = prompt
        
        if task_type == "code_generation" and language:
            formatted_prompt = f"As an expert {language} developer, {prompt}"
        elif task_type == "code_analysis" and focus:
            formatted_prompt = f"Analyze the following code with focus on {focus}: {prompt}"
        elif task_type == "debugging":
            formatted_prompt = f"Debug the following code and provide a fix: {prompt}"
        
        return formatted_prompt
    
    def _calculate_response_quality(self, response: str) -> float:
        """Calculate quality score for response"""
        if len(response) < 10:
            return 0.1
        
        score = 0.5  # Base score
        
        # Quality indicators
        if any(keyword in response.lower() for keyword in ["def ", "function", "class"]):
            score += 0.2  # Contains code structures
        
        if '"""' in response or "'''" in response:
            score += 0.15  # Has docstrings
        
        if "return" in response:
            score += 0.1  # Has return statements
        
        # Check for balanced brackets and parentheses
        if (response.count('(') == response.count(')') and 
            response.count('[') == response.count(']') and 
            response.count('{') == response.count('}')):
            score += 0.1
        
        # Penalty for too short responses
        if len(response) < 50:
            score -= 0.2
        
        return min(max(score, 0.0), 1.0)
    
    async def _make_api_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request to Google AI"""
        if not self.session:
            raise RuntimeError("Client session not initialized. Use async context manager.")
        
        url = f"{self.config.base_url}/models/{self.config.model_name}:generateContent"
        params = {"key": self.config.api_key}
        
        async with self.session.post(url, json=data, params=params) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Google AI API error {response.status}: {error_text}")
            
            return await response.json()
    
    async def _stream_api_request(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Make streaming API request to Google AI"""
        if not self.session:
            raise RuntimeError("Client session not initialized. Use async context manager.")
        
        url = f"{self.config.base_url}/models/{self.config.model_name}:streamGenerateContent"
        params = {"key": self.config.api_key}
        
        chunks = []
        async with self.session.post(url, json=data, params=params) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Google AI API error {response.status}: {error_text}")
            
            async for line in response.content:
                if line.strip():
                    try:
                        line_str = line.decode('utf-8').strip()
                        # Parse JSON lines
                        if line_str.startswith('{'):
                            chunk_data = json.loads(line_str)
                            chunks.append(chunk_data)
                    except json.JSONDecodeError:
                        continue
        
        return chunks
    
    def _parse_response(self, response_data: Dict[str, Any]) -> ModelResponse:
        """Parse Google AI API response into ModelResponse"""
        candidates = response_data.get("candidates", [])
        if not candidates:
            raise Exception("No candidates in Gemini response")
        
        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        
        # Extract text content
        text_content = ""
        function_call = None
        
        for part in parts:
            if "text" in part:
                text_content += part["text"]
            elif "functionCall" in part:
                function_call = part["functionCall"]
        
        # Extract usage metadata
        usage = response_data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = usage.get("totalTokenCount", input_tokens + output_tokens)
        
        cost = self.calculate_cost(input_tokens, output_tokens)
        quality_score = self._calculate_response_quality(text_content)
        
        metadata = {}
        
        # Handle function calls
        if function_call:
            metadata["function_call"] = function_call
        
        return ModelResponse(
            content=text_content,
            model=self.config.model_name,
            tokens_used=total_tokens,
            cost=cost,
            quality_score=quality_score,
            confidence=quality_score,  # Use quality as confidence for now
            metadata=metadata
        )