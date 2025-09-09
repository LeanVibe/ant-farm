"""
GPT-4 Client Implementation for AI Integration
"""
import asyncio
import json
import logging
import aiohttp
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass

from . import ModelResponse, ModelType, AIModelInterface

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

logger = logging.getLogger(__name__)


@dataclass
class GPT4Config:
    """Configuration for GPT-4 client"""
    api_key: str
    model_name: str = "gpt-4o-mini"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 30
    base_url: str = "https://api.openai.com/v1"
    max_context_tokens: int = 128000  # GPT-4o-mini context window


class GPT4Client(AIModelInterface):
    """GPT-4 client implementation"""
    
    def __init__(self, config: GPT4Config):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Initialize tiktoken encoding if available
        if TIKTOKEN_AVAILABLE and tiktoken:
            try:
                self._encoding = tiktoken.encoding_for_model("gpt-4o-mini")
            except Exception:
                self._encoding = None
        else:
            self._encoding = None
        
        # Model pricing (per token) - current OpenAI pricing
        self._pricing = {
            "gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
            "gpt-4": {"input": 0.00003, "output": 0.00006},
            "gpt-4-vision-preview": {"input": 0.00001, "output": 0.00003}
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def generate_response(
        self, 
        prompt: str, 
        task_context: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
        functions: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response using GPT-4"""
        
        # Format prompt based on task context
        formatted_prompt = self._format_prompt(prompt, task_context)
        
        # Manage context window
        managed_prompt = self._manage_context_window(formatted_prompt)
        
        # Prepare API request
        messages = [{"role": "user", "content": managed_prompt}]
        
        request_data = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": temperature or self.config.temperature
        }
        
        if functions:
            request_data["functions"] = functions
            request_data["function_call"] = "auto"
        
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
                    logger.warning(f"GPT-4 request failed, retrying in {wait_time}s: {e}")
                else:
                    logger.error(f"GPT-4 request failed after {max_retries} attempts: {e}")
        
        if last_exception:
            raise last_exception
        else:
            raise Exception("GPT-4 request failed after maximum retries")
    
    async def stream_response(
        self, 
        prompt: str, 
        task_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response from GPT-4"""
        
        formatted_prompt = self._format_prompt(prompt, task_context)
        managed_prompt = self._manage_context_window(formatted_prompt)
        
        request_data = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": managed_prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": True
        }
        
        stream_chunks = await self._stream_api_request(request_data)
        for chunk in stream_chunks:
            if chunk.get("choices") and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities"""
        model_capabilities = {
            "gpt-4o-mini": {
                "supports_vision": False,
                "max_context": 128000,
                "supports_function_calling": True,
                "capabilities": ["reasoning", "code_generation", "analysis", "writing"],
                "cost_per_1k_tokens": {"input": 0.15, "output": 0.6}
            },
            "gpt-4": {
                "supports_vision": False,
                "max_context": 8192,
                "supports_function_calling": True,
                "capabilities": ["reasoning", "code_generation", "analysis", "writing"],
                "cost_per_1k_tokens": {"input": 30.0, "output": 60.0}
            },
            "gpt-4-vision-preview": {
                "supports_vision": True,
                "max_context": 128000,
                "supports_function_calling": True,
                "capabilities": ["vision", "reasoning", "code_generation", "analysis", "writing"],
                "cost_per_1k_tokens": {"input": 10.0, "output": 30.0}
            }
        }
        
        return model_capabilities.get(self.config.model_name, model_capabilities["gpt-4o-mini"])
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token usage"""
        pricing = self._pricing.get(self.config.model_name, self._pricing["gpt-4o-mini"])
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
    
    def _manage_context_window(self, prompt: str) -> str:
        """Manage context window to fit within model limits"""
        token_count = self._count_tokens(prompt)
        
        if token_count <= self.config.max_context_tokens - self.config.max_tokens:
            return prompt
        
        # Truncate to fit within context window, leaving room for response
        max_prompt_tokens = self.config.max_context_tokens - self.config.max_tokens - 100  # Buffer
        
        # Simple truncation from the middle to preserve beginning and end
        if self._encoding:
            try:
                tokens = self._encoding.encode(prompt)
                if len(tokens) > max_prompt_tokens:
                    start_tokens = tokens[:max_prompt_tokens // 2]
                    end_tokens = tokens[-(max_prompt_tokens - max_prompt_tokens // 2):]
                    truncated_tokens = start_tokens + end_tokens
                    truncated_prompt = self._encoding.decode(truncated_tokens)
                    return truncated_prompt
            except Exception:
                # Fallback to character-based truncation
                half_chars = (max_prompt_tokens * 4) // 2  # Rough character estimation
                return prompt[:half_chars] + "..." + prompt[-half_chars:]
        else:
            # Fallback to character-based truncation when tiktoken is not available
            half_chars = (max_prompt_tokens * 4) // 2  # Rough character estimation
            return prompt[:half_chars] + "..." + prompt[-half_chars:]
        
        return prompt
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self._encoding:
            try:
                return len(self._encoding.encode(text))
            except Exception:
                # Fallback to rough estimation
                return len(text) // 4
        else:
            # Fallback to rough estimation when tiktoken is not available
            return len(text) // 4
    
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
        """Make API request to OpenAI"""
        if not self.session:
            raise RuntimeError("Client session not initialized. Use async context manager.")
        
        url = f"{self.config.base_url}/chat/completions"
        
        async with self.session.post(url, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"OpenAI API error {response.status}: {error_text}")
            
            return await response.json()
    
    async def _stream_api_request(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Make streaming API request to OpenAI"""
        if not self.session:
            raise RuntimeError("Client session not initialized. Use async context manager.")
        
        url = f"{self.config.base_url}/chat/completions"
        
        chunks = []
        async with self.session.post(url, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"OpenAI API error {response.status}: {error_text}")
            
            async for line in response.content:
                if line.strip():
                    try:
                        # Parse SSE format
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str != '[DONE]':
                                chunk_data = json.loads(data_str)
                                chunks.append(chunk_data)
                    except json.JSONDecodeError:
                        continue
        
        return chunks
    
    def _parse_response(self, response_data: Dict[str, Any]) -> ModelResponse:
        """Parse OpenAI API response into ModelResponse"""
        choice = response_data["choices"][0]
        message = choice["message"]
        content = message.get("content", "")
        
        usage = response_data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
        
        cost = self.calculate_cost(input_tokens, output_tokens)
        quality_score = self._calculate_response_quality(content)
        
        metadata = {}
        
        # Handle function calls
        if message.get("function_call"):
            metadata["function_call"] = message["function_call"]
        
        return ModelResponse(
            content=content,
            model=response_data.get("model", self.config.model_name),
            tokens_used=total_tokens,
            cost=cost,
            quality_score=quality_score,
            confidence=quality_score,  # Use quality as confidence for now
            metadata=metadata
        )