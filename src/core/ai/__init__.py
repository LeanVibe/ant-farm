"""
AI Model Integration - Core Types and Interfaces
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod


class ModelType(Enum):
    """Supported AI model types"""
    GPT4 = "gpt-4"
    GPT4_MINI = "gpt-4o-mini"
    GPT4_VISION = "gpt-4-vision"
    CLAUDE_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_HAIKU = "claude-3-5-haiku-20241022"
    GEMINI_PRO = "gemini-pro"
    GEMINI_FLASH = "gemini-1.5-flash"


@dataclass
class ModelResponse:
    """Response from an AI model"""
    content: str
    model: str
    tokens_used: int
    cost: float
    quality_score: float = 0.8
    confidence: float = 0.9
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TaskRequirements:
    """Requirements for a task that influence model selection"""
    task_type: str
    complexity: str
    context_length: int
    requirements: List[str]
    budget_constraint: bool = False
    requires_vision: bool = False
    max_tokens: int = 4096


class AIModelInterface(ABC):
    """Interface for AI model implementations"""
    
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using the model"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities"""
        pass
    
    @abstractmethod
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token usage"""
        pass