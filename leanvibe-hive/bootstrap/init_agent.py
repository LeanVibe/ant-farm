    def _template_base_agent(self) -> str:
        """Template for base agent class."""
        return '''"""Base agent class for LeanVibe Agent Hive 2.0."""

import asyncio
import json
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import redis
import structlog
from src.core.config import settings


logger = structlog.get_logger()


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, name: str, agent_type: str, role: str):
        self.name = name
        self.agent_type = agent_type
        self.role = role
        self.status = "inactive"
        self.redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        
        # Detect available CLI tools
        self.available_cli_tools = self._detect_cli_tools()
        self.preferred_cli_tool = self._select_preferred_tool()
        
        # Optional API clients
        self.anthropic_client = None
        
        if settings.anthropic_api_key:
            try:
                from anthropic import Anthropic
                self.anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
            except ImportError:
                logger.warning("Anthropic package not available")
    
    def _detect_cli_tools(self) -> Dict[str, Dict[str, Any]]:
        """Detect available CLI agentic coding tools."""
        tools = {}
        
        # Check opencode
        if self._check_opencode():
            tools['opencode'] = {
                'name': 'OpenCode',
                'command': 'opencode',
                'execute_pattern': 'opencode "{prompt}"',
                'install_url': 'https://opencode.ai'
            }
        
        # Check Claude Code CLI
        if self._check_claude_cli():
            tools['claude'] = {
                'name': 'Claude Code CLI', 
                'command': 'claude',
                'execute_pattern': 'claude --no-interactive "{prompt}"',
                'install_url': 'https://claude.ai/cli'
            }
        
        # Check Gemini CLI
        if self._check_gemini_cli():
            tools['gemini'] = {
                'name': 'Gemini CLI',
                'command': 'gemini', 
                'execute_pattern': 'gemini code "{prompt}"',
                'install_url': 'https://ai.google.dev/gemini-api/docs/cli'
            }
        
        return tools
    
    def _select_preferred_tool(self) -> Optional[str]:
        """Select preferred tool based on availability and priority."""
        # Priority order: opencode > claude > gemini
        priority = ['opencode', 'claude', 'gemini']
        
        for tool in priority:
            if tool in self.available_cli_tools:
                return tool
        return None
    
    def _check_opencode(self) -> bool:
        """Check if opencode CLI is available."""
        try:
            result = subprocess.run(["opencode", "--version"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _check_claude_cli(self) -> bool:
        """Check if Claude Code CLI is available."""
        try:
            result = subprocess.run(["claude", "--version"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _check_gemini_cli(self) -> bool:
        """Check if Gemini CLI is available."""
        try:
            # Try gemini command first
            result = subprocess.run(["gemini", "--version"], capture_output=True)
            if result.returncode == 0:
                return True
        except FileNotFoundError:
            pass
        
        try:
            # Try gcloud ai alternative
            result = subprocess.run(["gcloud", "ai", "--version"], capture_output=True)
            if result.returncode == 0:
                return True
        except FileNotFoundError:
            pass
        
        return False
    
    async def start(self) -> None:
        """Start the agent."""
        self.status = "active"
        logger.info("Agent started", agent=self.name, type=self.agent_type)
        
        try:
            await self.run()
        except Exception as e:
            logger.error("Agent error", agent=self.name, error=str(e))
            self.status = "error"
        finally:
            await self.cleanup()
    
    @abstractmethod
    async def run(self) -> None:
        """Main agent execution loop."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.status = "inactive"
        logger.info("Agent stopped", agent=self.name)
    
    async def execute_with_claude(self, prompt: str, tool_override: str = None) -> Optional[str]:
        """Execute prompt using available CLI tools or fallback to API."""
        # Select tool to use
        tool_name = tool_override or self.preferred_cli_tool
        
        # Try CLI tools first
        if tool_name and tool_name in self.available_cli_tools:
            try:
                tool_config = self.available_cli_tools[tool_name]
                cmd = tool_config['execute_pattern'].format(prompt=prompt)
                
                result = subprocess.run(
                    cmd.split(),
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    return result.stdout.strip()
                else:
                    logger.warning(f"{tool_config['name']} failed", error=result.stderr)
            except Exception as e:
                logger.warning(f"{tool_config['name']} failed", error=str(e))
        
        # Fallback to other available CLI tools
        for fallback_tool, tool_config in self.available_cli_tools.items():
            if fallback_tool != tool_name:
                try:
                    cmd = tool_config['execute_pattern'].format(prompt=prompt)
                    result = subprocess.run(
                        cmd.split(),
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    if result.returncode == 0:
                        logger.info(f"Used fallback tool: {tool_config['name']}")
                        return result.stdout.strip()
                except Exception as e:
                    logger.warning(f"Fallback {tool_config['name']} failed", error=str(e))
        
        # Fallback to API if no CLI tools work
        if self.anthropic_client:
            try:
                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except Exception as e:
                logger.warning("Anthropic API failed", error=str(e))
        
        logger.warning("No CLI tools or API access available")
        return None
    
    async def send_message(self, to_agent: str, content: Dict[str, Any]) -> None:
        """Send message to another agent."""
        message = {
            "from": self.name,
            "to": to_agent,
            "content": content,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Publish to Redis
        channel = f"agent_messages:{to_agent}"
        self.redis_client.publish(channel, json.dumps(message))
        
        logger.info("Message sent", from_agent=self.name, to_agent=to_agent)
    
    async def get_messages(self) -> List[Dict[str, Any]]:
        """Get pending messages for this agent."""
        channel = f"agent_messages:{self.name}"
        # In a real implementation, this would use Redis Streams or similar
        # For now, return empty list
        return []
'''

    def _template_task_queue(self) -> str:
        """Template for task queue."""
        return '''"""Redis-based task queue with priority support."""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from enum import IntEnum
from typing import Dict, List, Optional, Any
import redis
import structlog
from src.core.config import settings


logger = structlog.get_logger()


class TaskPriority(IntEnum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 3
    NORMAL = 5
    LOW = 7
    BACKGROUND = 9


class TaskStatus:
    """Task status constants."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task data structure."""
    id: str
    title: str
    description: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = TaskPriority.NORMAL
    status: str = TaskStatus.PENDING
    agent_id: Optional[str] = None
    dependencies: List[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = time.time()


class TaskQueue:
    """Redis-based priority task queue."""
    
    def __init__(self):
        self.redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.queue_prefix = "hive:queue"
    
    async def submit_task(self, task: Task) -> str:
        """Submit a task to the queue."""
        # Store task data
        task_key = f"hive:task:{task.id}"
        self.redis_client.hset(task_key, mapping=asdict(task))
        
        # Add to priority queue
        queue_key = f"{self.queue_prefix}:p{task.priority}"
        self.redis_client.lpush(queue_key, task.id)
        
        logger.info("Task submitted", task_id=task.id, priority=task.priority)
        return task.id
    
    async def get_task(self, agent_id: str, priorities: List[int] = None) -> Optional[Task]:
        """Get next task for agent."""
        if priorities is None:
            priorities = [1, 3, 5, 7, 9]  # All priorities
        
        # Check queues in priority order
        for priority in sorted(priorities):
            queue_key = f"{self.queue_prefix}:p{priority}"
            
            # Try to pop a task
            task_id = self.redis_client.brpop(queue_key, timeout=1)
            if task_id:
                task_id = task_id[1]  # brpop returns (key, value)
                
                # Get task data
                task_data = self.redis_client.hgetall(f"hive:task:{task_id}")
                if task_data:
                    task = Task(**task_data)
                    
                    # Check dependencies
                    if await self._dependencies_satisfied(task):
                        # Assign to agent
                        task.status = TaskStatus.ASSIGNED
                        task.agent_id = agent_id
                        task.started_at = time.time()
                        
                        # Update in Redis
                        self.redis_client.hset(f"hive:task:{task_id}", mapping=asdict(task))
                        
                        logger.info("Task assigned", task_id=task_id, agent_id=agent_id)
                        return task
                    else:
                        # Put back in queue
                        self.redis_client.lpush(queue_key, task_id)
        
        return None
    
    async def _dependencies_satisfied(self, task: Task) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            dep_data = self.redis_client.hgetall(f"hive:task:{dep_id}")
            if not dep_data or dep_data.get("status") != TaskStatus.COMPLETED:
                return False
        return True
    
    async def complete_task(self, task_id: str, result: Dict[str, Any] = None) -> None:
        """Mark task as completed."""
        task_key = f"hive:task:{task_id}"
        updates = {
            "status": TaskStatus.COMPLETED,
            "completed_at": time.time()
        }
        if result:
            updates["result"] = json.dumps(result)
        
        self.redis_client.hset(task_key, mapping=updates)
        logger.info("Task completed", task_id=task_id)
    
    async def fail_task(self, task_id: str, error: str) -> None:
        """Mark task as failed."""
        task_key = f"hive:task:{task_id}"
        self.redis_client.hset(task_key, mapping={
            "status": TaskStatus.FAILED,
            "error_message": error,
            "completed_at": time.time()
        })
        logger.info("Task failed", task_id=task_id, error=error)


# Global instance
task_queue = TaskQueue()
'''

    def _template_models(self) -> str:
        """Template for database models - using the existing init.sql structure."""
        return '''"""SQLAlchemy models for LeanVibe Agent Hive 2.0."""

from datetime import datetime
from typing import Dict, List, Optional
from sqlalchemy import Column, String, Text, Integer, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB, VECTOR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid


Base = declarative_base()


class Agent(Base):
    """Agent model."""
    __tablename__ = "agents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    type = Column(String(100), nullable=False)
    role = Column(String(100), nullable=False)
    capabilities = Column(JSONB, default={})
    system_prompt = Column(Text)
    status = Column(String(50), default="inactive")
    tmux_session = Column(String(255))
    last_heartbeat = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tasks = relationship("Task", back_populates="agent")
    contexts = relationship("Context", back_populates="agent")


class Task(Base):
    """Task model."""
    __tablename__ = "tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    type = Column(String(100), nullable=False)
    payload = Column(JSONB, default={})
    priority = Column(Integer, default=5)
    status = Column(String(50), default="pending")
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"))
    dependencies = Column(JSONB, default=[])
    result = Column(JSONB)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    timeout_seconds = Column(Integer, default=300)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    agent = relationship("Agent", back_populates="tasks")


class Session(Base):
    """Session model."""
    __tablename__ = "sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    type = Column(String(100), nullable=False)
    agents = Column(JSONB, default=[])
    state = Column(JSONB, default={})
    status = Column(String(50), default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    contexts = relationship("Context", back_populates="session")
    conversations = relationship("Conversation", back_populates="session")


class Context(Base):
    """Context model for semantic memory."""
    __tablename__ = "contexts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"))
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"))
    content = Column(Text, nullable=False)
    embedding = Column(VECTOR(1536))  # OpenAI ada-002 dimension
    importance_score = Column(Float, default=0.5)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("contexts.id"))
    tags = Column(JSONB, default=[])
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    accessed_at = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)
    
    # Relationships
    agent = relationship("Agent", back_populates="contexts")
    session = relationship("Session", back_populates="contexts")
    parent = relationship("Context", remote_side=[id])


class Conversation(Base):
    """Conversation model."""
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"))
    from_agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"))
    to_agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"))
    message_type = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(VECTOR(1536))
    reply_to = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("Session", back_populates="conversations")
    from_agent = relationship("Agent", foreign_keys=[from_agent_id])
    to_agent = relationship("Agent", foreign_keys=[to_agent_id])


class SystemCheckpoint(Base):
    """System checkpoint model."""
    __tablename__ = "system_checkpoints"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type = Column(String(100), nullable=False)
    description = Column(Text)
    state = Column(JSONB, nullable=False)
    git_commit_hash = Column(String(64))
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"))
    performance_metrics = Column(JSONB)
    rollback_data = Column(JSONB)
    status = Column(String(50), default="created")
    created_at = Column(DateTime, default=datetime.utcnow)
    applied_at = Column(DateTime)
    rolled_back_at = Column(DateTime)
    
    # Relationships
    agent = relationship("Agent")
'''