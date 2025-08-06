    def _template_message_broker(self) -> str:
        """Template for message broker."""
        return '''"""Redis-based message broker for inter-agent communication."""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable
import redis.asyncio as redis
import structlog
from src.core.config import settings


logger = structlog.get_logger()


class MessageBroker:
    """Redis-based message broker with pub/sub and persistence."""
    
    def __init__(self):
        self.redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.pubsub = self.redis_client.pubsub()
        self.message_handlers: Dict[str, Callable] = {}
    
    async def publish(self, topic: str, message: Dict[str, Any], persist: bool = True) -> None:
        """Publish message to topic."""
        message_data = {
            "id": str(time.time()),
            "topic": topic,
            "timestamp": time.time(),
            "payload": message
        }
        
        # Publish to Redis pub/sub
        await self.redis_client.publish(topic, json.dumps(message_data))
        
        # Persist message if requested
        if persist:
            message_key = f"hive:messages:{topic}:{message_data['id']}"
            await self.redis_client.setex(message_key, 86400, json.dumps(message_data))  # 24h TTL
        
        logger.info("Message published", topic=topic, message_id=message_data["id"])
    
    async def subscribe(self, topic: str, handler: Callable) -> None:
        """Subscribe to topic with message handler."""
        self.message_handlers[topic] = handler
        await self.pubsub.subscribe(topic)
        logger.info("Subscribed to topic", topic=topic)
    
    async def start_listening(self) -> None:
        """Start listening for messages."""
        logger.info("Message broker started listening")
        
        async for message in self.pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    topic = data["topic"]
                    
                    # Find handler for topic
                    handler = self.message_handlers.get(topic)
                    if handler:
                        await handler(data)
                    else:
                        logger.warning("No handler for topic", topic=topic)
                        
                except Exception as e:
                    logger.error("Message processing failed", error=str(e))
    
    async def send_agent_message(self, from_agent: str, to_agent: str, content: Dict[str, Any]) -> None:
        """Send direct message between agents."""
        message = {
            "from": from_agent,
            "to": to_agent,
            "content": content
        }
        
        topic = f"agent:{to_agent}"
        await self.publish(topic, message)
    
    async def broadcast(self, content: Dict[str, Any], sender: str = "system") -> None:
        """Broadcast message to all agents."""
        message = {
            "from": sender,
            "to": "all",
            "content": content
        }
        
        await self.publish("broadcast", message)
    
    async def get_message_history(self, topic: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get message history for topic."""
        pattern = f"hive:messages:{topic}:*"
        keys = await self.redis_client.keys(pattern)
        
        messages = []
        for key in keys[-limit:]:  # Get latest messages
            data = await self.redis_client.get(key)
            if data:
                messages.append(json.loads(data))
        
        # Sort by timestamp
        messages.sort(key=lambda x: x["timestamp"])
        return messages


# Global instance
message_broker = MessageBroker()
'''

    def _template_orchestrator(self) -> str:
        """Template for orchestrator."""
        return '''"""Agent orchestrator for lifecycle management."""

import asyncio
import subprocess
import time
from typing import Dict, List, Optional
import structlog
from src.core.config import settings
from src.core.task_queue import task_queue, Task
from src.core.message_broker import message_broker


logger = structlog.get_logger()


class AgentOrchestrator:
    """Manages agent lifecycle and task distribution."""
    
    def __init__(self):
        self.active_agents: Dict[str, Dict] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.running = False
    
    async def start(self) -> None:
        """Start the orchestrator."""
        self.running = True
        logger.info("Agent orchestrator started")
        
        # Start background tasks
        await asyncio.gather(
            self.monitor_agents(),
            self.distribute_tasks(),
            self.handle_messages()
        )
    
    async def stop(self) -> None:
        """Stop the orchestrator."""
        self.running = False
        logger.info("Agent orchestrator stopped")
    
    async def spawn_agent(self, agent_type: str, name: str = None) -> str:
        """Spawn a new agent in a tmux session."""
        if not name:
            name = f"{agent_type}-{int(time.time())}"
        
        session_name = f"hive-{name}"
        
        try:
            # Create tmux session
            subprocess.run([
                "tmux", "new-session", "-d", "-s", session_name,
                f"cd {settings.project_root} && python -m src.agents.{agent_type} --name {name}"
            ], check=True)
            
            # Register agent
            self.active_agents[name] = {
                "type": agent_type,
                "session": session_name,
                "status": "starting",
                "last_heartbeat": time.time(),
                "spawned_at": time.time()
            }
            
            logger.info("Agent spawned", name=name, type=agent_type, session=session_name)
            return name
            
        except subprocess.CalledProcessError as e:
            logger.error("Failed to spawn agent", name=name, error=str(e))
            raise
    
    async def terminate_agent(self, name: str) -> None:
        """Terminate an agent."""
        if name in self.active_agents:
            session = self.active_agents[name]["session"]
            
            try:
                # Kill tmux session
                subprocess.run(["tmux", "kill-session", "-t", session], check=True)
                
                # Remove from active agents
                del self.active_agents[name]
                
                logger.info("Agent terminated", name=name)
                
            except subprocess.CalledProcessError as e:
                logger.error("Failed to terminate agent", name=name, error=str(e))
    
    async def monitor_agents(self) -> None:
        """Monitor agent health."""
        while self.running:
            current_time = time.time()
            dead_agents = []
            
            for name, info in self.active_agents.items():
                # Check if agent is responsive
                if current_time - info["last_heartbeat"] > settings.agent_heartbeat_interval * 2:
                    logger.warning("Agent appears dead", name=name)
                    dead_agents.append(name)
            
            # Clean up dead agents
            for name in dead_agents:
                await self.terminate_agent(name)
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def distribute_tasks(self) -> None:
        """Distribute tasks to available agents."""
        while self.running:
            # Get next task
            task = await task_queue.get_task("orchestrator")
            
            if task:
                # Find suitable agent
                agent_name = await self.find_suitable_agent(task)
                
                if agent_name:
                    # Assign task to agent
                    await message_broker.send_agent_message(
                        "orchestrator",
                        agent_name,
                        {"type": "task_assignment", "task": task}
                    )
                    logger.info("Task assigned", task_id=task.id, agent=agent_name)
                else:
                    # No suitable agent, put task back
                    await task_queue.submit_task(task)
                    logger.warning("No suitable agent for task", task_id=task.id)
            
            await asyncio.sleep(1)  # Check for tasks every second
    
    async def find_suitable_agent(self, task: Task) -> Optional[str]:
        """Find agent suitable for task."""
        # Simple round-robin for now
        active_agents = [name for name, info in self.active_agents.items() 
                        if info["status"] == "active"]
        
        if active_agents:
            return active_agents[0]  # For simplicity, return first active agent
        
        return None
    
    async def handle_messages(self) -> None:
        """Handle messages from agents."""
        await message_broker.subscribe("orchestrator", self.process_message)
        await message_broker.start_listening()
    
    async def process_message(self, message: Dict) -> None:
        """Process message from agent."""
        content = message.get("content", {})
        msg_type = content.get("type")
        
        if msg_type == "heartbeat":
            agent_name = message.get("from")
            if agent_name in self.active_agents:
                self.active_agents[agent_name]["last_heartbeat"] = time.time()
                self.active_agents[agent_name]["status"] = "active"
        
        elif msg_type == "task_completed":
            task_id = content.get("task_id")
            result = content.get("result")
            await task_queue.complete_task(task_id, result)
        
        elif msg_type == "task_failed":
            task_id = content.get("task_id")
            error = content.get("error")
            await task_queue.fail_task(task_id, error)


# Global instance
orchestrator = AgentOrchestrator()
'''

    def _template_api_main(self) -> str:
        """Template for FastAPI main."""
        return '''"""FastAPI application for LeanVibe Agent Hive 2.0."""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from src.core.config import settings
from src.core.task_queue import task_queue, Task, TaskPriority
from src.core.orchestrator import orchestrator


app = FastAPI(
    title="LeanVibe Agent Hive 2.0",
    description="Self-improving multi-agent development platform",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class TaskCreate(BaseModel):
    title: str
    description: str
    task_type: str
    payload: Dict[str, Any] = {}
    priority: int = TaskPriority.NORMAL


class AgentCreate(BaseModel):
    name: str
    agent_type: str
    role: str = ""


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str


# Health check
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import time
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="2.0.0"
    )


# Agent endpoints
@app.post("/api/v1/agents")
async def create_agent(agent: AgentCreate):
    """Create a new agent."""
    try:
        agent_name = await orchestrator.spawn_agent(agent.agent_type, agent.name)
        return {"status": "success", "agent_name": agent_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agents")
async def list_agents():
    """List all active agents."""
    return {"agents": orchestrator.active_agents}


@app.delete("/api/v1/agents/{agent_name}")
async def terminate_agent(agent_name: str):
    """Terminate an agent."""
    try:
        await orchestrator.terminate_agent(agent_name)
        return {"status": "success", "message": f"Agent {agent_name} terminated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Task endpoints
@app.post("/api/v1/tasks")
async def create_task(task_data: TaskCreate):
    """Submit a new task."""
    import uuid
    
    task = Task(
        id=str(uuid.uuid4()),
        title=task_data.title,
        description=task_data.description,
        task_type=task_data.task_type,
        payload=task_data.payload,
        priority=task_data.priority
    )
    
    task_id = await task_queue.submit_task(task)
    return {"status": "success", "task_id": task_id}


@app.get("/api/v1/tasks")
async def list_tasks():
    """List all tasks."""
    # In a real implementation, this would query the database
    return {"tasks": []}


@app.get("/api/v1/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task details."""
    # In a real implementation, this would query the database
    return {"task_id": task_id, "status": "not_implemented"}


# System endpoints
@app.get("/api/v1/system/status")
async def system_status():
    """Get system status."""
    return {
        "orchestrator": "running" if orchestrator.running else "stopped",
        "active_agents": len(orchestrator.active_agents),
        "settings": {
            "max_workers": settings.max_workers,
            "embedding_model": settings.embedding_model
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.api_port)
'''

    def _template_meta_agent(self) -> str:
        """Template for meta agent."""
        return '''"""Meta-agent for system analysis and improvement."""

import asyncio
import json
import time
from typing import Dict, Any
import structlog
from src.agents.base_agent import BaseAgent
from src.core.config import settings
from src.core.task_queue import task_queue, Task, TaskPriority
from src.core.message_broker import message_broker


logger = structlog.get_logger()


class MetaAgent(BaseAgent):
    """Meta-agent responsible for system analysis and improvement."""
    
    def __init__(self, name: str = "meta-agent"):
        super().__init__(name, "meta", "system_improver")
        self.analysis_interval = 300  # 5 minutes
        self.last_analysis = 0
    
    async def run(self) -> None:
        """Main execution loop."""
        logger.info("Meta-agent started")
        
        # Send heartbeat
        await self.send_heartbeat()
        
        while self.status == "active":
            try:
                # Check for messages
                await self.process_messages()
                
                # Periodic system analysis
                if time.time() - self.last_analysis > self.analysis_interval:
                    await self.analyze_system()
                    self.last_analysis = time.time()
                
                # Send heartbeat
                await self.send_heartbeat()
                
                await asyncio.sleep(10)  # Main loop interval
                
            except Exception as e:
                logger.error("Meta-agent error", error=str(e))
                await asyncio.sleep(30)  # Wait before retry
    
    async def send_heartbeat(self) -> None:
        """Send heartbeat to orchestrator."""
        await message_broker.send_agent_message(
            self.name,
            "orchestrator",
            {"type": "heartbeat", "status": self.status}
        )
    
    async def process_messages(self) -> None:
        """Process incoming messages."""
        # In a real implementation, this would check for messages
        pass
    
    async def analyze_system(self) -> None:
        """Analyze system performance and propose improvements."""
        logger.info("Starting system analysis")
        
        analysis_prompt = """
        Analyze the current LeanVibe Agent Hive system and provide recommendations for improvement.
        
        Consider:
        1. System performance and bottlenecks
        2. Agent efficiency and coordination
        3. Code quality and architecture
        4. Potential new features or optimizations
        
        Provide specific, actionable recommendations.
        """
        
        analysis_result = await self.execute_with_claude(analysis_prompt)
        
        if analysis_result:
            # Create improvement task
            improvement_task = Task(
                id=f"improvement-{int(time.time())}",
                title="System Improvement",
                description="Implement meta-agent analysis recommendations",
                task_type="system_improvement",
                payload={"analysis": analysis_result},
                priority=TaskPriority.HIGH
            )
            
            await task_queue.submit_task(improvement_task)
            logger.info("Improvement task created", task_id=improvement_task.id)
        
        logger.info("System analysis completed")
    
    async def propose_code_changes(self, component: str, issue: str) -> None:
        """Propose specific code changes."""
        prompt = f"""
        Analyze the {component} component and propose code changes to address: {issue}
        
        Provide:
        1. Specific code changes needed
        2. Testing strategy
        3. Risk assessment
        4. Implementation steps
        """
        
        proposal = await self.execute_with_claude(prompt)
        
        if proposal:
            # Create code change task
            change_task = Task(
                id=f"code-change-{int(time.time())}",
                title=f"Code Change: {component}",
                description=f"Address issue: {issue}",
                task_type="code_modification",
                payload={
                    "component": component,
                    "issue": issue,
                    "proposal": proposal
                },
                priority=TaskPriority.HIGH
            )
            
            await task_queue.submit_task(change_task)
            logger.info("Code change task created", task_id=change_task.id)


async def main():
    """Main entry point."""
    import sys
    
    name = "meta-agent"
    if len(sys.argv) > 1 and sys.argv[1] == "--name":
        name = sys.argv[2]
    
    agent = MetaAgent(name)
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
'''