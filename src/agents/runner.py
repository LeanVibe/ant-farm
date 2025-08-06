"""Agent runner system for spawning and managing agents in LeanVibe Agent Hive 2.0."""

import asyncio
import argparse
import signal
import sys
import time
from typing import Dict, Type, Optional
import structlog

from .base_agent import BaseAgent
from .meta_agent import MetaAgent
from ..core.config import settings

logger = structlog.get_logger()


class DeveloperAgent(BaseAgent):
    """Developer agent specialized in code implementation and modification."""
    
    def __init__(self, name: str = "developer-agent"):
        super().__init__(name, "developer", "code_developer")
        logger.info("Developer agent initialized", name=self.name)
    
    async def run(self) -> None:
        """Main developer agent execution loop."""
        logger.info("Developer agent starting", agent=self.name)
        
        while self.status == "active":
            try:
                # Check for development tasks
                task = await task_queue.get_next_task(self.name, task_types=["development", "coding", "implementation"])
                
                if task:
                    await self.process_task(task)
                else:
                    # No specific tasks, do proactive development work
                    await self._proactive_development_work()
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error("Developer agent error", agent=self.name, error=str(e))
                await asyncio.sleep(30)
    
    async def _proactive_development_work(self) -> None:
        """Perform proactive development work when no specific tasks are assigned."""
        try:
            # Look for TODO comments in the codebase
            prompt = """
            Scan the codebase for TODO comments, incomplete implementations, 
            or areas that need improvement. Focus on:
            1. Missing functionality
            2. Code that needs refactoring
            3. Performance optimizations
            4. Better error handling
            
            If you find something to work on, implement it.
            Otherwise, analyze the codebase for potential improvements.
            """
            
            result = await self.execute_with_cli_tool(prompt)
            
            if result.success:
                await self.store_context(
                    content=f"Proactive development work completed: {result.output[:500]}...",
                    importance_score=0.6,
                    category="proactive_development",
                    metadata={"tool_used": result.tool_used}
                )
        
        except Exception as e:
            logger.error("Proactive development work failed", agent=self.name, error=str(e))


class QAAgent(BaseAgent):
    """QA agent specialized in testing and quality assurance."""
    
    def __init__(self, name: str = "qa-agent"):
        super().__init__(name, "qa", "quality_assurance")
        logger.info("QA agent initialized", name=self.name)
    
    async def run(self) -> None:
        """Main QA agent execution loop."""
        logger.info("QA agent starting", agent=self.name)
        
        while self.status == "active":
            try:
                # Check for QA tasks
                task = await task_queue.get_next_task(self.name, task_types=["testing", "qa", "validation"])
                
                if task:
                    await self.process_task(task)
                else:
                    # No specific tasks, do proactive QA work
                    await self._proactive_qa_work()
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error("QA agent error", agent=self.name, error=str(e))
                await asyncio.sleep(30)
    
    async def _proactive_qa_work(self) -> None:
        """Perform proactive QA work when no specific tasks are assigned."""
        try:
            prompt = """
            Perform quality assurance tasks on the current codebase:
            1. Run existing tests and check for failures
            2. Analyze test coverage and identify gaps
            3. Look for potential bugs or edge cases
            4. Check code quality and style compliance
            5. Validate documentation accuracy
            
            Report any issues found and suggest improvements.
            """
            
            result = await self.execute_with_cli_tool(prompt)
            
            if result.success:
                await self.store_context(
                    content=f"Proactive QA work completed: {result.output[:500]}...",
                    importance_score=0.7,
                    category="proactive_qa",
                    metadata={"tool_used": result.tool_used}
                )
        
        except Exception as e:
            logger.error("Proactive QA work failed", agent=self.name, error=str(e))


class ArchitectAgent(BaseAgent):
    """Architect agent specialized in system design and architecture decisions."""
    
    def __init__(self, name: str = "architect-agent"):
        super().__init__(name, "architect", "system_architect")
        logger.info("Architect agent initialized", name=self.name)
    
    async def run(self) -> None:
        """Main architect agent execution loop."""
        logger.info("Architect agent starting", agent=self.name)
        
        while self.status == "active":
            try:
                # Check for architecture tasks
                task = await task_queue.get_next_task(self.name, task_types=["architecture", "design", "planning"])
                
                if task:
                    await self.process_task(task)
                else:
                    # No specific tasks, do proactive architecture work
                    await self._proactive_architecture_work()
                
                await asyncio.sleep(15)
                
            except Exception as e:
                logger.error("Architect agent error", agent=self.name, error=str(e))
                await asyncio.sleep(30)
    
    async def _proactive_architecture_work(self) -> None:
        """Perform proactive architecture work when no specific tasks are assigned."""
        try:
            prompt = """
            Analyze the current system architecture and identify:
            1. Architectural debt and technical debt
            2. Scalability bottlenecks
            3. Design pattern improvements
            4. Component coupling issues
            5. Performance optimization opportunities
            
            Suggest architectural improvements and document findings.
            """
            
            result = await self.execute_with_cli_tool(prompt)
            
            if result.success:
                await self.store_context(
                    content=f"Proactive architecture work completed: {result.output[:500]}...",
                    importance_score=0.8,
                    category="proactive_architecture",
                    metadata={"tool_used": result.tool_used}
                )
        
        except Exception as e:
            logger.error("Proactive architecture work failed", agent=self.name, error=str(e))


class ResearchAgent(BaseAgent):
    """Research agent specialized in learning and knowledge acquisition."""
    
    def __init__(self, name: str = "research-agent"):
        super().__init__(name, "research", "knowledge_researcher")
        logger.info("Research agent initialized", name=self.name)
    
    async def run(self) -> None:
        """Main research agent execution loop."""
        logger.info("Research agent starting", agent=self.name)
        
        while self.status == "active":
            try:
                # Check for research tasks
                task = await task_queue.get_next_task(self.name, task_types=["research", "learning", "analysis"])
                
                if task:
                    await self.process_task(task)
                else:
                    # No specific tasks, do proactive research work
                    await self._proactive_research_work()
                
                await asyncio.sleep(20)
                
            except Exception as e:
                logger.error("Research agent error", agent=self.name, error=str(e))
                await asyncio.sleep(30)
    
    async def _proactive_research_work(self) -> None:
        """Perform proactive research work when no specific tasks are assigned."""
        try:
            prompt = """
            Conduct research on current trends and best practices relevant to our system:
            1. Latest developments in multi-agent systems
            2. AI/ML techniques for self-improving systems
            3. Best practices for autonomous software development
            4. New tools and frameworks that could benefit us
            5. Security considerations for self-modifying systems
            
            Summarize findings and suggest actionable improvements.
            """
            
            result = await self.execute_with_cli_tool(prompt)
            
            if result.success:
                await self.store_context(
                    content=f"Research findings: {result.output[:500]}...",
                    importance_score=0.6,
                    category="research_findings",
                    metadata={"tool_used": result.tool_used}
                )
        
        except Exception as e:
            logger.error("Proactive research work failed", agent=self.name, error=str(e))


# Agent registry
AGENT_TYPES: Dict[str, Type[BaseAgent]] = {
    "meta": MetaAgent,
    "developer": DeveloperAgent,
    "qa": QAAgent,
    "architect": ArchitectAgent,
    "research": ResearchAgent,
}


class AgentRunner:
    """Manages individual agent lifecycle."""
    
    def __init__(self, agent_type: str, agent_name: Optional[str] = None):
        self.agent_type = agent_type
        self.agent_name = agent_name or f"{agent_type}-{int(time.time())}"
        self.agent: Optional[BaseAgent] = None
        self.shutdown_requested = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received", signal=signum, agent=self.agent_name)
        self.shutdown_requested = True
    
    async def start(self) -> None:
        """Start the agent."""
        if self.agent_type not in AGENT_TYPES:
            raise ValueError(f"Unknown agent type: {self.agent_type}")
        
        agent_class = AGENT_TYPES[self.agent_type]
        self.agent = agent_class(self.agent_name)
        
        logger.info("Starting agent", type=self.agent_type, name=self.agent_name)
        
        try:
            # Start the agent in a task so we can monitor for shutdown
            agent_task = asyncio.create_task(self.agent.start())
            
            # Wait for either agent completion or shutdown signal
            while not self.shutdown_requested and not agent_task.done():
                await asyncio.sleep(1)
            
            if self.shutdown_requested:
                logger.info("Initiating graceful shutdown", agent=self.agent_name)
                self.agent.status = "stopping"
                
                # Give agent time to cleanup
                try:
                    await asyncio.wait_for(agent_task, timeout=30)
                except asyncio.TimeoutError:
                    logger.warning("Agent shutdown timeout, forcing termination", agent=self.agent_name)
                    agent_task.cancel()
        
        except Exception as e:
            logger.error("Agent runtime error", agent=self.agent_name, error=str(e))
            raise
        
        finally:
            if self.agent:
                await self.agent.cleanup()
            logger.info("Agent stopped", agent=self.agent_name)


async def main():
    """Main entry point for agent runner."""
    parser = argparse.ArgumentParser(description="LeanVibe Agent Hive 2.0 - Agent Runner")
    parser.add_argument("--type", required=True, choices=list(AGENT_TYPES.keys()),
                      help="Type of agent to run")
    parser.add_argument("--name", help="Name for the agent instance")
    parser.add_argument("--log-level", default="INFO", 
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Logging level")
    
    args = parser.parse_args()
    
    # Configure logging
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog.stdlib.logging, args.log_level)
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Create and start agent runner
    runner = AgentRunner(args.type, args.name)
    
    try:
        await runner.start()
    except KeyboardInterrupt:
        logger.info("Agent runner interrupted")
    except Exception as e:
        logger.error("Agent runner failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())