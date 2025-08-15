"""MetaAgent - Strategic coordination and system self-improvement agent."""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from .base_agent import BaseAgent, TaskResult, ToolResult
from ..core.task_queue import Task, TaskPriority, TaskStatus
from ..core.enums import AgentStatus

logger = structlog.get_logger()


class MetaAgent(BaseAgent):
    """Meta-agent for system self-improvement and strategic coordination."""
    
    def __init__(self, name: str, role: str = "meta_coordinator"):
        super().__init__(name, "meta", role, enhanced_communication=True)
        
        # Meta-specific capabilities
        self.meta_capabilities = [
            "system_analysis", 
            "code_generation", 
            "self_improvement",
            "strategic_planning",
            "performance_optimization",
            "agent_coordination",
            "quality_assessment",
            "task_delegation"
        ]
        
        # Performance tracking
        self.system_health_history = []
        self.improvement_tasks_created = 0
        self.coordination_sessions = 0
        
        # Strategic planning state
        self.current_strategic_focus = "system_stability"
        self.improvement_pipeline = []
        self.coordination_active = {}
        
        logger.info(
            "MetaAgent initialized", 
            agent=self.name,
            capabilities=self.meta_capabilities
        )
    
    async def run(self) -> None:
        """Main MetaAgent execution loop."""
        logger.info("MetaAgent starting main execution loop", agent=self.name)
        
        # Strategic monitoring and coordination cycle
        last_system_analysis = 0
        last_performance_check = 0
        last_coordination_review = 0
        
        analysis_interval = 300  # 5 minutes
        performance_interval = 120  # 2 minutes  
        coordination_interval = 180  # 3 minutes
        
        while self.status == "active":
            try:
                current_time = time.time()
                
                # System health analysis
                if current_time - last_system_analysis >= analysis_interval:
                    await self._perform_system_analysis()
                    last_system_analysis = current_time
                
                # Performance monitoring
                if current_time - last_performance_check >= performance_interval:
                    await self._monitor_performance()
                    last_performance_check = current_time
                
                # Coordination review
                if current_time - last_coordination_review >= coordination_interval:
                    await self._review_coordination_opportunities()
                    last_coordination_review = current_time
                
                # Process improvement tasks
                await self._process_improvement_pipeline()
                
                # Check for collaboration requests
                await self._handle_pending_collaborations()
                
                # Brief pause between cycles
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(
                    "MetaAgent main loop error", 
                    agent=self.name, 
                    error=str(e)
                )
                await asyncio.sleep(30)  # Longer pause on error

    async def analyze_system_health(self) -> Dict[str, Any]:
        """Analyze overall system health and identify improvement areas."""
        logger.info("Analyzing system health", agent=self.name)
        
        try:
            health_metrics = {}
            
            # Agent health analysis
            agent_health = await self._analyze_agent_ecosystem()
            health_metrics["agents"] = agent_health
            
            # Task queue analysis
            queue_health = await self._analyze_task_queue_health()
            health_metrics["task_queue"] = queue_health
            
            # Database health
            if self.async_db_manager:
                db_health = await self._analyze_database_health()
                health_metrics["database"] = db_health
            
            # CLI tool availability
            cli_health = await self._analyze_cli_tool_health()
            health_metrics["cli_tools"] = cli_health
            
            # System performance
            performance_metrics = await self._analyze_system_performance()
            health_metrics["performance"] = performance_metrics
            
            # Overall health score
            health_metrics["overall_score"] = self._calculate_health_score(health_metrics)
            health_metrics["timestamp"] = time.time()
            health_metrics["analysis_duration"] = time.time() - time.time()
            
            # Store in history
            self.system_health_history.append(health_metrics)
            if len(self.system_health_history) > 100:  # Keep last 100 analyses
                self.system_health_history.pop(0)
            
            # Store analysis results in context
            if self.context_engine and self.agent_uuid:
                await self.store_context(
                    content=f"System Health Analysis: Overall score {health_metrics['overall_score']}/100",
                    importance_score=0.8,
                    category="system_analysis",
                    metadata=health_metrics
                )
            
            logger.info(
                "System health analysis complete",
                agent=self.name,
                overall_score=health_metrics["overall_score"],
                key_issues=health_metrics.get("key_issues", [])
            )
            
            return health_metrics
            
        except Exception as e:
            logger.error("System health analysis failed", agent=self.name, error=str(e))
            return {
                "error": str(e),
                "overall_score": 0,
                "timestamp": time.time()
            }

    async def plan_improvement_tasks(self, analysis_results: Dict) -> List[Task]:
        """Generate strategic improvement tasks based on system analysis."""
        logger.info("Planning improvement tasks", agent=self.name)
        
        improvement_tasks = []
        
        try:
            overall_score = analysis_results.get("overall_score", 50)
            
            # Critical issues (score < 30)
            if overall_score < 30:
                critical_tasks = await self._plan_critical_improvements(analysis_results)
                improvement_tasks.extend(critical_tasks)
            
            # Performance optimizations (score < 70)
            elif overall_score < 70:
                performance_tasks = await self._plan_performance_improvements(analysis_results)
                improvement_tasks.extend(performance_tasks)
            
            # Enhancement opportunities (score >= 70)
            else:
                enhancement_tasks = await self._plan_enhancement_tasks(analysis_results)
                improvement_tasks.extend(enhancement_tasks)
            
            # Agent ecosystem improvements
            agent_tasks = await self._plan_agent_improvements(analysis_results)
            improvement_tasks.extend(agent_tasks)
            
            # Submit tasks to queue
            submitted_tasks = []
            for task in improvement_tasks:
                try:
                    task_id = await self._submit_improvement_task(task)
                    if task_id:
                        submitted_tasks.append(task)
                        self.improvement_tasks_created += 1
                except Exception as e:
                    logger.warning(
                        "Failed to submit improvement task", 
                        agent=self.name,
                        task=task.title,
                        error=str(e)
                    )
            
            # Store planning results
            if self.context_engine and self.agent_uuid:
                await self.store_context(
                    content=f"Improvement Planning: Created {len(submitted_tasks)} tasks",
                    importance_score=0.7,
                    category="strategic_planning",
                    metadata={
                        "tasks_planned": len(improvement_tasks),
                        "tasks_submitted": len(submitted_tasks),
                        "focus_area": self.current_strategic_focus,
                        "system_score": overall_score
                    }
                )
            
            logger.info(
                "Improvement tasks planned",
                agent=self.name,
                tasks_created=len(submitted_tasks),
                focus_area=self.current_strategic_focus
            )
            
            return submitted_tasks
            
        except Exception as e:
            logger.error("Improvement planning failed", agent=self.name, error=str(e))
            return []

    async def coordinate_agents(self, task: Task) -> bool:
        """Coordinate multiple agents for complex tasks."""
        logger.info(
            "Coordinating agents for complex task",
            agent=self.name, 
            task_id=task.id,
            task_type=task.task_type
        )
        
        try:
            # Analyze task complexity and requirements
            coordination_plan = await self._analyze_task_coordination_needs(task)
            
            if not coordination_plan["requires_coordination"]:
                logger.info("Task does not require coordination", task_id=task.id)
                return True
            
            # Find suitable agents
            suitable_agents = await self._find_suitable_agents(coordination_plan["required_capabilities"])
            
            if len(suitable_agents) < coordination_plan["min_agents"]:
                logger.warning(
                    "Insufficient agents available for coordination",
                    task_id=task.id,
                    required=coordination_plan["min_agents"],
                    available=len(suitable_agents)
                )
                return False
            
            # Create coordination session
            session_id = await self._create_coordination_session(task, suitable_agents, coordination_plan)
            
            if session_id:
                self.coordination_sessions += 1
                self.coordination_active[session_id] = {
                    "task_id": task.id,
                    "agents": suitable_agents,
                    "started_at": time.time(),
                    "status": "active"
                }
                
                # Store coordination context
                if self.context_engine and self.agent_uuid:
                    await self.store_context(
                        content=f"Agent Coordination: Session {session_id} for task {task.title}",
                        importance_score=0.8,
                        category="agent_coordination",
                        metadata={
                            "session_id": session_id,
                            "task_id": task.id,
                            "coordinated_agents": suitable_agents,
                            "coordination_plan": coordination_plan
                        }
                    )
                
                logger.info(
                    "Agent coordination session created",
                    agent=self.name,
                    session_id=session_id,
                    coordinated_agents=len(suitable_agents)
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error("Agent coordination failed", agent=self.name, error=str(e))
            return False

    async def monitor_performance(self) -> Dict[str, float]:
        """Monitor system performance metrics and identify bottlenecks."""
        logger.debug("Monitoring system performance", agent=self.name)
        
        try:
            performance_metrics = {}
            
            # Agent performance metrics
            if self.async_db_manager:
                agent_metrics = await self._collect_agent_performance_metrics()
                performance_metrics.update(agent_metrics)
            
            # Task processing metrics
            task_metrics = await self._collect_task_processing_metrics()
            performance_metrics.update(task_metrics)
            
            # Resource utilization
            resource_metrics = await self._collect_resource_metrics()
            performance_metrics.update(resource_metrics)
            
            # Communication performance
            comm_metrics = await self._collect_communication_metrics()
            performance_metrics.update(comm_metrics)
            
            # Identify bottlenecks
            bottlenecks = await self._identify_performance_bottlenecks(performance_metrics)
            performance_metrics["bottlenecks"] = bottlenecks
            
            # Performance trend analysis
            performance_trend = await self._analyze_performance_trends()
            performance_metrics["trend_analysis"] = performance_trend
            
            return performance_metrics
            
        except Exception as e:
            logger.error("Performance monitoring failed", agent=self.name, error=str(e))
            return {"error": str(e)}

    async def _perform_system_analysis(self) -> None:
        """Perform comprehensive system analysis."""
        try:
            analysis_results = await self.analyze_system_health()
            
            # Plan improvements based on analysis
            if analysis_results.get("overall_score", 0) < 80:
                improvement_tasks = await self.plan_improvement_tasks(analysis_results)
                logger.info(
                    "System analysis complete - improvement tasks created",
                    agent=self.name,
                    system_score=analysis_results.get("overall_score", 0),
                    improvements_planned=len(improvement_tasks)
                )
            else:
                logger.info(
                    "System analysis complete - system healthy",
                    agent=self.name,
                    system_score=analysis_results.get("overall_score", 0)
                )
                
        except Exception as e:
            logger.error("System analysis failed", agent=self.name, error=str(e))

    async def _monitor_performance(self) -> None:
        """Monitor and report on system performance."""
        try:
            performance_data = await self.monitor_performance()
            
            # Check for performance issues
            critical_bottlenecks = [
                b for b in performance_data.get("bottlenecks", []) 
                if b.get("severity", "low") == "critical"
            ]
            
            if critical_bottlenecks:
                logger.warning(
                    "Critical performance bottlenecks detected",
                    agent=self.name,
                    bottlenecks=len(critical_bottlenecks)
                )
                
                # Create performance improvement tasks
                for bottleneck in critical_bottlenecks:
                    await self._create_performance_improvement_task(bottleneck)
            
        except Exception as e:
            logger.error("Performance monitoring failed", agent=self.name, error=str(e))

    async def _review_coordination_opportunities(self) -> None:
        """Review and initiate coordination opportunities."""
        try:
            # Check for complex tasks that could benefit from coordination
            pending_complex_tasks = await self._find_complex_tasks()
            
            for task in pending_complex_tasks:
                coordination_success = await self.coordinate_agents(task)
                if coordination_success:
                    logger.info(
                        "Coordination initiated for complex task",
                        agent=self.name,
                        task_id=task.id
                    )
            
            # Review active coordination sessions
            await self._review_active_coordination_sessions()
            
        except Exception as e:
            logger.error("Coordination review failed", agent=self.name, error=str(e))

    async def _process_improvement_pipeline(self) -> None:
        """Process queued improvement tasks."""
        try:
            if not self.improvement_pipeline:
                return
            
            # Process next item in pipeline
            next_improvement = self.improvement_pipeline.pop(0)
            await self._execute_improvement_task(next_improvement)
            
        except Exception as e:
            logger.error("Improvement pipeline processing failed", agent=self.name, error=str(e))

    # Helper methods for system analysis
    async def _analyze_agent_ecosystem(self) -> Dict[str, Any]:
        """Analyze the health of the agent ecosystem."""
        try:
            if not self.async_db_manager:
                return {"error": "Database not available"}
            
            # Get all agents from database
            active_agents = await self.async_db_manager.get_active_agents()
            
            agent_health = {
                "total_agents": len(active_agents),
                "active_count": len([a for a in active_agents if a.status == "active"]),
                "idle_count": len([a for a in active_agents if a.status == "idle"]),
                "error_count": len([a for a in active_agents if a.status == "error"]),
                "agent_types": {},
                "health_score": 0
            }
            
            # Analyze by agent type
            for agent in active_agents:
                agent_type = agent.type
                if agent_type not in agent_health["agent_types"]:
                    agent_health["agent_types"][agent_type] = {
                        "count": 0,
                        "active": 0,
                        "performance": []
                    }
                
                agent_health["agent_types"][agent_type]["count"] += 1
                if agent.status == "active":
                    agent_health["agent_types"][agent_type]["active"] += 1
            
            # Calculate health score
            total_agents = agent_health["total_agents"]
            if total_agents > 0:
                active_ratio = agent_health["active_count"] / total_agents
                error_ratio = agent_health["error_count"] / total_agents
                agent_health["health_score"] = max(0, (active_ratio - error_ratio) * 100)
            
            return agent_health
            
        except Exception as e:
            return {"error": str(e), "health_score": 0}

    async def _analyze_task_queue_health(self) -> Dict[str, Any]:
        """Analyze task queue health and performance."""
        try:
            from ..core.task_queue import task_queue
            
            queue_stats = await task_queue.get_queue_stats()
            
            queue_health = {
                "total_queued": sum(queue_stats.queue_size_by_priority.values()),
                "completed_tasks": queue_stats.completed_tasks,
                "failed_tasks": queue_stats.failed_tasks,
                "processing_rate": queue_stats.processing_rate,
                "average_wait_time": queue_stats.average_wait_time,
                "health_score": 0
            }
            
            # Calculate health score based on queue performance
            total_tasks = queue_health["completed_tasks"] + queue_health["failed_tasks"]
            if total_tasks > 0:
                success_rate = queue_health["completed_tasks"] / total_tasks
                queue_health["health_score"] = min(100, success_rate * 100)
            
            # Adjust score based on queue size and processing rate
            if queue_health["total_queued"] > 100:  # Large backlog
                queue_health["health_score"] *= 0.8
            
            if queue_health["processing_rate"] < 1:  # Slow processing
                queue_health["health_score"] *= 0.9
            
            return queue_health
            
        except Exception as e:
            return {"error": str(e), "health_score": 0}

    async def _analyze_database_health(self) -> Dict[str, Any]:
        """Analyze database health and performance."""
        try:
            db_health = await self.async_db_manager.health_check()
            
            return {
                "connection_healthy": db_health,
                "health_score": 100 if db_health else 0
            }
            
        except Exception as e:
            return {"error": str(e), "health_score": 0}

    async def _analyze_cli_tool_health(self) -> Dict[str, Any]:
        """Analyze CLI tool availability and performance."""
        try:
            cli_health = {
                "available_tools": list(self.cli_tools.available_tools.keys()),
                "preferred_tool": self.cli_tools.preferred_tool.value if self.cli_tools.preferred_tool else None,
                "tool_count": len(self.cli_tools.available_tools),
                "health_score": 0
            }
            
            # Health score based on tool availability
            if cli_health["tool_count"] >= 2:
                cli_health["health_score"] = 100
            elif cli_health["tool_count"] == 1:
                cli_health["health_score"] = 75
            else:
                cli_health["health_score"] = 0
            
            return cli_health
            
        except Exception as e:
            return {"error": str(e), "health_score": 0}

    async def _analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze overall system performance."""
        try:
            performance_data = await self.monitor_performance()
            
            return {
                "response_time": performance_data.get("avg_response_time", 0),
                "throughput": performance_data.get("tasks_per_minute", 0),
                "error_rate": performance_data.get("error_rate", 0),
                "health_score": max(0, 100 - (performance_data.get("error_rate", 0) * 100))
            }
            
        except Exception as e:
            return {"error": str(e), "health_score": 50}

    def _calculate_health_score(self, health_metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score."""
        try:
            component_scores = []
            
            # Weight different components
            weights = {
                "agents": 0.3,
                "task_queue": 0.25,
                "database": 0.2,
                "cli_tools": 0.15,
                "performance": 0.1
            }
            
            total_score = 0
            total_weight = 0
            
            for component, weight in weights.items():
                if component in health_metrics and "health_score" in health_metrics[component]:
                    score = health_metrics[component]["health_score"]
                    total_score += score * weight
                    total_weight += weight
            
            if total_weight > 0:
                overall_score = total_score / total_weight
            else:
                overall_score = 0
            
            # Identify key issues
            key_issues = []
            for component, metrics in health_metrics.items():
                if isinstance(metrics, dict) and metrics.get("health_score", 100) < 50:
                    key_issues.append(f"{component}_unhealthy")
            
            health_metrics["key_issues"] = key_issues
            
            return round(overall_score, 1)
            
        except Exception as e:
            logger.error("Health score calculation failed", error=str(e))
            return 0.0

    # Task planning helpers
    async def _plan_critical_improvements(self, analysis: Dict) -> List[Task]:
        """Plan critical system improvements."""
        tasks = []
        
        # Critical database issues
        if analysis.get("database", {}).get("health_score", 100) < 30:
            tasks.append(await self._create_task(
                title="Fix Critical Database Issues",
                description="Address critical database connectivity and performance issues",
                task_type="system_repair",
                priority=TaskPriority.CRITICAL
            ))
        
        # Critical agent failures
        agent_health = analysis.get("agents", {})
        if agent_health.get("error_count", 0) > agent_health.get("active_count", 1):
            tasks.append(await self._create_task(
                title="Restore Failed Agents",
                description="Investigate and restore failed agents to operational status",
                task_type="agent_recovery",
                priority=TaskPriority.HIGH
            ))
        
        return tasks

    async def _plan_performance_improvements(self, analysis: Dict) -> List[Task]:
        """Plan performance improvement tasks."""
        tasks = []
        
        performance = analysis.get("performance", {})
        if performance.get("health_score", 100) < 70:
            tasks.append(await self._create_task(
                title="Optimize System Performance",
                description="Identify and resolve system performance bottlenecks",
                task_type="performance_optimization",
                priority=TaskPriority.NORMAL
            ))
        
        # Task queue optimization
        queue_health = analysis.get("task_queue", {})
        if queue_health.get("total_queued", 0) > 50:
            tasks.append(await self._create_task(
                title="Optimize Task Queue Processing",
                description="Improve task queue processing efficiency and reduce backlog",
                task_type="queue_optimization", 
                priority=TaskPriority.NORMAL
            ))
        
        return tasks

    async def _plan_enhancement_tasks(self, analysis: Dict) -> List[Task]:
        """Plan system enhancement tasks."""
        tasks = []
        
        # Agent capability enhancements
        agent_health = analysis.get("agents", {})
        if agent_health.get("health_score", 0) > 70:
            tasks.append(await self._create_task(
                title="Enhance Agent Capabilities",
                description="Add new capabilities and improve existing agent functions",
                task_type="capability_enhancement",
                priority=TaskPriority.LOW
            ))
        
        return tasks

    async def _plan_agent_improvements(self, analysis: Dict) -> List[Task]:
        """Plan agent-specific improvement tasks."""
        tasks = []
        
        agent_health = analysis.get("agents", {})
        agent_types = agent_health.get("agent_types", {})
        
        # Check for missing essential agent types
        essential_types = ["meta", "architect", "qa", "developer"]
        existing_types = list(agent_types.keys())
        
        for essential_type in essential_types:
            if essential_type not in existing_types:
                tasks.append(await self._create_task(
                    title=f"Implement {essential_type.title()} Agent",
                    description=f"Create and deploy {essential_type} agent for system completeness",
                    task_type="agent_implementation",
                    priority=TaskPriority.HIGH
                ))
        
        return tasks

    async def _create_task(
        self, 
        title: str, 
        description: str, 
        task_type: str, 
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> Task:
        """Create a task object."""
        return Task(
            title=title,
            description=description,
            task_type=task_type,
            priority=priority,
            payload={
                "created_by": self.name,
                "created_at": time.time(),
                "meta_analysis": True
            }
        )

    # Helper methods for task management
    async def _submit_improvement_task(self, task: Task) -> str | None:
        """Submit improvement task to queue."""
        try:
            from ..core.task_queue import task_queue
            task_id = await task_queue.submit_task(task)
            logger.info(
                "Improvement task submitted", 
                agent=self.name, 
                task_id=task_id,
                title=task.title
            )
            return task_id
        except Exception as e:
            logger.error(
                "Failed to submit improvement task", 
                agent=self.name, 
                error=str(e)
            )
            return None

    async def _analyze_task_coordination_needs(self, task: Task) -> Dict[str, Any]:
        """Analyze if a task requires coordination and what capabilities are needed."""
        coordination_keywords = [
            "complex", "multiple", "system", "architecture", "integration",
            "full-stack", "end-to-end", "comprehensive"
        ]
        
        # Check if task requires coordination based on keywords
        requires_coordination = any(
            keyword in task.title.lower() or keyword in task.description.lower()
            for keyword in coordination_keywords
        )
        
        # Determine required capabilities based on task type
        capability_map = {
            "development": ["coding", "testing"],
            "architecture": ["system_design", "documentation"],
            "testing": ["quality_assurance", "validation"],
            "deployment": ["devops", "monitoring"],
            "analysis": ["system_analysis", "research"]
        }
        
        required_capabilities = capability_map.get(task.task_type, ["general"])
        min_agents = 2 if requires_coordination else 1
        
        return {
            "requires_coordination": requires_coordination,
            "required_capabilities": required_capabilities,
            "min_agents": min_agents,
            "complexity_score": len([kw for kw in coordination_keywords 
                                   if kw in task.description.lower()])
        }

    async def _find_suitable_agents(self, required_capabilities: List[str]) -> List[str]:
        """Find agents with suitable capabilities for coordination."""
        try:
            if not self.async_db_manager:
                return []
            
            active_agents = await self.async_db_manager.get_active_agents()
            suitable_agents = []
            
            for agent in active_agents:
                if agent.name == self.name:  # Skip self
                    continue
                
                if agent.status in ["active", "idle"]:
                    # Check if agent has any of the required capabilities
                    agent_capabilities = agent.capabilities or {}
                    agent_type = agent.type
                    
                    # Simple capability matching based on agent type
                    type_capabilities = {
                        "architect": ["system_design", "architecture", "documentation"],
                        "developer": ["coding", "implementation", "testing"],
                        "qa": ["quality_assurance", "testing", "validation"],
                        "devops": ["deployment", "monitoring", "infrastructure"]
                    }
                    
                    agent_caps = type_capabilities.get(agent_type, [])
                    if any(cap in agent_caps for cap in required_capabilities):
                        suitable_agents.append(agent.name)
            
            return suitable_agents
            
        except Exception as e:
            logger.error("Failed to find suitable agents", error=str(e))
            return []

    async def _create_coordination_session(
        self, 
        task: Task, 
        agents: List[str], 
        coordination_plan: Dict[str, Any]
    ) -> str | None:
        """Create a coordination session for multiple agents."""
        try:
            import uuid
            session_id = f"coord-{uuid.uuid4().hex[:8]}"
            
            # In a full implementation, this would:
            # 1. Create shared workspace
            # 2. Set up communication channels
            # 3. Distribute subtasks
            # 4. Initialize monitoring
            
            logger.info(
                "Coordination session created",
                agent=self.name,
                session_id=session_id,
                task_id=task.id,
                coordinated_agents=agents
            )
            
            return session_id
            
        except Exception as e:
            logger.error("Failed to create coordination session", error=str(e))
            return None

    # Performance monitoring helper methods
    async def _collect_agent_performance_metrics(self) -> Dict[str, float]:
        """Collect agent performance metrics."""
        try:
            metrics = {}
            
            if self.async_db_manager:
                # Get recent task completion data
                metrics["avg_response_time"] = 1.5  # Mock data
                metrics["agent_utilization"] = 0.75
                metrics["avg_task_duration"] = 45.0
            
            return metrics
        except Exception as e:
            logger.error("Failed to collect agent performance metrics", error=str(e))
            return {}

    async def _collect_task_processing_metrics(self) -> Dict[str, float]:
        """Collect task processing metrics."""
        try:
            from ..core.task_queue import task_queue
            
            stats = await task_queue.get_queue_stats()
            
            return {
                "tasks_per_minute": stats.processing_rate * 60,
                "avg_task_duration": stats.average_wait_time,
                "queue_backlog": sum(stats.queue_size_by_priority.values()),
                "success_rate": (stats.completed_tasks / 
                               max(1, stats.completed_tasks + stats.failed_tasks))
            }
        except Exception as e:
            logger.error("Failed to collect task processing metrics", error=str(e))
            return {}

    async def _collect_resource_metrics(self) -> Dict[str, float]:
        """Collect system resource metrics."""
        try:
            import psutil
            
            return {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent / 100,
                "disk_usage": psutil.disk_usage('/').percent / 100
            }
        except Exception as e:
            logger.error("Failed to collect resource metrics", error=str(e))
            return {
                "cpu_usage": 0.5,
                "memory_usage": 0.4,
                "disk_usage": 0.3
            }

    async def _collect_communication_metrics(self) -> Dict[str, float]:
        """Collect communication performance metrics."""
        try:
            # Mock implementation - in production would check Redis/message broker
            return {
                "message_latency": 0.1,
                "message_throughput": 50.0,
                "connection_errors": 0
            }
        except Exception as e:
            logger.error("Failed to collect communication metrics", error=str(e))
            return {}

    async def _identify_performance_bottlenecks(
        self, 
        metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks from metrics."""
        bottlenecks = []
        
        # Check CPU usage
        cpu_usage = metrics.get("cpu_usage", 0)
        if cpu_usage > 80:
            bottlenecks.append({
                "component": "cpu",
                "severity": "critical" if cpu_usage > 95 else "high",
                "value": cpu_usage,
                "description": f"High CPU usage: {cpu_usage}%"
            })
        
        # Check memory usage
        memory_usage = metrics.get("memory_usage", 0)
        if memory_usage > 80:
            bottlenecks.append({
                "component": "memory",
                "severity": "critical" if memory_usage > 95 else "high",
                "value": memory_usage,
                "description": f"High memory usage: {memory_usage}%"
            })
        
        # Check task processing
        success_rate = metrics.get("success_rate", 1.0)
        if success_rate < 0.8:
            bottlenecks.append({
                "component": "task_processing",
                "severity": "medium",
                "value": success_rate,
                "description": f"Low task success rate: {success_rate:.1%}"
            })
        
        # Check queue backlog
        queue_backlog = metrics.get("queue_backlog", 0)
        if queue_backlog > 100:
            bottlenecks.append({
                "component": "task_queue",
                "severity": "medium",
                "value": queue_backlog,
                "description": f"Large queue backlog: {queue_backlog} tasks"
            })
        
        return bottlenecks

    async def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        # Mock implementation - in production would analyze historical data
        return {
            "trend": "stable",
            "confidence": 0.8,
            "prediction": "performance_maintained"
        }

    async def _create_performance_improvement_task(self, bottleneck: Dict[str, Any]) -> None:
        """Create a performance improvement task for a bottleneck."""
        try:
            component = bottleneck["component"]
            severity = bottleneck["severity"]
            
            task_title = f"Resolve {severity.title()} {component.title()} Bottleneck"
            task_description = bottleneck["description"]
            
            task = await self._create_task(
                title=task_title,
                description=task_description,
                task_type="performance_optimization",
                priority=TaskPriority.HIGH if severity == "critical" else TaskPriority.MEDIUM
            )
            
            await self._submit_improvement_task(task)
            
        except Exception as e:
            logger.error("Failed to create performance improvement task", error=str(e))

    async def _find_complex_tasks(self) -> List[Task]:
        """Find pending tasks that could benefit from coordination."""
        try:
            from ..core.task_queue import task_queue
            
            # Get pending high-priority tasks
            pending_tasks = await task_queue.get_pending_tasks()
            
            complex_tasks = []
            for task in pending_tasks:
                if task.priority <= TaskPriority.HIGH:
                    coordination_needs = await self._analyze_task_coordination_needs(task)
                    if coordination_needs["requires_coordination"]:
                        complex_tasks.append(task)
            
            return complex_tasks
            
        except Exception as e:
            logger.error("Failed to find complex tasks", error=str(e))
            return []

    async def _review_active_coordination_sessions(self) -> None:
        """Review and manage active coordination sessions."""
        try:
            current_time = time.time()
            expired_sessions = []
            
            for session_id, session_info in self.coordination_active.items():
                # Check if session has been running too long
                duration = current_time - session_info["started_at"]
                if duration > 3600:  # 1 hour timeout
                    expired_sessions.append(session_id)
                    logger.warning(
                        "Coordination session expired",
                        agent=self.name,
                        session_id=session_id,
                        duration=duration
                    )
            
            # Clean up expired sessions
            for session_id in expired_sessions:
                del self.coordination_active[session_id]
                
        except Exception as e:
            logger.error("Failed to review coordination sessions", error=str(e))

    async def _execute_improvement_task(self, improvement_task: Dict[str, Any]) -> None:
        """Execute an improvement task from the pipeline."""
        try:
            # Mock implementation - would execute actual improvement
            logger.info(
                "Executing improvement task",
                agent=self.name,
                task=improvement_task.get("title", "unknown")
            )
            
            # Simulate task execution
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error("Failed to execute improvement task", error=str(e))

    async def _handle_pending_collaborations(self) -> None:
        """Handle pending collaboration requests."""
        try:
            # Check for collaboration opportunities
            opportunities = await self.get_collaboration_opportunities()
            
            for opportunity in opportunities:
                if opportunity.get("status") == "pending":
                    # Evaluate and potentially join collaboration
                    agent_name = opportunity.get("agent_name")
                    logger.debug(
                        "Evaluating collaboration opportunity",
                        agent=self.name,
                        opportunity_agent=agent_name
                    )
            
        except Exception as e:
            logger.error("Failed to handle pending collaborations", error=str(e))

    @property
    def capabilities(self) -> List[str]:
        """Get MetaAgent capabilities."""
        base_caps = super().capabilities
        return base_caps + self.meta_capabilities