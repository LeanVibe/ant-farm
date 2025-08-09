"""Meta-Agent for LeanVibe Agent Hive 2.0 - Self-improvement and system coordination."""

import asyncio
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

# Handle both module and direct execution imports
try:
    from ..core.config import settings
    from ..core.constants import Intervals
    from ..core.context_engine import get_context_engine
    from ..core.message_broker import MessageType, message_broker
    from ..core.models import Agent as AgentModel
    from ..core.self_modifier import SelfModifier
    from ..core.task_queue import Task, task_queue
    from .base_agent import BaseAgent, TaskResult
except ImportError:
    # Direct execution - add src to path
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))
    from agents.base_agent import BaseAgent, TaskResult
    from core.config import settings
    from core.context_engine import get_context_engine
    from core.message_broker import MessageType, message_broker
    from core.models import Agent as AgentModel
    from core.self_modifier import SelfModifier
    from core.task_queue import Task, task_queue

logger = structlog.get_logger()


class ImprovementType(Enum):
    """Types of system improvements."""

    BUG_FIX = "bug_fix"
    FEATURE_ENHANCEMENT = "feature_enhancement"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    CODE_REFACTORING = "code_refactoring"
    ARCHITECTURE_IMPROVEMENT = "architecture_improvement"
    NEW_CAPABILITY = "new_capability"
    AGENT_OPTIMIZATION = "agent_optimization"


@dataclass
class ImprovementProposal:
    """A proposed improvement to the system."""

    id: str
    type: ImprovementType
    title: str
    description: str
    impact_score: float  # 0.0 to 1.0
    complexity_score: float  # 0.0 to 1.0
    risk_score: float  # 0.0 to 1.0
    affected_components: list[str]
    implementation_steps: list[str]
    success_metrics: list[str]
    created_at: float


@dataclass
class SystemAnalysis:
    """Analysis of current system state."""

    health_score: float
    performance_metrics: dict[str, float]
    agent_efficiency: dict[str, float]
    code_quality_score: float
    identified_issues: list[str]
    improvement_opportunities: list[str]
    resource_utilization: dict[str, float]


class MetaAgent(BaseAgent):
    """Meta-Agent responsible for self-improvement and system coordination."""

    def __init__(self, name: str = "meta-agent"):
        super().__init__(name, "meta", "system_coordinator")

        # Meta-agent specific state
        self.improvement_proposals: dict[str, ImprovementProposal] = {}
        self.active_improvements: set[str] = set()
        self.completed_improvements: set[str] = set()
        self.failed_improvements: set[str] = set()

        # Analysis state
        self.last_system_analysis: SystemAnalysis | None = None
        self.analysis_interval = 300  # 5 minutes
        self.last_analysis_time = 0.0

        # Performance optimization: Cache for active agents
        self._active_agents_cache: list[AgentModel] | None = None
        self._cache_timestamp = 0
        self._cache_ttl = 60  # Cache for 60 seconds

        # Self-modification safety
        self.modification_enabled = settings.enable_self_modification
        self.max_concurrent_improvements = 3
        self.backup_created = False

        # Initialize core components for self-improvement
        self.context_engine = None
        self.self_modifier = None

        logger.info(
            "Meta-Agent initialized",
            name=self.name,
            modification_enabled=self.modification_enabled,
        )

    async def initialize(self):
        """Initialize MetaAgent with required components."""
        # Initialize context engine
        from ..core.config import get_settings

        settings = get_settings()
        self.context_engine = await get_context_engine(settings.database_url)

        # Initialize self modifier with workspace path
        self.self_modifier = SelfModifier(workspace_path=str(settings.project_root))

        logger.info("MetaAgent components initialized", name=self.name)

    async def _get_active_agents_cached(self) -> list[AgentModel]:
        """Get active agents with caching to improve performance."""
        current_time = time.time()

        # Check if cache is valid
        if (
            self._active_agents_cache is not None
            and current_time - self._cache_timestamp < self._cache_ttl
        ):
            return self._active_agents_cache

        # Cache is invalid, fetch from database
        try:
            # Use async database manager if available
            if self.async_db_manager:
                # Query active agents with performance monitoring
                query_start = time.time()
                agents = await self.async_db_manager.get_active_agents()
                query_time = time.time() - query_start

                if query_time > 1.0:  # Log slow queries
                    logger.warning("Slow agent query detected", query_time=query_time)

                # Update cache
                self._active_agents_cache = agents
                self._cache_timestamp = current_time

                return agents
            else:
                # Fallback - return empty list if no database manager
                logger.warning(
                    "No async database manager available for fetching agents"
                )
                return []

        except Exception as e:
            logger.error("Failed to fetch active agents", error=str(e))
            # Return stale cache if available as fallback
            if self._active_agents_cache is not None:
                logger.warning("Using stale agent cache due to database error")
                return self._active_agents_cache
            raise

    def _invalidate_agents_cache(self) -> None:
        """Invalidate the active agents cache."""
        self._active_agents_cache = None
        self._cache_timestamp = 0

    async def run(self) -> None:
        """Main meta-agent execution loop."""
        logger.info("Meta-Agent starting main loop", agent=self.name)

        # Set last analysis time to now + 60 seconds to delay initial analysis
        self.last_analysis_time = time.time() + 60
        logger.info(
            "Meta-Agent initial analysis delayed by 60 seconds for stability",
            agent=self.name,
        )

        while self.status == "active":
            try:
                # Periodic system analysis
                if time.time() - self.last_analysis_time > self.analysis_interval:
                    logger.info("Starting periodic system analysis", agent=self.name)
                    await self._perform_system_analysis()
                    self.last_analysis_time = time.time()

                # Process any assigned tasks
                await self._process_pending_tasks()

                # Review and execute improvement proposals
                await self._execute_improvements()

                # Coordinate with other agents
                await self._coordinate_agents()

                # Brief pause before next iteration
                await asyncio.sleep(Intervals.META_AGENT_CYCLE)

            except Exception as e:
                logger.error(
                    "Meta-agent error in main loop", agent=self.name, error=str(e)
                )
                await asyncio.sleep(
                    Intervals.META_AGENT_ERROR_DELAY
                )  # Longer pause on error

    async def _perform_system_analysis(self) -> SystemAnalysis:
        """Perform comprehensive system analysis."""
        logger.info("Performing system analysis", agent=self.name)

        # Analyze system health
        health_score = await self._calculate_system_health()

        # Get performance metrics
        performance_metrics = await self._gather_performance_metrics()

        # Analyze agent efficiency
        agent_efficiency = await self._analyze_agent_efficiency()

        # Analyze code quality
        code_quality_score = await self._analyze_code_quality()

        # Identify issues and opportunities
        issues = await self._identify_system_issues()
        opportunities = await self._identify_improvement_opportunities()

        # Resource utilization
        resource_utilization = await self._analyze_resource_utilization()

        analysis = SystemAnalysis(
            health_score=health_score,
            performance_metrics=performance_metrics,
            agent_efficiency=agent_efficiency,
            code_quality_score=code_quality_score,
            identified_issues=issues,
            improvement_opportunities=opportunities,
            resource_utilization=resource_utilization,
        )

        self.last_system_analysis = analysis

        # Store analysis in context
        await self.store_context(
            content=f"System Analysis Results:\nHealth: {health_score:.2f}\nIssues: {len(issues)}\nOpportunities: {len(opportunities)}",
            importance_score=0.9,
            category="system_analysis",
            metadata=analysis.__dict__,
        )

        # Generate improvement proposals based on analysis
        await self._generate_improvement_proposals(analysis)

        return analysis

    async def _calculate_system_health(self) -> float:
        """Calculate overall system health score."""
        try:
            # Get all active agents using cached method
            agents = await self._get_active_agents_cached()

            if not agents:
                return 0.0

            # Calculate health score based on various factors
            healthy_agents = 0
            total_agents = len(agents)

            for agent in agents:
                # Check if agent is responsive (simplified)
                if (
                    agent.last_heartbeat
                    and (datetime.now(UTC) - agent.last_heartbeat).total_seconds() < 300
                ):
                    healthy_agents += 1

            agent_health = healthy_agents / total_agents if total_agents > 0 else 0.0

            # Check infrastructure health
            infrastructure_health = 1.0  # Simplified - would check Redis, DB, etc.

            # Overall health score
            return agent_health * 0.7 + infrastructure_health * 0.3

        except Exception as e:
            logger.error("Failed to calculate system health", error=str(e))
            return 0.0

    async def _gather_performance_metrics(self) -> dict[str, float]:
        """Gather system performance metrics."""
        metrics = {}

        try:
            # Task completion rate
            total_tasks = await task_queue.get_total_tasks()
            completed_tasks = await task_queue.get_completed_tasks_count()

            if total_tasks > 0:
                metrics["task_completion_rate"] = completed_tasks / total_tasks
            else:
                metrics["task_completion_rate"] = 0.0

            # Average task execution time
            metrics["avg_task_execution_time"] = await self._get_avg_execution_time()

            # Agent utilization
            metrics["agent_utilization"] = await self._calculate_agent_utilization()

            # Queue depth
            metrics["queue_depth"] = await task_queue.get_queue_depth()

        except Exception as e:
            logger.error("Failed to gather performance metrics", error=str(e))

        return metrics

    async def _analyze_agent_efficiency(self) -> dict[str, float]:
        """Analyze individual agent efficiency."""
        efficiency = {}

        try:
            agents = await self._get_active_agents_cached()

            for agent in agents:
                # Calculate efficiency based on task completion ratio
                # This is simplified - would use more sophisticated metrics
                if hasattr(agent, "tasks_completed") and hasattr(
                    agent, "tasks_assigned"
                ):
                    if agent.tasks_assigned > 0:
                        efficiency[agent.name] = (
                            agent.tasks_completed / agent.tasks_assigned
                        )
                    else:
                        efficiency[agent.name] = 0.0
                else:
                    efficiency[agent.name] = 0.5  # Default efficiency

        except Exception as e:
            logger.error("Failed to analyze agent efficiency", error=str(e))

        return efficiency

    async def _analyze_code_quality(self) -> float:
        """Analyze overall code quality."""
        try:
            # Run code quality analysis using CLI tools
            prompt = """
            Analyze the codebase in the current directory for:
            1. Code quality and maintainability
            2. Test coverage
            3. Documentation completeness
            4. Architecture adherence
            5. Performance bottlenecks

            Provide a score from 0.0 to 1.0 for overall code quality.
            """

            result = await self.execute_with_cli_tool(prompt)

            if result.success:
                # Extract quality score from output (simplified)
                # In practice, would parse structured output
                score = 0.7  # Default score

                # Store code quality analysis
                await self.store_context(
                    content=f"Code Quality Analysis:\n{result.output}",
                    importance_score=0.8,
                    category="code_quality",
                    metadata={"quality_score": score},
                )

                return score
            else:
                return 0.5  # Default score on analysis failure

        except Exception as e:
            logger.error("Failed to analyze code quality", error=str(e))
            return 0.5

    async def _identify_system_issues(self) -> list[str]:
        """Identify current system issues."""
        issues = []

        try:
            # Check for failed tasks
            failed_tasks = await task_queue.get_failed_tasks()
            if failed_tasks:
                issues.append(f"Failed tasks detected: {len(failed_tasks)}")

            # Check for unresponsive agents
            agents = await self._get_active_agents_cached()
            unresponsive_agents = [
                agent.name
                for agent in agents
                if agent.last_heartbeat
                and (datetime.now(UTC) - agent.last_heartbeat).total_seconds() > 300
            ]

            if unresponsive_agents:
                issues.append(f"Unresponsive agents: {', '.join(unresponsive_agents)}")

            # Check queue depth
            queue_depth = await task_queue.get_queue_depth()
            if queue_depth > 100:
                issues.append(f"High queue depth: {queue_depth}")

        except Exception as e:
            logger.error("Failed to identify system issues", error=str(e))
            issues.append(f"Analysis error: {str(e)}")

        return issues

    async def _identify_improvement_opportunities(self) -> list[str]:
        """Identify opportunities for system improvement."""
        opportunities = []

        try:
            # Analyze task patterns for optimization opportunities
            prompt = """
            Analyze the system logs and task patterns to identify:
            1. Repetitive tasks that could be automated
            2. Performance bottlenecks
            3. Agent coordination improvements
            4. New capabilities that would be beneficial
            5. Architecture optimizations

            List the top 5 improvement opportunities.
            """

            result = await self.execute_with_cli_tool(prompt)

            if result.success:
                # Parse opportunities from output (simplified)
                lines = result.output.split("\n")
                for line in lines:
                    if line.strip() and (line.startswith("-") or line.startswith("*")):
                        opportunities.append(line.strip())

            # Add some built-in opportunities based on metrics
            if self.last_system_analysis:
                if self.last_system_analysis.health_score < 0.8:
                    opportunities.append(
                        "Improve system reliability and health monitoring"
                    )

                if any(
                    eff < 0.6
                    for eff in self.last_system_analysis.agent_efficiency.values()
                ):
                    opportunities.append("Optimize low-performing agents")

        except Exception as e:
            logger.error("Failed to identify improvement opportunities", error=str(e))

        return opportunities

    async def _analyze_resource_utilization(self) -> dict[str, float]:
        """Analyze system resource utilization."""
        utilization = {}

        try:
            # Get basic resource metrics (simplified)
            utilization["cpu_usage"] = 0.5  # Would get actual CPU usage
            utilization["memory_usage"] = 0.6  # Would get actual memory usage
            utilization["agent_capacity"] = await self._calculate_agent_capacity()
            utilization["queue_utilization"] = min(
                1.0, (await task_queue.get_queue_depth()) / 1000
            )

        except Exception as e:
            logger.error("Failed to analyze resource utilization", error=str(e))

        return utilization

    async def _generate_improvement_proposals(self, analysis: SystemAnalysis) -> None:
        """Generate improvement proposals based on system analysis."""

        # Generate proposals for identified issues
        for issue in analysis.identified_issues:
            proposal = await self._create_improvement_proposal_for_issue(issue)
            if proposal:
                self.improvement_proposals[proposal.id] = proposal

        # Generate proposals for opportunities
        for opportunity in analysis.improvement_opportunities:
            proposal = await self._create_improvement_proposal_for_opportunity(
                opportunity
            )
            if proposal:
                self.improvement_proposals[proposal.id] = proposal

        # Generate performance-based proposals
        if analysis.health_score < 0.7:
            proposal = self._create_health_improvement_proposal(analysis)
            self.improvement_proposals[proposal.id] = proposal

        logger.info(
            "Generated improvement proposals",
            agent=self.name,
            proposal_count=len(self.improvement_proposals),
        )

    async def _create_improvement_proposal_for_issue(
        self, issue: str
    ) -> ImprovementProposal | None:
        """Create improvement proposal to address a specific issue."""
        try:
            prompt = f"""
            Create an improvement proposal to address this system issue:
            Issue: {issue}

            Provide:
            1. A clear title for the improvement
            2. Detailed description of the solution
            3. Implementation steps
            4. Success metrics
            5. Risk assessment (low/medium/high)
            6. Complexity assessment (low/medium/high)
            7. Impact assessment (low/medium/high)

            Format as JSON.
            """

            result = await self.execute_with_cli_tool(prompt)

            if result.success:
                # Parse the proposal (simplified - would use proper JSON parsing)
                return ImprovementProposal(
                    id=str(uuid.uuid4()),
                    type=ImprovementType.BUG_FIX,
                    title=f"Fix: {issue[:50]}...",
                    description=f"Address system issue: {issue}",
                    impact_score=0.7,
                    complexity_score=0.5,
                    risk_score=0.3,
                    affected_components=["system"],
                    implementation_steps=[
                        "Analyze issue",
                        "Implement fix",
                        "Test solution",
                    ],
                    success_metrics=["Issue resolved", "No regression"],
                    created_at=time.time(),
                )

        except Exception as e:
            logger.error(
                "Failed to create improvement proposal for issue",
                issue=issue,
                error=str(e),
            )

        return None

    async def _create_improvement_proposal_for_opportunity(
        self, opportunity: str
    ) -> ImprovementProposal | None:
        """Create improvement proposal for an identified opportunity."""
        try:
            return ImprovementProposal(
                id=str(uuid.uuid4()),
                type=ImprovementType.FEATURE_ENHANCEMENT,
                title=f"Enhancement: {opportunity[:50]}...",
                description=f"Implement improvement opportunity: {opportunity}",
                impact_score=0.6,
                complexity_score=0.6,
                risk_score=0.4,
                affected_components=["system"],
                implementation_steps=[
                    "Design enhancement",
                    "Implement",
                    "Test",
                    "Deploy",
                ],
                success_metrics=["Feature implemented", "Performance improved"],
                created_at=time.time(),
            )

        except Exception as e:
            logger.error(
                "Failed to create improvement proposal for opportunity",
                opportunity=opportunity,
                error=str(e),
            )

        return None

    def _create_health_improvement_proposal(
        self, analysis: SystemAnalysis
    ) -> ImprovementProposal:
        """Create proposal to improve system health."""
        return ImprovementProposal(
            id=str(uuid.uuid4()),
            type=ImprovementType.ARCHITECTURE_IMPROVEMENT,
            title="Improve System Health and Reliability",
            description=f"Address low system health score ({analysis.health_score:.2f})",
            impact_score=0.9,
            complexity_score=0.7,
            risk_score=0.5,
            affected_components=["orchestrator", "agents", "monitoring"],
            implementation_steps=[
                "Improve health monitoring",
                "Add agent recovery mechanisms",
                "Enhance error handling",
                "Implement redundancy",
            ],
            success_metrics=[
                "Health score > 0.8",
                "Reduced downtime",
                "Faster error recovery",
            ],
            created_at=time.time(),
        )

    async def _process_pending_tasks(self) -> None:
        """Process any tasks assigned to the meta-agent."""
        try:
            task = await task_queue.get_task(self.name)
            if task:
                logger.info(
                    "MetaAgent processing task",
                    task_id=task.id,
                    task_type=task.task_type,
                )

                # This is the core self-improvement workflow from PLAN.md
                if task.task_type in [
                    "refactor",
                    "self_improvement",
                    "code_modification",
                ]:
                    await self._process_self_improvement_task(task)
                else:
                    await self.process_task(task)
        except Exception as e:
            logger.error(
                "Failed to process pending tasks", agent=self.name, error=str(e)
            )

    async def _process_self_improvement_task(self, task: Task) -> None:
        """Process self-improvement tasks using the ContextEngine and SelfModifier workflow."""
        logger.info(
            "Processing self-improvement task",
            task_id=task.id,
            description=task.description,
        )

        try:
            # Step 1: Use ContextEngine to retrieve relevant context
            context_results = await self.context_engine.retrieve_context(
                query=task.description,
                agent_id=self.agent_uuid,
                limit=10,
                min_importance=0.4,
            )

            logger.info("Retrieved context", results_count=len(context_results))

            # Step 2: Build detailed prompt with context
            context_text = "\n\n".join(
                [
                    f"File: {r.context.metadata.get('file_path', 'unknown')}\n{r.context.content[:500]}..."
                    for r in context_results[:5]  # Top 5 most relevant
                ]
            )

            # Step 3: Determine which files need modification
            file_paths = []
            for result in context_results:
                if result.context.metadata and "file_path" in result.context.metadata:
                    file_path = result.context.metadata["file_path"]
                    if file_path not in file_paths:
                        file_paths.append(file_path)

            # Step 4: Use SelfModifier to propose and apply changes
            if file_paths:
                # For now, work on the most relevant file
                target_file = file_paths[0]

                change_description = f"""
                Task: {task.description}

                Relevant Context:
                {context_text}

                Please implement the requested changes to {target_file}.
                """

                # Step 5: Execute the self-modification
                success = await self.self_modifier.propose_and_apply_change(
                    file_path=target_file, change_description=change_description
                )

                if success:
                    # Step 6: Mark task as completed
                    await task_queue.complete_task(
                        task.id,
                        {
                            "success": True,
                            "modified_file": target_file,
                            "description": "Self-modification completed successfully",
                        },
                    )

                    # Store the successful modification in context
                    await self.context_engine.store_context(
                        agent_id=self.name,
                        content=f"Successfully completed self-improvement task: {task.description}. Modified file: {target_file}",
                        importance_score=0.9,
                        category="self_improvement",
                        metadata={
                            "task_id": str(task.id),
                            "modified_file": target_file,
                            "success": True,
                        },
                    )

                    logger.info(
                        "Self-improvement task completed successfully",
                        task_id=task.id,
                        file=target_file,
                    )
                else:
                    # Mark task as failed
                    await task_queue.fail_task(
                        task.id, "Self-modification failed validation"
                    )
                    logger.error("Self-improvement task failed", task_id=task.id)

            else:
                # No relevant files found
                await task_queue.fail_task(
                    task.id, "No relevant files found for modification"
                )
                logger.warning(
                    "No files found for self-improvement task", task_id=task.id
                )

        except Exception as e:
            logger.error(
                "Self-improvement task processing failed", task_id=task.id, error=str(e)
            )
            await task_queue.fail_task(task.id, f"Processing error: {str(e)}")
            raise

    async def _execute_improvements(self) -> None:
        """Execute approved improvement proposals."""
        if not self.modification_enabled:
            return

        # Limit concurrent improvements
        if len(self.active_improvements) >= self.max_concurrent_improvements:
            return

        # Select highest impact, lowest risk proposals
        available_proposals = [
            p
            for p in self.improvement_proposals.values()
            if p.id not in self.active_improvements
            and p.id not in self.completed_improvements
        ]

        if not available_proposals:
            return

        # Sort by impact/risk ratio
        available_proposals.sort(
            key=lambda p: p.impact_score / max(p.risk_score, 0.1), reverse=True
        )

        # Execute top proposal
        proposal = available_proposals[0]
        await self._execute_improvement_proposal(proposal)

    async def _execute_improvement_proposal(
        self, proposal: ImprovementProposal
    ) -> None:
        """Execute a specific improvement proposal."""
        logger.info(
            "Executing improvement proposal",
            agent=self.name,
            proposal_id=proposal.id,
            title=proposal.title,
        )

        self.active_improvements.add(proposal.id)

        try:
            # Create backup if not already done
            if not self.backup_created:
                await self._create_system_backup()
                self.backup_created = True

            # Execute implementation steps
            for step in proposal.implementation_steps:
                await self._execute_implementation_step(proposal, step)

            # Validate success metrics
            success = await self._validate_improvement_success(proposal)

            if success:
                self.completed_improvements.add(proposal.id)
                logger.info(
                    "Improvement proposal completed successfully",
                    proposal_id=proposal.id,
                )
            else:
                self.failed_improvements.add(proposal.id)
                await self._rollback_improvement(proposal)
                logger.warning(
                    "Improvement proposal failed validation", proposal_id=proposal.id
                )

        except Exception as e:
            self.failed_improvements.add(proposal.id)
            logger.error(
                "Improvement proposal execution failed",
                proposal_id=proposal.id,
                error=str(e),
            )
            await self._rollback_improvement(proposal)

        finally:
            self.active_improvements.discard(proposal.id)

    async def _execute_implementation_step(
        self, proposal: ImprovementProposal, step: str
    ) -> None:
        """Execute a single implementation step."""
        prompt = f"""
        Execute this implementation step for the improvement proposal:

        Proposal: {proposal.title}
        Description: {proposal.description}
        Step: {step}

        Affected Components: {", ".join(proposal.affected_components)}

        Please implement this step carefully, ensuring:
        1. Code quality and testing
        2. Backwards compatibility
        3. Proper error handling
        4. Documentation updates

        Make the necessary changes to the codebase.
        """

        result = await self.execute_with_cli_tool(prompt)

        if not result.success:
            raise Exception(f"Implementation step failed: {result.error}")

        # Store implementation context
        await self.store_context(
            content=f"Implemented step: {step}\nResult: {result.output[:500]}...",
            importance_score=0.8,
            category="improvement_implementation",
            metadata={
                "proposal_id": proposal.id,
                "step": step,
                "success": result.success,
            },
        )

    async def _validate_improvement_success(
        self, proposal: ImprovementProposal
    ) -> bool:
        """Validate that improvement proposal succeeded."""
        try:
            # Run validation for each success metric
            for metric in proposal.success_metrics:
                prompt = f"""
                Validate this success metric for the implemented improvement:

                Proposal: {proposal.title}
                Metric: {metric}

                Check if this metric has been achieved and return true/false.
                """

                result = await self.execute_with_cli_tool(prompt)

                if result.success and "true" not in result.output.lower():
                    return False

            return True

        except Exception as e:
            logger.error(
                "Failed to validate improvement success",
                proposal_id=proposal.id,
                error=str(e),
            )
            return False

    async def _rollback_improvement(self, proposal: ImprovementProposal) -> None:
        """Rollback a failed improvement."""
        logger.info("Rolling back improvement proposal", proposal_id=proposal.id)

        prompt = f"""
        Rollback the changes made for this improvement proposal:

        Proposal: {proposal.title}
        Description: {proposal.description}

        Restore the system to its previous state before this improvement was attempted.
        Use git or other version control mechanisms as appropriate.
        """

        result = await self.execute_with_cli_tool(prompt)

        if result.success:
            logger.info("Improvement rollback completed", proposal_id=proposal.id)
        else:
            logger.error(
                "Improvement rollback failed",
                proposal_id=proposal.id,
                error=result.error,
            )

    async def _create_system_backup(self) -> None:
        """Create a backup of the current system state."""
        try:
            prompt = """
            Create a backup of the current system state:
            1. Commit all current changes to git
            2. Create a backup branch
            3. Tag the current state for easy recovery

            This backup will be used for rollback if improvements fail.
            """

            result = await self.execute_with_cli_tool(prompt)

            if result.success:
                logger.info("System backup created successfully")
            else:
                logger.error("Failed to create system backup", error=result.error)

        except Exception as e:
            logger.error("Exception creating system backup", error=str(e))

    async def _coordinate_agents(self) -> None:
        """Coordinate with other agents in the system."""
        try:
            # Send health check to all agents
            agents = await self._get_active_agents()

            for agent_name in agents:
                if agent_name != self.name:
                    await self.send_message(
                        to_agent=agent_name,
                        topic="health_check",
                        content={"check_time": time.time()},
                        message_type=MessageType.DIRECT,
                    )

            # Broadcast system status
            if self.last_system_analysis:
                await message_broker.broadcast_message(
                    from_agent=self.name,
                    topic="system_status",
                    payload={
                        "health_score": self.last_system_analysis.health_score,
                        "active_improvements": len(self.active_improvements),
                        "pending_proposals": len(self.improvement_proposals),
                        "timestamp": time.time(),
                    },
                )

        except Exception as e:
            logger.error("Failed to coordinate with agents", error=str(e))

    async def _get_active_agents(self) -> list[str]:
        """Get list of active agent names."""
        try:
            agents = await self._get_active_agents_cached()
            agent_names = [agent.name for agent in agents]
            return agent_names
        except Exception as e:
            logger.error("Failed to get active agents", error=str(e))
            return []

    async def _get_avg_execution_time(self) -> float:
        """Get average task execution time."""
        # Simplified implementation
        return 30.0  # Would calculate from actual metrics

    async def _calculate_agent_utilization(self) -> float:
        """Calculate overall agent utilization."""
        # Simplified implementation
        return 0.7  # Would calculate from actual metrics

    async def _calculate_agent_capacity(self) -> float:
        """Calculate agent capacity utilization."""
        try:
            active_agents = len(await self._get_active_agents())
            max_agents = 10  # Would get from configuration
            return min(1.0, active_agents / max_agents)
        except Exception:
            return 0.5

    async def _process_task_implementation(self, task: Task) -> TaskResult:
        """Process meta-agent specific tasks."""

        if task.task_type == "system_analysis":
            analysis = await self._perform_system_analysis()
            return TaskResult(
                success=True,
                data=analysis.__dict__,
                metrics={"analysis_duration": time.time() - self.last_analysis_time},
            )

        elif task.task_type == "improvement_proposal":
            # Create improvement proposal from task description
            proposal = ImprovementProposal(
                id=str(uuid.uuid4()),
                type=ImprovementType.FEATURE_ENHANCEMENT,
                title=task.title,
                description=task.description,
                impact_score=0.6,
                complexity_score=0.5,
                risk_score=0.4,
                affected_components=["system"],
                implementation_steps=["Analyze", "Implement", "Test"],
                success_metrics=["Feature working", "No regressions"],
                created_at=time.time(),
            )

            self.improvement_proposals[proposal.id] = proposal

            return TaskResult(
                success=True,
                data={"proposal_id": proposal.id},
                metrics={"proposal_count": len(self.improvement_proposals)},
            )

        elif task.task_type == "agent_coordination":
            await self._coordinate_agents()
            return TaskResult(
                success=True,
                data={"coordinated_agents": len(await self._get_active_agents())},
                metrics={"coordination_time": time.time()},
            )

        elif task.task_type in ["refactor", "self_improvement", "code_modification"]:
            # This is handled by _process_self_improvement_task
            # Return a placeholder result since the task is processed asynchronously
            return TaskResult(
                success=True,
                data={"message": "Self-improvement task initiated"},
                metrics={"task_type": task.task_type},
            )

        else:
            # Use base implementation for other task types
            return await super()._process_task_implementation(task)

    async def _on_collaboration_completed(self, result: dict[str, Any]) -> None:
        """Called when a collaboration is completed."""
        logger.info("Collaboration completed", result=result)
        # Meta-agent can learn from collaboration outcomes

    async def _on_collaboration_failed(self, failure_info: dict[str, Any]) -> None:
        """Called when a collaboration fails."""
        logger.warning("Collaboration failed", failure_info=failure_info)
        # Meta-agent can analyze failures for system improvement
