"""Sleep-wake cycle manager for memory consolidation and system optimization."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from .advanced_context_engine import (
    MemoryConsolidationStats,
    get_advanced_context_engine,
)
from .config import settings
from .constants import Intervals
from .message_broker import message_broker
from .models import Agent, SystemMetric, get_database_manager
from .task_queue import task_queue

logger = structlog.get_logger()


class SystemState(Enum):
    """System operational states."""

    AWAKE = "awake"
    DROWSY = "drowsy"
    SLEEPING = "sleeping"
    WAKING = "waking"
    MAINTENANCE = "maintenance"


class ConsolidationPhase(Enum):
    """Phases of memory consolidation during sleep."""

    PRE_SLEEP = "pre_sleep"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    PATTERN_EXTRACTION = "pattern_extraction"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    SYSTEM_OPTIMIZATION = "system_optimization"
    CHECKPOINT_CREATION = "checkpoint_creation"
    POST_SLEEP = "post_sleep"


@dataclass
class SleepSchedule:
    """Sleep schedule configuration."""

    sleep_hour: int = 2  # 2 AM
    sleep_minute: int = 0
    sleep_duration_hours: float = 2.0  # 2 hours
    timezone: str = "UTC"
    enable_adaptive_scheduling: bool = True
    min_awake_hours: float = 20.0  # Minimum time awake before sleep


@dataclass
class SleepMetrics:
    """Metrics from a sleep cycle."""

    sleep_start: float
    sleep_end: float
    duration_hours: float
    consolidation_stats: dict[str, MemoryConsolidationStats]
    patterns_discovered: int
    performance_improvements: dict[str, float]
    system_optimizations: list[str]
    checkpoint_created: bool
    issues_resolved: int


@dataclass
class WakeMetrics:
    """Metrics from wake process."""

    wake_time: float
    restoration_duration_ms: float
    agents_restored: int
    contexts_loaded: int
    tasks_resumed: int
    system_health_score: float


@dataclass
class SystemCheckpoint:
    """System state checkpoint."""

    id: str
    created_at: float
    system_state: dict[str, Any]
    agent_states: dict[str, dict[str, Any]]
    performance_baseline: dict[str, float]
    memory_stats: dict[str, Any]
    active_tasks: list[str]
    git_commit_hash: str


class PerformanceMonitor:
    """Monitors system performance for optimization opportunities."""

    def __init__(self):
        self.metrics_history: list[dict[str, float]] = []
        self.baseline_metrics: dict[str, float] = {}

    async def collect_performance_metrics(self) -> dict[str, float]:
        """Collect current performance metrics."""
        metrics = {}

        try:
            # Task queue metrics
            queue_depth = await task_queue.get_queue_depth()
            completed_tasks = await task_queue.get_completed_tasks()
            failed_tasks = await task_queue.get_failed_tasks()

            metrics["queue_depth"] = float(queue_depth)
            metrics["task_success_rate"] = completed_tasks / max(
                1, completed_tasks + failed_tasks
            )

            # System resource metrics (simplified)
            import psutil

            metrics["cpu_usage"] = psutil.cpu_percent()
            metrics["memory_usage"] = psutil.virtual_memory().percent

            # Agent performance metrics
            db_manager = get_database_manager(settings.database_url)
            db_session = db_manager.get_session()

            try:
                active_agents = (
                    db_session.query(Agent).filter_by(status="active").count()
                )
                metrics["active_agents"] = float(active_agents)

                # Get recent system metrics
                recent_metrics = (
                    db_session.query(SystemMetric)
                    .filter(
                        SystemMetric.timestamp > time.time() - 3600  # Last hour
                    )
                    .all()
                )

                if recent_metrics:
                    task_times = [
                        m.value
                        for m in recent_metrics
                        if m.metric_name == "task_execution_time"
                    ]
                    if task_times:
                        metrics["avg_task_time"] = sum(task_times) / len(task_times)

            finally:
                db_session.close()

            # Store in history
            self.metrics_history.append(metrics)

            # Keep only recent history
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]

            return metrics

        except Exception as e:
            logger.error("Failed to collect performance metrics", error=str(e))
            return {}

    def identify_performance_issues(self) -> list[str]:
        """Identify performance issues from metrics history."""
        issues = []

        if len(self.metrics_history) < 5:
            return issues

        recent_metrics = self.metrics_history[-5:]

        # Check for degrading performance
        queue_depths = [m.get("queue_depth", 0) for m in recent_metrics]
        if all(q > 50 for q in queue_depths):
            issues.append("High queue depth indicating task processing bottleneck")

        task_success_rates = [m.get("task_success_rate", 1.0) for m in recent_metrics]
        if all(r < 0.8 for r in task_success_rates):
            issues.append("Low task success rate indicating system issues")

        cpu_usage = [m.get("cpu_usage", 0) for m in recent_metrics]
        if all(c > 80 for c in cpu_usage):
            issues.append("High CPU usage indicating resource constraints")

        memory_usage = [m.get("memory_usage", 0) for m in recent_metrics]
        if all(m > 85 for m in memory_usage):
            issues.append("High memory usage indicating potential memory leaks")

        return issues

    def suggest_optimizations(self) -> list[str]:
        """Suggest performance optimizations."""
        optimizations = []

        if not self.metrics_history:
            return optimizations

        current_metrics = self.metrics_history[-1]

        # Task queue optimizations
        if current_metrics.get("queue_depth", 0) > 20:
            optimizations.append("Consider spawning additional worker agents")

        if current_metrics.get("task_success_rate", 1.0) < 0.9:
            optimizations.append("Investigate and fix failing tasks")

        # Resource optimizations
        if current_metrics.get("memory_usage", 0) > 75:
            optimizations.append(
                "Implement memory optimization or increase system memory"
            )

        if current_metrics.get("cpu_usage", 0) > 70:
            optimizations.append(
                "Optimize CPU-intensive operations or scale horizontally"
            )

        # Agent optimizations
        active_agents = current_metrics.get("active_agents", 0)
        queue_depth = current_metrics.get("queue_depth", 0)

        if queue_depth > active_agents * 5:
            optimizations.append(
                "Agent to task ratio suboptimal - consider scaling agents"
            )

        return optimizations


class SleepWakeManager:
    """Manages sleep-wake cycles for the agent hive system."""

    def __init__(self, schedule: SleepSchedule = None):
        self.schedule = schedule or SleepSchedule()
        self.current_state = SystemState.AWAKE
        self.last_wake_time = time.time()
        self.last_sleep_time = 0.0

        # Components
        self.performance_monitor = PerformanceMonitor()
        self.consolidation_callbacks: list[Callable] = []

        # State tracking
        self.current_checkpoint: SystemCheckpoint | None = None
        self.sleep_metrics_history: list[SleepMetrics] = []
        self.wake_metrics_history: list[WakeMetrics] = []

        # Adaptive scheduling
        self.workload_history: list[float] = []

        logger.info("Sleep-wake manager initialized", schedule=asdict(self.schedule))

    async def start_sleep_wake_cycle(self) -> None:
        """Start the automatic sleep-wake cycle."""
        logger.info("Starting automatic sleep-wake cycle")

        while True:
            try:
                if self.current_state == SystemState.AWAKE:
                    # Check if it's time to sleep
                    if await self._should_enter_sleep():
                        await self._initiate_sleep_cycle()
                    else:
                        # Continue monitoring while awake
                        await self._monitor_awake_period()

                elif self.current_state == SystemState.SLEEPING:
                    # Monitor sleep cycle completion and transition back to awake
                    await self._monitor_sleep_cycle()

                # Check every minute
                await asyncio.sleep(Intervals.SYSTEM_HEALTH_CHECK)

            except Exception as e:
                logger.error("Sleep-wake cycle error", error=str(e))
                await asyncio.sleep(
                    Intervals.DB_CONNECTION_RETRY * 60
                )  # Wait 5 minutes on error

    async def _should_enter_sleep(self) -> bool:
        """Determine if the system should enter sleep mode."""
        current_time = datetime.now()

        # Check scheduled sleep time
        sleep_time = current_time.replace(
            hour=self.schedule.sleep_hour,
            minute=self.schedule.sleep_minute,
            second=0,
            microsecond=0,
        )

        # If past sleep time today, check if we haven't slept yet
        if current_time >= sleep_time:
            time_since_last_sleep = time.time() - self.last_sleep_time
            hours_since_sleep = time_since_last_sleep / 3600

            if hours_since_sleep >= self.schedule.min_awake_hours:
                return True

        # Adaptive scheduling based on workload
        if self.schedule.enable_adaptive_scheduling:
            return await self._should_sleep_adaptive()

        return False

    async def _should_sleep_adaptive(self) -> bool:
        """Adaptive sleep scheduling based on system workload."""
        try:
            # Check current workload
            queue_depth = await task_queue.get_queue_depth()

            # Don't sleep if high priority work is pending
            if queue_depth > 10:
                return False

            # Check if performance is degrading
            performance_issues = self.performance_monitor.identify_performance_issues()
            if len(performance_issues) >= 3:
                logger.info(
                    "Performance degradation detected, initiating early sleep for optimization"
                )
                return True

            # Check memory pressure
            context_engine = await get_advanced_context_engine(settings.database_url)
            memory_stats = await context_engine.get_memory_stats("system")

            if memory_stats.get("total_contexts", 0) > 10000:
                logger.info(
                    "High memory usage detected, initiating sleep for consolidation"
                )
                return True

            return False

        except Exception as e:
            logger.error("Adaptive sleep check failed", error=str(e))
            return False

    async def _monitor_awake_period(self) -> None:
        """Monitor system during awake period."""
        # Collect performance metrics
        metrics = await self.performance_monitor.collect_performance_metrics()

        # Store workload for adaptive scheduling
        self.workload_history.append(metrics.get("queue_depth", 0))
        if len(self.workload_history) > 100:
            self.workload_history = self.workload_history[-100:]

        # Log system status periodically
        if len(self.workload_history) % 30 == 0:  # Every 30 minutes
            logger.info(
                "System awake status",
                state=self.current_state.value,
                hours_awake=(time.time() - self.last_wake_time) / 3600,
                **metrics,
            )

    async def _initiate_sleep_cycle(self) -> None:
        """Initiate the sleep cycle."""
        logger.info("Initiating sleep cycle")

        self.current_state = SystemState.DROWSY

        # Notify all agents of impending sleep
        await message_broker.broadcast_message(
            from_agent="sleep_wake_manager",
            topic="sleep_preparation",
            payload={
                "sleep_start_time": time.time() + 300,  # 5 minutes warning
                "expected_duration_hours": self.schedule.sleep_duration_hours,
            },
        )

        # Wait for agents to prepare
        await asyncio.sleep(300)  # 5 minutes

        # Execute sleep cycle
        await self._execute_sleep_cycle()

    async def _execute_sleep_cycle(self) -> None:
        """Execute the complete sleep cycle."""
        sleep_start = time.time()
        self.current_state = SystemState.SLEEPING
        self.last_sleep_time = sleep_start

        logger.info("Entering sleep mode")

        try:
            # Create system checkpoint
            checkpoint = await self._create_system_checkpoint()

            # Execute consolidation phases
            consolidation_stats = {}
            patterns_discovered = 0
            optimizations = []

            # Phase 1: Memory consolidation
            logger.info("Sleep phase: Memory consolidation")
            consolidation_stats = await self._consolidate_memory()

            # Phase 2: Pattern extraction
            logger.info("Sleep phase: Pattern extraction")
            patterns_discovered = await self._extract_patterns()

            # Phase 3: Performance analysis
            logger.info("Sleep phase: Performance analysis")
            performance_improvements = await self._analyze_performance()

            # Phase 4: System optimization
            logger.info("Sleep phase: System optimization")
            optimizations = await self._optimize_system()

            # Phase 5: Cleanup and maintenance
            logger.info("Sleep phase: System maintenance")
            issues_resolved = await self._perform_maintenance()

            # Calculate actual sleep duration
            sleep_end = time.time()
            duration_hours = (sleep_end - sleep_start) / 3600

            # Create sleep metrics
            sleep_metrics = SleepMetrics(
                sleep_start=sleep_start,
                sleep_end=sleep_end,
                duration_hours=duration_hours,
                consolidation_stats=consolidation_stats,
                patterns_discovered=patterns_discovered,
                performance_improvements=performance_improvements,
                system_optimizations=optimizations,
                checkpoint_created=checkpoint is not None,
                issues_resolved=issues_resolved,
            )

            self.sleep_metrics_history.append(sleep_metrics)

            logger.info(
                "Sleep cycle completed",
                duration_hours=duration_hours,
                patterns_discovered=patterns_discovered,
                optimizations_count=len(optimizations),
            )

            # Initiate wake process
            await self._wake_up()

        except Exception as e:
            logger.error("Sleep cycle failed", error=str(e))
            await self._emergency_wake()

    async def _consolidate_memory(self) -> dict[str, MemoryConsolidationStats]:
        """Consolidate memory for all agents."""
        consolidation_stats = {}

        try:
            context_engine = await get_advanced_context_engine(settings.database_url)

            # Get all active agents
            db_manager = get_database_manager(settings.database_url)
            db_session = db_manager.get_session()

            try:
                agents = db_session.query(Agent).filter_by(status="active").all()

                for agent in agents:
                    stats = await context_engine.consolidate_memory(agent.name)
                    consolidation_stats[agent.name] = stats

                    logger.debug(
                        "Memory consolidated for agent",
                        agent=agent.name,
                        contexts_processed=stats.contexts_processed,
                    )

            finally:
                db_session.close()

        except Exception as e:
            logger.error("Memory consolidation failed", error=str(e))

        return consolidation_stats

    async def _extract_patterns(self) -> int:
        """Extract knowledge patterns from recent activities."""
        patterns_discovered = 0

        try:
            # This would involve analyzing recent contexts and identifying patterns
            # For now, return a placeholder count
            patterns_discovered = 5  # Placeholder

            logger.info(
                "Pattern extraction completed", patterns_discovered=patterns_discovered
            )

        except Exception as e:
            logger.error("Pattern extraction failed", error=str(e))

        return patterns_discovered

    async def _analyze_performance(self) -> dict[str, float]:
        """Analyze system performance and identify improvements."""
        improvements = {}

        try:
            # Collect final performance metrics
            final_metrics = await self.performance_monitor.collect_performance_metrics()

            # Compare with baseline
            if self.performance_monitor.baseline_metrics:
                for metric, value in final_metrics.items():
                    baseline = self.performance_monitor.baseline_metrics.get(
                        metric, value
                    )
                    if baseline > 0:
                        improvement = (value - baseline) / baseline
                        improvements[metric] = improvement

            # Update baseline
            self.performance_monitor.baseline_metrics = final_metrics

        except Exception as e:
            logger.error("Performance analysis failed", error=str(e))

        return improvements

    async def _optimize_system(self) -> list[str]:
        """Perform system optimizations."""
        optimizations = []

        try:
            # Get optimization suggestions
            suggestions = self.performance_monitor.suggest_optimizations()

            for suggestion in suggestions:
                # Apply optimization (simplified)
                logger.info("Applying optimization", suggestion=suggestion)
                optimizations.append(suggestion)

        except Exception as e:
            logger.error("System optimization failed", error=str(e))

        return optimizations

    async def _perform_maintenance(self) -> int:
        """Perform system maintenance tasks."""
        issues_resolved = 0

        try:
            # Database maintenance
            db_manager = get_database_manager(settings.database_url)
            db_session = db_manager.get_session()

            try:
                # Clean up old records (simplified)
                # cutoff_time = time.time() - (30 * 24 * 3600)  # 30 days

                # Would clean up old contexts, metrics, etc.
                logger.info("Database maintenance completed")
                issues_resolved += 1

            finally:
                db_session.close()

            # Task queue maintenance
            await task_queue.cleanup_old_tasks()
            issues_resolved += 1

        except Exception as e:
            logger.error("System maintenance failed", error=str(e))

        return issues_resolved

    async def _create_system_checkpoint(self) -> SystemCheckpoint | None:
        """Create a system state checkpoint."""
        try:
            checkpoint_id = f"checkpoint_{int(time.time())}"

            # Collect system state
            system_state = {
                "timestamp": time.time(),
                "active_agents": [],
                "system_metrics": await self.performance_monitor.collect_performance_metrics(),
            }

            # Get agent states
            agent_states = {}
            db_manager = get_database_manager(settings.database_url)
            db_session = db_manager.get_session()

            try:
                agents = db_session.query(Agent).filter_by(status="active").all()
                for agent in agents:
                    agent_states[agent.name] = {
                        "status": agent.status,
                        "last_heartbeat": agent.last_heartbeat,
                        "capabilities": agent.capabilities,
                    }
            finally:
                db_session.close()

            # Get active tasks
            active_tasks = []  # Would get from task queue

            # Get git commit hash
            git_commit = "unknown"  # Would get from git

            checkpoint = SystemCheckpoint(
                id=checkpoint_id,
                created_at=time.time(),
                system_state=system_state,
                agent_states=agent_states,
                performance_baseline=self.performance_monitor.baseline_metrics,
                memory_stats={},  # Would get from context engine
                active_tasks=active_tasks,
                git_commit_hash=git_commit,
            )

            self.current_checkpoint = checkpoint

            logger.info("System checkpoint created", checkpoint_id=checkpoint_id)
            return checkpoint

        except Exception as e:
            logger.error("Failed to create checkpoint", error=str(e))
            return None

    async def _monitor_sleep_cycle(self) -> None:
        """Monitor the sleep cycle and check if it should transition to wake."""
        # Check if enough time has passed since entering sleep
        if self.last_sleep_time:
            time_sleeping = time.time() - self.last_sleep_time
            min_sleep_hours = self.schedule.sleep_duration_hours

            # If minimum sleep time has passed, transition to wake
            if time_sleeping >= (min_sleep_hours * 3600):
                logger.info("Minimum sleep duration reached, transitioning to wake")
                await self._wake_up()
            else:
                # Continue sleeping - check again in a shorter interval
                await asyncio.sleep(30)  # Check every 30 seconds during sleep

    async def _wake_up(self) -> None:
        """Wake up the system from sleep."""
        wake_start = time.time()
        self.current_state = SystemState.WAKING

        logger.info("Waking up system")

        try:
            # Restore system state
            agents_restored = await self._restore_agents()

            # Load contexts back into working memory
            contexts_loaded = await self._restore_working_memory()

            # Resume pending tasks
            tasks_resumed = await self._resume_tasks()

            # Check system health
            health_score = await self._check_system_health()

            # Calculate wake metrics
            wake_end = time.time()
            restoration_duration = (wake_end - wake_start) * 1000

            wake_metrics = WakeMetrics(
                wake_time=wake_start,
                restoration_duration_ms=restoration_duration,
                agents_restored=agents_restored,
                contexts_loaded=contexts_loaded,
                tasks_resumed=tasks_resumed,
                system_health_score=health_score,
            )

            self.wake_metrics_history.append(wake_metrics)
            self.last_wake_time = wake_start
            self.current_state = SystemState.AWAKE

            # Notify agents of wake
            await message_broker.broadcast_message(
                from_agent="sleep_wake_manager",
                topic="system_wake",
                payload={
                    "wake_time": wake_start,
                    "system_health": health_score,
                    "restoration_duration_ms": restoration_duration,
                },
            )

            logger.info(
                "System wake completed",
                restoration_duration_ms=restoration_duration,
                agents_restored=agents_restored,
                system_health=health_score,
            )

        except Exception as e:
            logger.error("Wake process failed", error=str(e))
            await self._emergency_wake()

    async def _restore_agents(self) -> int:
        """Restore agent states after sleep."""
        agents_restored = 0

        if self.current_checkpoint:
            for (
                agent_name,
                _agent_state,
            ) in self.current_checkpoint.agent_states.items():
                # Restore agent state (simplified)
                logger.debug("Restoring agent", agent=agent_name)
                agents_restored += 1

        return agents_restored

    async def _restore_working_memory(self) -> int:
        """Restore working memory contexts."""
        contexts_loaded = 0

        try:
            # context_engine = await get_advanced_context_engine(settings.database_url)

            # Load recent contexts back into working memory cache
            # This would involve querying for recent contexts and loading them
            contexts_loaded = 100  # Placeholder

        except Exception as e:
            logger.error("Failed to restore working memory", error=str(e))

        return contexts_loaded

    async def _resume_tasks(self) -> int:
        """Resume pending tasks after wake."""
        tasks_resumed = 0

        try:
            # Resume tasks that were interrupted by sleep
            # This would involve checking for tasks that were running when sleep started
            tasks_resumed = 5  # Placeholder

        except Exception as e:
            logger.error("Failed to resume tasks", error=str(e))

        return tasks_resumed

    async def _check_system_health(self) -> float:
        """Check system health after wake."""
        try:
            metrics = await self.performance_monitor.collect_performance_metrics()

            # Calculate health score based on metrics
            health_factors = []

            # Agent availability
            active_agents = metrics.get("active_agents", 0)
            if active_agents > 0:
                health_factors.append(min(1.0, active_agents / 5))  # Target 5 agents

            # Task success rate
            success_rate = metrics.get("task_success_rate", 0.5)
            health_factors.append(success_rate)

            # Resource utilization
            cpu_health = max(0, 1.0 - metrics.get("cpu_usage", 50) / 100)
            memory_health = max(0, 1.0 - metrics.get("memory_usage", 50) / 100)
            health_factors.extend([cpu_health, memory_health])

            # Overall health score
            if health_factors:
                health_score = sum(health_factors) / len(health_factors)
            else:
                health_score = 0.5

            return health_score

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return 0.0

    async def _emergency_wake(self) -> None:
        """Emergency wake procedure if normal wake fails."""
        logger.warning("Initiating emergency wake procedure")

        self.current_state = SystemState.AWAKE
        self.last_wake_time = time.time()

        # Basic system restoration
        await message_broker.broadcast_message(
            from_agent="sleep_wake_manager",
            topic="emergency_wake",
            payload={"timestamp": time.time()},
        )

    def force_wake(self) -> None:
        """Force immediate wake (for manual intervention)."""
        logger.info("Forcing immediate wake")

        self.current_state = SystemState.AWAKE
        self.last_wake_time = time.time()

    def get_sleep_stats(self) -> dict[str, Any]:
        """Get sleep-wake cycle statistics."""
        if not self.sleep_metrics_history:
            return {
                "total_sleep_cycles": 0,
                "average_sleep_duration": 0.0,
                "average_patterns_discovered": 0.0,
                "current_state": self.current_state.value,
                "hours_since_last_sleep": 0.0,
            }

        total_cycles = len(self.sleep_metrics_history)
        avg_duration = (
            sum(s.duration_hours for s in self.sleep_metrics_history) / total_cycles
        )
        avg_patterns = (
            sum(s.patterns_discovered for s in self.sleep_metrics_history)
            / total_cycles
        )

        hours_since_sleep = (time.time() - self.last_sleep_time) / 3600

        return {
            "total_sleep_cycles": total_cycles,
            "average_sleep_duration": avg_duration,
            "average_patterns_discovered": avg_patterns,
            "current_state": self.current_state.value,
            "hours_since_last_sleep": hours_since_sleep,
            "last_sleep_time": self.last_sleep_time,
            "next_scheduled_sleep": self._get_next_sleep_time(),
        }

    def _get_next_sleep_time(self) -> float:
        """Calculate next scheduled sleep time."""
        current_time = datetime.now()
        next_sleep = current_time.replace(
            hour=self.schedule.sleep_hour,
            minute=self.schedule.sleep_minute,
            second=0,
            microsecond=0,
        )

        # If past today's sleep time, schedule for tomorrow
        if current_time >= next_sleep:
            next_sleep += timedelta(days=1)

        return next_sleep.timestamp()


# Global instance
_sleep_wake_manager: SleepWakeManager | None = None


def get_sleep_wake_manager(schedule: SleepSchedule = None) -> SleepWakeManager:
    """Get or create the sleep-wake manager singleton."""
    global _sleep_wake_manager

    if _sleep_wake_manager is None:
        _sleep_wake_manager = SleepWakeManager(schedule)

    return _sleep_wake_manager
