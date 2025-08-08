"""Async database manager for LeanVibe Agent Hive 2.0."""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Handle both module and direct execution imports
try:
    from .config import settings
    from .models import Agent as AgentModel, Base, Context, SystemMetric, Task
except ImportError:
    from config import settings
    from models import Agent as AgentModel, Base, Context, SystemMetric, Task

logger = structlog.get_logger()


class DatabaseConnectionError(Exception):
    """Database connection related errors."""

    pass


class DatabaseOperationError(Exception):
    """Database operation related errors."""

    pass


class AsyncDatabaseManager:
    """Async database manager for handling all database operations."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        # Ensure we use asyncpg for async operations
        if "postgresql://" in database_url:
            self.database_url = database_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )

        try:
            self.engine = create_async_engine(
                self.database_url,
                echo=settings.debug,
                pool_size=getattr(settings, "database_pool_size", 5),
                max_overflow=getattr(settings, "database_max_overflow", 10),
                pool_pre_ping=True,  # Validate connections before use
                pool_recycle=3600,  # Recycle connections every hour
            )
            self.async_session_maker = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        except Exception as e:
            logger.error("Failed to initialize database engine", error=str(e))
            raise DatabaseConnectionError(f"Failed to initialize database: {e}")

    async def health_check(self) -> bool:
        """Check database health."""
        try:
            async with self.async_session_maker() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False

    async def ensure_connection(self) -> bool:
        """Ensure database connection is available."""
        try:
            return await self.health_check()
        except Exception as e:
            logger.error("Database connection check failed", error=str(e))
            raise DatabaseConnectionError(f"Database connection failed: {e}")

    async def create_tables(self):
        """Create all database tables."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                logger.info("Database tables created successfully")
        except Exception as e:
            logger.error("Failed to create database tables", error=str(e))
            raise DatabaseConnectionError(f"Failed to create tables: {e}")

    async def register_agent(
        self,
        name: str,
        agent_type: str,
        role: str,
        capabilities: dict[str, Any] = None,
        tmux_session: str = None,
    ) -> str:
        """Register an agent in the database."""
        async with self.async_session_maker() as session:
            try:
                # Check if agent exists
                from sqlalchemy import select

                stmt = select(AgentModel).where(AgentModel.name == name)
                result = await session.execute(stmt)
                existing_agent = result.scalar_one_or_none()

                if existing_agent:
                    # Update existing agent
                    existing_agent.status = "active"
                    existing_agent.last_heartbeat = datetime.now(timezone.utc)
                    existing_agent.tmux_session = tmux_session
                    if capabilities:
                        existing_agent.capabilities = capabilities
                    await session.commit()
                    logger.info(
                        "Updated existing agent",
                        agent_name=name,
                        agent_id=str(existing_agent.id),
                    )
                    return str(existing_agent.id)
                else:
                    # Create new agent
                    agent = AgentModel(
                        name=name,
                        type=agent_type,
                        role=role,
                        capabilities=capabilities or {},
                        status="active",
                        tmux_session=tmux_session,
                        last_heartbeat=datetime.now(timezone.utc),
                    )
                    session.add(agent)
                    await session.commit()
                    await session.refresh(agent)
                    logger.info(
                        "Created new agent", agent_name=name, agent_id=str(agent.id)
                    )
                    return str(agent.id)

            except Exception as e:
                await session.rollback()
                logger.error("Failed to register agent", agent=name, error=str(e))
                raise DatabaseOperationError(f"Failed to register agent {name}: {e}")

    async def update_agent_heartbeat(self, agent_name: str) -> bool:
        """Update agent heartbeat."""
        async with self.async_session_maker() as session:
            try:
                from sqlalchemy import select, update

                stmt = (
                    update(AgentModel)
                    .where(AgentModel.name == agent_name)
                    .values(last_heartbeat=datetime.now(timezone.utc))
                )
                result = await session.execute(stmt)
                await session.commit()
                success = result.rowcount > 0

                if success:
                    logger.debug("Updated agent heartbeat", agent_name=agent_name)
                else:
                    logger.warning(
                        "Agent not found for heartbeat update", agent_name=agent_name
                    )

                return success

            except Exception as e:
                await session.rollback()
                logger.error(
                    "Failed to update heartbeat", agent=agent_name, error=str(e)
                )
                return False

    async def get_agent_by_name(self, name: str) -> Optional[AgentModel]:
        """Get agent by name."""
        async with self.async_session_maker() as session:
            try:
                from sqlalchemy import select

                stmt = select(AgentModel).where(AgentModel.name == name)
                result = await session.execute(stmt)
                return result.scalar_one_or_none()

            except Exception as e:
                logger.error("Failed to get agent", agent=name, error=str(e))
                return None

    async def get_agent_by_id(self, agent_id: str) -> Optional[AgentModel]:
        """Get agent by ID."""
        async with self.async_session_maker() as session:
            try:
                from sqlalchemy import select

                stmt = select(AgentModel).where(AgentModel.id == uuid.UUID(agent_id))
                result = await session.execute(stmt)
                return result.scalar_one_or_none()

            except Exception as e:
                logger.error(
                    "Failed to get agent by ID", agent_id=agent_id, error=str(e)
                )
                return None

    async def store_context(
        self,
        agent_id: str,
        content: str,
        importance_score: float = 0.5,
        category: str = "general",
        topic: str = None,
        metadata: dict[str, Any] = None,
        session_id: str = None,
    ) -> str:
        """Store context in the database."""
        async with self.async_session_maker() as session:
            try:
                context = Context(
                    agent_id=uuid.UUID(agent_id),
                    session_id=uuid.UUID(session_id) if session_id else None,
                    content=content,
                    importance_score=importance_score,
                    category=category,
                    topic=topic,
                    meta_data=metadata or {},
                    created_at=datetime.now(timezone.utc),
                    last_accessed=datetime.now(timezone.utc),
                )
                session.add(context)
                await session.commit()
                await session.refresh(context)
                logger.debug(
                    "Stored context", context_id=str(context.id), agent_id=agent_id
                )
                return str(context.id)

            except Exception as e:
                await session.rollback()
                logger.error("Failed to store context", agent_id=agent_id, error=str(e))
                raise DatabaseOperationError(f"Failed to store context: {e}")

    async def record_system_metric(
        self,
        metric_name: str,
        metric_type: str,
        value: float,
        unit: str = None,
        agent_id: str = None,
        task_id: str = None,
        session_id: str = None,
        labels: dict[str, str] = None,
    ) -> str:
        """Record a system metric."""
        async with self.async_session_maker() as session:
            try:
                metric = SystemMetric(
                    metric_name=metric_name,
                    metric_type=metric_type,
                    value=value,
                    unit=unit,
                    agent_id=uuid.UUID(agent_id) if agent_id else None,
                    task_id=uuid.UUID(task_id) if task_id else None,
                    session_id=uuid.UUID(session_id) if session_id else None,
                    labels=labels or {},
                    timestamp=datetime.now(timezone.utc),
                )
                session.add(metric)
                await session.commit()
                await session.refresh(metric)
                logger.debug(
                    "Recorded system metric", metric_name=metric_name, value=value
                )
                return str(metric.id)

            except Exception as e:
                await session.rollback()
                logger.error(
                    "Failed to record metric", metric_name=metric_name, error=str(e)
                )
                raise DatabaseOperationError(f"Failed to record metric: {e}")

    async def get_active_agents(self) -> list[AgentModel]:
        """Get all active agents."""
        async with self.async_session_maker() as session:
            try:
                from sqlalchemy import select

                stmt = select(AgentModel).where(AgentModel.status == "active")
                result = await session.execute(stmt)
                agents = result.scalars().all()
                logger.debug("Retrieved active agents", count=len(agents))
                return list(agents)

            except Exception as e:
                logger.error("Failed to get active agents", error=str(e))
                return []

    async def cleanup_stale_agents(self, threshold_minutes: int = 10) -> int:
        """Clean up agents that haven't sent heartbeat in threshold_minutes."""
        async with self.async_session_maker() as session:
            try:
                from sqlalchemy import select, update

                threshold_time = datetime.now(timezone.utc) - timedelta(
                    minutes=threshold_minutes
                )

                # Update stale agents to inactive status
                stmt = (
                    update(AgentModel)
                    .where(
                        AgentModel.status == "active",
                        AgentModel.last_heartbeat < threshold_time,
                    )
                    .values(status="inactive")
                )
                result = await session.execute(stmt)
                await session.commit()

                cleaned_count = result.rowcount
                if cleaned_count > 0:
                    logger.info("Cleaned up stale agents", count=cleaned_count)

                return cleaned_count

            except Exception as e:
                await session.rollback()
                logger.error("Failed to cleanup stale agents", error=str(e))
                return 0

    async def get_database_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        async with self.async_session_maker() as session:
            try:
                from sqlalchemy import func, select

                stats = {}

                # Agent counts
                agent_count = await session.execute(select(func.count(AgentModel.id)))
                stats["total_agents"] = agent_count.scalar()

                active_count = await session.execute(
                    select(func.count(AgentModel.id)).where(
                        AgentModel.status == "active"
                    )
                )
                stats["active_agents"] = active_count.scalar()

                # Context counts
                context_count = await session.execute(select(func.count(Context.id)))
                stats["total_contexts"] = context_count.scalar()

                # Task counts (if Task model exists)
                try:
                    task_count = await session.execute(select(func.count(Task.id)))
                    stats["total_tasks"] = task_count.scalar()
                except Exception:
                    stats["total_tasks"] = 0

                return stats

            except Exception as e:
                logger.error("Failed to get database stats", error=str(e))
                return {}

    async def close(self):
        """Close database connections."""
        try:
            if self.engine:
                await self.engine.dispose()
                logger.info("Database connections closed")
        except Exception as e:
            logger.error("Error closing database connections", error=str(e))


# Global async database manager instance
_async_db_manager: Optional[AsyncDatabaseManager] = None


async def get_async_database_manager(database_url: str = None) -> AsyncDatabaseManager:
    """Get the async database manager instance."""
    global _async_db_manager

    if database_url is None:
        database_url = settings.database_url

    if _async_db_manager is None:
        _async_db_manager = AsyncDatabaseManager(database_url)
        # Ensure connection is working
        await _async_db_manager.ensure_connection()

    return _async_db_manager


async def get_async_session(database_url: str = None) -> AsyncSession:
    """Get an async database session."""
    db_manager = await get_async_database_manager(database_url)
    return db_manager.async_session_maker()


async def close_async_database():
    """Close the async database connection."""
    global _async_db_manager

    if _async_db_manager:
        await _async_db_manager.close()
        _async_db_manager = None
