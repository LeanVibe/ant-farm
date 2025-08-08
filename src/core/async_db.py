"""Async database manager for LeanVibe Agent Hive 2.0."""

import asyncio
import uuid
from datetime import datetime
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


class AsyncDatabaseManager:
    """Async database manager for handling all database operations."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        # Ensure we use asyncpg for async operations
        if "postgresql://" in database_url:
            self.database_url = database_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )

        self.engine = create_async_engine(
            self.database_url,
            echo=settings.debug,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
        )
        self.async_session_maker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def health_check(self) -> bool:
        """Check database health."""
        try:
            async with self.async_session_maker() as session:
                result = await session.execute(text("SELECT 1"))
                return await result.scalar() == 1
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False

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
                existing_agent = await result.scalar_one_or_none()

                if existing_agent:
                    # Update existing agent
                    existing_agent.status = "active"
                    existing_agent.last_heartbeat = datetime.utcnow()
                    existing_agent.tmux_session = tmux_session
                    if capabilities:
                        existing_agent.capabilities = capabilities
                    await session.commit()
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
                        last_heartbeat=datetime.utcnow(),
                    )
                    session.add(agent)
                    await session.commit()
                    await session.refresh(agent)
                    return str(agent.id)

            except Exception as e:
                await session.rollback()
                logger.error("Failed to register agent", agent=name, error=str(e))
                raise

    async def update_agent_heartbeat(self, agent_name: str) -> bool:
        """Update agent heartbeat."""
        async with self.async_session_maker() as session:
            try:
                from sqlalchemy import select, update

                stmt = (
                    update(AgentModel)
                    .where(AgentModel.name == agent_name)
                    .values(last_heartbeat=datetime.utcnow())
                )
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount > 0

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
                return await result.scalar_one_or_none()

            except Exception as e:
                logger.error("Failed to get agent", agent=name, error=str(e))
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
                    created_at=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                )
                session.add(context)
                await session.commit()
                await session.refresh(context)
                return str(context.id)

            except Exception as e:
                await session.rollback()
                logger.error("Failed to store context", agent_id=agent_id, error=str(e))
                raise

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
                    timestamp=datetime.utcnow(),
                )
                session.add(metric)
                await session.commit()
                await session.refresh(metric)
                return str(metric.id)

            except Exception as e:
                await session.rollback()
                logger.error(
                    "Failed to record metric", metric_name=metric_name, error=str(e)
                )
                raise

    async def get_active_agents(self) -> list[AgentModel]:
        """Get all active agents."""
        async with self.async_session_maker() as session:
            try:
                from sqlalchemy import select

                stmt = select(AgentModel).where(AgentModel.status == "active")
                result = await session.execute(stmt)
                scalars_result = await result.scalars()
                return await scalars_result.all()

            except Exception as e:
                logger.error("Failed to get active agents", error=str(e))
                return []

    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")


# Global async database manager instance
_async_db_manager: Optional[AsyncDatabaseManager] = None


async def get_async_database_manager(database_url: str = None) -> AsyncDatabaseManager:
    """Get the async database manager instance."""
    if database_url is None:
        database_url = settings.database_url

    return AsyncDatabaseManager(database_url)


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
