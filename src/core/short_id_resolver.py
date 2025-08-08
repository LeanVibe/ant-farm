"""Database service for short ID resolution."""

from typing import Optional, Dict, Any
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_

from .models import Agent, Task, Session
from .short_id import ShortIDGenerator


class ShortIDResolver:
    """Service for resolving short IDs to full UUIDs."""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.generator = ShortIDGenerator()

    async def resolve_agent_id(self, identifier: str) -> Optional[uuid.UUID]:
        """Resolve agent identifier (short ID, name, or UUID) to UUID."""
        # Check if it's already a UUID
        try:
            return uuid.UUID(identifier)
        except ValueError:
            pass

        # Check if it's a valid short ID format
        if self.generator.is_valid_short_id(identifier, "agent"):
            stmt = select(Agent.id).where(Agent.short_id == identifier)
            result = await self.db.execute(stmt)
            agent_uuid = result.scalar_one_or_none()
            return agent_uuid

        # Try to find by name
        stmt = select(Agent.id).where(Agent.name == identifier)
        result = await self.db.execute(stmt)
        agent_uuid = result.scalar_one_or_none()
        return agent_uuid

    async def resolve_task_id(self, identifier: str) -> Optional[uuid.UUID]:
        """Resolve task identifier (short ID or UUID) to UUID."""
        # Check if it's already a UUID
        try:
            return uuid.UUID(identifier)
        except ValueError:
            pass

        # Check if it's a valid short ID format
        if self.generator.is_valid_short_id(identifier, "task"):
            stmt = select(Task.id).where(Task.short_id == identifier)
            result = await self.db.execute(stmt)
            task_uuid = result.scalar_one_or_none()
            return task_uuid

        return None

    async def resolve_session_id(self, identifier: str) -> Optional[uuid.UUID]:
        """Resolve session identifier (short ID, name, or UUID) to UUID."""
        # Check if it's already a UUID
        try:
            return uuid.UUID(identifier)
        except ValueError:
            pass

        # Check if it's a valid short ID format
        if self.generator.is_valid_short_id(identifier, "session"):
            stmt = select(Session.id).where(Session.short_id == identifier)
            result = await self.db.execute(stmt)
            session_uuid = result.scalar_one_or_none()
            return session_uuid

        # Try to find by name
        stmt = select(Session.id).where(Session.name == identifier)
        result = await self.db.execute(stmt)
        session_uuid = result.scalar_one_or_none()
        return session_uuid

    async def get_agent_by_identifier(
        self, identifier: str
    ) -> Optional[Dict[str, Any]]:
        """Get agent by any identifier and return as dict."""
        agent_id = await self.resolve_agent_id(identifier)
        if not agent_id:
            return None

        stmt = select(Agent).where(Agent.id == agent_id)
        result = await self.db.execute(stmt)
        agent = result.scalar_one_or_none()
        return agent.to_dict() if agent else None

    async def get_task_by_identifier(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Get task by any identifier and return as dict."""
        task_id = await self.resolve_task_id(identifier)
        if not task_id:
            return None

        stmt = select(Task).where(Task.id == task_id)
        result = await self.db.execute(stmt)
        task = result.scalar_one_or_none()
        return task.to_dict() if task else None

    async def search_agents_by_partial_id(
        self, partial: str, limit: int = 10
    ) -> list[Dict[str, Any]]:
        """Search agents by partial short ID or name."""
        conditions = []

        # Search by partial short ID
        if partial and len(partial) <= 5:
            conditions.append(Agent.short_id.like(f"{partial}%"))

        # Search by partial name
        conditions.append(Agent.name.like(f"%{partial}%"))

        if not conditions:
            return []

        stmt = select(Agent).where(or_(*conditions)).limit(limit)
        result = await self.db.execute(stmt)
        agents = result.scalars().all()

        return [agent.to_dict() for agent in agents]

    async def search_tasks_by_partial_id(
        self, partial: str, limit: int = 10
    ) -> list[Dict[str, Any]]:
        """Search tasks by partial short ID or title."""
        conditions = []

        # Search by partial short ID (for numeric partial matches)
        if partial and partial.isdigit() and len(partial) <= 4:
            conditions.append(Task.short_id.like(f"{partial}%"))

        # Search by partial title
        conditions.append(Task.title.like(f"%{partial}%"))

        if not conditions:
            return []

        stmt = select(Task).where(or_(*conditions)).limit(limit)
        result = await self.db.execute(stmt)
        tasks = result.scalars().all()

        return [task.to_dict() for task in tasks]

    async def generate_and_assign_short_ids(self) -> Dict[str, int]:
        """Generate short IDs for existing records that don't have them."""
        updated_counts = {"agents": 0, "tasks": 0, "sessions": 0}

        # Update agents without short IDs
        stmt = select(Agent).where(Agent.short_id.is_(None))
        result = await self.db.execute(stmt)
        agents = result.scalars().all()

        for agent in agents:
            agent.short_id = self.generator.generate_agent_short_id(
                agent.name, str(agent.id)
            )
            updated_counts["agents"] += 1

        # Update tasks without short IDs
        stmt = select(Task).where(Task.short_id.is_(None))
        result = await self.db.execute(stmt)
        tasks = result.scalars().all()

        for task in tasks:
            task.short_id = self.generator.generate_task_short_id(
                task.title, str(task.id)
            )
            updated_counts["tasks"] += 1

        # Update sessions without short IDs
        stmt = select(Session).where(Session.short_id.is_(None))
        result = await self.db.execute(stmt)
        sessions = result.scalars().all()

        for session in sessions:
            session.short_id = self.generator.generate_session_short_id(
                session.name, str(session.id)
            )
            updated_counts["sessions"] += 1

        await self.db.commit()
        return updated_counts
