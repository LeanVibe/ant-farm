"""Integration test for Context Engine async operations - TDD approach."""

import asyncio
import pytest
import uuid
from src.core.context_engine import ContextEngine
from src.core.config import get_settings
from src.core.models import Agent, Base
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession


@pytest.mark.asyncio
async def test_context_engine_full_workflow():
    """Test that Context Engine can store and retrieve contexts asynchronously.

    This test defines the expected behavior:
    1. Initialize engine with async database connection
    2. Store context successfully
    3. Search and retrieve context with semantic similarity
    4. All operations must be truly async (no sync DB calls)
    """
    settings = get_settings()
    engine = ContextEngine(settings.database_url)

    # Set up test agent in database
    async_engine = create_async_engine(settings.database_url)
    AsyncSessionLocal = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )

    test_agent_id = uuid.uuid4()

    try:
        # Create test agent first (required for foreign key constraint)
        async with AsyncSessionLocal() as session:
            test_agent = Agent(
                id=test_agent_id,
                name="test-agent",
                type="test",
                role="test_runner",
                status="active",
                capabilities=["testing"],
                config={},
            )
            session.add(test_agent)
            await session.commit()

        # Should initialize without blocking
        await engine.initialize()

        # Should store context and return ID
        context_id = await engine.store_context(
            content="Machine learning algorithms for autonomous agents",
            agent_id=str(test_agent_id),
            metadata={"source": "test", "importance": 0.8},
        )

        assert context_id is not None
        assert len(context_id) > 0

        # Should find related content via semantic search
        results = await engine.search_context(query="machine learning", limit=5)

        assert len(results) > 0
        assert results[0].content == "Machine learning algorithms for autonomous agents"
        assert results[0].score > 0.0  # Some similarity

        # Should retrieve by ID
        retrieved = await engine.get_context(context_id)
        assert retrieved is not None
        assert retrieved.content == "Machine learning algorithms for autonomous agents"

        print("✅ Context Engine async test passed!")

    finally:
        # Cleanup test agent
        async with AsyncSessionLocal() as session:
            result = await session.get(Agent, test_agent_id)
            if result:
                await session.delete(result)
                await session.commit()

        await engine.close()
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(test_context_engine_full_workflow())
