"""Comprehensive isolation tests for EnhancedMessageBroker component.

These tests ensure the EnhancedMessageBroker operates correctly in complete isolation
from Redis, database, and other external dependencies. All external operations are mocked
to verify enhanced messaging features without system dependencies.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Set

import pytest
import structlog

from src.core.enhanced_message_broker import (
    ContextShareType,
    EnhancedMessageBroker,
    MessagePriority,
    SharedContext,
    SyncMode,
    AgentState,
)
from src.core.message_broker import MessageType
from tests.unit.test_component_isolation_framework import (
    ComponentIsolationTestFramework,
    ComponentTestConfig,
    create_isolated_enhanced_message_broker,
)

logger = structlog.get_logger()


class TestEnhancedMessageBrokerIsolation:
    """Comprehensive isolation tests for EnhancedMessageBroker."""

    @pytest.fixture
    async def isolated_enhanced_broker(self):
        """Fixture providing an isolated EnhancedMessageBroker."""
        broker, framework = await create_isolated_enhanced_message_broker()
        yield broker, framework

    @pytest.mark.asyncio
    async def test_enhanced_broker_initialization_isolation(
        self, isolated_enhanced_broker
    ):
        """Test enhanced broker initialization without external dependencies."""
        broker, framework = isolated_enhanced_broker

        # Initialize broker - should only interact with mocked Redis
        await broker.initialize()

        # Verify enhanced initialization succeeded
        assert broker.pubsub is not None
        assert broker.shared_contexts == {}
        assert broker.agent_states == {}
        assert broker.context_subscribers == {}
        assert len(broker.sync_tasks) == 2  # context_sync and state_sync tasks

        # Verify only expected Redis interactions
        framework.assert_redis_interactions([{"method": "ping"}])

        # Ensure no external calls leaked
        framework.assert_no_external_calls()

    @pytest.mark.asyncio
    async def test_create_shared_context_isolation(self, isolated_enhanced_broker):
        """Test shared context creation in complete isolation."""
        broker, framework = isolated_enhanced_broker
        await broker.initialize()

        # Clear previous interactions
        framework.mock_redis_client.interactions.clear()

        # Create shared context
        context_id = await broker.create_shared_context(
            context_type=ContextShareType.WORK_SESSION,
            owner_agent="test_agent",
            initial_data={"project": "test_project", "phase": "development"},
            participants={"test_agent", "collaborator_agent"},
            sync_mode=SyncMode.REAL_TIME,
            ttl=3600,  # 1 hour
        )

        # Verify context creation succeeded
        assert context_id is not None
        assert context_id in broker.shared_contexts

        # Verify context properties
        context = broker.shared_contexts[context_id]
        assert context.type == ContextShareType.WORK_SESSION
        assert context.owner_agent == "test_agent"
        assert "test_agent" in context.participants
        assert "collaborator_agent" in context.participants
        assert context.data["project"] == "test_project"
        assert context.sync_mode == SyncMode.REAL_TIME
        assert context.ttl is not None

        # Verify Redis persistence
        redis_interactions = framework.mock_redis_client.interactions
        hset_calls = [i for i in redis_interactions if i.method == "hset"]
        expire_calls = [i for i in redis_interactions if i.method == "expire"]

        assert len(hset_calls) >= 1  # Context persistence
        assert len(expire_calls) >= 1  # TTL setting

        # Verify context was persisted with correct key format
        context_keys = [
            call.args[0] for call in hset_calls if "shared_context" in call.args[0]
        ]
        assert len(context_keys) >= 1
        assert context_keys[0] == f"hive:shared_context:{context_id}"

    @pytest.mark.asyncio
    async def test_join_shared_context_isolation(self, isolated_enhanced_broker):
        """Test joining shared context in isolation."""
        broker, framework = isolated_enhanced_broker
        await broker.initialize()

        # First create a context
        context_id = await broker.create_shared_context(
            context_type=ContextShareType.TASK_STATE,
            owner_agent="owner_agent",
            initial_data={"task": "development", "status": "in_progress"},
            participants={"owner_agent"},
        )

        # Clear previous interactions
        framework.mock_redis_client.interactions.clear()

        # Another agent joins the context
        success = await broker.join_shared_context(context_id, "new_participant")

        # Verify join succeeded
        assert success is True

        # Verify participant was added
        context = broker.shared_contexts[context_id]
        assert "new_participant" in context.participants
        assert "new_participant" in broker.context_subscribers.get(context_id, set())

        # Verify Redis operations for persistence and notification
        redis_interactions = framework.mock_redis_client.interactions
        hset_calls = [i for i in redis_interactions if i.method == "hset"]
        publish_calls = [i for i in redis_interactions if i.method == "publish"]

        assert len(hset_calls) >= 1  # Updated context persistence
        assert len(publish_calls) >= 1  # Notification to new participant

        # Verify notification was sent correctly
        notification_call = publish_calls[0]
        channel, message_data = notification_call.args
        assert channel == "agent:new_participant"

        message = json.loads(message_data)
        assert message["topic"] == "context_joined"
        assert message["payload"]["context_id"] == context_id

    @pytest.mark.asyncio
    async def test_update_shared_context_isolation(self, isolated_enhanced_broker):
        """Test shared context updates in isolation."""
        broker, framework = isolated_enhanced_broker
        await broker.initialize()

        # Create initial context
        context_id = await broker.create_shared_context(
            context_type=ContextShareType.WORK_SESSION,
            owner_agent="agent1",
            initial_data={"version": 1, "status": "active"},
            participants={"agent1", "agent2"},
        )

        # Clear previous interactions
        framework.mock_redis_client.interactions.clear()

        # Update context
        success = await broker.update_shared_context(
            context_id=context_id,
            agent_name="agent1",
            updates={"version": 2, "status": "updated", "new_field": "added"},
            merge_strategy="deep_merge",
        )

        # Verify update succeeded
        assert success is True

        # Verify context was updated
        context = broker.shared_contexts[context_id]
        assert context.version == 1  # Version should have incremented from 0
        assert context.data["version"] == 2
        assert context.data["status"] == "updated"
        assert context.data["new_field"] == "added"
        assert context.last_updated_by == "agent1"

        # Verify Redis operations
        redis_interactions = framework.mock_redis_client.interactions
        hset_calls = [i for i in redis_interactions if i.method == "hset"]
        publish_calls = [i for i in redis_interactions if i.method == "publish"]

        assert len(hset_calls) >= 1  # Context persistence

        # For real-time sync mode, should notify other participants
        assert len(publish_calls) >= 1  # Notification to other participants

        # Verify notification was sent to other participant
        notification_call = publish_calls[0]
        channel, message_data = notification_call.args
        assert channel == "agent:agent2"  # Notification to non-updating participant

        message = json.loads(message_data)
        assert message["topic"] == "context_updated"
        assert message["payload"]["context_id"] == context_id
        assert message["payload"]["new_version"] == 1

    @pytest.mark.asyncio
    async def test_agent_state_management_isolation(self, isolated_enhanced_broker):
        """Test agent state management in isolation."""
        broker, framework = isolated_enhanced_broker
        await broker.initialize()

        # Clear previous interactions
        framework.mock_redis_client.interactions.clear()

        # Update agent state
        await broker.update_agent_state(
            agent_name="test_agent",
            state_updates={
                "status": "busy",
                "current_task": "development_task",
                "capabilities": ["coding", "testing", "debugging"],
                "performance_metrics": {"tasks_completed": 15, "avg_time": 45.5},
            },
        )

        # Verify state was updated
        assert "test_agent" in broker.agent_states
        agent_state = broker.agent_states["test_agent"]

        assert agent_state.status == "busy"
        assert agent_state.current_task == "development_task"
        assert "coding" in agent_state.capabilities
        assert agent_state.performance_metrics["tasks_completed"] == 15
        assert agent_state.last_activity > 0

        # Verify Redis persistence
        redis_interactions = framework.mock_redis_client.interactions
        hset_calls = [i for i in redis_interactions if i.method == "hset"]
        expire_calls = [i for i in redis_interactions if i.method == "expire"]

        assert len(hset_calls) >= 1  # State persistence
        assert len(expire_calls) >= 1  # TTL setting

        # Verify state key format
        state_keys = [
            call.args[0] for call in hset_calls if "agent_state" in call.args[0]
        ]
        assert len(state_keys) >= 1
        assert state_keys[0] == "hive:agent_state:test_agent"

    @pytest.mark.asyncio
    async def test_get_agent_states_isolation(self, isolated_enhanced_broker):
        """Test retrieving agent states in isolation."""
        broker, framework = isolated_enhanced_broker
        await broker.initialize()

        # Create multiple agent states
        agents = ["agent1", "agent2", "agent3"]
        for i, agent in enumerate(agents):
            await broker.update_agent_state(
                agent_name=agent,
                state_updates={
                    "status": "active",
                    "current_task": f"task_{i}",
                    "capabilities": [f"skill_{i}", "common_skill"],
                    "performance_metrics": {"score": float(i * 10)},
                },
            )

        # Get all agent states
        states = await broker.get_agent_states("requesting_agent")

        # Verify all states were returned
        assert len(states) == len(agents)

        for agent in agents:
            assert agent in states
            agent_data = states[agent]
            assert agent_data["status"] == "active"
            assert agent_data["current_task"].startswith("task_")
            assert "common_skill" in agent_data["capabilities"]

    @pytest.mark.asyncio
    async def test_context_aware_messaging_isolation(self, isolated_enhanced_broker):
        """Test context-aware messaging in isolation."""
        broker, framework = isolated_enhanced_broker
        await broker.initialize()

        # Create shared contexts
        context_id1 = await broker.create_shared_context(
            context_type=ContextShareType.TASK_STATE,
            owner_agent="sender",
            initial_data={"task": "feature_dev", "progress": 50},
        )

        context_id2 = await broker.create_shared_context(
            context_type=ContextShareType.KNOWLEDGE_BASE,
            owner_agent="sender",
            initial_data={"domain": "ai", "expertise": "high"},
        )

        # Clear previous interactions
        framework.mock_redis_client.interactions.clear()

        # Send context-aware message
        success = await broker.send_context_aware_message(
            from_agent="sender",
            to_agent="receiver",
            topic="context_aware_update",
            payload={"update": "feature completed"},
            context_ids=[context_id1, context_id2],
            include_relevant_context=True,
        )

        # Verify message sent successfully
        assert success is True

        # Verify Redis operations
        redis_interactions = framework.mock_redis_client.interactions
        publish_calls = [i for i in redis_interactions if i.method == "publish"]

        assert len(publish_calls) >= 1

        # Verify message includes context data
        message_call = publish_calls[0]
        channel, message_data = message_call.args

        assert channel == "agent:receiver"
        message = json.loads(message_data)

        # Should include shared contexts in payload
        assert "shared_contexts" in message["payload"]
        assert len(message["payload"]["shared_contexts"]) == 2

        # Should include sender state
        assert "sender_state" in message["payload"]
        assert message["payload"]["sender_state"]["status"] == "idle"  # Default status

    @pytest.mark.asyncio
    async def test_priority_messaging_isolation(self, isolated_enhanced_broker):
        """Test priority messaging in isolation."""
        broker, framework = isolated_enhanced_broker
        await broker.initialize()

        # Clear previous interactions
        framework.mock_redis_client.interactions.clear()

        # Send priority messages
        priorities = [
            MessagePriority.CRITICAL,
            MessagePriority.HIGH,
            MessagePriority.NORMAL,
            MessagePriority.LOW,
        ]

        message_ids = []
        for priority in priorities:
            message_id = await broker.send_priority_message(
                from_agent="sender",
                to_agent="receiver",
                topic="priority_test",
                payload={"priority_level": priority.name},
                priority=priority,
            )
            message_ids.append(message_id)

        # Verify all messages got IDs
        assert len(message_ids) == len(priorities)
        assert all(msg_id for msg_id in message_ids)

        # Verify Redis operations for priority queues
        redis_interactions = framework.mock_redis_client.interactions
        zadd_calls = [i for i in redis_interactions if i.method == "zadd"]
        publish_calls = [i for i in redis_interactions if i.method == "publish"]

        # Should have zadd calls for priority queues
        assert len(zadd_calls) >= len(priorities)

        # Should have publish calls for both priority notifications and normal messaging
        assert (
            len(publish_calls) >= len(priorities) * 2
        )  # Priority notification + normal publish

        # Verify priority queue keys
        priority_keys = set()
        for call in zadd_calls:
            key = call.args[0]
            if "priority_queue" in key:
                priority_keys.add(key)

        expected_keys = {f"hive:priority_queue:{p.name.lower()}" for p in priorities}
        assert priority_keys == expected_keys

    @pytest.mark.asyncio
    async def test_broadcast_context_update_isolation(self, isolated_enhanced_broker):
        """Test broadcasting context updates in isolation."""
        broker, framework = isolated_enhanced_broker
        await broker.initialize()

        # Create shared context with multiple participants
        context_id = await broker.create_shared_context(
            context_type=ContextShareType.WORK_SESSION,
            owner_agent="coordinator",
            initial_data={"session": "team_standup"},
            participants={"coordinator", "dev1", "dev2", "qa1"},
        )

        # Clear previous interactions
        framework.mock_redis_client.interactions.clear()

        # Broadcast context update
        await broker.broadcast_context_update(
            from_agent="coordinator",
            context_id=context_id,
            update_data={"announcement": "standup starts in 5 minutes"},
            target_participants={"dev1", "dev2", "qa1"},  # Exclude coordinator
        )

        # Verify Redis operations
        redis_interactions = framework.mock_redis_client.interactions
        publish_calls = [i for i in redis_interactions if i.method == "publish"]

        # Should have one publish per target participant
        assert len(publish_calls) == 3  # dev1, dev2, qa1

        # Verify each participant got the message
        published_channels = {call.args[0] for call in publish_calls}
        expected_channels = {"agent:dev1", "agent:dev2", "agent:qa1"}
        assert published_channels == expected_channels

        # Verify message content
        for call in publish_calls:
            channel, message_data = call.args
            message = json.loads(message_data)

            assert message["topic"] == "context_broadcast_update"
            assert message["payload"]["context_id"] == context_id
            assert (
                message["payload"]["update_data"]["announcement"]
                == "standup starts in 5 minutes"
            )

    @pytest.mark.asyncio
    async def test_intelligent_message_routing_isolation(
        self, isolated_enhanced_broker
    ):
        """Test intelligent message routing in isolation."""
        broker, framework = isolated_enhanced_broker
        await broker.initialize()

        # Setup agent load information
        agents = ["worker1", "worker2", "worker3"]
        loads = [0.2, 0.8, 0.5]  # Different load levels
        capacities = [10, 10, 10]

        for agent, load, capacity in zip(agents, loads, capacities):
            await broker.update_agent_load(agent, load, capacity)

        # Clear previous interactions
        framework.mock_redis_client.interactions.clear()

        # Route message to best available agent
        selected_agent = await broker.route_message_intelligently(
            topic="processing_task",
            message_priority=MessagePriority.HIGH,
            required_capacity=1,
        )

        # Should select agent with lowest load (worker1 with 0.2)
        assert selected_agent == "worker1"

        # Verify Redis operations for load checking
        redis_interactions = framework.mock_redis_client.interactions
        keys_calls = [i for i in redis_interactions if i.method == "keys"]
        hgetall_calls = [i for i in redis_interactions if i.method == "hgetall"]

        assert len(keys_calls) >= 1  # To find agent load keys
        assert len(hgetall_calls) >= len(agents)  # To get each agent's load data

    @pytest.mark.asyncio
    async def test_batch_messaging_isolation(self, isolated_enhanced_broker):
        """Test batch messaging in isolation."""
        broker, framework = isolated_enhanced_broker
        await broker.initialize()

        # Clear previous interactions
        framework.mock_redis_client.interactions.clear()

        # Prepare batch messages
        batch_messages = [
            {
                "from_agent": "batch_sender",
                "to_agent": f"receiver_{i}",
                "topic": "batch_notification",
                "payload": {"batch_index": i, "data": f"message_{i}"},
                "priority": MessagePriority.NORMAL,
                "message_type": MessageType.DIRECT,
            }
            for i in range(5)
        ]

        # Send batch
        success = await broker.send_message_batch(batch_messages)

        # Verify batch succeeded
        assert success is True

        # Verify all messages were processed
        redis_interactions = framework.mock_redis_client.interactions
        publish_calls = [i for i in redis_interactions if i.method == "publish"]

        # Should have multiple publish calls (at least one per message in batch)
        assert len(publish_calls) >= len(batch_messages)

        # Verify messages went to correct recipients
        published_channels = [
            call.args[0] for call in publish_calls if call.args[0].startswith("agent:")
        ]
        expected_channels = [f"agent:receiver_{i}" for i in range(5)]

        for expected_channel in expected_channels:
            assert expected_channel in published_channels

    @pytest.mark.asyncio
    async def test_performance_metrics_isolation(self, isolated_enhanced_broker):
        """Test enhanced communication performance metrics in isolation."""
        broker, framework = isolated_enhanced_broker
        await broker.initialize()

        # Create some shared contexts and agent states
        await broker.create_shared_context(
            context_type=ContextShareType.TASK_STATE,
            owner_agent="metrics_agent",
            initial_data={"test": "data"},
        )

        await broker.update_agent_state("metrics_agent", {"status": "active"})

        # Get performance metrics
        metrics = await broker.get_communication_performance_metrics()

        # Verify metrics structure
        assert "enhanced_features" in metrics
        enhanced_metrics = metrics["enhanced_features"]

        assert "shared_contexts_active" in enhanced_metrics
        assert enhanced_metrics["shared_contexts_active"] >= 1

        assert "agents_with_state" in enhanced_metrics
        assert enhanced_metrics["agents_with_state"] >= 1

        assert "context_subscribers" in enhanced_metrics
        assert "sync_tasks_running" in enhanced_metrics

        # Verify context-related metrics
        if enhanced_metrics["shared_contexts_active"] > 0:
            assert "average_participants_per_context" in enhanced_metrics
            assert "real_time_sync_ratio" in enhanced_metrics

    @pytest.mark.asyncio
    async def test_error_scenarios_isolation(self, isolated_enhanced_broker):
        """Test enhanced broker error handling in isolation."""
        broker, framework = isolated_enhanced_broker
        await broker.initialize()

        # Test invalid context access
        context_data = await broker.get_shared_context(
            "nonexistent_context", "test_agent"
        )
        assert context_data is None

        # Test updating non-existent context
        success = await broker.update_shared_context(
            context_id="nonexistent", agent_name="test_agent", updates={"data": "value"}
        )
        assert success is False

        # Test joining context as non-participant
        context_id = await broker.create_shared_context(
            context_type=ContextShareType.TASK_STATE,
            owner_agent="owner",
            initial_data={"private": "data"},
        )

        # Try to update as non-participant
        success = await broker.update_shared_context(
            context_id=context_id,
            agent_name="unauthorized_agent",
            updates={"hack": "attempt"},
        )
        assert success is False

        # Verify context data wasn't modified
        context_data = await broker.get_shared_context(context_id, "owner")
        assert "hack" not in context_data["data"]

    @pytest.mark.asyncio
    async def test_context_expiration_isolation(self, isolated_enhanced_broker):
        """Test context TTL and expiration handling in isolation."""
        broker, framework = isolated_enhanced_broker
        await broker.initialize()

        # Create context with short TTL
        context_id = await broker.create_shared_context(
            context_type=ContextShareType.WORK_SESSION,
            owner_agent="test_agent",
            initial_data={"session": "temporary"},
            ttl=1,  # 1 second
        )

        # Verify context exists
        assert context_id in broker.shared_contexts

        # Simulate expiration by manipulating mock Redis
        context_key = f"hive:shared_context:{context_id}"
        if context_key in framework.mock_redis_client.expirations:
            # Set expiration to past time
            framework.mock_redis_client.expirations[context_key] = time.time() - 1

        # Try to load expired context
        loaded_context = await broker._load_shared_context(context_id)

        # Should return None for expired context
        assert loaded_context is None

    @pytest.mark.asyncio
    async def test_enhanced_broker_shutdown_isolation(self, isolated_enhanced_broker):
        """Test enhanced broker shutdown in isolation."""
        broker, framework = isolated_enhanced_broker
        await broker.initialize()

        # Verify sync tasks are running
        assert len(broker.sync_tasks) == 2
        running_tasks = [t for t in broker.sync_tasks.values() if not t.done()]
        assert len(running_tasks) == 2

        # Clear previous interactions
        framework.mock_redis_client.interactions.clear()

        # Shutdown broker
        await broker.shutdown()

        # Verify sync tasks were cancelled
        for task in broker.sync_tasks.values():
            assert task.done()

        # Verify Redis connection was closed
        redis_interactions = framework.mock_redis_client.interactions
        close_calls = [i for i in redis_interactions if i.method in ["close", "aclose"]]
        assert len(close_calls) >= 1

        # Verify no external resources were leaked
        framework.assert_no_external_calls()


class TestEnhancedMessageBrokerPerformance:
    """Performance tests for EnhancedMessageBroker in isolation."""

    @pytest.mark.asyncio
    async def test_high_volume_context_operations_isolation(self):
        """Test high-volume context operations performance in isolation."""
        config = ComponentTestConfig(record_interactions=False)
        framework = ComponentIsolationTestFramework(config)

        async with framework.isolate_component(
            lambda: None, redis_url="redis://mock:6379"
        ) as _:
            from src.core.enhanced_message_broker import EnhancedMessageBroker

            broker = EnhancedMessageBroker("redis://mock:6379")
            broker.redis_client = framework.mock_redis_client

            await broker.initialize()

            # Create many shared contexts
            start_time = time.time()
            context_count = 50

            context_ids = []
            for i in range(context_count):
                context_id = await broker.create_shared_context(
                    context_type=ContextShareType.TASK_STATE,
                    owner_agent=f"agent_{i % 10}",  # 10 different agents
                    initial_data={"task_id": i, "status": "active"},
                )
                context_ids.append(context_id)

            end_time = time.time()
            creation_duration = end_time - start_time

            # Test context updates performance
            start_time = time.time()

            for i, context_id in enumerate(context_ids):
                await broker.update_shared_context(
                    context_id=context_id,
                    agent_name=f"agent_{i % 10}",
                    updates={"status": "updated", "timestamp": time.time()},
                )

            end_time = time.time()
            update_duration = end_time - start_time

            # Performance assertions
            contexts_per_second = context_count / creation_duration
            updates_per_second = context_count / update_duration

            assert contexts_per_second > 10  # Should create at least 10 contexts/second
            assert updates_per_second > 10  # Should update at least 10 contexts/second

            logger.info(
                f"Context performance: {context_count} contexts created in {creation_duration:.2f}s "
                f"({contexts_per_second:.1f} ctx/s), updated in {update_duration:.2f}s "
                f"({updates_per_second:.1f} upd/s)"
            )

            # Verify all contexts were created and updated
            assert len(broker.shared_contexts) == context_count

            for context in broker.shared_contexts.values():
                assert context.data["status"] == "updated"
                assert "timestamp" in context.data


if __name__ == "__main__":
    # Run individual test for debugging
    async def run_single_test():
        test_instance = TestEnhancedMessageBrokerIsolation()
        broker, framework = await create_isolated_enhanced_message_broker()
        await test_instance.test_create_shared_context_isolation((broker, framework))
        print("Enhanced broker test completed successfully")

    asyncio.run(run_single_test())
