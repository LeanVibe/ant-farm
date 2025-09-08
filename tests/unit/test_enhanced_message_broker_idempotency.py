"""Test for Enhanced Message Broker idempotency behavior."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.core.enhanced_message_broker import EnhancedMessageBroker
from src.core.contracts import BrokerSendResult, BrokerSendReason


class TestEnhancedMessageBrokerIdempotency:
    """Test idempotency behavior specifically."""
    
    @pytest.mark.asyncio
    async def test_send_message_with_result_idempotency_duplicate(self):
        """Test that duplicate messages are detected correctly."""
        # Given: An initialized EnhancedMessageBroker
        redis_url = "redis://localhost:6379"
        
        with patch('src.core.enhanced_message_broker.redis_async_from_url') as mock_redis, \
             patch('src.core.enhanced_message_broker.get_message_broker_breaker') as mock_breaker:
            
            mock_redis_client = AsyncMock()
            mock_redis_client.ping = AsyncMock()
            mock_redis_client.pubsub.return_value = AsyncMock()
            mock_redis.return_value = mock_redis_client
            
            mock_circuit_breaker = AsyncMock()
            mock_breaker.return_value = mock_circuit_breaker
            
            # Mock Redis operations - first call succeeds, second call returns False (duplicate)
            call_count = 0
            def mock_call_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:  # First call to set idempotency key
                    return True  # Key is new
                elif call_count == 2:  # Second call to set the same key  
                    return False  # Key already exists (duplicate)
                return True  # Default for other calls
                
            mock_circuit_breaker.call = AsyncMock(side_effect=mock_call_side_effect)
            
            broker = EnhancedMessageBroker(redis_url=redis_url)
            await broker.initialize()
            
            # When: Sending first message with idempotency key
            result1 = await broker.send_message_with_result(
                from_agent="test_agent", 
                to_agent="target_agent", 
                topic="test_topic", 
                payload={"test": "data1"},
                idempotency_key="test_key_duplicate"
            )
            
            # Then: First message should succeed
            assert result1.success is True
            assert result1.reason == BrokerSendReason.OK
            
            # When: Sending second message with same idempotency key
            result2 = await broker.send_message_with_result(
                from_agent="test_agent", 
                to_agent="target_agent", 
                topic="test_topic", 
                payload={"test": "data2"},
                idempotency_key="test_key_duplicate"
            )
            
            # Then: Second message should be detected as duplicate
            assert result2.success is True  # Still successful from client perspective
            assert result2.reason == BrokerSendReason.IDEMPOTENT_DUPLICATE
            assert result2.message_id is None
            
    @pytest.mark.asyncio 
    async def test_send_message_with_result_idempotency_new(self):
        """Test that new messages with idempotency key work correctly."""
        # Given: An initialized EnhancedMessageBroker  
        redis_url = "redis://localhost:6379"
        
        with patch('src.core.enhanced_message_broker.redis_async_from_url') as mock_redis, \
             patch('src.core.enhanced_message_broker.get_message_broker_breaker') as mock_breaker:
            
            mock_redis_client = AsyncMock()
            mock_redis_client.ping = AsyncMock()
            mock_redis_client.pubsub.return_value = AsyncMock()
            mock_redis.return_value = mock_redis_client
            
            mock_circuit_breaker = AsyncMock()
            mock_breaker.return_value = mock_circuit_breaker
            
            # Mock Redis operations for new message
            mock_circuit_breaker.call = AsyncMock()
            mock_circuit_breaker.call.return_value = True  # Key is new
            
            broker = EnhancedMessageBroker(redis_url=redis_url)
            await broker.initialize()
            
            # When: Sending a new message with idempotency key
            result = await broker.send_message_with_result(
                from_agent="test_agent",
                to_agent="target_agent", 
                topic="test_topic",
                payload={"test": "data"},
                idempotency_key="test_key_new"
            )
            
            # Then: Should succeed
            assert result.success is True
            assert result.reason == BrokerSendReason.OK