"""Test for Enhanced Message Broker redis_breaker integration."""

import pytest
from unittest.mock import AsyncMock, patch
from src.core.enhanced_message_broker import EnhancedMessageBroker


class TestEnhancedMessageBrokerRedisBreaker:
    """Test that EnhancedMessageBroker properly initializes redis_breaker."""
    
    @pytest.mark.asyncio
    async def test_enhanced_message_broker_has_redis_breaker(self):
        """Test that EnhancedMessageBroker initializes with redis_breaker attribute."""
        # Given: A mock redis URL and context engine
        redis_url = "redis://localhost:6379"
        
        # When: Creating an EnhancedMessageBroker instance
        with patch('src.core.enhanced_message_broker.redis_async_from_url') as mock_redis, \
             patch('src.core.enhanced_message_broker.get_message_broker_breaker') as mock_breaker:
            
            mock_redis_client = AsyncMock()
            mock_redis_client.ping = AsyncMock()
            mock_redis_client.pubsub.return_value = AsyncMock()
            mock_redis.return_value = mock_redis_client
            
            mock_circuit_breaker = AsyncMock()
            mock_breaker.return_value = mock_circuit_breaker
            
            broker = EnhancedMessageBroker(redis_url=redis_url)
            await broker.initialize()
            
            # Then: The broker should have redis_breaker attribute
            assert hasattr(broker, 'redis_breaker')
            assert broker.redis_breaker is not None
    
    @pytest.mark.asyncio
    async def test_enhanced_message_broker_send_message_uses_circuit_breaker(self):
        """Test that send_message uses circuit breaker for Redis operations."""
        # Given: An initialized EnhancedMessageBroker
        redis_url = "redis://localhost:6379"
        
        with patch('src.core.enhanced_message_broker.redis_async_from_url') as mock_redis, \
             patch('src.core.enhanced_message_broker.get_message_broker_breaker') as mock_breaker:
            
            mock_redis_client = AsyncMock()
            mock_redis_client.ping = AsyncMock()
            mock_redis_client.pubsub.return_value = AsyncMock()
            mock_redis.return_value = mock_redis_client
            
            mock_circuit_breaker = AsyncMock()
            mock_circuit_breaker.call = AsyncMock(return_value=True)
            mock_breaker.return_value = mock_circuit_breaker
            
            broker = EnhancedMessageBroker(redis_url=redis_url)
            await broker.initialize()
            
            # When: Sending a message
            result = await broker.send_message("agent1", "agent2", "test topic", {"test": "data"})
            
            # Then: Circuit breaker should have been called
            assert mock_circuit_breaker.call.called
            
    def test_enhanced_message_broker_inherits_from_message_broker(self):
        """Test that EnhancedMessageBroker properly inherits from MessageBroker."""
        from src.core.message_broker import MessageBroker
        
        # Given/When: Creating EnhancedMessageBroker class
        broker_class = EnhancedMessageBroker
        
        # Then: It should inherit from MessageBroker
        assert issubclass(broker_class, MessageBroker)