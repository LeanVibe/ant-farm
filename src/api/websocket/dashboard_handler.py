"""Enhanced WebSocket Infrastructure for Real-time Dashboard Communication.

This module provides comprehensive WebSocket support for the observability dashboard,
integrating with the metrics collection and event aggregation systems.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum

try:
    import structlog

    logger = structlog.get_logger()
    use_structlog = True
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    use_structlog = False

try:
    from fastapi import WebSocket, WebSocketDisconnect

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    WebSocket = None
    WebSocketDisconnect = None


def log_info(message: str, **kwargs):
    """Helper function for logging."""
    if use_structlog:
        logger.info(message, **kwargs)
    else:
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(f"{message} {extra_info}" if extra_info else message)


def log_error(message: str, **kwargs):
    """Helper function for logging."""
    if use_structlog:
        logger.error(message, **kwargs)
    else:
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.error(f"{message} {extra_info}" if extra_info else message)


class MessageType(Enum):
    """Types of WebSocket messages."""

    SYSTEM_STATUS = "system_status"
    AGENT_UPDATE = "agent_update"
    TASK_UPDATE = "task_update"
    METRICS_UPDATE = "metrics_update"
    ALERT_UPDATE = "alert_update"
    ERROR = "error"
    SUBSCRIPTION_CONFIRMED = "subscription_confirmed"
    HEARTBEAT = "heartbeat"
    DASHBOARD_DATA = "dashboard_data"


@dataclass
class ClientConnection:
    """Represents a WebSocket client connection."""

    client_id: str
    websocket: Optional["WebSocket"]
    connected_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    subscriptions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_alive(self, timeout: float = 60.0) -> bool:
        """Check if connection is still alive based on heartbeat."""
        return (time.time() - self.last_heartbeat) < timeout

    def update_heartbeat(self):
        """Update the last heartbeat timestamp."""
        self.last_heartbeat = time.time()


class DashboardWebSocketManager:
    """Enhanced WebSocket manager for real-time dashboard communication."""

    def __init__(self):
        self.connections: Dict[str, ClientConnection] = {}
        self.subscription_handlers: Dict[str, List[Callable]] = {}

        # Heartbeat and cleanup
        self.heartbeat_interval = 30.0  # seconds
        self.connection_timeout = 120.0  # seconds
        self.is_running = False
        self.heartbeat_task: Optional[asyncio.Task] = None

        # Message broadcasting
        self.broadcast_queue: asyncio.Queue = asyncio.Queue()
        self.broadcast_task: Optional[asyncio.Task] = None

        # Integration with observability system
        self.metrics_collector = None
        self.event_aggregator = None
        self.alert_manager = None

        log_info("DashboardWebSocketManager initialized")

    async def start(self):
        """Start the WebSocket manager."""
        if self.is_running:
            return

        self.is_running = True

        # Initialize observability integrations
        await self._initialize_integrations()

        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.broadcast_task = asyncio.create_task(self._broadcast_loop())

        log_info("WebSocket manager started")

    async def stop(self):
        """Stop the WebSocket manager."""
        self.is_running = False

        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass

        if self.broadcast_task:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass

        # Disconnect all clients
        for client in list(self.connections.values()):
            await self.disconnect_client(client.client_id)

        log_info("WebSocket manager stopped")

    async def _initialize_integrations(self):
        """Initialize integrations with observability systems."""
        try:
            from ..core.observability import (
                get_metrics_collector,
                get_event_aggregator,
                get_alert_manager,
            )

            self.metrics_collector = get_metrics_collector()
            self.event_aggregator = get_event_aggregator()
            self.alert_manager = get_alert_manager()

            # Subscribe to relevant events
            self.event_aggregator.subscribe_global(self._handle_system_event)
            self.alert_manager.add_alert_handler(self._handle_alert_event)

            log_info("WebSocket integrations initialized")

        except ImportError as e:
            log_error("Failed to initialize WebSocket integrations", error=str(e))

    async def connect_client(
        self, websocket: "WebSocket", client_id: Optional[str] = None
    ) -> str:
        """Connect a new WebSocket client."""
        if not HAS_FASTAPI:
            raise RuntimeError("FastAPI WebSocket support not available")

        if client_id is None:
            client_id = str(uuid.uuid4())

        await websocket.accept()

        client = ClientConnection(client_id=client_id, websocket=websocket)

        self.connections[client_id] = client

        # Send welcome message
        await self.send_to_client(
            client_id,
            {
                "type": MessageType.SYSTEM_STATUS.value,
                "payload": {
                    "status": "connected",
                    "client_id": client_id,
                    "timestamp": time.time(),
                    "server_capabilities": [
                        "metrics_streaming",
                        "event_notifications",
                        "alert_monitoring",
                        "real_time_dashboard",
                    ],
                },
            },
        )

        # Send initial dashboard data
        await self._send_initial_dashboard_data(client_id)

        # Emit connection event
        if self.event_aggregator:
            await self.event_aggregator.emit_websocket_event(
                "connected",
                client_id,
                {"user_agent": websocket.headers.get("user-agent")},
            )

        log_info(
            "WebSocket client connected",
            client_id=client_id,
            total_connections=len(self.connections),
        )

        return client_id

    async def disconnect_client(self, client_id: str):
        """Disconnect a WebSocket client."""
        if client_id not in self.connections:
            return

        client = self.connections[client_id]

        try:
            if client.websocket:
                await client.websocket.close()
        except Exception as e:
            log_error("Error closing WebSocket", client_id=client_id, error=str(e))

        del self.connections[client_id]

        # Emit disconnection event
        if self.event_aggregator:
            await self.event_aggregator.emit_websocket_event(
                "disconnected",
                client_id,
                {"duration": time.time() - client.connected_at},
            )

        log_info(
            "WebSocket client disconnected",
            client_id=client_id,
            total_connections=len(self.connections),
        )

    async def send_to_client(self, client_id: str, message: Dict[str, Any]) -> bool:
        """Send a message to a specific client."""
        if client_id not in self.connections:
            return False

        client = self.connections[client_id]

        try:
            if client.websocket:
                message_data = json.dumps(
                    {**message, "timestamp": time.time(), "client_id": client_id}
                )
                await client.websocket.send_text(message_data)
                return True
        except Exception as e:
            log_error(
                "Failed to send message to client", client_id=client_id, error=str(e)
            )
            await self.disconnect_client(client_id)

        return False

    async def broadcast_to_all(
        self, message: Dict[str, Any], subscription_filter: Optional[str] = None
    ):
        """Broadcast a message to all connected clients."""
        if not self.connections:
            return

        # Add to broadcast queue for processing
        await self.broadcast_queue.put(
            {"message": message, "subscription_filter": subscription_filter}
        )

    async def broadcast_to_subscribers(
        self, subscription: str, message: Dict[str, Any]
    ):
        """Broadcast a message to clients subscribed to a specific topic."""
        subscribers = [
            client_id
            for client_id, client in self.connections.items()
            if subscription in client.subscriptions
        ]

        for client_id in subscribers:
            await self.send_to_client(client_id, message)

    async def subscribe_client(self, client_id: str, subscriptions: List[str]):
        """Subscribe a client to specific message types."""
        if client_id not in self.connections:
            return False

        client = self.connections[client_id]
        client.subscriptions.update(subscriptions)

        await self.send_to_client(
            client_id,
            {
                "type": MessageType.SUBSCRIPTION_CONFIRMED.value,
                "payload": {
                    "subscriptions": list(client.subscriptions),
                    "newly_subscribed": subscriptions,
                },
            },
        )

        log_info("Client subscribed", client_id=client_id, subscriptions=subscriptions)
        return True

    async def unsubscribe_client(self, client_id: str, subscriptions: List[str]):
        """Unsubscribe a client from specific message types."""
        if client_id not in self.connections:
            return False

        client = self.connections[client_id]
        client.subscriptions.difference_update(subscriptions)

        await self.send_to_client(
            client_id,
            {
                "type": MessageType.SUBSCRIPTION_CONFIRMED.value,
                "payload": {
                    "subscriptions": list(client.subscriptions),
                    "unsubscribed_from": subscriptions,
                },
            },
        )

        return True

    async def handle_client_message(self, client_id: str, message_data: str):
        """Handle incoming message from a client."""
        try:
            message = json.loads(message_data)
            message_type = message.get("type")
            payload = message.get("payload", {})

            if message_type == "subscribe":
                subscriptions = payload.get("subscriptions", [])
                await self.subscribe_client(client_id, subscriptions)

            elif message_type == "unsubscribe":
                subscriptions = payload.get("subscriptions", [])
                await self.unsubscribe_client(client_id, subscriptions)

            elif message_type == "heartbeat":
                if client_id in self.connections:
                    self.connections[client_id].update_heartbeat()
                    await self.send_to_client(
                        client_id,
                        {
                            "type": MessageType.HEARTBEAT.value,
                            "payload": {"status": "pong", "server_time": time.time()},
                        },
                    )

            elif message_type == "request_dashboard_data":
                await self._send_dashboard_data(client_id)

            elif message_type == "request_metrics":
                await self._send_metrics_data(client_id, payload)

            elif message_type == "request_alerts":
                await self._send_alerts_data(client_id)

            else:
                await self.send_to_client(
                    client_id,
                    {
                        "type": MessageType.ERROR.value,
                        "payload": {
                            "message": f"Unknown message type: {message_type}",
                            "received_message": message,
                        },
                    },
                )

        except json.JSONDecodeError:
            await self.send_to_client(
                client_id,
                {
                    "type": MessageType.ERROR.value,
                    "payload": {"message": "Invalid JSON format"},
                },
            )
        except Exception as e:
            log_error(
                "Error handling client message", client_id=client_id, error=str(e)
            )
            await self.send_to_client(
                client_id,
                {
                    "type": MessageType.ERROR.value,
                    "payload": {"message": f"Error processing message: {str(e)}"},
                },
            )

    async def _heartbeat_loop(self):
        """Background task for sending heartbeats and checking connections."""
        while self.is_running:
            try:
                current_time = time.time()
                disconnected_clients = []

                for client_id, client in self.connections.items():
                    # Check for stale connections
                    if not client.is_alive(self.connection_timeout):
                        disconnected_clients.append(client_id)
                        continue

                    # Send periodic heartbeat
                    if (current_time - client.last_heartbeat) > self.heartbeat_interval:
                        success = await self.send_to_client(
                            client_id,
                            {
                                "type": MessageType.HEARTBEAT.value,
                                "payload": {
                                    "status": "ping",
                                    "server_time": current_time,
                                },
                            },
                        )

                        if success:
                            client.update_heartbeat()

                # Clean up disconnected clients
                for client_id in disconnected_clients:
                    await self.disconnect_client(client_id)

                await asyncio.sleep(self.heartbeat_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error("Error in heartbeat loop", error=str(e))
                await asyncio.sleep(10)

    async def _broadcast_loop(self):
        """Background task for processing broadcast messages."""
        while self.is_running:
            try:
                # Wait for broadcast message
                broadcast_item = await asyncio.wait_for(
                    self.broadcast_queue.get(), timeout=1.0
                )

                message = broadcast_item["message"]
                subscription_filter = broadcast_item["subscription_filter"]

                # Send to appropriate clients
                if subscription_filter:
                    await self.broadcast_to_subscribers(subscription_filter, message)
                else:
                    # Send to all clients
                    for client_id in list(self.connections.keys()):
                        await self.send_to_client(client_id, message)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error("Error in broadcast loop", error=str(e))
                await asyncio.sleep(1)

    async def _send_initial_dashboard_data(self, client_id: str):
        """Send initial dashboard data to a newly connected client."""
        await self._send_dashboard_data(client_id)
        await self._send_metrics_data(client_id)
        await self._send_alerts_data(client_id)

    async def _send_dashboard_data(self, client_id: str):
        """Send comprehensive dashboard data."""
        try:
            dashboard_data = {}

            if self.metrics_collector:
                dashboard_data["metrics_summary"] = (
                    self.metrics_collector.get_metrics_summary()
                )

            if self.event_aggregator:
                dashboard_data["event_statistics"] = (
                    self.event_aggregator.get_event_statistics()
                )
                dashboard_data["recent_events"] = [
                    event.to_dict()
                    for event in self.event_aggregator.get_recent_events(300)
                ]

            if self.alert_manager:
                dashboard_data["alert_statistics"] = (
                    self.alert_manager.get_alert_statistics()
                )
                dashboard_data["active_alerts"] = [
                    alert.to_dict() for alert in self.alert_manager.get_active_alerts()
                ]

            await self.send_to_client(
                client_id,
                {"type": MessageType.DASHBOARD_DATA.value, "payload": dashboard_data},
            )

        except Exception as e:
            log_error("Error sending dashboard data", client_id=client_id, error=str(e))

    async def _send_metrics_data(
        self, client_id: str, filters: Optional[Dict[str, Any]] = None
    ):
        """Send metrics data with optional filtering."""
        try:
            if not self.metrics_collector:
                return

            metrics_data = {
                "summary": self.metrics_collector.get_metrics_summary(),
                "recent_metrics": [],
            }

            # Get recent metrics based on filters
            time_window = filters.get("time_window", 300) if filters else 300
            metric_name_filter = filters.get("metric_name") if filters else None
            tags_filter = filters.get("tags") if filters else None

            recent_metrics = self.metrics_collector.get_recent_metrics(
                time_window_seconds=time_window,
                metric_name_filter=metric_name_filter,
                tags_filter=tags_filter,
            )

            metrics_data["recent_metrics"] = [
                {
                    "name": point.name,
                    "value": point.value,
                    "timestamp": point.timestamp,
                    "metric_type": point.metric_type.value,
                    "tags": point.tags,
                    "metadata": point.metadata,
                }
                for point in recent_metrics[-100:]  # Limit to last 100 points
            ]

            await self.send_to_client(
                client_id,
                {"type": MessageType.METRICS_UPDATE.value, "payload": metrics_data},
            )

        except Exception as e:
            log_error("Error sending metrics data", client_id=client_id, error=str(e))

    async def _send_alerts_data(self, client_id: str):
        """Send current alerts data."""
        try:
            if not self.alert_manager:
                return

            alerts_data = {
                "statistics": self.alert_manager.get_alert_statistics(),
                "active_alerts": [
                    alert.to_dict() for alert in self.alert_manager.get_active_alerts()
                ],
                "recent_resolved": [
                    alert.to_dict()
                    for alert in self.alert_manager.get_resolved_alerts(20)
                ],
            }

            await self.send_to_client(
                client_id,
                {"type": MessageType.ALERT_UPDATE.value, "payload": alerts_data},
            )

        except Exception as e:
            log_error("Error sending alerts data", client_id=client_id, error=str(e))

    async def _handle_system_event(self, event):
        """Handle system events from the event aggregator."""
        try:
            await self.broadcast_to_subscribers(
                "system_events",
                {
                    "type": MessageType.SYSTEM_STATUS.value,
                    "payload": {
                        "event": event.to_dict(),
                        "event_type": event.event_type.value,
                    },
                },
            )
        except Exception as e:
            log_error("Error handling system event", error=str(e))

    async def _handle_alert_event(self, alert, action):
        """Handle alert events from the alert manager."""
        try:
            await self.broadcast_to_subscribers(
                "alerts",
                {
                    "type": MessageType.ALERT_UPDATE.value,
                    "payload": {
                        "alert": alert.to_dict(),
                        "action": action,
                        "timestamp": time.time(),
                    },
                },
            )
        except Exception as e:
            log_error("Error handling alert event", error=str(e))

    def get_connection_statistics(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        total_connections = len(self.connections)

        # Subscription statistics
        subscription_counts = {}
        for client in self.connections.values():
            for subscription in client.subscriptions:
                subscription_counts[subscription] = (
                    subscription_counts.get(subscription, 0) + 1
                )

        # Connection health
        healthy_connections = sum(
            1 for client in self.connections.values() if client.is_alive()
        )

        return {
            "total_connections": total_connections,
            "healthy_connections": healthy_connections,
            "subscription_distribution": subscription_counts,
            "average_connection_duration": sum(
                time.time() - client.connected_at
                for client in self.connections.values()
            )
            / total_connections
            if total_connections > 0
            else 0,
            "heartbeat_interval": self.heartbeat_interval,
            "connection_timeout": self.connection_timeout,
            "is_running": self.is_running,
        }


# Global WebSocket manager instance
_global_websocket_manager: Optional[DashboardWebSocketManager] = None


def get_websocket_manager() -> DashboardWebSocketManager:
    """Get the global WebSocket manager instance."""
    global _global_websocket_manager

    if _global_websocket_manager is None:
        _global_websocket_manager = DashboardWebSocketManager()

    return _global_websocket_manager


async def start_websocket_manager():
    """Start the global WebSocket manager."""
    manager = get_websocket_manager()
    await manager.start()


async def stop_websocket_manager():
    """Stop the global WebSocket manager."""
    global _global_websocket_manager

    if _global_websocket_manager:
        await _global_websocket_manager.stop()
