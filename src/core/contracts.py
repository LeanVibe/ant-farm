"""Core contracts (pydantic) for broker/task envelopes.

These provide explicit shapes for messages and results, without
breaking existing call sites.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MessagePriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class BrokerMessageEnvelope(BaseModel):
    id: Optional[str] = None
    from_agent: str
    to_agent: str
    topic: str
    payload: dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    idempotency_key: Optional[str] = None


class BrokerSendReason(str, Enum):
    OK = "ok"
    IDEMPOTENT_DUPLICATE = "idempotent_duplicate"
    PUBLISH_ERROR = "publish_error"
    INVALID = "invalid"


class BrokerSendResult(BaseModel):
    success: bool = Field(..., description="Whether the message was accepted for delivery")
    reason: BrokerSendReason = BrokerSendReason.OK
    message_id: Optional[str] = None
    details: Optional[str] = None
