"""Contract testing framework for message formats and system interfaces.

This framework validates that all components adhere to their defined contracts,
ensuring compatibility and preventing breaking changes across the system.

Key Features:
- Message format validation for all communication protocols
- API endpoint contract validation
- Database schema contract validation
- Agent behavior contract validation
- Version compatibility checking
- Backward compatibility testing
- Contract evolution tracking
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

import pytest
import structlog
from pydantic import BaseModel, ValidationError, validator

logger = structlog.get_logger()


class ContractType(Enum):
    """Types of contracts that can be validated."""

    MESSAGE_FORMAT = "message_format"
    API_ENDPOINT = "api_endpoint"
    DATABASE_SCHEMA = "database_schema"
    AGENT_BEHAVIOR = "agent_behavior"
    EVENT_FORMAT = "event_format"
    CONFIGURATION = "configuration"


class ContractSeverity(Enum):
    """Severity levels for contract violations."""

    CRITICAL = "critical"  # Breaking change that prevents operation
    MAJOR = "major"  # Breaking change with workaround possible
    MINOR = "minor"  # Non-breaking change, may affect performance
    WARNING = "warning"  # Style or best practice violation


@dataclass
class ContractViolation:
    """Represents a contract violation."""

    contract_id: str
    violation_type: str
    severity: ContractSeverity
    field_path: str
    expected: Any
    actual: Any
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ContractValidationResult:
    """Result of contract validation."""

    contract_id: str
    contract_type: ContractType
    valid: bool
    violations: List[ContractViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Message Format Contracts


class MessageContract(BaseModel):
    """Contract for message format validation."""

    id: str
    from_agent: str
    to_agent: str
    topic: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: float
    expires_at: Optional[float] = None
    reply_to: Optional[str] = None
    correlation_id: Optional[str] = None
    priority: int = 5
    delivery_count: int = 0
    max_retries: int = 3

    @validator("message_type")
    def validate_message_type(cls, v):
        valid_types = [
            "direct",
            "broadcast",
            "multicast",
            "request",
            "reply",
            "notification",
        ]
        if v not in valid_types:
            raise ValueError(f"message_type must be one of {valid_types}")
        return v

    @validator("priority")
    def validate_priority(cls, v):
        if not 1 <= v <= 9:
            raise ValueError("priority must be between 1 and 9")
        return v

    @validator("from_agent", "to_agent")
    def validate_agent_names(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("agent names cannot be empty")
        return v

    @validator("topic")
    def validate_topic(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("topic cannot be empty")
        if len(v) > 100:
            raise ValueError("topic cannot exceed 100 characters")
        return v


class SharedContextContract(BaseModel):
    """Contract for shared context format validation."""

    id: str
    type: str
    owner_agent: str
    participants: List[str]
    data: Dict[str, Any]
    version: int
    last_updated: float
    last_updated_by: str
    sync_mode: str
    ttl: Optional[float] = None
    metadata: Dict[str, Any] = {}

    @validator("type")
    def validate_context_type(cls, v):
        valid_types = [
            "work_session",
            "knowledge_base",
            "task_state",
            "performance_metrics",
            "error_patterns",
            "decision_history",
        ]
        if v not in valid_types:
            raise ValueError(f"context type must be one of {valid_types}")
        return v

    @validator("sync_mode")
    def validate_sync_mode(cls, v):
        valid_modes = ["real_time", "batched", "on_demand", "conflict_resolution"]
        if v not in valid_modes:
            raise ValueError(f"sync_mode must be one of {valid_modes}")
        return v

    @validator("participants")
    def validate_participants(cls, v):
        if not v:
            raise ValueError("participants list cannot be empty")
        if len(set(v)) != len(v):
            raise ValueError("participants list cannot contain duplicates")
        return v


class TaskContract(BaseModel):
    """Contract for task format validation."""

    id: str
    title: str
    description: str
    task_type: str
    payload: Dict[str, Any]
    priority: int
    status: str = "pending"
    assigned_agent: Optional[str] = None
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @validator("priority")
    def validate_priority(cls, v):
        if not 1 <= v <= 9:
            raise ValueError("priority must be between 1 and 9")
        return v

    @validator("status")
    def validate_status(cls, v):
        valid_statuses = [
            "pending",
            "assigned",
            "in_progress",
            "completed",
            "failed",
            "cancelled",
        ]
        if v not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")
        return v

    @validator("task_type")
    def validate_task_type(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("task_type cannot be empty")
        return v


# API Endpoint Contracts


class APIResponseContract(BaseModel):
    """Contract for API response format validation."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str
    request_id: str

    @validator("timestamp")
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("timestamp must be in ISO format")
        return v

    @validator("request_id")
    def validate_request_id(cls, v):
        if not v:
            raise ValueError("request_id cannot be empty")
        return v


class AgentStatusContract(BaseModel):
    """Contract for agent status information."""

    name: str
    agent_type: str
    status: str
    capabilities: List[str]
    current_task: Optional[str] = None
    last_activity: float
    performance_metrics: Dict[str, float] = {}

    @validator("status")
    def validate_status(cls, v):
        valid_statuses = ["active", "inactive", "busy", "error", "stopping"]
        if v not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")
        return v

    @validator("capabilities")
    def validate_capabilities(cls, v):
        if not isinstance(v, list):
            raise ValueError("capabilities must be a list")
        return v


# Database Schema Contracts


class DatabaseContract:
    """Base class for database schema contracts."""

    @staticmethod
    def validate_agent_table_schema(columns: Dict[str, str]) -> List[ContractViolation]:
        """Validate agent table schema."""
        violations = []

        required_columns = {
            "id": "UUID",
            "name": "VARCHAR",
            "agent_type": "VARCHAR",
            "role": "VARCHAR",
            "status": "VARCHAR",
            "created_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP",
        }

        for col_name, expected_type in required_columns.items():
            if col_name not in columns:
                violations.append(
                    ContractViolation(
                        contract_id="agent_table_schema",
                        violation_type="missing_column",
                        severity=ContractSeverity.CRITICAL,
                        field_path=f"columns.{col_name}",
                        expected=expected_type,
                        actual=None,
                        message=f"Required column '{col_name}' is missing",
                    )
                )
            elif not columns[col_name].upper().startswith(expected_type.upper()):
                violations.append(
                    ContractViolation(
                        contract_id="agent_table_schema",
                        violation_type="incorrect_column_type",
                        severity=ContractSeverity.MAJOR,
                        field_path=f"columns.{col_name}",
                        expected=expected_type,
                        actual=columns[col_name],
                        message=f"Column '{col_name}' has incorrect type",
                    )
                )

        return violations

    @staticmethod
    def validate_task_table_schema(columns: Dict[str, str]) -> List[ContractViolation]:
        """Validate task table schema."""
        violations = []

        required_columns = {
            "id": "UUID",
            "title": "VARCHAR",
            "description": "TEXT",
            "task_type": "VARCHAR",
            "priority": "INTEGER",
            "status": "VARCHAR",
            "payload": "JSONB",
            "created_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP",
        }

        for col_name, expected_type in required_columns.items():
            if col_name not in columns:
                violations.append(
                    ContractViolation(
                        contract_id="task_table_schema",
                        violation_type="missing_column",
                        severity=ContractSeverity.CRITICAL,
                        field_path=f"columns.{col_name}",
                        expected=expected_type,
                        actual=None,
                        message=f"Required column '{col_name}' is missing",
                    )
                )

        return violations


class ContractTestFramework:
    """Framework for validating system contracts."""

    def __init__(self):
        self.contracts: Dict[str, Any] = {}
        self.validation_results: List[ContractValidationResult] = []

    def register_contract(
        self, contract_id: str, contract_type: ContractType, contract_spec: Any
    ):
        """Register a contract for validation."""
        self.contracts[contract_id] = {
            "type": contract_type,
            "spec": contract_spec,
            "registered_at": datetime.now(),
        }

        logger.info(f"Contract registered: {contract_id} ({contract_type.value})")

    def validate_message_format(
        self, message_data: Dict[str, Any], contract_id: str = "default_message"
    ) -> ContractValidationResult:
        """Validate message against message format contract."""
        violations = []
        warnings = []

        try:
            # Validate using Pydantic model
            MessageContract(**message_data)

        except ValidationError as e:
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])

                violation = ContractViolation(
                    contract_id=contract_id,
                    violation_type="validation_error",
                    severity=ContractSeverity.CRITICAL,
                    field_path=field_path,
                    expected=error.get("ctx", {}).get("limit_value", "valid value"),
                    actual=error.get("input", "invalid value"),
                    message=error["msg"],
                )
                violations.append(violation)

        # Additional business logic validations
        if message_data.get("message_type") == "reply" and not message_data.get(
            "reply_to"
        ):
            violations.append(
                ContractViolation(
                    contract_id=contract_id,
                    violation_type="business_logic_error",
                    severity=ContractSeverity.MAJOR,
                    field_path="reply_to",
                    expected="message_id",
                    actual=None,
                    message="Reply messages must have reply_to field",
                )
            )

        # Check for deprecated fields
        deprecated_fields = ["legacy_field", "old_format"]
        for field in deprecated_fields:
            if field in message_data:
                warnings.append(f"Deprecated field '{field}' should be removed")

        result = ContractValidationResult(
            contract_id=contract_id,
            contract_type=ContractType.MESSAGE_FORMAT,
            valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            metadata={"message_size": len(json.dumps(message_data))},
        )

        self.validation_results.append(result)
        return result

    def validate_shared_context_format(
        self, context_data: Dict[str, Any], contract_id: str = "default_context"
    ) -> ContractValidationResult:
        """Validate shared context against format contract."""
        violations = []
        warnings = []

        try:
            SharedContextContract(**context_data)

        except ValidationError as e:
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])

                violation = ContractViolation(
                    contract_id=contract_id,
                    violation_type="validation_error",
                    severity=ContractSeverity.CRITICAL,
                    field_path=field_path,
                    expected="valid value",
                    actual=error.get("input", "invalid value"),
                    message=error["msg"],
                )
                violations.append(violation)

        # Business logic validations
        owner = context_data.get("owner_agent")
        participants = context_data.get("participants", [])

        if owner and owner not in participants:
            warnings.append("Owner agent should be included in participants list")

        # Check context data size
        data_size = len(json.dumps(context_data.get("data", {})))
        if data_size > 10000:  # 10KB limit
            violations.append(
                ContractViolation(
                    contract_id=contract_id,
                    violation_type="size_limit_exceeded",
                    severity=ContractSeverity.MINOR,
                    field_path="data",
                    expected="<= 10KB",
                    actual=f"{data_size} bytes",
                    message="Context data exceeds recommended size limit",
                )
            )

        result = ContractValidationResult(
            contract_id=contract_id,
            contract_type=ContractType.MESSAGE_FORMAT,
            valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            metadata={"data_size": data_size},
        )

        self.validation_results.append(result)
        return result

    def validate_task_format(
        self, task_data: Dict[str, Any], contract_id: str = "default_task"
    ) -> ContractValidationResult:
        """Validate task against format contract."""
        violations = []
        warnings = []

        try:
            TaskContract(**task_data)

        except ValidationError as e:
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])

                violation = ContractViolation(
                    contract_id=contract_id,
                    violation_type="validation_error",
                    severity=ContractSeverity.CRITICAL,
                    field_path=field_path,
                    expected="valid value",
                    actual=error.get("input", "invalid value"),
                    message=error["msg"],
                )
                violations.append(violation)

        # Business logic validations
        status = task_data.get("status")
        if status in ["completed", "failed"] and not task_data.get("completed_at"):
            warnings.append("Completed/failed tasks should have completed_at timestamp")

        if status == "completed" and not task_data.get("result"):
            warnings.append("Completed tasks should have result data")

        if status == "failed" and not task_data.get("error"):
            warnings.append("Failed tasks should have error message")

        result = ContractValidationResult(
            contract_id=contract_id,
            contract_type=ContractType.MESSAGE_FORMAT,
            valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
        )

        self.validation_results.append(result)
        return result

    def validate_api_response_format(
        self, response_data: Dict[str, Any], contract_id: str = "default_api"
    ) -> ContractValidationResult:
        """Validate API response against format contract."""
        violations = []
        warnings = []

        try:
            APIResponseContract(**response_data)

        except ValidationError as e:
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])

                violation = ContractViolation(
                    contract_id=contract_id,
                    violation_type="validation_error",
                    severity=ContractSeverity.CRITICAL,
                    field_path=field_path,
                    expected="valid value",
                    actual=error.get("input", "invalid value"),
                    message=error["msg"],
                )
                violations.append(violation)

        # Business logic validations
        success = response_data.get("success")
        data = response_data.get("data")
        error = response_data.get("error")

        if success and error:
            violations.append(
                ContractViolation(
                    contract_id=contract_id,
                    violation_type="business_logic_error",
                    severity=ContractSeverity.MAJOR,
                    field_path="error",
                    expected=None,
                    actual=error,
                    message="Successful responses should not have error field",
                )
            )

        if not success and not error:
            violations.append(
                ContractViolation(
                    contract_id=contract_id,
                    violation_type="business_logic_error",
                    severity=ContractSeverity.MAJOR,
                    field_path="error",
                    expected="error message",
                    actual=None,
                    message="Failed responses must have error field",
                )
            )

        result = ContractValidationResult(
            contract_id=contract_id,
            contract_type=ContractType.API_ENDPOINT,
            valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
        )

        self.validation_results.append(result)
        return result

    def validate_agent_status_format(
        self, status_data: Dict[str, Any], contract_id: str = "default_agent_status"
    ) -> ContractValidationResult:
        """Validate agent status against format contract."""
        violations = []
        warnings = []

        try:
            AgentStatusContract(**status_data)

        except ValidationError as e:
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])

                violation = ContractViolation(
                    contract_id=contract_id,
                    violation_type="validation_error",
                    severity=ContractSeverity.CRITICAL,
                    field_path=field_path,
                    expected="valid value",
                    actual=error.get("input", "invalid value"),
                    message=error["msg"],
                )
                violations.append(violation)

        # Business logic validations
        status = status_data.get("status")
        current_task = status_data.get("current_task")

        if status == "busy" and not current_task:
            warnings.append("Busy agents should have current_task specified")

        if status == "inactive" and current_task:
            warnings.append("Inactive agents should not have current_task")

        result = ContractValidationResult(
            contract_id=contract_id,
            contract_type=ContractType.AGENT_BEHAVIOR,
            valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
        )

        self.validation_results.append(result)
        return result

    def validate_database_schema(
        self, table_name: str, columns: Dict[str, str], contract_id: str = None
    ) -> ContractValidationResult:
        """Validate database table schema against contract."""
        if not contract_id:
            contract_id = f"{table_name}_schema"

        violations = []

        if table_name == "agents":
            violations.extend(DatabaseContract.validate_agent_table_schema(columns))
        elif table_name == "tasks":
            violations.extend(DatabaseContract.validate_task_table_schema(columns))
        else:
            # Generic table validation
            required_base_columns = ["id", "created_at", "updated_at"]
            for col in required_base_columns:
                if col not in columns:
                    violations.append(
                        ContractViolation(
                            contract_id=contract_id,
                            violation_type="missing_base_column",
                            severity=ContractSeverity.MAJOR,
                            field_path=f"columns.{col}",
                            expected="required base column",
                            actual=None,
                            message=f"Table missing required base column '{col}'",
                        )
                    )

        result = ContractValidationResult(
            contract_id=contract_id,
            contract_type=ContractType.DATABASE_SCHEMA,
            valid=len(violations) == 0,
            violations=violations,
        )

        self.validation_results.append(result)
        return result

    def validate_batch(
        self, validations: List[tuple]
    ) -> List[ContractValidationResult]:
        """Validate multiple contracts in batch."""
        results = []

        for validation_type, data, contract_id in validations:
            if validation_type == "message":
                result = self.validate_message_format(data, contract_id)
            elif validation_type == "context":
                result = self.validate_shared_context_format(data, contract_id)
            elif validation_type == "task":
                result = self.validate_task_format(data, contract_id)
            elif validation_type == "api":
                result = self.validate_api_response_format(data, contract_id)
            elif validation_type == "agent_status":
                result = self.validate_agent_status_format(data, contract_id)
            else:
                logger.warning(f"Unknown validation type: {validation_type}")
                continue

            results.append(result)

        return results

    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of all contract violations."""
        total_validations = len(self.validation_results)
        successful_validations = sum(1 for r in self.validation_results if r.valid)
        failed_validations = total_validations - successful_validations

        violations_by_severity = {}
        violations_by_type = {}

        for result in self.validation_results:
            for violation in result.violations:
                # Count by severity
                severity = violation.severity.value
                violations_by_severity[severity] = (
                    violations_by_severity.get(severity, 0) + 1
                )

                # Count by type
                violation_type = violation.violation_type
                violations_by_type[violation_type] = (
                    violations_by_type.get(violation_type, 0) + 1
                )

        return {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "failed_validations": failed_validations,
            "success_rate": successful_validations / total_validations
            if total_validations > 0
            else 0,
            "violations_by_severity": violations_by_severity,
            "violations_by_type": violations_by_type,
            "contracts_validated": list(
                set(r.contract_id for r in self.validation_results)
            ),
            "most_common_violations": sorted(
                violations_by_type.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    def assert_no_critical_violations(self):
        """Assert that no critical contract violations exist."""
        critical_violations = []

        for result in self.validation_results:
            for violation in result.violations:
                if violation.severity == ContractSeverity.CRITICAL:
                    critical_violations.append(violation)

        if critical_violations:
            violation_messages = [
                f"{v.contract_id}: {v.message}" for v in critical_violations
            ]
            pytest.fail(
                f"Critical contract violations found:\n" + "\n".join(violation_messages)
            )

    def assert_contract_compatibility(
        self, old_contract: Dict[str, Any], new_contract: Dict[str, Any]
    ):
        """Assert that contract changes maintain backward compatibility."""
        # This is a simplified compatibility check
        # In practice, this would be more sophisticated

        if old_contract.get("version", 1) > new_contract.get("version", 1):
            pytest.fail("Contract version cannot go backwards")

        # Check for removed required fields
        old_required = set(old_contract.get("required_fields", []))
        new_required = set(new_contract.get("required_fields", []))

        removed_required = old_required - new_required
        if removed_required:
            pytest.fail(f"Cannot remove required fields: {removed_required}")


# Example usage and test helpers
def create_valid_message() -> Dict[str, Any]:
    """Create a valid message for testing."""
    return {
        "id": str(uuid.uuid4()),
        "from_agent": "test_agent",
        "to_agent": "target_agent",
        "topic": "test_topic",
        "message_type": "direct",
        "payload": {"data": "test_value"},
        "timestamp": 1234567890.0,
        "priority": 5,
    }


def create_valid_shared_context() -> Dict[str, Any]:
    """Create a valid shared context for testing."""
    return {
        "id": str(uuid.uuid4()),
        "type": "work_session",
        "owner_agent": "owner",
        "participants": ["owner", "participant1"],
        "data": {"session_id": "test_session"},
        "version": 1,
        "last_updated": 1234567890.0,
        "last_updated_by": "owner",
        "sync_mode": "real_time",
    }


def create_valid_task() -> Dict[str, Any]:
    """Create a valid task for testing."""
    return {
        "id": str(uuid.uuid4()),
        "title": "Test Task",
        "description": "A test task",
        "task_type": "development",
        "payload": {"instructions": "Do something"},
        "priority": 5,
        "status": "pending",
    }


def create_valid_api_response() -> Dict[str, Any]:
    """Create a valid API response for testing."""
    return {
        "success": True,
        "data": {"result": "operation completed"},
        "timestamp": "2024-01-01T00:00:00Z",
        "request_id": str(uuid.uuid4()),
    }


if __name__ == "__main__":
    # Example usage
    framework = ContractTestFramework()

    # Test message validation
    valid_message = create_valid_message()
    result = framework.validate_message_format(valid_message)
    print(f"Message validation: {'PASS' if result.valid else 'FAIL'}")

    # Test with invalid message
    invalid_message = create_valid_message()
    invalid_message["priority"] = 15  # Invalid priority
    result = framework.validate_message_format(invalid_message, "invalid_message")
    print(f"Invalid message validation: {'PASS' if result.valid else 'FAIL'}")

    # Print summary
    summary = framework.get_violation_summary()
    print(f"Validation summary: {summary}")
