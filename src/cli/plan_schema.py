"""Plan schema for YAML-driven batch execution.

Defines strict Pydantic models used by the plan runner to validate
and materialize Claude-like plan/delegate workflows in Cursor.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator


class TaskType(str, Enum):
    EDIT = "edit"
    TESTS = "tests"
    COMMAND = "command"


class Task(BaseModel):
    """A single task within a batch.

    - edit: describe a high-level edit to perform in a file
    - tests: declare test files to be added or executed
    - command: arbitrary command to run (short-lived)
    """

    type: TaskType
    file: Optional[str] = Field(default=None, description="Target file for edit")
    apply: Optional[str] = Field(default=None, description="High-level edit description")
    files: Optional[List[str]] = Field(default=None, description="List of files for tests task")
    run: Optional[str] = Field(default=None, description="Command for command task")

    @model_validator(mode="after")
    def _validate_shape(self) -> "Task":
        if self.type == TaskType.EDIT and not self.file:
            raise ValueError("edit task requires 'file'")
        if self.type == TaskType.TESTS and not self.files:
            raise ValueError("tests task requires 'files'")
        if self.type == TaskType.COMMAND and not self.run:
            raise ValueError("command task requires 'run'")
        return self


class VerifyStep(BaseModel):
    run: str


class Commit(BaseModel):
    message: str


class Batch(BaseModel):
    name: str
    tasks: List[Task] = Field(default_factory=list)
    verify: List[VerifyStep] = Field(default_factory=list)
    commit: Optional[Commit] = None
    depends_on: List[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class Plan(BaseModel):
    version: int = 1
    plan_id: str
    goal: str
    batches: List[Batch]


def validate_plan_dict(data: dict) -> Plan:
    """Validate a raw dict into a Plan model, raising a clear error on failure."""
    try:
        return Plan.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid plan: {exc}") from exc
