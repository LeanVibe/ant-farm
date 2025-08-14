import pytest
from src.cli.plan_schema import Plan, TaskType, validate_plan_dict


def test_validate_minimal_plan():
    data = {
        "version": 1,
        "plan_id": "p1",
        "goal": "demo",
        "batches": [
            {
                "name": "b1",
                "tasks": [{"type": "command", "run": "echo hi"}],
                "verify": [{"run": "true"}],
            }
        ],
    }
    plan = validate_plan_dict(data)
    assert isinstance(plan, Plan)
    assert plan.plan_id == "p1"


def test_invalid_edit_without_file():
    data = {
        "version": 1,
        "plan_id": "p1",
        "goal": "demo",
        "batches": [
            {
                "name": "b1",
                "tasks": [{"type": "edit", "apply": "missing file"}],
            }
        ],
    }
    with pytest.raises(ValueError):
        validate_plan_dict(data)


def test_depends_on_and_env_fields():
    data = {
        "version": 1,
        "plan_id": "p2",
        "goal": "demo",
        "batches": [
            {"name": "a", "tasks": []},
            {
                "name": "b",
                "depends_on": ["a"],
                "env": {"FOO": "bar"},
                "tasks": [{"type": "command", "run": "echo hi"}],
            },
        ],
    }
    plan = validate_plan_dict(data)
    assert plan.batches[1].depends_on == ["a"]
    assert plan.batches[1].env == {"FOO": "bar"}
