import pytest
from src.core.task_queue import TaskQueue


@pytest.mark.asyncio
def test_task_queue_interface_compliance(task_queue_instance):
    """
    Ensure TaskQueue and any mock/subclass implements required interface.
    """
    assert hasattr(task_queue_instance, "get_failed_tasks"), (
        "TaskQueue must implement get_failed_tasks()"
    )
    assert callable(getattr(task_queue_instance, "get_failed_tasks", None)), (
        "get_failed_tasks must be callable"
    )
    assert hasattr(task_queue_instance, "get_failed_tasks_count"), (
        "TaskQueue must implement get_failed_tasks_count()"
    )
    assert callable(getattr(task_queue_instance, "get_failed_tasks_count", None)), (
        "get_failed_tasks_count must be callable"
    )
    assert hasattr(task_queue_instance, "get_total_tasks"), (
        "TaskQueue must implement get_total_tasks()"
    )
    assert callable(getattr(task_queue_instance, "get_total_tasks", None)), (
        "get_total_tasks must be callable"
    )
    assert hasattr(task_queue_instance, "get_completed_tasks"), (
        "TaskQueue must implement get_completed_tasks()"
    )
    assert callable(getattr(task_queue_instance, "get_completed_tasks", None)), (
        "get_completed_tasks must be callable"
    )
    assert hasattr(task_queue_instance, "get_queue_depth"), (
        "TaskQueue must implement get_queue_depth()"
    )
    assert callable(getattr(task_queue_instance, "get_queue_depth", None)), (
        "get_queue_depth must be callable"
    )
