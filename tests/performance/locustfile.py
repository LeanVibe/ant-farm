"""Locust performance test file for LeanVibe Agent Hive API."""

from locust import HttpUser, task, between, events
import random
import string


class LeanVibeUser(HttpUser):
    """Simulate a user interacting with the LeanVibe API."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    def on_start(self):
        """Initialize user session."""
        # Create a test agent for this user
        self.agent_name = f"test_agent_{random.randint(1000, 9999)}"
        self.task_counter = 0

    @task(3)
    def health_check(self):
        """Check API health endpoint."""
        self.client.get("/health")

    @task(2)
    def get_agents(self):
        """Get list of agents."""
        self.client.get("/api/v1/agents")

    @task(1)
    def create_task(self):
        """Create a new task."""
        self.task_counter += 1
        task_data = {
            "title": f"Performance Test Task {self.task_counter}",
            "description": f"Task created during performance testing by {self.agent_name}",
            "type": "performance_test",
            "priority": random.choice(["low", "normal", "high", "critical"]),
        }
        self.client.post("/api/v1/tasks", json=task_data)

    @task(1)
    def get_tasks(self):
        """Get list of tasks."""
        self.client.get("/api/v1/tasks")

    def on_stop(self):
        """Clean up user session."""
        pass


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when a new test is started."""
    print("Starting performance test for LeanVibe Agent Hive API")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when a test is stopped."""
    print("Performance test completed for LeanVibe Agent Hive API")
