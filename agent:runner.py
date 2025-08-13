#!/usr/bin/env python3
"""Agent runner that executes in tmux sessions on the host.

This polls tasks from Redis and executes them using Claude Code CLI.
No API key needed - uses the host's Claude Code installation.
"""

import json
import subprocess
import time
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import psycopg2
import redis
import typer
from psycopg2.extras import RealDictCursor

# Add src to path for imports
src_path = Path(__file__).parent / "src"
if src_path.exists():
    import sys

    sys.path.insert(0, str(src_path))

# Import configuration
try:
    from src.core.config import get_settings

    settings = get_settings()
except ImportError:
    # Fallback configuration if imports fail
    class MockSettings:
        database_url = "postgresql://hive_user:hive_pass@localhost:5433/leanvibe_hive"
        redis_url = "redis://localhost:6381"

    settings = MockSettings()

app = typer.Typer()


class AgentRunner:
    """Runs in tmux session, polls tasks, executes via Claude Code."""

    def __init__(self, agent_type: str, agent_name: str):
        self.agent_type = agent_type
        self.agent_name = agent_name
        self.agent_id = agent_name
        self.redis_client = None
        self.db_conn = None
        self.running = True

    def connect(self):
        """Connect to Docker services using proper configuration."""
        try:
            # Parse Redis configuration
            redis_url = getattr(settings, "redis_url", "redis://localhost:6381")
            redis_host, redis_port = self._parse_redis_url(redis_url)

            # Redis on localhost (exposed from Docker)
            self.redis_client = redis.Redis(
                host=redis_host, port=redis_port, decode_responses=True
            )

            # Parse database configuration
            database_url = getattr(
                settings,
                "database_url",
                "postgresql://hive_user:hive_pass@localhost:5433/leanvibe_hive",
            )
            db_host, db_port, db_name, db_user, db_pass = self._parse_database_url(
                database_url
            )

            # PostgreSQL on localhost (exposed from Docker)
            self.db_conn = psycopg2.connect(
                host=db_host,
                port=db_port,
                database=db_name,
                user=db_user,
                password=db_pass,
                cursor_factory=RealDictCursor,
            )

            print(
                f"[{self.agent_name}] Connected to services (Redis: {redis_host}:{redis_port}, DB: {db_host}:{db_port}/{db_name})"
            )
        except Exception as e:
            print(f"[{self.agent_name}] Failed to connect to services: {e}")
            raise

    def _parse_database_url(self, database_url: str) -> tuple:
        """Parse database URL into connection parameters."""
        # Handle different URL formats
        if database_url.startswith("postgresql://") or database_url.startswith(
            "postgres://"
        ):
            # Remove protocol
            if database_url.startswith("postgresql://"):
                db_part = database_url[13:]
            else:
                db_part = database_url[11:]

            # Parse user:pass@host:port/dbname
            if "@" in db_part:
                user_pass, host_db = db_part.split("@", 1)
                if ":" in user_pass:
                    db_user, db_pass = user_pass.split(":", 1)
                else:
                    db_user, db_pass = user_pass, ""
            else:
                db_user, db_pass = "hive_user", "hive_pass"
                host_db = db_part

            # Parse host:port/dbname
            if ":" in host_db:
                host_port, db_name = (
                    host_db.split("/", 1)
                    if "/" in host_db
                    else (host_db, "leanvibe_hive")
                )
                host_parts = host_port.split(":")
                if len(host_parts) == 2:
                    db_host, db_port_str = host_parts
                    try:
                        db_port = int(db_port_str)
                    except ValueError:
                        db_port = 5433
                else:
                    db_host = host_parts[0]
                    db_port = 5433
            else:
                host_db_parts = (
                    host_db.split("/", 1)
                    if "/" in host_db
                    else [host_db, "leanvibe_hive"]
                )
                db_host = host_db_parts[0]
                db_name = host_db_parts[1]
                db_port = 5433

            return db_host, db_port, db_name, db_user, db_pass
        else:
            # Default fallback
            return "localhost", 5433, "leanvibe_hive", "hive_user", "hive_pass"

    def _parse_redis_url(self, redis_url: str) -> tuple:
        """Parse Redis URL into connection parameters."""
        if redis_url.startswith("redis://"):
            redis_part = redis_url[8:]  # Remove 'redis://'
            if "/" in redis_part:
                host_port = redis_part.split("/")[0]
            else:
                host_port = redis_part

            if ":" in host_port:
                host, port_str = host_port.split(":", 1)
                try:
                    port = int(port_str)
                except ValueError:
                    port = 6381
            else:
                host, port = host_port, 6381

            return host, port
        else:
            return "localhost", 6381

    def update_status(self, status: str):
        """Update agent status in database."""
        try:
            with self.db_conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE agents
                    SET status = %s, last_heartbeat = NOW()
                    WHERE name = %s
                """,
                    (status, self.agent_name),
                )
                self.db_conn.commit()
        except Exception as e:
            print(f"[{self.agent_name}] Failed to update status: {e}")
            # Try to reconnect
            try:
                self.connect()
            except:
                pass

    def get_next_task(self) -> dict[str, Any] | None:
        """Get next task from Redis queue."""
        try:
            # Check priority queues in order
            for priority in ["critical", "high", "normal", "low", "background"]:
                queue_key = f"task_queue:{priority}"

                # Try to pop task (non-blocking)
                task_json = self.redis_client.rpop(queue_key)
                if task_json:
                    try:
                        task = json.loads(task_json)
                        print(
                            f"[{self.agent_name}] Got task: {task.get('title', 'Untitled')}"
                        )
                        return task
                    except json.JSONDecodeError:
                        print(f"[{self.agent_name}] Invalid task JSON: {task_json}")

            return None
        except Exception as e:
            print(f"[{self.agent_name}] Failed to get task from Redis: {e}")
            # Try to reconnect to Redis
            try:
                self.connect()
            except:
                pass
            return None

    def execute_claude_code(self, prompt: str) -> dict[str, Any]:
        """Execute Claude Code CLI command."""
        print(f"[{self.agent_name}] Executing: {prompt[:100]}...")

        start_time = time.time()

        try:
            # Run Claude Code CLI
            result = subprocess.run(
                ["claude", "--no-interactive", prompt],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            elapsed = time.time() - start_time

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "elapsed_seconds": elapsed,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Claude Code timeout after 5 minutes",
                "elapsed_seconds": 300,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "elapsed_seconds": time.time() - start_time,
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def process_task(self, task: dict[str, Any]):
        """Process a single task."""
        task_id = task.get("id", str(uuid.uuid4()))

        # Update task status
        self.redis_client.hset(
            f"task:{task_id}",
            mapping={
                "status": "in_progress",
                "agent_id": self.agent_name,
                "started_at": datetime.now().isoformat(),
            },
        )

        # Build prompt based on task type
        prompt = self.build_prompt(task)

        # Execute via Claude Code
        result = self.execute_claude_code(prompt)

        # Store result
        self.redis_client.hset(
            f"task:{task_id}",
            mapping={
                "status": "completed" if result["success"] else "failed",
                "completed_at": datetime.now().isoformat(),
                "result": json.dumps(result),
            },
        )

        # Store in database for persistence
        try:
            with self.db_conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO tasks (id, title, type, status, assigned_agent_id, result, completed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (id) DO UPDATE
                    SET status = EXCLUDED.status,
                        result = EXCLUDED.result,
                        completed_at = EXCLUDED.completed_at
                """,
                    (
                        task_id,
                        task.get("title", ""),
                        task.get("type", "general"),
                        "completed" if result["success"] else "failed",
                        self.agent_name,
                        json.dumps(result),
                    ),
                )
                self.db_conn.commit()
        except Exception as e:
            print(f"[{self.agent_name}] Failed to store task result in database: {e}")
            # Try to reconnect
            try:
                self.connect()
            except:
                pass

        print(
            f"[{self.agent_name}] Task {task_id} {'completed' if result['success'] else 'failed'}"
        )

    def build_prompt(self, task: dict[str, Any]) -> str:
        """Build Claude prompt based on task type and agent specialization."""
        base_prompt = f"""You are {self.agent_name}, a {self.agent_type} agent in the LeanVibe Agent Hive system.

Task: {task.get("title", "No title")}
Type: {task.get("type", "general")}
Description: {task.get("description", "No description")}

Context:
- Working directory: {Path.cwd()}
- Previous context: {task.get("context", "None")}
"""

        # Add agent-specific instructions
        if self.agent_type == "meta":
            base_prompt += """
As a meta-agent, your role is to:
1. Analyze the current system state
2. Identify areas for improvement
3. Generate concrete improvement proposals
4. Test improvements safely
5. Document changes

Focus on making the system more efficient, reliable, and self-improving.
"""
        elif self.agent_type == "developer":
            base_prompt += """
As a developer agent, your role is to:
1. Implement new features
2. Fix bugs
3. Write clean, well-documented code
4. Create comprehensive tests
5. Follow best practices

Use Python 3.11+, type hints, async/await, and maintain >90% test coverage.
"""
        elif self.agent_type == "qa":
            base_prompt += """
As a QA agent, your role is to:
1. Write comprehensive tests
2. Find and report bugs
3. Verify fixes work correctly
4. Ensure code quality
5. Check test coverage

Use pytest for testing and aim for >90% coverage.
"""

        # Add specific task instructions
        if task.get("instructions"):
            base_prompt += f"\nSpecific instructions:\n{task['instructions']}"

        return base_prompt

    def run(self):
        """Main agent loop."""
        print(f"[{self.agent_name}] Starting agent runner...")

        # Connect to services
        self.connect()

        # Register agent
        try:
            with self.db_conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO agents (name, type, status, capabilities)
                    VALUES (%s, %s, 'active', %s)
                    ON CONFLICT (name) DO UPDATE
                    SET status = 'active', last_heartbeat = NOW()
                """,
                    (
                        self.agent_name,
                        self.agent_type,
                        json.dumps(self.get_capabilities()),
                    ),
                )
                self.db_conn.commit()
                print(f"[{self.agent_name}] Agent registered successfully")
        except Exception as e:
            print(f"[{self.agent_name}] Failed to register agent: {e}")
            # Continue running even if registration fails

        print(f"[{self.agent_name}] Agent registered and ready")

        # Main processing loop
        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.running:
            try:
                # Update heartbeat
                self.update_status("active")

                # Get next task
                task = self.get_next_task()

                if task:
                    self.update_status("busy")
                    self.process_task(task)
                    self.update_status("active")
                    consecutive_errors = 0  # Reset error counter on success
                else:
                    # No tasks, wait a bit
                    time.sleep(5)

                consecutive_errors = (
                    0  # Reset error counter on successful loop iteration
                )

            except KeyboardInterrupt:
                print(f"[{self.agent_name}] Shutting down...")
                self.running = False
            except Exception as e:
                consecutive_errors += 1
                print(
                    f"[{self.agent_name}] Error (attempt {consecutive_errors}/{max_consecutive_errors}): {e}"
                )

                # If we have too many consecutive errors, try to reconnect
                if consecutive_errors >= max_consecutive_errors:
                    print(
                        f"[{self.agent_name}] Too many consecutive errors, attempting to reconnect..."
                    )
                    try:
                        self.connect()
                        consecutive_errors = 0  # Reset on successful reconnect
                        print(f"[{self.agent_name}] Reconnected successfully")
                    except Exception as reconnect_error:
                        print(
                            f"[{self.agent_name}] Reconnect failed: {reconnect_error}"
                        )
                        time.sleep(30)  # Wait longer before retrying if reconnect fails
                        continue

                time.sleep(10)  # Wait before retrying

        # Clean shutdown
        self.update_status("terminated")
        self.cleanup()

    def get_capabilities(self) -> list:
        """Get agent capabilities based on type."""
        capabilities_map = {
            "meta": ["analysis", "improvement", "optimization", "monitoring"],
            "developer": ["coding", "debugging", "implementation", "refactoring"],
            "qa": ["testing", "validation", "quality_assurance", "coverage"],
            "architect": ["design", "planning", "architecture", "patterns"],
            "devops": ["deployment", "monitoring", "infrastructure", "ci_cd"],
        }
        return capabilities_map.get(self.agent_type, ["general"])

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.db_conn:
                self.db_conn.close()
        except Exception as e:
            print(f"[{self.agent_name}] Error closing database connection: {e}")

        try:
            if self.redis_client:
                self.redis_client.close()
        except Exception as e:
            print(f"[{self.agent_name}] Error closing Redis connection: {e}")

        print(f"[{self.agent_name}] Shutdown complete")


@app.command()
def run(
    agent_type: str = typer.Option("worker", "--type", help="Agent type"),
    agent_name: str = typer.Option(None, "--name", help="Agent name"),
):
    """Run an agent in a tmux session."""
    if not agent_name:
        agent_name = f"{agent_type}-{uuid.uuid4().hex[:8]}"

    runner = AgentRunner(agent_type, agent_name)
    runner.run()


if __name__ == "__main__":
    app()
