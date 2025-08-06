#!/usr/bin/env python3
"""Bootstrap agent for LeanVibe Agent Hive 2.0 - Hybrid Architecture.

This runs on the host machine and:
1. Uses Claude Code CLI directly (no API key needed in code)
2. Spawns other agents in tmux sessions
3. Coordinates with Docker services (PostgreSQL, Redis)
"""
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import typer

console = Console()
app = typer.Typer()

class BootstrapAgent:
    """Bootstrap agent that runs on host and spawns tmux sessions."""
    
    def __init__(self):
        self.agent_id = "bootstrap-agent"
        self.session_name = f"{self.agent_id}-{uuid.uuid4().hex[:8]}"
        self.workspace = Path.cwd()
        self.redis_client = None
        self.db_conn = None
        
    def connect_services(self):
        """Connect to Docker services."""
        # Connect to Redis (in Docker, exposed on localhost)
        try:
            self.redis_client = redis.Redis(
                host="localhost", 
                port=6379, 
                decode_responses=True
            )
            self.redis_client.ping()
            console.print("[green]✓[/green] Redis connected")
        except Exception as e:
            console.print(f"[red]✗[/red] Redis connection failed: {e}")
            console.print("Make sure Docker services are running: docker-compose up -d")
            sys.exit(1)
            
        # Connect to PostgreSQL (in Docker, exposed on localhost)
        try:
            self.db_conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="leanvibe_hive",
                user="hive_user",
                password="hive_pass"
            )
            console.print("[green]✓[/green] PostgreSQL connected")
        except Exception as e:
            console.print(f"[red]✗[/red] PostgreSQL connection failed: {e}")
            console.print("Make sure Docker services are running: docker-compose up -d")
            sys.exit(1)
    
    def check_claude_code(self) -> bool:
        """Check if Claude Code CLI is installed."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                console.print(f"[green]✓[/green] Claude Code found: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            pass
        
        console.print("[red]✗[/red] Claude Code CLI not found")
        console.print("Please install: https://claude.ai/cli")
        return False
    
    def check_tmux(self) -> bool:
        """Check if tmux is installed."""
        try:
            result = subprocess.run(
                ["tmux", "-V"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                console.print(f"[green]✓[/green] tmux found: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            pass
        
        console.print("[red]✗[/red] tmux not found")
        console.print("Please install: brew install tmux")
        return False
    
    def spawn_tmux_session(self, session_name: str, command: str = None) -> bool:
        """Spawn a new tmux session."""
        try:
            # Create detached tmux session
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", session_name],
                check=True
            )
            
            # Send initial command if provided
            if command:
                subprocess.run(
                    ["tmux", "send-keys", "-t", session_name, command, "Enter"],
                    check=True
                )
            
            console.print(f"[green]✓[/green] Spawned tmux session: {session_name}")
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]✗[/red] Failed to spawn tmux session: {e}")
            return False
    
    def execute_claude_task(self, task: str, session_name: str = None) -> Dict[str, Any]:
        """Execute a task using Claude Code in a tmux session."""
        if not session_name:
            session_name = self.session_name
        
        console.print(f"\n[cyan]Executing in {session_name}:[/cyan] {task[:100]}...")
        
        # Build Claude command
        claude_cmd = f'claude --no-interactive "{task}"'
        
        # Send command to tmux session
        subprocess.run(
            ["tmux", "send-keys", "-t", session_name, claude_cmd, "Enter"],
            check=True
        )
        
        # Store task in Redis
        task_id = str(uuid.uuid4())
        task_data = {
            "id": task_id,
            "agent_id": self.agent_id,
            "session": session_name,
            "task": task,
            "status": "executing",
            "created_at": datetime.now().isoformat()
        }
        
        self.redis_client.hset(f"task:{task_id}", mapping=task_data)
        self.redis_client.lpush(f"agent_tasks:{self.agent_id}", task_id)
        
        return task_data
    
    def spawn_agent(self, agent_type: str, agent_name: str = None) -> str:
        """Spawn a new agent in a tmux session."""
        if not agent_name:
            agent_name = f"{agent_type}-{uuid.uuid4().hex[:8]}"
        
        console.print(f"\n[cyan]Spawning agent:[/cyan] {agent_name}")
        
        # Create tmux session for agent
        session_name = agent_name
        self.spawn_tmux_session(session_name)
        
        # Start agent runner in the session
        runner_cmd = f"python src/agents/runner.py --type {agent_type} --name {agent_name}"
        subprocess.run(
            ["tmux", "send-keys", "-t", session_name, runner_cmd, "Enter"],
            check=True
        )
        
        # Register agent in database
        with self.db_conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO agents (name, type, status, created_at)
                VALUES (%s, %s, 'active', NOW())
                ON CONFLICT (name) DO UPDATE SET status = 'active'
            """, (agent_name, agent_type))
            self.db_conn.commit()
        
        console.print(f"[green]✓[/green] Agent spawned: {agent_name}")
        return session_name
    
    def create_project_structure(self):
        """Create the initial project structure."""
        directories = [
            "src/core",
            "src/agents",
            "src/api",
            "src/web/dashboard",
            "tests/unit",
            "tests/integration",
            "scripts",
            "logs",
            "workspace"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        # Create __init__.py files
        for dir_path in ["src", "src/core", "src/agents", "src/api"]:
            (Path(dir_path) / "__init__.py").touch()
            
        console.print("[green]✓[/green] Project structure created")
    
    def generate_core_components(self):
        """Generate core system components using Claude Code."""
        components = [
            ("Task Queue", "Create src/core/task_queue.py - Redis-based task queue with priorities, retries, and dependencies"),
            ("Agent Orchestrator", "Create src/core/orchestrator.py - Manages agent lifecycle and tmux sessions"),
            ("Message Broker", "Create src/core/message_broker.py - Inter-agent communication via Redis pub/sub"),
            ("Database Models", "Create src/core/models.py - SQLAlchemy models for agents, tasks, and contexts"),
            ("Agent Runner", "Create src/agents/runner.py - Base agent that polls tasks and executes via Claude Code"),
            ("Meta Agent", "Create src/agents/meta_agent.py - Self-improvement coordinator"),
            ("API Server", "Create src/api/main.py - FastAPI endpoints for system control"),
        ]
        
        # Execute each component generation in bootstrap session
        for name, task in components:
            console.print(f"\n[bold cyan]Generating {name}...[/bold cyan]")
            self.execute_claude_task(task)
            time.sleep(5)  # Give Claude time to work
    
    def bootstrap_system(self):
        """Bootstrap the entire system."""
        console.print("\n[bold cyan]Starting LeanVibe Agent Hive 2.0 Bootstrap[/bold cyan]\n")
        
        # Check prerequisites
        if not self.check_claude_code() or not self.check_tmux():
            return
        
        # Connect to services
        self.connect_services()
        
        # Create project structure
        self.create_project_structure()
        
        # Create bootstrap tmux session
        console.print(f"\n[cyan]Creating bootstrap session: {self.session_name}[/cyan]")
        self.spawn_tmux_session(self.session_name)
        
        # Generate core components
        self.generate_core_components()
        
        # Spawn initial agents
        console.print("\n[bold cyan]Spawning initial agents...[/bold cyan]")
        self.spawn_agent("meta", "meta-agent-001")
        self.spawn_agent("developer", "dev-agent-001")
        
        # Submit first improvement task
        console.print("\n[cyan]Submitting first self-improvement task...[/cyan]")
        self.redis_client.lpush("task_queue:high", json.dumps({
            "id": str(uuid.uuid4()),
            "type": "analyze_and_improve",
            "title": "Analyze system and propose improvements",
            "assigned_to": "meta-agent-001",
            "created_at": datetime.now().isoformat()
        }))
        
        console.print("\n[bold green]Bootstrap Complete![/bold green]")
        console.print("\nActive tmux sessions:")
        subprocess.run(["tmux", "ls"])
        
        console.print("\nNext steps:")
        console.print("1. View bootstrap session: tmux attach -t " + self.session_name)
        console.print("2. View meta-agent: tmux attach -t meta-agent-001")
        console.print("3. Check API: curl http://localhost:8000/health")
        console.print("4. View logs: tail -f logs/*.log")
    
    def list_agents(self):
        """List all active agent sessions."""
        console.print("\n[bold]Active Agent Sessions:[/bold]")
        result = subprocess.run(
            ["tmux", "ls"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            sessions = result.stdout.strip().split('\n')
            agent_sessions = [s for s in sessions if 'agent' in s]
            for session in agent_sessions:
                console.print(f"  • {session}")
        else:
            console.print("No active sessions")
    
    def cleanup(self):
        """Clean up resources."""
        if self.db_conn:
            self.db_conn.close()
        if self.redis_client:
            self.redis_client.close()

@app.command()
def bootstrap():
    """Bootstrap the LeanVibe Agent Hive system."""
    agent = BootstrapAgent()
    try:
        agent.bootstrap_system()
    finally:
        agent.cleanup()

@app.command()
def spawn(
    agent_type: str = typer.Argument(..., help="Type of agent to spawn"),
    name: Optional[str] = typer.Option(None, help="Agent name")
):
    """Spawn a new agent."""
    agent = BootstrapAgent()
    try:
        agent.connect_services()
        agent.spawn_agent(agent_type, name)
    finally:
        agent.cleanup()

@app.command()
def list_agents():
    """List all active agents."""
    agent = BootstrapAgent()
    agent.list_agents()

@app.command()
def status():
    """Check system status."""
    agent = BootstrapAgent()
    
    # Check Claude Code
    agent.check_claude_code()
    
    # Check tmux
    agent.check_tmux()
    
    # Check Docker services
    try:
        agent.connect_services()
        console.print("\n[bold green]System is ready![/bold green]")
    except:
        console.print("\n[bold red]System not ready - start Docker services[/bold red]")
    finally:
        agent.cleanup()

def main():
    """Main entry point."""
    app()

if __name__ == "__main__":
    main()