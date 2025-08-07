#!/usr/bin/env python3
"""Unified CLI for LeanVibe Agent Hive 2.0.

This consolidates all startup scripts into a single `hive` command with subcommands:
- hive run-agent: Start an agent instance
- hive bootstrap: Bootstrap the entire system
- hive start-api: Start the API server
- hive init-db: Initialize database
- hive tools: Check available CLI tools
- hive status: System status
- hive coordination: Agent coordination commands
"""

import asyncio
import subprocess
import sys
import uuid
from pathlib import Path

import typer
from rich.console import Console

# Add src to Python path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from core.config import settings

console = Console()
app = typer.Typer(
    name="hive",
    help="LeanVibe Agent Hive 2.0 - Self-improving multi-agent development platform",
    no_args_is_help=True,
)

# Add coordination subcommands
try:
    from .coordination import app as coordination_app

    app.add_typer(
        coordination_app,
        name="coordination",
        help="Agent coordination and collaboration",
    )
except ImportError:
    console.print(
        "[yellow]Warning: Coordination commands not available (missing dependencies)[/yellow]"
    )


@app.command("run-agent")
def run_agent(
    agent_type: str = typer.Argument(
        ..., help="Type of agent to run (meta, developer, qa, architect, research)"
    ),
    name: str | None = typer.Option(None, "--name", "-n", help="Agent name"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
) -> None:
    """Start an agent instance."""
    console.print(f"[cyan]Starting {agent_type} agent...[/cyan]")

    if not name:
        name = f"{agent_type}-{uuid.uuid4().hex[:8]}"

    try:
        # Import and run agent runner
        from agents.runner import AgentRunner

        runner = AgentRunner(agent_type, name)
        asyncio.run(runner.start())

    except KeyboardInterrupt:
        console.print(f"\n[yellow]Agent {name} stopped by user[/yellow]")
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("Make sure all dependencies are installed: uv sync")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error starting agent: {e}[/red]")
        sys.exit(1)


@app.command("bootstrap")
def bootstrap() -> None:
    """Bootstrap the LeanVibe Agent Hive system."""
    console.print("[cyan]Bootstrapping LeanVibe Agent Hive 2.0...[/cyan]")

    try:
        # Import bootstrap functionality
        from bootstrap import BootstrapAgent

        agent = BootstrapAgent()
        agent.bootstrap_system()

    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("Make sure all dependencies are installed: uv sync")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Bootstrap failed: {e}[/red]")
        sys.exit(1)


@app.command("start-api")
def start_api(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Auto-reload on changes"),
) -> None:
    """Start the FastAPI server."""
    console.print(f"[cyan]Starting API server on {host}:{port}...[/cyan]")

    try:
        cmd = [
            "uvicorn",
            "src.api.main:app",
            "--host",
            host,
            "--port",
            str(port),
        ]

        if reload:
            cmd.append("--reload")

        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to start API server: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]API server stopped[/yellow]")


@app.command("init-db")
def init_db() -> None:
    """Initialize database tables."""
    console.print("[cyan]Initializing database...[/cyan]")

    try:
        from core.models import get_database_manager

        db_manager = get_database_manager(settings.database_url)
        db_manager.create_tables()

        console.print("[green]✓ Database tables created successfully![/green]")

    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("Make sure all dependencies are installed: uv sync")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Database initialization failed: {e}[/red]")
        console.print("Make sure PostgreSQL is running and accessible")
        sys.exit(1)


@app.command("tools")
def tools() -> None:
    """Check available CLI agentic coding tools."""
    console.print("[cyan]Checking available CLI tools...[/cyan]")

    try:
        from bootstrap import BootstrapAgent

        agent = BootstrapAgent()

        if not agent.available_tools:
            console.print("[red]No CLI agentic coding tools found![/red]")
            console.print("\nInstall options:")
            console.print("• opencode: curl -fsSL https://opencode.ai/install | bash")
            console.print("• Claude Code CLI: https://claude.ai/cli")
            console.print("• Gemini CLI: https://ai.google.dev/gemini-api/docs/cli")
            return

        console.print(
            f"\n[bold]Available CLI Tools ({len(agent.available_tools)}):[/bold]"
        )
        for tool_key, tool_config in agent.available_tools.items():
            status = (
                "✓ [green]PREFERRED[/green]"
                if tool_key == agent.preferred_tool
                else "✓"
            )
            console.print(
                f"  {status} {tool_config['name']} ({tool_config['command']})"
            )

        console.print(
            f"\nPreferred tool: {agent.available_tools[agent.preferred_tool]['name']}"
            if agent.preferred_tool
            else "No preferred tool"
        )

    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command("status")
def status() -> None:
    """Check system status."""
    console.print("[cyan]Checking system status...[/cyan]")

    try:
        from bootstrap import BootstrapAgent

        agent = BootstrapAgent()

        console.print("\n[bold]CLI Agentic Coding Tools:[/bold]")

        # Check each tool
        agent.check_opencode()
        agent.check_claude_code()
        agent.check_gemini_cli()

        # Check tmux
        agent.check_tmux()

        # Check Docker services
        try:
            agent.connect_services()
            console.print("\n[bold green]System is ready![/bold green]")
            if agent.preferred_tool:
                console.print(
                    f"Preferred tool: {agent.available_tools[agent.preferred_tool]['name']}"
                )
        except:
            console.print(
                "\n[bold red]System not ready - start Docker services[/bold red]"
            )
            console.print("Run: docker compose up -d")
        finally:
            agent.cleanup()

    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command("spawn")
def spawn(
    agent_type: str = typer.Argument(..., help="Type of agent to spawn"),
    name: str | None = typer.Option(None, "--name", "-n", help="Agent name"),
) -> None:
    """Spawn a new agent in tmux session."""
    console.print(f"[cyan]Spawning {agent_type} agent...[/cyan]")

    try:
        from bootstrap import BootstrapAgent

        agent = BootstrapAgent()
        agent.connect_services()
        session_name = agent.spawn_agent(agent_type, name)

        console.print(f"[green]✓ Agent spawned in session: {session_name}[/green]")
        console.print(f"Attach with: tmux attach -t {session_name}")

    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Failed to spawn agent: {e}[/red]")
        sys.exit(1)


@app.command("list")
def list_agents() -> None:
    """List all active agent sessions."""
    console.print("[cyan]Active agent sessions:[/cyan]")

    try:
        result = subprocess.run(["tmux", "ls"], capture_output=True, text=True)

        if result.returncode == 0:
            sessions = result.stdout.strip().split("\n")
            agent_sessions = [s for s in sessions if "agent" in s]

            if agent_sessions:
                for session in agent_sessions:
                    console.print(f"  • {session}")
            else:
                console.print("  No agent sessions found")
        else:
            console.print("  No active tmux sessions")

    except FileNotFoundError:
        console.print("[red]tmux not found - install with: brew install tmux[/red]")
    except Exception as e:
        console.print(f"[red]Error listing sessions: {e}[/red]")


if __name__ == "__main__":
    app()
