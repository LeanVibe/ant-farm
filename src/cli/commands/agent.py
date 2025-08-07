"""
Agent management commands for the Hive CLI
"""

import asyncio
import subprocess
import sys
import uuid
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.table import Table

from ..utils import (
    API_BASE_URL,
    create_status_table,
    error_handler,
    info_message,
    success_message,
)

app = typer.Typer(help="Agent management commands")
console = Console()


@app.command()
def list():
    """List all registered agents"""
    asyncio.run(_list_agents())


async def _list_agents():
    """Internal async agent listing"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{API_BASE_URL}/api/v1/agents")

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    agents = data.get("data", [])

                    if not agents:
                        info_message("No agents are currently registered")
                        return

                    # Create table data
                    table_data = []
                    for agent in agents:
                        table_data.append(
                            {
                                "name": agent.get("name", "unknown"),
                                "type": agent.get("type", "unknown"),
                                "status": agent.get("status", "unknown"),
                                "uptime": f"{agent.get('uptime', 0):.1f}s",
                                "last_heartbeat": agent.get("last_heartbeat", "never"),
                            }
                        )

                    table = create_status_table("🤖 Active Agents", table_data)
                    console.print(table)

                    success_message(f"Found {len(agents)} registered agents")
                else:
                    error_handler(Exception(data.get("error", "Unknown API error")))
            else:
                error_handler(Exception(f"API returned status {response.status_code}"))

    except httpx.ConnectError:
        error_handler(
            Exception(
                "Cannot connect to API server. Is it running? Try: hive system start"
            )
        )
    except Exception as e:
        error_handler(e)


@app.command()
def describe(
    agent_name: str = typer.Argument(..., help="Name of the agent to describe"),
):
    """Show detailed information about a specific agent"""
    asyncio.run(_describe_agent(agent_name))


async def _describe_agent(agent_name: str):
    """Internal async agent description"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{API_BASE_URL}/api/v1/agents/{agent_name}")

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    agent = data.get("data", {})

                    console.print(f"\n🤖 [bold cyan]Agent: {agent_name}[/bold cyan]")
                    console.print(f"Type: {agent.get('type', 'unknown')}")
                    console.print(f"Role: {agent.get('role', 'unknown')}")
                    console.print(f"Status: {agent.get('status', 'unknown')}")
                    console.print(f"Uptime: {agent.get('uptime', 0):.1f} seconds")
                    console.print(f"Tasks Completed: {agent.get('tasks_completed', 0)}")
                    console.print(f"Tasks Failed: {agent.get('tasks_failed', 0)}")

                    capabilities = agent.get("capabilities", {})
                    if capabilities:
                        console.print("\n📋 [bold]Capabilities:[/bold]")
                        for key, value in capabilities.items():
                            console.print(f"  • {key}: {value}")

                    if agent.get("last_heartbeat"):
                        console.print(f"\n💓 Last Heartbeat: {agent['last_heartbeat']}")

                else:
                    error_handler(Exception(data.get("error", "Unknown API error")))
            elif response.status_code == 404:
                error_handler(Exception(f"Agent '{agent_name}' not found"))
            else:
                error_handler(Exception(f"API returned status {response.status_code}"))

    except httpx.ConnectError:
        error_handler(
            Exception(
                "Cannot connect to API server. Is it running? Try: hive system start"
            )
        )
    except Exception as e:
        error_handler(e)


@app.command()
def spawn(
    agent_type: str = typer.Argument(
        ..., help="Type of agent to spawn (meta, architect, qa, devops)"
    ),
    name: str = typer.Option(None, "--name", "-n", help="Custom name for the agent"),
):
    """Spawn a new agent instance"""
    asyncio.run(_spawn_agent(agent_type, name))


async def _spawn_agent(agent_type: str, name: str = None):
    """Internal async agent spawning"""
    try:
        params = {"agent_type": agent_type}
        if name:
            params["agent_name"] = name

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{API_BASE_URL}/api/v1/agents", params=params)

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    result = data.get("data", {})
                    agent_name = result.get("agent_name", "unknown")
                    session_name = result.get("session_name", "unknown")

                    success_message(f"Agent '{agent_name}' spawned successfully!")
                    info_message(f"Session: {session_name}")
                    info_message(f"Type: {agent_type}")
                    console.print(
                        f"\n💡 [dim]Tip: Use 'hive agent describe {agent_name}' to check status[/dim]"
                    )

                else:
                    error_handler(Exception(data.get("error", "Unknown API error")))
            else:
                error_handler(Exception(f"API returned status {response.status_code}"))

    except httpx.ConnectError:
        error_handler(
            Exception(
                "Cannot connect to API server. Is it running? Try: hive system start"
            )
        )
    except Exception as e:
        error_handler(e)


@app.command()
def stop(agent_name: str = typer.Argument(..., help="Name of the agent to stop")):
    """Stop a specific agent"""
    asyncio.run(_stop_agent(agent_name))


async def _stop_agent(agent_name: str):
    """Internal async agent stopping"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v1/agents/{agent_name}/stop"
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    success_message(f"Agent '{agent_name}' stopped successfully!")
                else:
                    error_handler(Exception(data.get("error", "Unknown API error")))
            elif response.status_code == 404:
                error_handler(Exception(f"Agent '{agent_name}' not found"))
            else:
                error_handler(Exception(f"API returned status {response.status_code}"))

    except httpx.ConnectError:
        error_handler(
            Exception(
                "Cannot connect to API server. Is it running? Try: hive system start"
            )
        )
    except Exception as e:
        error_handler(e)


@app.command()
def health(
    agent_name: str = typer.Argument(..., help="Name of the agent to health check"),
):
    """Check the health of a specific agent"""
    asyncio.run(_check_agent_health(agent_name))


async def _check_agent_health(agent_name: str):
    """Internal async agent health check"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v1/agents/{agent_name}/health"
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    success_message(f"Health check sent to agent '{agent_name}'")
                    info_message("Check agent logs for response")
                else:
                    error_handler(Exception(data.get("error", "Unknown API error")))
            elif response.status_code == 404:
                error_handler(Exception(f"Agent '{agent_name}' not found"))
            else:
                error_handler(Exception(f"API returned status {response.status_code}"))

    except httpx.ConnectError:
        error_handler(
            Exception(
                "Cannot connect to API server. Is it running? Try: hive system start"
            )
        )
    except Exception as e:
        error_handler(e)


# NEW COMMANDS from old CLI
@app.command("run")
def run_agent(
    agent_type: str = typer.Argument(
        ..., help="Type of agent to run (meta, developer, qa, architect, research)"
    ),
    name: str | None = typer.Option(None, "--name", "-n", help="Agent name"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
) -> None:
    """Start an agent instance directly (not in tmux)"""
    console.print(f"[cyan]Starting {agent_type} agent...[/cyan]")

    if not name:
        name = f"{agent_type}-{uuid.uuid4().hex[:8]}"

    try:
        # Import and run agent runner
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
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


@app.command("sessions")
def list_sessions() -> None:
    """List all active agent tmux sessions"""
    console.print("[cyan]Active agent sessions:[/cyan]")

    try:
        result = subprocess.run(["tmux", "ls"], capture_output=True, text=True)

        if result.returncode == 0:
            sessions = result.stdout.strip().split("\n")
            agent_sessions = [s for s in sessions if "agent" in s or "hive" in s]

            if agent_sessions:
                table = Table(title="🖥️  Active Tmux Sessions", show_header=True)
                table.add_column("Session", style="cyan")
                table.add_column("Status", style="green")

                for session in agent_sessions:
                    parts = session.split(":")
                    session_name = parts[0]
                    status = "active" if len(parts) > 1 else "unknown"
                    table.add_row(session_name, status)

                console.print(table)
            else:
                console.print("  No agent sessions found")
        else:
            console.print("  No active tmux sessions")

    except FileNotFoundError:
        console.print("[red]tmux not found - install with: brew install tmux[/red]")
    except Exception as e:
        console.print(f"[red]Error listing sessions: {e}[/red]")


@app.command()
def bootstrap():
    """Bootstrap the LeanVibe Agent Hive system"""
    console.print("[cyan]Bootstrapping LeanVibe Agent Hive 2.0...[/cyan]")

    try:
        # Import bootstrap functionality
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from bootstrap import BootstrapAgent

        agent = BootstrapAgent()
        agent.bootstrap_system()

        success_message("Bootstrap completed successfully!")

    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("Make sure all dependencies are installed: uv sync")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Bootstrap failed: {e}[/red]")
        sys.exit(1)


@app.command()
def tools():
    """Check available CLI agentic coding tools"""
    console.print("[cyan]Checking available CLI tools...[/cyan]")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
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

        table = Table(title="🛠️  CLI Tools", show_header=True)
        table.add_column("Tool", style="cyan")
        table.add_column("Command", style="yellow")
        table.add_column("Status", style="green")

        for tool_key, tool_config in agent.available_tools.items():
            status = (
                "✅ PREFERRED" if tool_key == agent.preferred_tool else "✅ Available"
            )
            table.add_row(tool_config["name"], tool_config["command"], status)

        console.print(table)

        console.print(
            f"\nPreferred tool: {agent.available_tools[agent.preferred_tool]['name']}"
            if agent.preferred_tool
            else "No preferred tool"
        )

    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    app()
