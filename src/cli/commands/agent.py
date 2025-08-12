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
def list(
    status: str = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status (active, starting, stopping, inactive, error)",
    ),
    agent_type: str = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by agent type (meta, developer, qa, architect, devops)",
    ),
    name_pattern: str = typer.Option(
        None, "--name", "-n", help="Filter by name pattern (partial match)"
    ),
    sort_by: str = typer.Option(
        "name", "--sort", help="Sort by: name, type, status, uptime"
    ),
    reverse: bool = typer.Option(False, "--reverse", "-r", help="Reverse sort order"),
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of results"),
):
    """List all registered agents with advanced filtering"""
    asyncio.run(
        _list_agents_filtered(status, agent_type, name_pattern, sort_by, reverse, limit)
    )


async def _list_agents_filtered(
    status=None,
    agent_type=None,
    name_pattern=None,
    sort_by="name",
    reverse=False,
    limit=None,
):
    """Internal async agent listing with filtering"""
    try:
        from ..utils import get_api_headers

        headers = get_api_headers()

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v1/cli/agents", headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    agents = data.get("data", [])

                    if not agents:
                        info_message("No agents are currently registered")
                        return

                    # Apply filters
                    filtered_agents = agents

                    if status:
                        filtered_agents = [
                            a
                            for a in filtered_agents
                            if a.get("status", "").lower() == status.lower()
                        ]

                    if agent_type:
                        filtered_agents = [
                            a
                            for a in filtered_agents
                            if a.get("type", "").lower() == agent_type.lower()
                        ]

                    if name_pattern:
                        filtered_agents = [
                            a
                            for a in filtered_agents
                            if name_pattern.lower() in a.get("name", "").lower()
                        ]

                    # Sort agents
                    sort_key_map = {
                        "name": lambda x: x.get("name", ""),
                        "type": lambda x: x.get("type", ""),
                        "status": lambda x: x.get("status", ""),
                        "uptime": lambda x: x.get("uptime", 0),
                    }

                    if sort_by in sort_key_map:
                        filtered_agents.sort(key=sort_key_map[sort_by], reverse=reverse)

                    # Apply limit
                    if limit and limit > 0:
                        filtered_agents = filtered_agents[:limit]

                    if not filtered_agents:
                        filter_desc = []
                        if status:
                            filter_desc.append(f"status={status}")
                        if agent_type:
                            filter_desc.append(f"type={agent_type}")
                        if name_pattern:
                            filter_desc.append(f"name contains '{name_pattern}'")
                        filter_text = (
                            f" ({', '.join(filter_desc)})" if filter_desc else ""
                        )
                        info_message(f"No agents found matching filters{filter_text}")
                        return

                    # Create table data
                    table_data = []
                    for agent in filtered_agents:
                        table_data.append(
                            {
                                "short_id": agent.get("short_id", "n/a"),
                                "name": agent.get("name", "unknown"),
                                "type": agent.get("type", "unknown"),
                                "status": agent.get("status", "unknown"),
                                "uptime": f"{agent.get('uptime', 0):.1f}s",
                                "last_heartbeat": agent.get("last_heartbeat", "never"),
                            }
                        )

                    # Create title with filter info
                    filter_info = []
                    if status:
                        filter_info.append(f"status={status}")
                    if agent_type:
                        filter_info.append(f"type={agent_type}")
                    if name_pattern:
                        filter_info.append(f"name=*{name_pattern}*")
                    if limit:
                        filter_info.append(f"limit={limit}")

                    title = "🤖 Active Agents"
                    if filter_info:
                        title += f" ({', '.join(filter_info)})"

                    table = create_status_table(title, table_data)
                    console.print(table)

                    success_message(f"Found {len(filtered_agents)} agents")
                    if len(filtered_agents) < len(agents):
                        info_message(
                            f"Showing {len(filtered_agents)} of {len(agents)} total agents"
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


async def _list_agents():
    """Internal async agent listing (legacy function)"""
    await _list_agents_filtered()


@app.command()
def describe(
    agent_identifier: str = typer.Argument(
        ..., help="Name, short ID, or UUID of the agent to describe"
    ),
):
    """Show detailed information about a specific agent"""
    asyncio.run(_describe_agent(agent_identifier))


async def _describe_agent(agent_identifier: str):
    """Internal async agent description"""
    try:
        from ..utils import get_api_headers

        headers = get_api_headers()

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v1/cli/agents/{agent_identifier}", headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    agent = data.get("data", {})

                    console.print(
                        f"\n🤖 [bold cyan]Agent: {agent.get('name', 'unknown')}[/bold cyan]"
                    )

                    # Show short ID prominently
                    short_id = agent.get("short_id")
                    if short_id:
                        console.print(f"[bold green]Short ID: {short_id}[/bold green]")

                    console.print(f"UUID: {agent.get('id', 'unknown')}")
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
                error_handler(Exception(f"Agent '{agent_identifier}' not found"))
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
        from ..utils import get_api_headers

        headers = get_api_headers()
        params = {"agent_type": agent_type}
        if name:
            params["agent_name"] = name

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v1/cli/agents", params=params, headers=headers
            )

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
def stop(
    agent_identifier: str = typer.Argument(
        ..., help="Name, short ID, or UUID of the agent to stop"
    ),
):
    """Stop a specific agent"""
    asyncio.run(_stop_agent(agent_identifier))


async def _stop_agent(agent_identifier: str):
    """Internal async agent stopping"""
    try:
        from ..utils import get_api_headers

        headers = get_api_headers()

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v1/cli/agents/{agent_identifier}/stop",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    result = data.get("data", {})
                    agent_name = result.get("agent_name", agent_identifier)
                    success_message(f"Agent '{agent_name}' stopped successfully!")
                else:
                    error_handler(Exception(data.get("error", "Unknown API error")))
            elif response.status_code == 404:
                error_handler(Exception(f"Agent '{agent_identifier}' not found"))
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
    agent_identifier: str = typer.Argument(
        ..., help="Name, short ID, or UUID of the agent to health check"
    ),
):
    """Check the health of a specific agent"""
    asyncio.run(_check_agent_health(agent_identifier))


async def _check_agent_health(agent_identifier: str):
    """Internal async agent health check"""
    try:
        from ..utils import get_api_headers

        headers = get_api_headers()

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v1/cli/agents/{agent_identifier}/health",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    success_message(f"Health check sent to agent '{agent_identifier}'")
                    info_message("Check agent logs for response")
                else:
                    error_handler(Exception(data.get("error", "Unknown API error")))
            elif response.status_code == 404:
                error_handler(Exception(f"Agent '{agent_identifier}' not found"))
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
def search(
    query: str = typer.Argument(..., help="Search term (partial short ID or name)"),
):
    """Search for agents by partial short ID or name"""
    asyncio.run(_search_agents(query))


async def _search_agents(query: str):
    """Internal async agent search"""
    try:
        from ..utils import get_api_headers

        headers = get_api_headers()

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v1/search/agents",
                params={"q": query},
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    agents = data.get("data", {}).get("agents", [])

                    if not agents:
                        info_message(f"No agents found matching '{query}'")
                        return

                    console.print(
                        f"\n🔍 [bold cyan]Search Results for '{query}'[/bold cyan]"
                    )

                    # Create table data
                    table_data = []
                    for agent in agents:
                        table_data.append(
                            {
                                "short_id": agent.get("short_id", "n/a"),
                                "name": agent.get("name", "unknown"),
                                "type": agent.get("type", "unknown"),
                                "status": agent.get("status", "unknown"),
                            }
                        )

                    table = create_status_table("🤖 Matching Agents", table_data)
                    console.print(table)

                    success_message(f"Found {len(agents)} matching agents")

                    if len(agents) == 1:
                        agent = agents[0]
                        short_id = agent.get("short_id")
                        name = agent.get("name")
                        console.print(
                            f"\n💡 [dim]Tip: Use 'hive agent describe {short_id or name}' for more details[/dim]"
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
        # Simply spawn a MetaAgent to start the self-improvement process
        console.print("🤖 Spawning Meta Agent for system bootstrap...")

        # Call the spawn command to create a meta agent
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli.main",
                "agent",
                "spawn",
                "meta",
                "--name",
                "bootstrap-meta",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            success_message("Meta Agent spawned successfully!")
            console.print("🎯 System bootstrap initiated. The Meta Agent will:")
            console.print("   • Analyze system health and performance")
            console.print("   • Identify optimization opportunities")
            console.print("   • Propose and implement improvements")
            console.print("   • Monitor system metrics continuously")
            console.print("\n💡 Use 'hive task submit' to give the Meta Agent tasks")
            console.print("💡 Use 'hive agent list' to see active agents")
        else:
            console.print(f"[red]Failed to spawn Meta Agent: {result.stderr}[/red]")
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


@app.command()
def monitor(
    refresh_interval: int = typer.Option(
        3, "--interval", "-i", help="Refresh interval in seconds"
    ),
    agent_filter: str = typer.Option(
        None, "--filter", "-f", help="Filter by agent name/type"
    ),
):
    """Real-time agent monitoring dashboard"""
    asyncio.run(_monitor_agents(refresh_interval, agent_filter))


async def _monitor_agents(refresh_interval: int, agent_filter: str = None):
    """Internal async agent monitoring"""
    try:
        from rich import box
        from rich.layout import Layout
        from rich.live import Live
        from rich.panel import Panel
        from rich.table import Table

        console.clear()
        filter_desc = f" (filter: {agent_filter})" if agent_filter else ""
        info_message(
            f"Starting real-time agent monitor{filter_desc} (refresh every {refresh_interval}s)"
        )
        info_message("Press Ctrl+C to exit")

        def create_agent_display():
            """Create the agent monitoring display layout"""
            layout = Layout()

            # Split into sections
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="agents", ratio=2),
                Layout(name="details", ratio=1),
                Layout(name="footer", size=3),
            )

            return layout

        async def update_agent_display():
            """Update the display with current agent data"""
            try:
                # Get agent data
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        "http://localhost:9001/api/v1/cli/agents"
                    )

                    if response.status_code != 200:
                        raise Exception(f"API returned status {response.status_code}")

                    data = response.json()
                    if not data.get("success"):
                        raise Exception(data.get("error", "Unknown API error"))

                    agents = data.get("data", [])

                    # Apply filter if specified
                    if agent_filter:
                        agents = [
                            a
                            for a in agents
                            if agent_filter.lower() in a.get("name", "").lower()
                            or agent_filter.lower() in a.get("type", "").lower()
                        ]

                # Create agents table
                agents_table = Table(title="🤖 Active Agents", box=box.ROUNDED)
                agents_table.add_column(
                    "Short ID", style="green", no_wrap=True, width=8
                )
                agents_table.add_column("Name", style="cyan", no_wrap=True)
                agents_table.add_column("Type", style="yellow", no_wrap=True)
                agents_table.add_column("Status", style="bold", no_wrap=True)
                agents_table.add_column("Uptime", style="dim", no_wrap=True)
                agents_table.add_column("Heartbeat", style="dim", no_wrap=True)

                for agent in agents:
                    status = agent.get("status", "unknown")
                    status_icon = {
                        "active": "🟢",
                        "starting": "🟡",
                        "stopping": "🟠",
                        "inactive": "🔴",
                        "error": "❌",
                    }.get(status, "⚪")

                    uptime = agent.get("uptime", 0)
                    uptime_str = (
                        f"{uptime:.1f}s" if uptime < 60 else f"{uptime / 60:.1f}m"
                    )

                    heartbeat = agent.get("last_heartbeat", "never")
                    if heartbeat != "never" and isinstance(heartbeat, (int, float)):
                        heartbeat_str = f"{time.time() - heartbeat:.1f}s ago"
                    else:
                        heartbeat_str = str(heartbeat)

                    agents_table.add_row(
                        agent.get("short_id", "n/a"),
                        agent.get("name", "unknown"),
                        agent.get("type", "unknown"),
                        f"{status_icon} {status}",
                        uptime_str,
                        heartbeat_str,
                    )

                # Create summary table
                summary_table = Table(title="📊 Agent Summary", box=box.ROUNDED)
                summary_table.add_column("Metric", style="cyan", no_wrap=True)
                summary_table.add_column("Count", style="bold", no_wrap=True)

                # Count by status
                status_counts = {}
                type_counts = {}
                for agent in agents:
                    status = agent.get("status", "unknown")
                    agent_type = agent.get("type", "unknown")
                    status_counts[status] = status_counts.get(status, 0) + 1
                    type_counts[agent_type] = type_counts.get(agent_type, 0) + 1

                summary_table.add_row("Total Agents", str(len(agents)))
                for status, count in status_counts.items():
                    summary_table.add_row(f"{status.title()} Agents", str(count))

                # Type breakdown
                for agent_type, count in type_counts.items():
                    summary_table.add_row(f"{agent_type.title()} Agents", str(count))

                # Create layout
                layout = create_agent_display()

                # Header
                filter_text = f" | Filter: {agent_filter}" if agent_filter else ""
                layout["header"].update(
                    Panel(
                        f"[bold cyan]Agent Monitor[/bold cyan] | "
                        f"Refresh: {refresh_interval}s | "
                        f"Time: {time.strftime('%H:%M:%S')}{filter_text}",
                        border_style="blue",
                    )
                )

                # Main content
                layout["agents"].update(Panel(agents_table, border_style="green"))
                layout["details"].update(Panel(summary_table, border_style="yellow"))

                # Footer
                layout["footer"].update(
                    Panel(
                        "[dim]Press Ctrl+C to exit | Use --filter to filter agents | Use 'hive agent describe <name>' for details[/dim]",
                        border_style="dim",
                    )
                )

                return layout

            except Exception as e:
                # Error display
                layout = create_agent_display()
                layout["header"].update(
                    Panel(
                        "[bold red]Agent Monitor - Error[/bold red]", border_style="red"
                    )
                )

                error_panel = Panel(
                    f"[red]Error fetching agent data: {str(e)}[/red]\n\n"
                    "• Check if API server is running: hive system status\n"
                    "• Try starting the system: hive system start",
                    title="❌ Connection Error",
                    border_style="red",
                )
                layout["agents"].update(error_panel)
                layout["details"].update(Panel("", border_style="dim"))
                layout["footer"].update(
                    Panel(
                        "[dim]Press Ctrl+C to exit | Fix connection issues to see agent data[/dim]",
                        border_style="dim",
                    )
                )

                return layout

        # Start live monitoring
        with Live(await update_agent_display(), refresh_per_second=1) as live:
            while True:
                await asyncio.sleep(refresh_interval)
                live.update(await update_agent_display())

    except KeyboardInterrupt:
        console.clear()
        success_message("Agent monitoring stopped")
    except Exception as e:
        error_handler(e)


@app.command()
def start(
    agent_types: str = typer.Option(
        "meta",
        "--types",
        "-t",
        help="Comma-separated agent types to start (meta,developer,qa,architect)",
    ),
    count: int = typer.Option(
        1, "--count", "-c", help="Number of each agent type to start"
    ),
    all_types: bool = typer.Option(
        False, "--all", help="Start one agent of each available type"
    ),
    prefix: str = typer.Option(
        "batch", "--prefix", "-p", help="Prefix for agent names"
    ),
):
    """Start multiple agents in batch"""
    asyncio.run(_start_agents_batch(agent_types, count, all_types, prefix))


async def _start_agents_batch(
    agent_types: str, count: int, all_types: bool, prefix: str
):
    """Internal async batch agent starting"""
    try:
        if all_types:
            types_to_start = ["meta", "developer", "qa", "architect", "devops"]
            info_message("Starting one agent of each type...")
        else:
            types_to_start = [t.strip() for t in agent_types.split(",")]
            info_message(
                f"Starting {count} agent(s) for each type: {', '.join(types_to_start)}"
            )

        started_agents = []
        failed_agents = []

        for agent_type in types_to_start:
            for i in range(count):
                try:
                    agent_name = (
                        f"{prefix}-{agent_type}-{i + 1}"
                        if count > 1
                        else f"{prefix}-{agent_type}"
                    )

                    console.print(f"🤖 Starting {agent_type} agent: {agent_name}")

                    # Use the existing spawn logic
                    params = {"agent_type": agent_type, "agent_name": agent_name}

                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.post(
                            f"{API_BASE_URL}/api/v1/cli/agents", params=params
                        )

                        if response.status_code == 200:
                            data = response.json()
                            if data.get("success"):
                                result = data.get("data", {})
                                started_agents.append(
                                    {
                                        "name": result.get("agent_name", agent_name),
                                        "type": agent_type,
                                        "session": result.get(
                                            "session_name", "unknown"
                                        ),
                                    }
                                )
                                console.print(f"  ✅ {agent_name} started successfully")
                            else:
                                failed_agents.append(
                                    {
                                        "name": agent_name,
                                        "type": agent_type,
                                        "error": data.get("error", "Unknown error"),
                                    }
                                )
                                console.print(
                                    f"  ❌ {agent_name} failed: {data.get('error', 'Unknown error')}"
                                )
                        else:
                            failed_agents.append(
                                {
                                    "name": agent_name,
                                    "type": agent_type,
                                    "error": f"HTTP {response.status_code}",
                                }
                            )
                            console.print(
                                f"  ❌ {agent_name} failed: HTTP {response.status_code}"
                            )

                    # Small delay between starts to avoid overwhelming the system
                    await asyncio.sleep(0.5)

                except Exception as e:
                    failed_agents.append(
                        {"name": agent_name, "type": agent_type, "error": str(e)}
                    )
                    console.print(f"  ❌ {agent_name} failed: {str(e)}")

        # Summary
        console.print("\n📊 Batch Agent Start Summary:")
        if started_agents:
            success_message(f"✅ Successfully started {len(started_agents)} agents:")
            for agent in started_agents:
                console.print(
                    f"  • {agent['name']} ({agent['type']}) - Session: {agent['session']}"
                )

        if failed_agents:
            console.print(f"\n❌ Failed to start {len(failed_agents)} agents:")
            for agent in failed_agents:
                console.print(
                    f"  • {agent['name']} ({agent['type']}) - Error: {agent['error']}"
                )

        if started_agents:
            console.print(
                "\n💡 [dim]Use 'hive agent list' to see all active agents[/dim]"
            )

    except httpx.ConnectError:
        error_handler(
            Exception(
                "Cannot connect to API server. Is it running? Try: hive system start"
            )
        )
    except Exception as e:
        error_handler(e)


@app.command()
def stop_all(
    agent_type_filter: str = typer.Option(
        None, "--type", "-t", help="Only stop agents of this type"
    ),
    force: bool = typer.Option(
        False, "--force", help="Force stop without confirmation"
    ),
):
    """Stop all agents or agents of a specific type"""
    asyncio.run(_stop_agents_batch(agent_type_filter, force))


async def _stop_agents_batch(agent_type_filter: str = None, force: bool = False):
    """Internal async batch agent stopping"""
    try:
        # Get current agents
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{API_BASE_URL}/api/v1/cli/agents")

            if response.status_code != 200:
                error_handler(
                    Exception(f"Failed to get agent list: HTTP {response.status_code}")
                )
                return

            data = response.json()
            if not data.get("success"):
                error_handler(Exception(data.get("error", "Failed to get agent list")))
                return

            agents = data.get("data", [])

            # Filter by type if specified
            if agent_type_filter:
                agents = [a for a in agents if a.get("type") == agent_type_filter]
                filter_desc = f" of type '{agent_type_filter}'"
            else:
                filter_desc = ""

            if not agents:
                info_message(f"No agents found{filter_desc}")
                return

            # Confirm unless forced
            if not force:
                console.print(f"\n📋 Agents to stop{filter_desc}:")
                for agent in agents:
                    console.print(
                        f"  • {agent.get('name', 'unknown')} ({agent.get('type', 'unknown')}) - {agent.get('status', 'unknown')}"
                    )

                if not typer.confirm(f"\nStop {len(agents)} agent(s)?"):
                    info_message("Batch stop cancelled")
                    return

            # Stop agents
            stopped_agents = []
            failed_agents = []

            info_message(f"Stopping {len(agents)} agent(s)...")

            for agent in agents:
                agent_name = agent.get("name", "unknown")
                try:
                    console.print(f"🛑 Stopping {agent_name}...")

                    response = await client.post(
                        f"{API_BASE_URL}/api/v1/cli/agents/{agent_name}/stop"
                    )

                    if response.status_code == 200:
                        data = response.json()
                        if data.get("success"):
                            stopped_agents.append(agent_name)
                            console.print(f"  ✅ {agent_name} stopped successfully")
                        else:
                            failed_agents.append(
                                {
                                    "name": agent_name,
                                    "error": data.get("error", "Unknown error"),
                                }
                            )
                            console.print(
                                f"  ❌ {agent_name} failed: {data.get('error', 'Unknown error')}"
                            )
                    else:
                        failed_agents.append(
                            {
                                "name": agent_name,
                                "error": f"HTTP {response.status_code}",
                            }
                        )
                        console.print(
                            f"  ❌ {agent_name} failed: HTTP {response.status_code}"
                        )

                except Exception as e:
                    failed_agents.append({"name": agent_name, "error": str(e)})
                    console.print(f"  ❌ {agent_name} failed: {str(e)}")

                # Small delay between stops
                await asyncio.sleep(0.2)

            # Summary
            console.print("\n📊 Batch Agent Stop Summary:")
            if stopped_agents:
                success_message(
                    f"✅ Successfully stopped {len(stopped_agents)} agents:"
                )
                for name in stopped_agents:
                    console.print(f"  • {name}")

            if failed_agents:
                console.print(f"\n❌ Failed to stop {len(failed_agents)} agents:")
                for agent in failed_agents:
                    console.print(f"  • {agent['name']} - Error: {agent['error']}")

    except httpx.ConnectError:
        error_handler(
            Exception(
                "Cannot connect to API server. Is it running? Try: hive system start"
            )
        )
    except Exception as e:
        error_handler(e)


if __name__ == "__main__":
    app()
