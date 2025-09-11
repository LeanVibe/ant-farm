"""
Real-time monitoring commands for the Hive CLI
"""

import asyncio
import time
from typing import Optional

import httpx
import typer
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from ..utils import (
    API_BASE_URL,
    create_status_table,
    create_system_status_table,
    error_handler,
    get_api_headers,
    info_message,
    success_message,
)

app = typer.Typer(help="Real-time monitoring commands")
console = Console()


@app.command()
def agents(
    live: bool = typer.Option(False, "--live", "-l", help="Enable live monitoring mode"),
    refresh_interval: int = typer.Option(
        3, "--interval", "-i", help="Refresh interval in seconds (for live mode)"
    ),
    filter_type: str = typer.Option(
        None, "--filter-type", "-t", help="Filter by agent type"
    ),
    filter_status: str = typer.Option(
        None, "--filter-status", "-s", help="Filter by agent status"
    ),
):
    """Monitor agents with real-time updates"""
    if live:
        asyncio.run(_monitor_agents_live(refresh_interval, filter_type, filter_status))
    else:
        asyncio.run(_list_agents_snapshot(filter_type, filter_status))


@app.command()
def tasks(
    live: bool = typer.Option(False, "--live", "-l", help="Enable live monitoring mode"),
    refresh_interval: int = typer.Option(
        3, "--interval", "-i", help="Refresh interval in seconds (for live mode)"
    ),
    filter_status: str = typer.Option(
        None, "--filter-status", "-s", help="Filter by task status"
    ),
    filter_type: str = typer.Option(
        None, "--filter-type", "-t", help="Filter by task type"
    ),
):
    """Monitor tasks with real-time updates"""
    if live:
        asyncio.run(_monitor_tasks_live(refresh_interval, filter_status, filter_type))
    else:
        asyncio.run(_list_tasks_snapshot(filter_status, filter_type))


@app.command()
def system(
    live: bool = typer.Option(False, "--live", "-l", help="Enable live monitoring mode"),
    refresh_interval: int = typer.Option(
        3, "--interval", "-i", help="Refresh interval in seconds (for live mode)"
    ),
):
    """Monitor system health and metrics"""
    if live:
        asyncio.run(_monitor_system_live(refresh_interval))
    else:
        asyncio.run(_show_system_snapshot())


@app.command()
def resources(
    live: bool = typer.Option(False, "--live", "-l", help="Enable live monitoring mode"),
    refresh_interval: int = typer.Option(
        3, "--interval", "-i", help="Refresh interval in seconds (for live mode)"
    ),
):
    """Monitor system resources (CPU, memory, disk)"""
    if live:
        asyncio.run(_monitor_resources_live(refresh_interval))
    else:
        asyncio.run(_show_resources_snapshot())


@app.command()
def metrics(
    target: str = typer.Argument(..., help="Metrics target: agents, tasks, or system"),
    period: str = typer.Option(
        "1h", "--period", "-p", help="Time period (e.g., 1h, 6h, 1d, 1w)"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json, csv"
    ),
):
    """Show performance metrics and analytics"""
    if target not in ["agents", "tasks", "system"]:
        error_handler(Exception(f"Invalid target: {target}. Must be agents, tasks, or system"))
        return

    asyncio.run(_show_metrics(target, period, format))


async def _monitor_agents_live(
    refresh_interval: int, filter_type: str | None, filter_status: str | None
):
    """Live monitoring of agents"""
    try:
        console.clear()
        filter_desc = ""
        if filter_type or filter_status:
            filters = []
            if filter_type:
                filters.append(f"type={filter_type}")
            if filter_status:
                filters.append(f"status={filter_status}")
            filter_desc = f" ({', '.join(filters)})"

        info_message(
            f"Starting live agent monitor{filter_desc} (refresh every {refresh_interval}s)"
        )
        info_message("Press Ctrl+C to exit")

        async def update_display():
            """Update the agent monitoring display"""
            try:
                headers = get_api_headers()
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{API_BASE_URL}/api/v1/cli/agents", headers=headers
                    )

                    if response.status_code != 200:
                        raise Exception(f"API returned status {response.status_code}")

                    data = response.json()
                    if not data.get("success"):
                        raise Exception(data.get("error", "Unknown API error"))

                    agents = data.get("data", [])

                    # Apply filters
                    if filter_type:
                        agents = [a for a in agents if a.get("type") == filter_type]
                    if filter_status:
                        agents = [a for a in agents if a.get("status") == filter_status]

                    # Create agents table
                    agents_table = Table(
                        title="🤖 Active Agents",
                        box=box.ROUNDED,
                        show_header=True,
                        header_style="bold cyan"
                    )
                    agents_table.add_column("Short ID", style="green", no_wrap=True, width=8)
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
                        uptime_str = f"{uptime:.1f}s" if uptime < 60 else f"{uptime / 60:.1f}m"

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

                    # Create summary panel
                    summary_text = f"Total Agents: {len(agents)}"
                    if filter_type:
                        summary_text += f" | Type: {filter_type}"
                    if filter_status:
                        summary_text += f" | Status: {filter_status}"
                    summary_text += f" | Updated: {time.strftime('%H:%M:%S')}"

                    layout = Layout()
                    layout.split_column(
                        Layout(name="summary", size=3),
                        Layout(name="agents", ratio=1),
                    )

                    layout["summary"].update(
                        Panel(
                            summary_text,
                            title="📊 Summary",
                            border_style="blue",
                        )
                    )
                    layout["agents"].update(Panel(agents_table, border_style="green"))

                    return layout

            except Exception as e:
                # Error display
                layout = Layout()
                layout.split_column(
                    Layout(name="summary", size=3),
                    Layout(name="agents", ratio=1),
                )

                layout["summary"].update(
                    Panel(
                        "[bold red]Agent Monitor - Error[/bold red]",
                        border_style="red",
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

                return layout

        # Start live monitoring
        with Live(await update_display(), refresh_per_second=1) as live:
            while True:
                await asyncio.sleep(refresh_interval)
                live.update(await update_display())

    except KeyboardInterrupt:
        console.clear()
        success_message("Agent monitoring stopped")
    except Exception as e:
        error_handler(e)


async def _list_agents_snapshot(filter_type: str | None = None, filter_status: str | None = None):
    """Show snapshot of agents"""
    try:
        headers = get_api_headers()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v1/cli/agents", headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    agents = data.get("data", [])

                    # Apply filters
                    if filter_type:
                        agents = [a for a in agents if a.get("type") == filter_type]
                    if filter_status:
                        agents = [a for a in agents if a.get("status") == filter_status]

                    if not agents:
                        filter_desc = ""
                        if filter_type or filter_status:
                            filters = []
                            if filter_type:
                                filters.append(f"type={filter_type}")
                            if filter_status:
                                filters.append(f"status={filter_status}")
                            filter_desc = f" matching {', '.join(filters)}"
                        info_message(f"No agents found{filter_desc}")
                        return

                    # Create table
                    table_data = []
                    for agent in agents:
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

                    title = "🤖 Agents"
                    if filter_type or filter_status:
                        filters = []
                        if filter_type:
                            filters.append(f"type={filter_type}")
                        if filter_status:
                            filters.append(f"status={filter_status}")
                        title += f" ({', '.join(filters)})"

                    table = create_status_table(title, table_data)
                    console.print(table)

                    success_message(f"Found {len(agents)} agents")
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


async def _monitor_tasks_live(
    refresh_interval: int, filter_status: str | None, filter_type: str | None
):
    """Live monitoring of tasks"""
    try:
        console.clear()
        filter_desc = ""
        if filter_type or filter_status:
            filters = []
            if filter_type:
                filters.append(f"type={filter_type}")
            if filter_status:
                filters.append(f"status={filter_status}")
            filter_desc = f" ({', '.join(filters)})"

        info_message(
            f"Starting live task monitor{filter_desc} (refresh every {refresh_interval}s)"
        )
        info_message("Press Ctrl+C to exit")

        async def update_display():
            """Update the task monitoring display"""
            try:
                headers = get_api_headers()
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{API_BASE_URL}/api/v1/cli/tasks", headers=headers
                    )

                    if response.status_code != 200:
                        raise Exception(f"API returned status {response.status_code}")

                    data = response.json()
                    if not data.get("success"):
                        raise Exception(data.get("error", "Unknown API error"))

                    tasks = data.get("data", [])

                    # Apply filters
                    if filter_type:
                        tasks = [t for t in tasks if t.get("type") == filter_type]
                    if filter_status:
                        tasks = [t for t in tasks if t.get("status") == filter_status]

                    # Create tasks table
                    tasks_table = Table(
                        title="📋 Active Tasks",
                        box=box.ROUNDED,
                        show_header=True,
                        header_style="bold cyan"
                    )
                    tasks_table.add_column("Short ID", style="green", no_wrap=True, width=8)
                    tasks_table.add_column("Title", style="cyan", no_wrap=True)
                    tasks_table.add_column("Type", style="yellow", no_wrap=True)
                    tasks_table.add_column("Status", style="bold", no_wrap=True)
                    tasks_table.add_column("Priority", style="magenta", no_wrap=True)
                    tasks_table.add_column("Assigned To", style="dim", no_wrap=True)

                    for task in tasks:
                        status = task.get("status", "unknown")
                        status_icon = {
                            "pending": "⏳",
                            "in_progress": "🔄",
                            "completed": "✅",
                            "failed": "❌",
                            "cancelled": "🚫",
                        }.get(status, "⚪")

                        priority_val = task.get("priority", 5)
                        priority_name = {
                            1: "critical",
                            3: "high",
                            5: "normal",
                            7: "low",
                            9: "background",
                        }.get(priority_val, str(priority_val))

                        tasks_table.add_row(
                            task.get("short_id", "n/a"),
                            task.get("title", "untitled")[:20],
                            task.get("type", "unknown"),
                            f"{status_icon} {status}",
                            priority_name,
                            task.get("assigned_to", "unassigned"),
                        )

                    # Create summary panel
                    summary_text = f"Total Tasks: {len(tasks)}"
                    if filter_type:
                        summary_text += f" | Type: {filter_type}"
                    if filter_status:
                        summary_text += f" | Status: {filter_status}"
                    summary_text += f" | Updated: {time.strftime('%H:%M:%S')}"

                    layout = Layout()
                    layout.split_column(
                        Layout(name="summary", size=3),
                        Layout(name="tasks", ratio=1),
                    )

                    layout["summary"].update(
                        Panel(
                            summary_text,
                            title="📊 Summary",
                            border_style="blue",
                        )
                    )
                    layout["tasks"].update(Panel(tasks_table, border_style="green"))

                    return layout

            except Exception as e:
                # Error display
                layout = Layout()
                layout.split_column(
                    Layout(name="summary", size=3),
                    Layout(name="tasks", ratio=1),
                )

                layout["summary"].update(
                    Panel(
                        "[bold red]Task Monitor - Error[/bold red]",
                        border_style="red",
                    )
                )

                error_panel = Panel(
                    f"[red]Error fetching task data: {str(e)}[/red]\n\n"
                    "• Check if API server is running: hive system status\n"
                    "• Try starting the system: hive system start",
                    title="❌ Connection Error",
                    border_style="red",
                )
                layout["tasks"].update(error_panel)

                return layout

        # Start live monitoring
        with Live(await update_display(), refresh_per_second=1) as live:
            while True:
                await asyncio.sleep(refresh_interval)
                live.update(await update_display())

    except KeyboardInterrupt:
        console.clear()
        success_message("Task monitoring stopped")
    except Exception as e:
        error_handler(e)


async def _list_tasks_snapshot(filter_status: str | None = None, filter_type: str | None = None):
    """Show snapshot of tasks"""
    try:
        headers = get_api_headers()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v1/cli/tasks", headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    tasks = data.get("data", [])

                    # Apply filters
                    if filter_type:
                        tasks = [t for t in tasks if t.get("type") == filter_type]
                    if filter_status:
                        tasks = [t for t in tasks if t.get("status") == filter_status]

                    if not tasks:
                        filter_desc = ""
                        if filter_type or filter_status:
                            filters = []
                            if filter_type:
                                filters.append(f"type={filter_type}")
                            if filter_status:
                                filters.append(f"status={filter_status}")
                            filter_desc = f" matching {', '.join(filters)}"
                        info_message(f"No tasks found{filter_desc}")
                        return

                    # Create table
                    table_data = []
                    for task in tasks:
                        priority_val = task.get("priority", 5)
                        priority_name = {
                            1: "critical",
                            3: "high",
                            5: "normal",
                            7: "low",
                            9: "background",
                        }.get(priority_val, str(priority_val))

                        table_data.append(
                            {
                                "short_id": task.get("short_id", "n/a"),
                                "title": task.get("title", "untitled")[:30],
                                "type": task.get("type", "unknown"),
                                "status": task.get("status", "unknown"),
                                "priority": priority_name,
                                "assigned_to": task.get("assigned_to", "unassigned"),
                            }
                        )

                    title = "📋 Tasks"
                    if filter_type or filter_status:
                        filters = []
                        if filter_type:
                            filters.append(f"type={filter_type}")
                        if filter_status:
                            filters.append(f"status={filter_status}")
                        title += f" ({', '.join(filters)})"

                    table = create_status_table(title, table_data)
                    console.print(table)

                    success_message(f"Found {len(tasks)} tasks")
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


async def _monitor_system_live(refresh_interval: int):
    """Live monitoring of system health"""
    try:
        console.clear()
        info_message(
            f"Starting live system monitor (refresh every {refresh_interval}s)"
        )
        info_message("Press Ctrl+C to exit")

        async def update_display():
            """Update the system monitoring display"""
            try:
                headers = get_api_headers()
                async with httpx.AsyncClient(timeout=5.0) as client:
                    # Get system health
                    response = await client.get(
                        f"{API_BASE_URL}/api/v1/health", headers=headers
                    )

                    if response.status_code != 200:
                        raise Exception(f"API returned status {response.status_code}")

                    health_data = response.json()

                    # Get system metrics (if available)
                    try:
                        metrics_response = await client.get(
                            f"{API_BASE_URL}/api/v1/metrics/system", headers=headers
                        )
                        metrics_data = (
                            metrics_response.json() if metrics_response.status_code == 200
                            else {}
                        )
                    except:
                        metrics_data = {}

                    # Create system status table
                    system_table = Table(
                        title="🏥 System Health",
                        box=box.ROUNDED,
                        show_header=True,
                        header_style="bold cyan"
                    )
                    system_table.add_column("Component", style="cyan", no_wrap=True)
                    system_table.add_column("Status", style="bold", no_wrap=True)
                    system_table.add_column("Details", style="dim")

                    # API status
                    api_status = "✅ Online" if health_data.get("status") == "healthy" else "❌ Offline"
                    system_table.add_row("API Server", api_status, "Port 9001")

                    # Database status
                    db_status = health_data.get("database", {}).get("status", "unknown")
                    db_status_icon = "✅ Online" if db_status == "healthy" else "❌ Offline"
                    system_table.add_row("Database", db_status_icon, "PostgreSQL")

                    # Redis status
                    redis_status = health_data.get("redis", {}).get("status", "unknown")
                    redis_status_icon = "✅ Online" if redis_status == "healthy" else "❌ Offline"
                    system_table.add_row("Redis", redis_status_icon, "Cache/Queue")

                    # Add metrics if available
                    if metrics_data.get("success"):
                        metrics = metrics_data.get("data", {})
                        if "cpu_percent" in metrics:
                            system_table.add_row("CPU Usage", f"{metrics['cpu_percent']:.1f}%", "System load")
                        if "memory_percent" in metrics:
                            system_table.add_row("Memory", f"{metrics['memory_percent']:.1f}%", "RAM usage")
                        if "disk_percent" in metrics:
                            system_table.add_row("Disk", f"{metrics['disk_percent']:.1f}%", "Storage usage")

                    # Create summary panel
                    summary_text = f"System Status: {'Healthy' if health_data.get('status') == 'healthy' else 'Issues Detected'}"
                    summary_text += f" | Updated: {time.strftime('%H:%M:%S')}"

                    layout = Layout()
                    layout.split_column(
                        Layout(name="summary", size=3),
                        Layout(name="system", ratio=1),
                    )

                    layout["summary"].update(
                        Panel(
                            summary_text,
                            title="📊 Summary",
                            border_style="blue",
                        )
                    )
                    layout["system"].update(Panel(system_table, border_style="green"))

                    return layout

            except Exception as e:
                # Error display
                layout = Layout()
                layout.split_column(
                    Layout(name="summary", size=3),
                    Layout(name="system", ratio=1),
                )

                layout["summary"].update(
                    Panel(
                        "[bold red]System Monitor - Error[/bold red]",
                        border_style="red",
                    )
                )

                error_panel = Panel(
                    f"[red]Error fetching system data: {str(e)}[/red]\n\n"
                    "• Check if API server is running: hive system status\n"
                    "• Try starting the system: hive system start",
                    title="❌ Connection Error",
                    border_style="red",
                )
                layout["system"].update(error_panel)

                return layout

        # Start live monitoring
        with Live(await update_display(), refresh_per_second=1) as live:
            while True:
                await asyncio.sleep(refresh_interval)
                live.update(await update_display())

    except KeyboardInterrupt:
        console.clear()
        success_message("System monitoring stopped")
    except Exception as e:
        error_handler(e)


async def _show_system_snapshot():
    """Show snapshot of system health"""
    try:
        headers = get_api_headers()
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get system health
            response = await client.get(
                f"{API_BASE_URL}/api/v1/health", headers=headers
            )

            if response.status_code == 200:
                health_data = response.json()

                # Create system status data
                services = {
                    "api": {
                        "status": "online" if health_data.get("status") == "healthy" else "offline",
                        "details": "FastAPI server on port 9001"
                    },
                    "database": {
                        "status": "online" if health_data.get("database", {}).get("status") == "healthy" else "offline",
                        "details": "PostgreSQL database"
                    },
                    "redis": {
                        "status": "online" if health_data.get("redis", {}).get("status") == "healthy" else "offline",
                        "details": "Redis cache and message broker"
                    }
                }

                table = create_system_status_table(services)
                console.print(table)

                overall_status = "healthy" if all(
                    service["status"] == "online" for service in services.values()
                ) else "degraded"

                if overall_status == "healthy":
                    success_message("System is healthy")
                else:
                    from ..utils import warning_message
                    warning_message("System has issues - check status details above")
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


async def _monitor_resources_live(refresh_interval: int):
    """Live monitoring of system resources"""
    try:
        console.clear()
        info_message(
            f"Starting live resource monitor (refresh every {refresh_interval}s)"
        )
        info_message("Press Ctrl+C to exit")

        async def update_display():
            """Update the resource monitoring display"""
            try:
                headers = get_api_headers()
                async with httpx.AsyncClient(timeout=5.0) as client:
                    # Get system metrics
                    response = await client.get(
                        f"{API_BASE_URL}/api/v1/metrics/system", headers=headers
                    )

                    if response.status_code != 200:
                        raise Exception(f"API returned status {response.status_code}")

                    data = response.json()
                    if not data.get("success"):
                        raise Exception(data.get("error", "Unknown API error"))

                    metrics = data.get("data", {})

                    # Create resources table
                    resources_table = Table(
                        title="💻 System Resources",
                        box=box.ROUNDED,
                        show_header=True,
                        header_style="bold cyan"
                    )
                    resources_table.add_column("Resource", style="cyan", no_wrap=True)
                    resources_table.add_column("Usage", style="bold", no_wrap=True)
                    resources_table.add_column("Status", style="green", no_wrap=True)
                    resources_table.add_column("Details", style="dim")

                    # CPU
                    cpu_percent = metrics.get("cpu_percent", 0)
                    cpu_status = "🟢 Normal" if cpu_percent < 80 else "🟠 High" if cpu_percent < 95 else "🔴 Critical"
                    resources_table.add_row("CPU", f"{cpu_percent:.1f}%", cpu_status, "System processor load")

                    # Memory
                    memory_percent = metrics.get("memory_percent", 0)
                    memory_status = "🟢 Normal" if memory_percent < 80 else "🟠 High" if memory_percent < 95 else "🔴 Critical"
                    memory_used = metrics.get("memory_used", 0)
                    memory_total = metrics.get("memory_total", 0)
                    memory_details = f"{memory_used/1024/1024/1024:.1f}GB / {memory_total/1024/1024/1024:.1f}GB"
                    resources_table.add_row("Memory", f"{memory_percent:.1f}%", memory_status, memory_details)

                    # Disk
                    disk_percent = metrics.get("disk_percent", 0)
                    disk_status = "🟢 Normal" if disk_percent < 80 else "🟠 High" if disk_percent < 95 else "🔴 Critical"
                    disk_used = metrics.get("disk_used", 0)
                    disk_total = metrics.get("disk_total", 0)
                    disk_details = f"{disk_used/1024/1024/1024:.1f}GB / {disk_total/1024/1024/1024:.1f}GB"
                    resources_table.add_row("Disk", f"{disk_percent:.1f}%", disk_status, disk_details)

                    # Network (if available)
                    if "network_rx" in metrics and "network_tx" in metrics:
                        network_rx = metrics.get("network_rx", 0)
                        network_tx = metrics.get("network_tx", 0)
                        network_details = f"↓{network_rx/1024/1024:.1f}MB ↑{network_tx/1024/1024:.1f}MB"
                        resources_table.add_row("Network", "Active", "🟢 Normal", network_details)

                    # Create summary panel
                    summary_text = f"Resource Monitor | Updated: {time.strftime('%H:%M:%S')}"

                    layout = Layout()
                    layout.split_column(
                        Layout(name="summary", size=3),
                        Layout(name="resources", ratio=1),
                    )

                    layout["summary"].update(
                        Panel(
                            summary_text,
                            title="📊 Summary",
                            border_style="blue",
                        )
                    )
                    layout["resources"].update(Panel(resources_table, border_style="green"))

                    return layout

            except Exception as e:
                # Error display
                layout = Layout()
                layout.split_column(
                    Layout(name="summary", size=3),
                    Layout(name="resources", ratio=1),
                )

                layout["summary"].update(
                    Panel(
                        "[bold red]Resource Monitor - Error[/bold red]",
                        border_style="red",
                    )
                )

                error_panel = Panel(
                    f"[red]Error fetching resource data: {str(e)}[/red]\n\n"
                    "• Check if API server is running: hive system status\n"
                    "• Try starting the system: hive system start\n"
                    "• Ensure system metrics collection is enabled",
                    title="❌ Connection Error",
                    border_style="red",
                )
                layout["resources"].update(error_panel)

                return layout

        # Start live monitoring
        with Live(await update_display(), refresh_per_second=1) as live:
            while True:
                await asyncio.sleep(refresh_interval)
                live.update(await update_display())

    except KeyboardInterrupt:
        console.clear()
        success_message("Resource monitoring stopped")
    except Exception as e:
        error_handler(e)


async def _show_resources_snapshot():
    """Show snapshot of system resources"""
    try:
        headers = get_api_headers()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v1/metrics/system", headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    metrics = data.get("data", {})

                    # Create resource data
                    resources = []
                    if "cpu_percent" in metrics:
                        resources.append({
                            "resource": "CPU",
                            "usage": f"{metrics['cpu_percent']:.1f}%",
                            "status": "Normal" if metrics['cpu_percent'] < 80 else "High",
                            "details": "System processor load"
                        })

                    if "memory_percent" in metrics:
                        memory_used = metrics.get("memory_used", 0)
                        memory_total = metrics.get("memory_total", 0)
                        resources.append({
                            "resource": "Memory",
                            "usage": f"{metrics['memory_percent']:.1f}%",
                            "status": "Normal" if metrics['memory_percent'] < 80 else "High",
                            "details": f"{memory_used/1024/1024/1024:.1f}GB / {memory_total/1024/1024/1024:.1f}GB"
                        })

                    if "disk_percent" in metrics:
                        disk_used = metrics.get("disk_used", 0)
                        disk_total = metrics.get("disk_total", 0)
                        resources.append({
                            "resource": "Disk",
                            "usage": f"{metrics['disk_percent']:.1f}%",
                            "status": "Normal" if metrics['disk_percent'] < 80 else "High",
                            "details": f"{disk_used/1024/1024/1024:.1f}GB / {disk_total/1024/1024/1024:.1f}GB"
                        })

                    if resources:
                        table = create_status_table("💻 System Resources", resources)
                        console.print(table)
                        success_message("Resource metrics retrieved")
                    else:
                        info_message("No resource metrics available")
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


async def _show_metrics(target: str, period: str, format_type: str):
    """Show metrics for the specified target"""
    try:
        headers = get_api_headers()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = None

            # Get metrics based on target
            if target == "agents":
                response = await client.get(
                    f"{API_BASE_URL}/api/v1/metrics/agents",
                    params={"period": period},
                    headers=headers
                )
            elif target == "tasks":
                response = await client.get(
                    f"{API_BASE_URL}/api/v1/metrics/tasks",
                    params={"period": period},
                    headers=headers
                )
            elif target == "system":
                response = await client.get(
                    f"{API_BASE_URL}/api/v1/metrics/system",
                    params={"period": period},
                    headers=headers
                )

            if response and response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    metrics = data.get("data", {})

                    if format_type == "json":
                        import json
                        console.print(json.dumps(metrics, indent=2))
                        return
                    elif format_type == "csv":
                        _output_metrics_csv(target, metrics)
                        return

                    # Default table format
                    if target == "agents":
                        _display_agent_metrics(metrics)
                    elif target == "tasks":
                        _display_task_metrics(metrics)
                    elif target == "system":
                        _display_system_metrics(metrics)

                    success_message(f"{target.title()} metrics for period: {period}")
                else:
                    error_handler(Exception(data.get("error", "Unknown API error")))
            elif response:
                error_handler(Exception(f"API returned status {response.status_code}"))
            else:
                error_handler(Exception("Failed to make API request"))

    except httpx.ConnectError:
        error_handler(
            Exception(
                "Cannot connect to API server. Is it running? Try: hive system start"
            )
        )
    except Exception as e:
        error_handler(e)


def _display_agent_metrics(metrics: dict):
    """Display agent metrics in table format"""
    # Agent performance metrics
    performance_data = []
    if "performance" in metrics:
        perf = metrics["performance"]
        performance_data.append({
            "metric": "Total Agents",
            "value": str(perf.get("total_agents", 0)),
            "description": "Active agents in system"
        })
        performance_data.append({
            "metric": "Avg Tasks/Agent",
            "value": f"{perf.get('avg_tasks_per_agent', 0):.1f}",
            "description": "Average tasks completed per agent"
        })
        performance_data.append({
            "metric": "Success Rate",
            "value": f"{perf.get('success_rate', 0):.1f}%",
            "description": "Task completion success rate"
        })

    if performance_data:
        table = create_status_table("🤖 Agent Performance Metrics", performance_data)
        console.print(table)

    # Agent type breakdown
    if "by_type" in metrics:
        console.print("\n[bold cyan]📊 Agents by Type:[/bold cyan]")
        type_data = []
        for agent_type, count in metrics["by_type"].items():
            type_data.append({
                "type": agent_type,
                "count": str(count),
                "percentage": f"{count / sum(metrics['by_type'].values()) * 100:.1f}%"
            })

        if type_data:
            type_table = create_status_table("Agent Types", type_data)
            console.print(type_table)

    # Handle empty metrics
    if not performance_data and "by_type" not in metrics:
        console.print("[dim]No agent metrics available[/dim]")


def _display_task_metrics(metrics: dict):
    """Display task metrics in table format"""
    # Task performance metrics
    performance_data = []
    if "performance" in metrics:
        perf = metrics["performance"]
        performance_data.append({
            "metric": "Total Tasks",
            "value": str(perf.get("total_tasks", 0)),
            "description": "Tasks processed in period"
        })
        performance_data.append({
            "metric": "Avg Completion Time",
            "value": f"{perf.get('avg_completion_time', 0):.1f}s",
            "description": "Average time to complete tasks"
        })
        performance_data.append({
            "metric": "Success Rate",
            "value": f"{perf.get('success_rate', 0):.1f}%",
            "description": "Task completion success rate"
        })

    if performance_data:
        table = create_status_table("📋 Task Performance Metrics", performance_data)
        console.print(table)

    # Task status breakdown
    if "by_status" in metrics:
        console.print("\n[bold cyan]📊 Tasks by Status:[/bold cyan]")
        status_data = []
        for status, count in metrics["by_status"].items():
            status_data.append({
                "status": status,
                "count": str(count),
                "percentage": f"{count / sum(metrics['by_status'].values()) * 100:.1f}%"
            })

        if status_data:
            status_table = create_status_table("Task Status", status_data)
            console.print(status_table)


def _display_system_metrics(metrics: dict):
    """Display system metrics in table format"""
    # System performance metrics
    performance_data = []
    if "performance" in metrics:
        perf = metrics["performance"]
        if "avg_cpu_percent" in perf:
            performance_data.append({
                "metric": "Avg CPU Usage",
                "value": f"{perf['avg_cpu_percent']:.1f}%",
                "description": "Average CPU utilization"
            })
        if "avg_memory_percent" in perf:
            performance_data.append({
                "metric": "Avg Memory Usage",
                "value": f"{perf['avg_memory_percent']:.1f}%",
                "description": "Average memory utilization"
            })
        if "avg_response_time" in perf:
            performance_data.append({
                "metric": "Avg Response Time",
                "value": f"{perf['avg_response_time']:.1f}ms",
                "description": "Average API response time"
            })

    if performance_data:
        table = create_status_table("💻 System Performance Metrics", performance_data)
        console.print(table)

    # System health indicators
    if "health" in metrics:
        console.print("\n[bold cyan]🏥 System Health Indicators:[/bold cyan]")
        health_data = []
        health = metrics["health"]
        health_data.append({
            "indicator": "Uptime",
            "value": f"{health.get('uptime_hours', 0):.1f}h",
            "status": "Good" if health.get('uptime_hours', 0) > 1 else "Starting"
        })
        health_data.append({
            "indicator": "Error Rate",
            "value": f"{health.get('error_rate', 0):.2f}%",
            "status": "Good" if health.get('error_rate', 0) < 5 else "High"
        })

        if health_data:
            health_table = create_status_table("Health Indicators", health_data)
            console.print(health_table)


def _output_metrics_csv(target: str, metrics: dict):
    """Output metrics in CSV format"""
    import csv
    import sys

    writer = csv.writer(sys.stdout)

    # Write header
    writer.writerow(["Metric", "Value", "Description"])

    # Write data based on target
    if target == "agents" and "performance" in metrics:
        perf = metrics["performance"]
        writer.writerow(["Total Agents", perf.get("total_agents", 0), "Active agents in system"])
        writer.writerow(["Avg Tasks/Agent", f"{perf.get('avg_tasks_per_agent', 0):.1f}", "Average tasks completed per agent"])
        writer.writerow(["Success Rate", f"{perf.get('success_rate', 0):.1f}%", "Task completion success rate"])

    elif target == "tasks" and "performance" in metrics:
        perf = metrics["performance"]
        writer.writerow(["Total Tasks", perf.get("total_tasks", 0), "Tasks processed in period"])
        writer.writerow(["Avg Completion Time", f"{perf.get('avg_completion_time', 0):.1f}s", "Average time to complete tasks"])
        writer.writerow(["Success Rate", f"{perf.get('success_rate', 0):.1f}%", "Task completion success rate"])

    elif target == "system" and "performance" in metrics:
        perf = metrics["performance"]
        if "avg_cpu_percent" in perf:
            writer.writerow(["Avg CPU Usage", f"{perf['avg_cpu_percent']:.1f}%", "Average CPU utilization"])
        if "avg_memory_percent" in perf:
            writer.writerow(["Avg Memory Usage", f"{perf['avg_memory_percent']:.1f}%", "Average memory utilization"])
        if "avg_response_time" in perf:
            writer.writerow(["Avg Response Time", f"{perf['avg_response_time']:.1f}ms", "Average API response time"])


if __name__ == "__main__":
    app()
