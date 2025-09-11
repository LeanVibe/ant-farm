"""
System management commands for the Hive CLI
"""

import asyncio
import subprocess
import sys
import time

import httpx
import redis.asyncio as redis
import typer
from rich.console import Console
from sqlalchemy.ext.asyncio import create_async_engine

# Import constants
try:
    from ...core.constants import Intervals
except ImportError:
    # Fallback for when running outside package
    class Intervals:
        SYSTEM_RESTART_DELAY = 2
        SYSTEM_STARTUP_DELAY = 3


from ..utils import (
    create_system_status_table,
    error_handler,
    info_message,
    success_message,
    warning_message,
)

app = typer.Typer(help="System management commands")
console = Console()


@app.command()
def status():
    """Check system status and health"""
    asyncio.run(_status())


async def _status():
    """Internal async status check"""
    try:
        # Check core services
        services = await _check_all_services()

        # Create and display status table
        table = create_system_status_table(services)
        console.print(table)

        # Overall health summary
        healthy_services = sum(1 for s in services.values() if s["status"] == "online")
        total_services = len(services)

        if healthy_services == total_services:
            success_message(f"All {total_services} services are healthy! 🎉")
        elif healthy_services > 0:
            warning_message(f"{healthy_services}/{total_services} services are healthy")
        else:
            console.print(
                "❌ [bold red]System is down - all services are offline[/bold red]"
            )

    except Exception as e:
        error_handler(e)


async def _check_all_services():
    """Check all core system services"""
    services = {}

    # Check API server
    try:
        from ..utils import get_api_headers

        headers = get_api_headers()

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                "http://localhost:9001/api/v1/health", headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    services["API"] = {
                        "status": "online",
                        "details": "http://localhost:9001",
                    }
                else:
                    services["API"] = {
                        "status": "offline",
                        "details": "Health check failed",
                    }
            else:
                services["API"] = {
                    "status": "offline",
                    "details": f"HTTP {response.status_code}",
                }
    except Exception as e:
        services["API"] = {
            "status": "offline",
            "details": f"Connection failed: {str(e)[:50]}",
        }

    # Check Database
    try:
        from ...core.config import get_settings

        settings = get_settings()

        engine = create_async_engine(settings.database_url)
        async with engine.begin() as conn:
            from sqlalchemy import text

            await conn.execute(text("SELECT 1"))
        await engine.dispose()

        services["Database"] = {"status": "online", "details": "PostgreSQL connected"}
    except Exception as e:
        services["Database"] = {
            "status": "offline",
            "details": f"Connection failed: {str(e)[:50]}",
        }

    # Check Redis
    try:
        from ...core.config import get_settings

        settings = get_settings()

        redis_client = redis.from_url(settings.redis_url)
        await redis_client.ping()
        await redis_client.close()

        services["Redis"] = {"status": "online", "details": "Cache and message broker"}
    except Exception as e:
        services["Redis"] = {
            "status": "offline",
            "details": f"Connection failed: {str(e)[:50]}",
        }

    return services


@app.command()
def start():
    """Start the hive system (API server and core services)"""
    asyncio.run(_start())


async def _start():
    """Start system services"""
    try:
        info_message("Starting LeanVibe Hive system...")

        # Check if services are already running
        services = await _check_all_services()

        # Start API server if not running
        if services.get("API", {}).get("status") != "online":
            info_message("Starting API server...")

            # Start API server in background
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "src.api.main:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "9001",
                    "--reload",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait a moment for startup
            await asyncio.sleep(3)

        # Check if it started successfully
        try:
            from ..utils import get_api_headers

            headers = get_api_headers()

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    "http://localhost:9001/api/v1/health", headers=headers
                )
                if response.status_code == 200:
                    success_message("API server started successfully!")
                else:
                    warning_message("API server may not have started properly")
        except Exception as e:
            warning_message(f"Could not verify API server startup: {e}")
        else:
            info_message("API server is already running")

        # Start orchestrator and default agents
        await _start_agents()

        success_message("Hive system startup complete!")

    except Exception as e:
        error_handler(e)


@app.command()
def stop():
    """Stop the hive system gracefully"""
    try:
        info_message("Stopping LeanVibe Hive system...")

        # Use the async stop function
        asyncio.run(_stop_services())

        success_message("Hive system stopped")

    except Exception as e:
        error_handler(e)


@app.command()
def restart():
    """Restart the hive system"""
    try:
        info_message("Restarting LeanVibe Hive system...")

        # Stop first
        asyncio.run(_stop_services())

        # Wait a moment
        time.sleep(Intervals.SYSTEM_RESTART_DELAY)

        # Start again
        asyncio.run(_start())

        success_message("Hive system restarted!")

    except Exception as e:
        error_handler(e)


async def _start_agents():
    """Internal async function to start orchestrator and agents"""
    # Start orchestrator and agent processes
    try:
        from ...core.orchestrator import orchestrator

        info_message("Starting agent orchestrator...")
        await orchestrator.start()
        success_message("Agent orchestrator started")
    except Exception as e:
        warning_message(f"Could not start orchestrator: {e}")

    # Initialize default agents if configured
    try:
        from ...core.config import settings

        if hasattr(settings, "AUTO_START_AGENTS") and settings.AUTO_START_AGENTS:
            info_message("Initializing default agents...")
            # Start meta agent by default
            await orchestrator.spawn_agent("meta", "meta-system")
            success_message("Default agents initialized")
    except Exception as e:
        warning_message(f"Could not initialize default agents: {e}")


async def _stop_services():
    """Internal async stop function"""
    try:
        # Stop API server
        subprocess.run(
            ["pkill", "-f", "uvicorn.*src.api.main:app"], capture_output=True, text=True
        )
    except Exception:
        pass

    # Stop agent processes and orchestrator
    try:
        from ...core.orchestrator import orchestrator

        await orchestrator.stop()
    except Exception:
        pass  # Orchestrator may not be running

    # Graceful shutdown of services
    try:
        # Stop message broker
        try:
            from ...core.message_broker import message_broker

            await message_broker.stop()
        except Exception:
            pass  # Message broker may not be running

        # Stop task queue
        try:
            from ...core.task_queue import task_queue

            await task_queue.stop()
        except Exception:
            pass  # Task queue may not be running
    except Exception:
        pass  # Services may not be running


@app.command()
def monitor(
    refresh_interval: int = typer.Option(
        5, "--interval", "-i", help="Refresh interval in seconds"
    ),
    show_detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed metrics"
    ),
):
    """Real-time system monitoring dashboard"""
    asyncio.run(_monitor_system(refresh_interval, show_detailed))


async def _monitor_system(refresh_interval: int, show_detailed: bool):
    """Internal async system monitoring"""
    try:
        from rich import box
        from rich.layout import Layout
        from rich.live import Live
        from rich.panel import Panel
        from rich.table import Table

        console.clear()
        info_message(
            f"Starting real-time system monitor (refresh every {refresh_interval}s)"
        )
        info_message("Press Ctrl+C to exit")

        def create_monitor_display():
            """Create the monitoring display layout"""
            layout = Layout()

            # Split into top and bottom sections
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="footer", size=3),
            )

            # Split main into left and right
            layout["main"].split_row(
                Layout(name="services", ratio=1), Layout(name="metrics", ratio=1)
            )

            return layout

        async def update_display():
            """Update the display with current data"""
            # Get service status
            services = await _check_all_services()

            # Create services table
            services_table = Table(title="🔧 System Services", box=box.ROUNDED)
            services_table.add_column("Service", style="cyan", no_wrap=True)
            services_table.add_column("Status", style="bold")
            services_table.add_column("Details", style="dim")

            for service_name, service_info in services.items():
                status_style = "green" if service_info["status"] == "online" else "red"
                status_icon = "🟢" if service_info["status"] == "online" else "🔴"
                services_table.add_row(
                    service_name,
                    f"{status_icon} {service_info['status']}",
                    service_info["details"],
                )

            # Get system metrics
            metrics_table = Table(title="📊 System Metrics", box=box.ROUNDED)
            metrics_table.add_column("Metric", style="cyan", no_wrap=True)
            metrics_table.add_column("Value", style="bold")

            try:
                from ..utils import get_api_headers

                headers = get_api_headers()

                # Get metrics from API
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        "http://localhost:9001/api/v1/cli/agents", headers=headers
                    )
                    if response.status_code == 200:
                        data = response.json()
                        agents = data.get("data", [])
                        active_agents = len(
                            [a for a in agents if a.get("status") == "active"]
                        )
                        metrics_table.add_row("Active Agents", str(active_agents))
                        metrics_table.add_row("Total Agents", str(len(agents)))

                    # Get tasks
                    response = await client.get(
                        "http://localhost:9001/api/v1/cli/tasks", headers=headers
                    )
                    if response.status_code == 200:
                        data = response.json()
                        tasks = data.get("data", [])
                        pending_tasks = len(
                            [t for t in tasks if t.get("status") == "pending"]
                        )
                        completed_tasks = len(
                            [t for t in tasks if t.get("status") == "completed"]
                        )
                        metrics_table.add_row("Pending Tasks", str(pending_tasks))
                        metrics_table.add_row("Completed Tasks", str(completed_tasks))
                        metrics_table.add_row("Total Tasks", str(len(tasks)))

            except Exception as e:
                metrics_table.add_row(
                    "Error", f"Failed to fetch metrics: {str(e)[:30]}"
                )

            # Create layout
            layout = create_monitor_display()

            # Header
            layout["header"].update(
                Panel(
                    f"[bold cyan]LeanVibe Hive System Monitor[/bold cyan] | "
                    f"Refresh: {refresh_interval}s | "
                    f"Time: {time.strftime('%H:%M:%S')}",
                    border_style="blue",
                )
            )

            # Services and metrics
            layout["services"].update(Panel(services_table, border_style="green"))
            layout["metrics"].update(Panel(metrics_table, border_style="yellow"))

            # Footer
            layout["footer"].update(
                Panel(
                    "[dim]Press Ctrl+C to exit | Use --detailed for more metrics[/dim]",
                    border_style="dim",
                )
            )

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


@app.command()
def logs():
    """Show system logs"""
    try:
        info_message("Showing system logs (press Ctrl+C to exit)...")

        # For now, show recent Python/uvicorn logs
        # In a production system, this would tail proper log files
        try:
            subprocess.run(
                [
                    "tail",
                    "-f",
                    "/dev/null",  # Placeholder - would be actual log file
                ]
            )
        except KeyboardInterrupt:
            console.print("\n👋 Log viewing stopped")

    except Exception as e:
        error_handler(e)


@app.command("init-db")
def init_db() -> None:
    """Initialize database tables"""
    console.print("[cyan]Initializing database...[/cyan]")

    try:
        from ...core.config import get_settings
        from ...core.models import get_database_manager

        settings = get_settings()
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


@app.command("start-api")
def start_api(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(9001, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Auto-reload on changes"),
) -> None:
    """Start the FastAPI server"""
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


@app.command("batch-config")
def batch_config(
    config_file: str = typer.Option(
        ..., "--file", "-f", help="JSON file with system configuration updates"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be updated without actually updating",
    ),
):
    """Apply batch configuration updates to the system"""
    asyncio.run(_batch_config_update(config_file, dry_run))


@app.command("batch-export")
def batch_export(
    types: str = typer.Option(
        "agents,tasks,metrics",
        "--type",
        "-t",
        help="Comma-separated types to export (agents,tasks,metrics,system)",
    ),
    output_file: str = typer.Option(
        None, "--output", "-o", help="Output file path (default: stdout)"
    ),
    format_type: str = typer.Option(
        "json", "--format", "-f", help="Output format: json, csv"
    ),
):
    """Export system data in batch"""
    asyncio.run(_batch_export_data(types, output_file, format_type))


async def _batch_config_update(config_file: str, dry_run: bool):
    """Internal async batch configuration update"""
    try:
        from pathlib import Path
        import json

        # Load configuration updates
        config_path = Path(config_file)
        if not config_path.exists():
            error_handler(Exception(f"Config file not found: {config_file}"))
            return

        try:
            with open(config_path) as f:
                config_updates = json.load(f)
            info_message(f"Loaded configuration updates from {config_file}")
        except json.JSONDecodeError as e:
            error_handler(Exception(f"Invalid JSON in {config_file}: {e}"))
            return
        except Exception as e:
            error_handler(Exception(f"Error reading {config_file}: {e}"))
            return

        # Dry run output
        if dry_run:
            console.print(
                "\n🔍 [bold cyan]Dry Run - System configuration updates:[/bold cyan]"
            )
            console.print(json.dumps(config_updates, indent=2))
            console.print("\n📊 Summary: Configuration would be applied to system")
            return

        # Apply configuration updates
        info_message("Applying system configuration updates...")

        # For now, we'll simulate the configuration update since the API may not support system config updates yet
        console.print("⚠️ System configuration update not yet implemented in API")
        console.print(f"Would apply: {json.dumps(config_updates, indent=2)}")

        success_message("System configuration update completed (simulated)")

        console.print(
            "\n💡 [dim]Note: System configuration update functionality requires API support[/dim]"
        )

    except Exception as e:
        error_handler(e)


async def _batch_export_data(types: str, output_file: str = None, format_type: str = "json"):
    """Internal async batch data export"""
    try:
        export_types = [t.strip() for t in types.split(",")]
        exported_data = {}

        # Export each requested type
        for export_type in export_types:
            if export_type not in ["agents", "tasks", "metrics", "system"]:
                warning_message(f"Unknown export type: {export_type}, skipping")
                continue

            info_message(f"Exporting {export_type} data...")

            try:
                if export_type == "agents":
                    # Get agents data
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.get("http://localhost:9001/api/v1/cli/agents")
                        if response.status_code == 200:
                            data = response.json()
                            exported_data["agents"] = data.get("data", [])

                elif export_type == "tasks":
                    # Get tasks data
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.get("http://localhost:9001/api/v1/cli/tasks")
                        if response.status_code == 200:
                            data = response.json()
                            exported_data["tasks"] = data.get("data", [])

                elif export_type == "metrics":
                    # Get system metrics
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.get("http://localhost:9001/api/v1/metrics/system")
                        if response.status_code == 200:
                            data = response.json()
                            exported_data["metrics"] = data.get("data", {})

                elif export_type == "system":
                    # Get system health
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.get("http://localhost:9001/api/v1/health")
                        if response.status_code == 200:
                            exported_data["system"] = response.json()

                success_message(f"✅ {export_type} data exported")

            except Exception as e:
                warning_message(f"Failed to export {export_type}: {e}")

        if not exported_data:
            error_handler(Exception("No data was successfully exported"))
            return

        # Output the data
        if format_type == "json":
            import json
            output_content = json.dumps(exported_data, indent=2)
        elif format_type == "csv":
            # For CSV, we'll create a simple summary
            output_content = f"Export Summary\nTypes: {', '.join(exported_data.keys())}\n"
            for key, value in exported_data.items():
                if isinstance(value, list):
                    output_content += f"{key}: {len(value)} items\n"
                elif isinstance(value, dict):
                    output_content += f"{key}: {len(value)} metrics\n"
                else:
                    output_content += f"{key}: {value}\n"
        else:
            error_handler(Exception(f"Unsupported format: {format_type}"))
            return

        if output_file:
            with open(output_file, "w") as f:
                f.write(output_content)
            success_message(f"Data exported to {output_file}")
        else:
            console.print(output_content)

        success_message(f"Batch export completed for types: {', '.join(exported_data.keys())}")

    except Exception as e:
        error_handler(e)


if __name__ == "__main__":
    app()
