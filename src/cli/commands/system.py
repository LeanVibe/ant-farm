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
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:9001/api/v1/health")
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
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get("http://localhost:9001/api/v1/health")
                    if response.status_code == 200:
                        success_message("API server started successfully!")
                    else:
                        warning_message("API server may not have started properly")
            except Exception as e:
                warning_message(f"Could not verify API server startup: {e}")
        else:
            info_message("API server is already running")

        # TODO: Start agent processes
        # TODO: Initialize default agents

        success_message("Hive system startup complete!")

    except Exception as e:
        error_handler(e)


@app.command()
def stop():
    """Stop the hive system gracefully"""
    try:
        info_message("Stopping LeanVibe Hive system...")

        # Find and stop API server process
        try:
            result = subprocess.run(
                ["pkill", "-f", "uvicorn.*src.api.main:app"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                success_message("API server stopped")
            else:
                warning_message("No API server process found")
        except Exception as e:
            warning_message(f"Could not stop API server: {e}")

        # TODO: Stop agent processes
        # TODO: Graceful shutdown of services

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


async def _stop_services():
    """Internal async stop function"""
    try:
        subprocess.run(
            ["pkill", "-f", "uvicorn.*src.api.main:app"], capture_output=True, text=True
        )
    except Exception:
        pass


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


if __name__ == "__main__":
    app()
