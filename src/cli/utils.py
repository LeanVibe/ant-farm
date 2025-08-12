"""
Utility functions for the Hive CLI
"""

import os
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.table import Table

# Import CLI auth module
try:
    from .auth import get_authenticated_cli_user, get_current_auth_token

    CLI_AUTH_AVAILABLE = True
except ImportError:
    CLI_AUTH_AVAILABLE = False

console = Console()

# Updated API URL for new port
API_BASE_URL = "http://localhost:9001"


def get_api_headers() -> dict:
    """
    Get appropriate headers for API requests, including authentication if available.

    Returns:
        dict: Headers for API requests
    """
    headers = {"Content-Type": "application/json"}

    # Check for CLI auth token in environment
    token = os.getenv("HIVE_CLI_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif CLI_AUTH_AVAILABLE:
        # Check for token from auth module
        token = get_current_auth_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

    return headers


def error_handler(error: Exception) -> None:
    """Handle CLI errors with rich formatting"""
    console.print(f"[bold red]Error:[/bold red] {error}")
    sys.exit(1)


def success_message(message: str) -> None:
    """Print success message with formatting"""
    console.print(f"[bold green]✅ {message}[/bold green]")


def warning_message(message: str) -> None:
    """Print warning message with formatting"""
    console.print(f"[bold yellow]⚠️  {message}[/bold yellow]")


def info_message(message: str) -> None:
    """Print info message with formatting"""
    console.print(f"[bold blue]ℹ️  {message}[/bold blue]")


def create_status_table(title: str, data: list[dict[str, Any]]) -> Table:
    """Create a formatted table for status information"""
    table = Table(title=title, show_header=True, header_style="bold magenta")

    if not data:
        return table

    # Add columns based on first row keys
    for key in data[0].keys():
        table.add_column(key.replace("_", " ").title())

    # Add rows
    for row in data:
        values = []
        for value in row.values():
            if isinstance(value, bool):
                values.append("✅ Yes" if value else "❌ No")
            elif value is None:
                values.append("N/A")
            else:
                values.append(str(value))
        table.add_row(*values)

    return table


def create_system_status_table(services: dict[str, dict[str, Any]]) -> Table:
    """Create a system status table"""
    table = Table(
        title="🏠 LeanVibe Hive System Status",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Service", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    for service, info in services.items():
        status = info.get("status", "unknown")
        details = info.get("details", "")

        if status == "online":
            status_icon = "[bold green]✅ ONLINE[/bold green]"
        elif status == "offline":
            status_icon = "[bold red]❌ OFFLINE[/bold red]"
        else:
            status_icon = "[bold yellow]⚠️  UNKNOWN[/bold yellow]"

        table.add_row(service.title(), status_icon, details)

    return table


def spinner_task(message: str):
    """Context manager for showing a spinner during async operations"""
    return Status(message, console=console, spinner="dots")


async def run_with_spinner(coro, message: str):
    """Run async operation with spinner"""
    with spinner_task(message):
        return await coro


def print_panel(content: str, title: str = None, style: str = "blue") -> None:
    """Print content in a styled panel"""
    console.print(Panel(content, title=title, border_style=style))


def confirm_action(message: str) -> bool:
    """Ask for user confirmation"""
    response = console.input(f"[bold yellow]{message} (y/N):[/bold yellow] ")
    return response.lower() in ("y", "yes")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def format_bytes(bytes_count: int) -> str:
    """Format bytes in human-readable format"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f}{unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f}PB"
