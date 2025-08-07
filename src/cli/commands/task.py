"""
Task management commands for the Hive CLI
"""

import asyncio
import json
from pathlib import Path

import httpx
import typer
from rich.console import Console

from ..utils import (
    confirm_action,
    create_status_table,
    error_handler,
    info_message,
    success_message,
)

app = typer.Typer(help="Task management commands")
console = Console()


@app.command()
def list(
    status: str = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status (pending, in_progress, completed, failed)",
    ),
    assigned_to: str = typer.Option(
        None, "--assigned", "-a", help="Filter by assigned agent"
    ),
):
    """List tasks with optional filtering"""
    asyncio.run(_list_tasks(status, assigned_to))


async def _list_tasks(status_filter: str = None, assigned_filter: str = None):
    """Internal async task listing"""
    try:
        params = {}
        if status_filter:
            params["status"] = status_filter
        if assigned_filter:
            params["assigned_to"] = assigned_filter

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "http://localhost:8001/api/v1/tasks", params=params
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    tasks = data.get("data", [])

                    if not tasks:
                        info_message("No tasks found")
                        return

                    # Create table data
                    table_data = []
                    for task in tasks:
                        table_data.append(
                            {
                                "id": task.get("id", "unknown")[:8] + "...",
                                "title": task.get("title", "untitled")[:30],
                                "type": task.get("type", "unknown"),
                                "status": task.get("status", "unknown"),
                                "priority": task.get("priority", "normal"),
                                "assigned_to": task.get("assigned_to", "unassigned"),
                            }
                        )

                    filter_desc = []
                    if status_filter:
                        filter_desc.append(f"status={status_filter}")
                    if assigned_filter:
                        filter_desc.append(f"assigned={assigned_filter}")

                    title = "📋 Tasks"
                    if filter_desc:
                        title += f" ({', '.join(filter_desc)})"

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


@app.command()
def submit(
    type: str = typer.Option("general", "--type", "-t", help="Task type"),
    title: str = typer.Option(None, "--title", help="Task title"),
    description: str = typer.Option(
        None, "--description", "-d", help="Task description"
    ),
    file: str = typer.Option(
        None, "--file", "-f", help="JSON file with task definition"
    ),
    priority: str = typer.Option(
        "normal", "--priority", "-p", help="Task priority (low, normal, high, critical)"
    ),
    assigned_to: str = typer.Option(
        "meta-agent", "--assigned", "-a", help="Agent to assign the task to"
    ),
):
    """Submit a new task"""
    asyncio.run(_submit_task(type, title, description, file, priority, assigned_to))


async def _submit_task(
    task_type: str,
    title: str,
    description: str,
    file: str,
    priority: str,
    assigned_to: str,
):
    """Internal async task submission"""
    try:
        task_data = {}

        # Load from file if specified
        if file:
            file_path = Path(file)
            if not file_path.exists():
                error_handler(Exception(f"File not found: {file}"))
                return

            try:
                with open(file_path) as f:
                    task_data = json.load(f)
                    info_message(f"Loaded task from {file}")
            except json.JSONDecodeError as e:
                error_handler(Exception(f"Invalid JSON in {file}: {e}"))
                return

        # Override with command line arguments
        if title:
            task_data["title"] = title
        if description:
            task_data["description"] = description
        if task_type != "general":
            task_data["type"] = task_type
        if priority != "normal":
            task_data["priority"] = priority
        if assigned_to != "meta-agent":
            task_data["assigned_to"] = assigned_to

        # Validate required fields
        if not task_data.get("title"):
            task_data["title"] = typer.prompt("Task title")
        if not task_data.get("description"):
            task_data["description"] = typer.prompt("Task description")
        if not task_data.get("type"):
            task_data["type"] = task_type

        # Submit the task
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8001/api/v1/tasks", json=task_data
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    result = data.get("data", {})
                    task_id = result.get("task_id", "unknown")

                    success_message("Task submitted successfully!")
                    info_message(f"Task ID: {task_id}")
                    info_message(f"Title: {task_data['title']}")
                    info_message(
                        f"Assigned to: {task_data.get('assigned_to', 'meta-agent')}"
                    )

                    console.print(
                        f"\n💡 [dim]Use 'hive task logs {task_id}' to monitor progress[/dim]"
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
def logs(
    task_id: str = typer.Argument(..., help="Task ID to show logs for"),
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Follow logs in real-time"
    ),
):
    """Show logs for a specific task"""
    asyncio.run(_show_task_logs(task_id, follow))


async def _show_task_logs(task_id: str, follow: bool):
    """Internal async task log viewing"""
    try:
        # First, get task info
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"http://localhost:8001/api/v1/tasks/{task_id}")

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    task = data.get("data", {})

                    console.print(
                        f"\n📋 [bold cyan]Task: {task.get('title', 'Unknown')}[/bold cyan]"
                    )
                    console.print(f"ID: {task_id}")
                    console.print(f"Status: {task.get('status', 'unknown')}")
                    console.print(f"Type: {task.get('type', 'unknown')}")
                    console.print(
                        f"Assigned to: {task.get('assigned_to', 'unassigned')}"
                    )

                    if task.get("error"):
                        console.print(
                            f"\n❌ [bold red]Error:[/bold red] {task['error']}"
                        )

                    if task.get("result"):
                        console.print("\n✅ [bold green]Result:[/bold green]")
                        console.print(json.dumps(task["result"], indent=2))

                    # TODO: In a real implementation, this would tail actual log files
                    # For now, show task status
                    if follow:
                        info_message("Following task logs (press Ctrl+C to exit)...")
                        info_message(
                            "Note: Real-time log following not yet implemented"
                        )

                else:
                    error_handler(Exception(data.get("error", "Unknown API error")))
            elif response.status_code == 404:
                error_handler(Exception(f"Task '{task_id}' not found"))
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
def cancel(
    task_id: str = typer.Argument(..., help="Task ID to cancel"),
    force: bool = typer.Option(
        False, "--force", help="Force cancellation without confirmation"
    ),
):
    """Cancel a specific task"""
    asyncio.run(_cancel_task(task_id, force))


async def _cancel_task(task_id: str, force: bool):
    """Internal async task cancellation"""
    try:
        # Confirm cancellation unless forced
        if not force:
            if not confirm_action(f"Cancel task {task_id}?"):
                info_message("Cancellation aborted")
                return

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"http://localhost:8001/api/v1/tasks/{task_id}/cancel"
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    success_message(f"Task {task_id} cancelled successfully!")
                else:
                    error_handler(Exception(data.get("error", "Unknown API error")))
            elif response.status_code == 404:
                error_handler(Exception(f"Task '{task_id}' not found"))
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


@app.command("self-improvement")
def self_improvement(
    title: str = typer.Option(None, "--title", help="Self-improvement task title"),
    description: str = typer.Option(
        None, "--description", "-d", help="What should the system improve?"
    ),
):
    """Submit a self-improvement task to the MetaAgent"""
    asyncio.run(_submit_self_improvement(title, description))


async def _submit_self_improvement(title: str, description: str):
    """Internal async self-improvement task submission"""
    try:
        # Get title and description if not provided
        if not title:
            title = typer.prompt("Self-improvement task title")
        if not description:
            description = typer.prompt("What should the system improve?")

        # Submit the self-improvement task
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8001/api/v1/tasks/self-improvement",
                params={"title": title, "description": description},
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    result = data.get("data", {})
                    task_id = result.get("task_id", "unknown")

                    success_message("🤖 Self-improvement task submitted!")
                    info_message(f"Task ID: {task_id}")
                    info_message(f"Title: {title}")
                    info_message("The MetaAgent will process this automatically")

                    console.print(
                        f"\n💡 [dim]Use 'hive task logs {task_id}' to monitor progress[/dim]"
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


if __name__ == "__main__":
    app()
