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
    API_BASE_URL,
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
    task_type: str = typer.Option(None, "--type", "-t", help="Filter by task type"),
    priority: str = typer.Option(
        None,
        "--priority",
        "-p",
        help="Filter by priority (critical, high, normal, low, background)",
    ),
    title_pattern: str = typer.Option(
        None, "--title", help="Filter by title pattern (partial match)"
    ),
    since: str = typer.Option(
        None, "--since", help="Show tasks created since (e.g., '1h', '1d', '1w')"
    ),
    sort_by: str = typer.Option(
        "created_at",
        "--sort",
        help="Sort by: title, type, status, priority, created_at, assigned_to",
    ),
    reverse: bool = typer.Option(False, "--reverse", "-r", help="Reverse sort order"),
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of results"),
):
    """List tasks with advanced filtering"""
    asyncio.run(
        _list_tasks_filtered(
            status,
            assigned_to,
            task_type,
            priority,
            title_pattern,
            since,
            sort_by,
            reverse,
            limit,
        )
    )


async def _list_tasks_filtered(
    status_filter=None,
    assigned_filter=None,
    task_type=None,
    priority=None,
    title_pattern=None,
    since=None,
    sort_by="created_at",
    reverse=False,
    limit=None,
):
    """Internal async task listing with advanced filtering"""
    try:
        from ..utils import get_api_headers

        headers = get_api_headers()
        params = {}
        if status_filter:
            params["status"] = status_filter
        if assigned_filter:
            params["assigned_to"] = assigned_filter

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v1/cli/tasks", params=params, headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    tasks = data.get("data", [])

                    if not tasks:
                        info_message("No tasks found")
                        return

                    # Apply additional filters
                    filtered_tasks = tasks

                    if task_type:
                        filtered_tasks = [
                            t
                            for t in filtered_tasks
                            if t.get("type", "").lower() == task_type.lower()
                        ]

                    if priority:
                        # Map priority names to values for comparison
                        priority_map = {
                            "critical": 1,
                            "high": 3,
                            "normal": 5,
                            "low": 7,
                            "background": 9,
                        }
                        if priority.lower() in priority_map:
                            target_priority = priority_map[priority.lower()]
                            filtered_tasks = [
                                t
                                for t in filtered_tasks
                                if t.get("priority") == target_priority
                                or str(t.get("priority", "")).lower()
                                == priority.lower()
                            ]

                    if title_pattern:
                        filtered_tasks = [
                            t
                            for t in filtered_tasks
                            if title_pattern.lower() in t.get("title", "").lower()
                        ]

                    if since:
                        # Parse since parameter (e.g., '1h', '1d', '1w')
                        import time

                        current_time = time.time()
                        since_seconds = _parse_time_delta(since)
                        if since_seconds:
                            cutoff_time = current_time - since_seconds
                            filtered_tasks = [
                                t
                                for t in filtered_tasks
                                if t.get("created_at", 0) >= cutoff_time
                            ]

                    # Sort tasks
                    sort_key_map = {
                        "title": lambda x: x.get("title", ""),
                        "type": lambda x: x.get("type", ""),
                        "status": lambda x: x.get("status", ""),
                        "priority": lambda x: x.get("priority", 5),
                        "created_at": lambda x: x.get("created_at", 0),
                        "assigned_to": lambda x: x.get("assigned_to", ""),
                    }

                    if sort_by in sort_key_map:
                        filtered_tasks.sort(key=sort_key_map[sort_by], reverse=reverse)

                    # Apply limit
                    if limit and limit > 0:
                        filtered_tasks = filtered_tasks[:limit]

                    if not filtered_tasks:
                        filter_desc = []
                        if status_filter:
                            filter_desc.append(f"status={status_filter}")
                        if assigned_filter:
                            filter_desc.append(f"assigned={assigned_filter}")
                        if task_type:
                            filter_desc.append(f"type={task_type}")
                        if priority:
                            filter_desc.append(f"priority={priority}")
                        if title_pattern:
                            filter_desc.append(f"title contains '{title_pattern}'")
                        if since:
                            filter_desc.append(f"since {since}")
                        filter_text = (
                            f" ({', '.join(filter_desc)})" if filter_desc else ""
                        )
                        info_message(f"No tasks found matching filters{filter_text}")
                        return

                    # Create table data
                    table_data = []
                    for task in filtered_tasks:
                        # Format priority
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

                    # Create title with filter info
                    filter_info = []
                    if status_filter:
                        filter_info.append(f"status={status_filter}")
                    if assigned_filter:
                        filter_info.append(f"assigned={assigned_filter}")
                    if task_type:
                        filter_info.append(f"type={task_type}")
                    if priority:
                        filter_info.append(f"priority={priority}")
                    if title_pattern:
                        filter_info.append(f"title=*{title_pattern}*")
                    if since:
                        filter_info.append(f"since {since}")
                    if limit:
                        filter_info.append(f"limit={limit}")

                    title = "📋 Tasks"
                    if filter_info:
                        title += f" ({', '.join(filter_info)})"

                    table = create_status_table(title, table_data)
                    console.print(table)

                    success_message(f"Found {len(filtered_tasks)} tasks")
                    if len(filtered_tasks) < len(tasks):
                        info_message(
                            f"Showing {len(filtered_tasks)} of {len(tasks)} total tasks"
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


def _parse_time_delta(time_str):
    """Parse time delta strings like '1h', '2d', '1w' into seconds"""
    if not time_str:
        return None

    time_str = time_str.lower().strip()

    # Extract number and unit
    import re

    match = re.match(r"^(\d+)([smhdw])$", time_str)
    if not match:
        return None

    value, unit = match.groups()
    value = int(value)

    multipliers = {
        "s": 1,  # seconds
        "m": 60,  # minutes
        "h": 3600,  # hours
        "d": 86400,  # days
        "w": 604800,  # weeks
    }

    return value * multipliers.get(unit, 0)


async def _list_tasks(status_filter: str = None, assigned_filter: str = None):
    """Internal async task listing (legacy function)"""
    await _list_tasks_filtered(status_filter, assigned_filter)


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
        from ..utils import get_api_headers

        headers = get_api_headers()
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

        # Map string priority to integer values for API
        priority_mapping = {
            "critical": 1,
            "high": 3,
            "normal": 5,
            "low": 7,
            "background": 9,
        }

        # Override with command line arguments
        if title:
            task_data["title"] = title
        if description:
            task_data["description"] = description
        if task_type != "general":
            task_data["type"] = task_type
        if priority != "normal":
            task_data["priority"] = priority_mapping.get(
                priority, 5
            )  # Convert to integer
        else:
            task_data["priority"] = 5  # Default normal priority
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
                f"{API_BASE_URL}/api/v1/cli/tasks", json=task_data, headers=headers
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
    task_identifier: str = typer.Argument(
        ..., help="Task short ID or UUID to show logs for"
    ),
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Follow logs in real-time"
    ),
):
    """Show logs for a specific task"""
    asyncio.run(_show_task_logs(task_identifier, follow))


async def _show_task_logs(task_identifier: str, follow: bool):
    """Internal async task log viewing"""
    try:
        from ..utils import get_api_headers

        headers = get_api_headers()

        # First, get task info
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v1/tasks/{task_identifier}", headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    task = data.get("data", {})

                    console.print(
                        f"\n📋 [bold cyan]Task: {task.get('title', 'Unknown')}[/bold cyan]"
                    )

                    # Show both short ID and UUID if available
                    short_id = task.get("short_id")
                    if short_id:
                        console.print(f"[bold green]Short ID: {short_id}[/bold green]")
                    console.print(f"UUID: {task.get('id', 'unknown')}")

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

                    # Real log file tailing implementation
                    if follow:
                        info_message("Following task logs (press Ctrl+C to exit)...")
                        await _follow_task_logs(task_identifier)

                    return  # Exit after showing logs

                else:
                    error_handler(Exception(data.get("error", "Unknown API error")))
            elif response.status_code == 404:
                error_handler(Exception(f"Task '{task_identifier}' not found"))
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
    task_identifier: str = typer.Argument(..., help="Task short ID or UUID to cancel"),
    force: bool = typer.Option(
        False, "--force", help="Force cancellation without confirmation"
    ),
):
    """Cancel a specific task"""
    asyncio.run(_cancel_task(task_identifier, force))


async def _cancel_task(task_identifier: str, force: bool):
    """Internal async task cancellation"""
    try:
        from ..utils import get_api_headers

        headers = get_api_headers()

        # Confirm cancellation unless forced
        if not force:
            if not confirm_action(f"Cancel task {task_identifier}?"):
                info_message("Cancellation aborted")
                return

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v1/tasks/{task_identifier}/cancel", headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    success_message(f"Task {task_identifier} cancelled successfully!")
                else:
                    error_handler(Exception(data.get("error", "Unknown API error")))
            elif response.status_code == 404:
                error_handler(Exception(f"Task '{task_identifier}' not found"))
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
        from ..utils import get_api_headers

        headers = get_api_headers()

        # Get title and description if not provided
        if not title:
            title = typer.prompt("Self-improvement task title")
        if not description:
            description = typer.prompt("What should the system improve?")

        # Submit the self-improvement task
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v1/tasks/self-improvement",
                params={"title": title, "description": description},
                headers=headers,
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


@app.command()
def search(
    query: str = typer.Argument(..., help="Search term (partial short ID or title)"),
):
    """Search for tasks by partial short ID or title"""
    asyncio.run(_search_tasks(query))


async def _search_tasks(query: str):
    """Internal async task search"""
    try:
        from ..utils import get_api_headers

        headers = get_api_headers()

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v1/search/tasks",
                params={"q": query},
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    tasks = data.get("data", {}).get("tasks", [])

                    if not tasks:
                        info_message(f"No tasks found matching '{query}'")
                        return

                    console.print(
                        f"\n🔍 [bold cyan]Search Results for '{query}'[/bold cyan]"
                    )

                    # Create table data
                    table_data = []
                    for task in tasks:
                        table_data.append(
                            {
                                "short_id": task.get("short_id", "n/a"),
                                "title": task.get("title", "untitled")[:30],
                                "type": task.get("type", "unknown"),
                                "status": task.get("status", "unknown"),
                                "priority": task.get("priority", "normal"),
                            }
                        )

                    table = create_status_table("📋 Matching Tasks", table_data)
                    console.print(table)

                    success_message(f"Found {len(tasks)} matching tasks")

                    if len(tasks) == 1:
                        task = tasks[0]
                        short_id = task.get("short_id")
                        task_id = task.get("id")
                        console.print(
                            f"\n💡 [dim]Tip: Use 'hive task logs {short_id or task_id}' for more details[/dim]"
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


async def _follow_task_logs(task_identifier: str):
    """Follow task logs in real-time by tailing log files."""
    try:
        # Look for log files related to the task
        log_dir = Path(".")

        # Common log file patterns for tasks
        log_patterns = [
            f"*{task_identifier}*.log",
            f"task-{task_identifier}.log",
            f"{task_identifier}.log",
            "*.log",  # Fallback to all log files
        ]

        log_file = None
        for pattern in log_patterns:
            matching_files = list(log_dir.glob(pattern))
            if matching_files:
                # Use the most recently modified log file
                log_file = max(matching_files, key=lambda f: f.stat().st_mtime)
                break

        if not log_file:
            # Create log directory structure if it doesn't exist
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            # Check agent log files
            agent_logs = list(log_dir.glob("*agent*.log"))
            if agent_logs:
                log_file = max(agent_logs, key=lambda f: f.stat().st_mtime)
            else:
                info_message("No log files found for this task")
                return

        console.print(f"📋 Tailing log file: {log_file}")

        # Follow the log file
        try:
            with open(log_file) as f:
                # Seek to end of file
                f.seek(0, 2)

                while True:
                    line = f.readline()
                    if line:
                        # Filter for relevant log entries
                        if task_identifier in line or "task" in line.lower():
                            console.print(line.strip())
                    else:
                        # Wait for new content
                        await asyncio.sleep(0.5)

        except KeyboardInterrupt:
            info_message("\nStopped following logs")
        except Exception as e:
            error_handler(Exception(f"Error reading log file: {e}"))

    except Exception as e:
        error_handler(Exception(f"Error setting up log following: {e}"))


@app.command()
def batch_submit(
    file: str = typer.Option(
        ..., "--file", "-f", help="JSON file with task definitions"
    ),
    assign_round_robin: bool = typer.Option(
        False, "--round-robin", help="Assign tasks in round-robin to available agents"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be submitted without actually submitting",
    ),
):
    """Submit multiple tasks from a JSON file"""
    asyncio.run(_batch_submit_tasks(file, assign_round_robin, dry_run))


async def _batch_submit_tasks(file: str, assign_round_robin: bool, dry_run: bool):
    """Internal async batch task submission"""
    try:
        from pathlib import Path

        file_path = Path(file)
        if not file_path.exists():
            error_handler(Exception(f"File not found: {file}"))
            return

        # Load task definitions
        try:
            with open(file_path) as f:
                task_definitions = json.load(f)
            info_message(f"Loaded {len(task_definitions)} task definitions from {file}")
        except json.JSONDecodeError as e:
            error_handler(Exception(f"Invalid JSON in {file}: {e}"))
            return
        except Exception as e:
            error_handler(Exception(f"Error reading {file}: {e}"))
            return

        # Validate task definitions
        valid_tasks = []
        for i, task_def in enumerate(task_definitions):
            if not isinstance(task_def, dict):
                warning_message(f"Task {i + 1}: Not a valid task object, skipping")
                continue

            # Ensure required fields
            if not task_def.get("title"):
                task_def["title"] = f"Batch Task {i + 1}"
            if not task_def.get("description"):
                task_def["description"] = "Auto-generated batch task"
            if not task_def.get("type"):
                task_def["type"] = "general"
            if not task_def.get("priority"):
                task_def["priority"] = 5  # Normal priority

            valid_tasks.append(task_def)

        if not valid_tasks:
            error_handler(Exception("No valid tasks found in file"))
            return

        # Get available agents for round-robin assignment
        available_agents = []
        if assign_round_robin:
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
                            available_agents = [
                                a.get("name")
                                for a in agents
                                if a.get("status") == "active"
                            ]
                            if available_agents:
                                info_message(
                                    f"Found {len(available_agents)} active agents for round-robin assignment"
                                )
                            else:
                                warning_message(
                                    "No active agents found, using meta-agent for all tasks"
                                )
                                available_agents = ["meta-agent"]
            except Exception as e:
                warning_message(
                    f"Could not get agent list: {e}. Using meta-agent for all tasks"
                )
                available_agents = ["meta-agent"]

        # Dry run output
        if dry_run:
            console.print(
                "\n🔍 [bold cyan]Dry Run - Tasks to be submitted:[/bold cyan]"
            )
            for i, task_def in enumerate(valid_tasks):
                agent_name = task_def.get("assigned_to", "meta-agent")
                if assign_round_robin and available_agents:
                    agent_name = available_agents[i % len(available_agents)]

                console.print(
                    f"  {i + 1}. {task_def['title']} ({task_def['type']}) → {agent_name}"
                )

            console.print(f"\n📊 Summary: {len(valid_tasks)} tasks would be submitted")
            return

        # Submit tasks
        submitted_tasks = []
        failed_tasks = []

        info_message(f"Submitting {len(valid_tasks)} tasks...")

        for i, task_def in enumerate(valid_tasks):
            try:
                # Assign agent for round-robin
                if assign_round_robin and available_agents:
                    task_def["assigned_to"] = available_agents[
                        i % len(available_agents)
                    ]
                elif not task_def.get("assigned_to"):
                    task_def["assigned_to"] = "meta-agent"

                console.print(
                    f"📋 Submitting: {task_def['title']} → {task_def['assigned_to']}"
                )

                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{API_BASE_URL}/api/v1/cli/tasks",
                        json=task_def,
                        headers=headers,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        if data.get("success"):
                            result = data.get("data", {})
                            submitted_tasks.append(
                                {
                                    "id": result.get("task_id", "unknown"),
                                    "title": task_def["title"],
                                    "assigned_to": task_def["assigned_to"],
                                }
                            )
                            console.print(
                                f"  ✅ Task submitted with ID: {result.get('task_id', 'unknown')}"
                            )
                        else:
                            failed_tasks.append(
                                {
                                    "title": task_def["title"],
                                    "error": data.get("error", "Unknown error"),
                                }
                            )
                            console.print(
                                f"  ❌ Failed: {data.get('error', 'Unknown error')}"
                            )
                    else:
                        failed_tasks.append(
                            {
                                "title": task_def["title"],
                                "error": f"HTTP {response.status_code}",
                            }
                        )
                        console.print(f"  ❌ Failed: HTTP {response.status_code}")

                # Small delay between submissions
                await asyncio.sleep(0.3)

            except Exception as e:
                failed_tasks.append({"title": task_def["title"], "error": str(e)})
                console.print(f"  ❌ Failed: {str(e)}")

        # Summary
        console.print("\n📊 Batch Task Submission Summary:")
        if submitted_tasks:
            success_message(f"✅ Successfully submitted {len(submitted_tasks)} tasks:")
            for task in submitted_tasks:
                console.print(
                    f"  • {task['title']} (ID: {task['id']}) → {task['assigned_to']}"
                )

        if failed_tasks:
            console.print(f"\n❌ Failed to submit {len(failed_tasks)} tasks:")
            for task in failed_tasks:
                console.print(f"  • {task['title']} - Error: {task['error']}")

        if submitted_tasks:
            console.print("\n💡 [dim]Use 'hive task list' to see all tasks[/dim]")

    except httpx.ConnectError:
        error_handler(
            Exception(
                "Cannot connect to API server. Is it running? Try: hive system start"
            )
        )
    except Exception as e:
        error_handler(e)


@app.command()
def batch_cancel(
    status_filter: str = typer.Option(
        None,
        "--status",
        "-s",
        help="Only cancel tasks with this status (pending, in_progress)",
    ),
    type_filter: str = typer.Option(
        None, "--type", "-t", help="Only cancel tasks of this type"
    ),
    force: bool = typer.Option(
        False, "--force", help="Force cancellation without confirmation"
    ),
):
    """Cancel multiple tasks in batch"""
    asyncio.run(_batch_cancel_tasks(status_filter, type_filter, force))


async def _batch_cancel_tasks(
    status_filter: str = None, type_filter: str = None, force: bool = False
):
    """Internal async batch task cancellation"""
    try:
        # Get current tasks
        params = {}
        if status_filter:
            params["status"] = status_filter

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v1/cli/tasks", params=params, headers=headers
            )

            if response.status_code != 200:
                error_handler(
                    Exception(f"Failed to get task list: HTTP {response.status_code}")
                )
                return

            data = response.json()
            if not data.get("success"):
                error_handler(Exception(data.get("error", "Failed to get task list")))
                return

            tasks = data.get("data", [])

            # Additional filtering
            if type_filter:
                tasks = [t for t in tasks if t.get("type") == type_filter]

            # Only allow cancellation of pending/in_progress tasks
            cancelable_tasks = [
                t for t in tasks if t.get("status") in ["pending", "in_progress"]
            ]

            if not cancelable_tasks:
                filter_desc = []
                if status_filter:
                    filter_desc.append(f"status={status_filter}")
                if type_filter:
                    filter_desc.append(f"type={type_filter}")
                filter_text = f" ({', '.join(filter_desc)})" if filter_desc else ""
                info_message(f"No cancelable tasks found{filter_text}")
                return

            # Confirm unless forced
            if not force:
                console.print("\n📋 Tasks to cancel:")
                for task in cancelable_tasks:
                    console.print(
                        f"  • {task.get('title', 'untitled')} ({task.get('type', 'unknown')}) - {task.get('status', 'unknown')}"
                    )

                if not typer.confirm(f"\nCancel {len(cancelable_tasks)} task(s)?"):
                    info_message("Batch cancellation aborted")
                    return

            # Cancel tasks
            cancelled_tasks = []
            failed_tasks = []

            info_message(f"Cancelling {len(cancelable_tasks)} task(s)...")

            for task in cancelable_tasks:
                task_id = task.get("id", "unknown")
                task_title = task.get("title", "untitled")
                try:
                    console.print(f"🛑 Cancelling: {task_title} ({task_id})")

                    response = await client.post(
                        f"{API_BASE_URL}/api/v1/tasks/{task_id}/cancel", headers=headers
                    )

                    if response.status_code == 200:
                        data = response.json()
                        if data.get("success"):
                            cancelled_tasks.append({"id": task_id, "title": task_title})
                            console.print(f"  ✅ {task_title} cancelled")
                        else:
                            failed_tasks.append(
                                {
                                    "id": task_id,
                                    "title": task_title,
                                    "error": data.get("error", "Unknown error"),
                                }
                            )
                            console.print(
                                f"  ❌ {task_title} failed: {data.get('error', 'Unknown error')}"
                            )
                    else:
                        failed_tasks.append(
                            {
                                "id": task_id,
                                "title": task_title,
                                "error": f"HTTP {response.status_code}",
                            }
                        )
                        console.print(
                            f"  ❌ {task_title} failed: HTTP {response.status_code}"
                        )

                except Exception as e:
                    failed_tasks.append(
                        {"id": task_id, "title": task_title, "error": str(e)}
                    )
                    console.print(f"  ❌ {task_title} failed: {str(e)}")

                # Small delay between cancellations
                await asyncio.sleep(0.2)

            # Summary
            console.print("\n📊 Batch Task Cancellation Summary:")
            if cancelled_tasks:
                success_message(
                    f"✅ Successfully cancelled {len(cancelled_tasks)} tasks:"
                )
                for task in cancelled_tasks:
                    console.print(f"  • {task['title']} (ID: {task['id']})")

            if failed_tasks:
                console.print(f"\n❌ Failed to cancel {len(failed_tasks)} tasks:")
                for task in failed_tasks:
                    console.print(
                        f"  • {task['title']} (ID: {task['id']}) - Error: {task['error']}"
                    )

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
