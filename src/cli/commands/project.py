"""
CLI commands for large project coordination and management.
"""

import asyncio
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from ..utils import error_handler, info_message, success_message, warning_message

app = typer.Typer()
console = Console()


@app.command()
def create(
    name: str = typer.Argument(..., help="Project name"),
    description: str = typer.Argument(..., help="Project description"),
    scale: str = typer.Option(
        "medium", help="Project scale: small, medium, large, massive"
    ),
    lead_agent: str = typer.Option("meta-agent", help="Lead agent for the project"),
    root_path: str | None = typer.Option(
        None, help="Root path for project workspace"
    ),
):
    """Create a new large project workspace."""
    try:
        asyncio.run(_create_project(name, description, scale, lead_agent, root_path))
    except Exception as e:
        error_handler(e)


async def _create_project(
    name: str, description: str, scale: str, lead_agent: str, root_path: str | None
):
    """Async implementation of project creation."""
    from ...core.collaboration import ProjectScale, get_large_project_coordinator

    try:
        # Validate scale
        scale_enum = ProjectScale(scale.lower())
    except ValueError:
        raise ValueError(
            f"Invalid scale '{scale}'. Must be one of: small, medium, large, massive"
        )

    coordinator = await get_large_project_coordinator()

    info_message(f"Creating large project workspace: {name}")

    project_id = await coordinator.create_project_workspace(
        name=name,
        description=description,
        scale=scale_enum,
        lead_agent=lead_agent,
        root_path=root_path,
    )

    success_message("Project workspace created successfully!")
    info_message(f"Project ID: {project_id}")
    info_message(f"Scale: {scale}")
    info_message(f"Lead Agent: {lead_agent}")

    if root_path:
        info_message(f"Workspace Path: {root_path}")


@app.command()
def join(
    project_id: str = typer.Argument(..., help="Project ID to join"),
    agent_id: str = typer.Argument(..., help="Agent ID to add to project"),
    roles: list[str] = typer.Option(["contributor"], help="Agent roles in the project"),
):
    """Add an agent to a project workspace."""
    try:
        asyncio.run(_join_project(project_id, agent_id, roles))
    except Exception as e:
        error_handler(e)


async def _join_project(project_id: str, agent_id: str, roles: list[str]):
    """Async implementation of joining a project."""
    from ...core.collaboration import get_large_project_coordinator

    coordinator = await get_large_project_coordinator()

    info_message(f"Adding agent {agent_id} to project {project_id}")

    success = await coordinator.join_project(project_id, agent_id, roles)

    if success:
        success_message(f"Agent {agent_id} successfully joined project")
        info_message(f"Roles: {', '.join(roles)}")
    else:
        warning_message(f"Failed to join project {project_id}")


@app.command()
def decompose(
    project_id: str = typer.Argument(..., help="Project ID"),
    description: str = typer.Argument(..., help="Task description to decompose"),
    complexity: int = typer.Option(
        5, min=1, max=10, help="Estimated complexity (1-10)"
    ),
    target_agents: list[str] | None = typer.Option(
        None, help="Target agents for assignment"
    ),
):
    """Decompose a large task into coordinated sub-tasks."""
    try:
        asyncio.run(_decompose_task(project_id, description, complexity, target_agents))
    except Exception as e:
        error_handler(e)


async def _decompose_task(
    project_id: str,
    description: str,
    complexity: int,
    target_agents: list[str] | None,
):
    """Async implementation of task decomposition."""
    from ...core.collaboration import get_large_project_coordinator

    coordinator = await get_large_project_coordinator()

    info_message(f"Decomposing task for project {project_id}")
    info_message(f"Task: {description}")
    info_message(f"Complexity: {complexity}/10")

    result = await coordinator.decompose_large_task(
        project_id=project_id,
        task_description=description,
        estimated_complexity=complexity,
        target_agents=target_agents,
    )

    success_message("Task decomposition completed!")

    # Display results in a tree structure
    tree = Tree(f"[bold blue]Task: {description}[/bold blue]")
    tree.add(f"[green]Task ID: {result['task_id']}[/green]")

    subtasks_node = tree.add("[yellow]Sub-tasks:[/yellow]")
    for subtask in result["sub_tasks"]:
        subtask_node = subtasks_node.add(
            f"[cyan]{subtask['id']}[/cyan] ({subtask['type']})"
        )
        subtask_node.add(f"Agent Type: {subtask['agent_type']}")
        if "depends_on" in subtask:
            subtask_node.add(f"Dependencies: {', '.join(subtask['depends_on'])}")

    assignments_node = tree.add("[magenta]Assignments:[/magenta]")
    for task_id, agent_id in result["assignments"].items():
        assignments_node.add(f"{task_id} → {agent_id}")

    console.print(tree)


@app.command()
def status(
    project_id: str = typer.Argument(..., help="Project ID"),
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed status"
    ),
):
    """Get comprehensive status of a large project."""
    try:
        asyncio.run(_get_project_status(project_id, detailed))
    except Exception as e:
        error_handler(e)


async def _get_project_status(project_id: str, detailed: bool):
    """Async implementation of project status retrieval."""
    from ...core.collaboration import get_large_project_coordinator

    coordinator = await get_large_project_coordinator()

    status = await coordinator.get_project_status(project_id)

    workspace = status["workspace"]
    progress = status["progress"]

    # Basic project information
    info_table = Table(title=f"Project Status: {workspace['name']}")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")

    info_table.add_row("Project ID", workspace["id"])
    info_table.add_row("Description", workspace["description"])
    info_table.add_row("Scale", workspace["scale"].upper())
    info_table.add_row("Phase", workspace["phase"].upper())
    info_table.add_row("Lead Agent", workspace["lead_agent"])
    info_table.add_row(
        "Participating Agents", str(len(workspace["participating_agents"]))
    )

    console.print(info_table)

    # Progress information
    progress_table = Table(title="Progress Metrics")
    progress_table.add_column("Metric", style="cyan")
    progress_table.add_column("Value", style="green")

    progress_table.add_row("Completion", f"{progress['completion_percentage']:.1f}%")
    progress_table.add_row("Total Tasks", str(progress["total_tasks"]))
    progress_table.add_row("Completed Tasks", str(progress["completed_tasks"]))
    progress_table.add_row("Active Tasks", str(progress["active_tasks"]))
    progress_table.add_row(
        "Critical Path Progress", f"{progress['critical_path_completion']:.1f}%"
    )

    console.print(progress_table)

    if detailed:
        # Agent activity details
        if progress["agent_activity"]:
            agent_table = Table(title="Agent Activity")
            agent_table.add_column("Agent ID", style="cyan")
            agent_table.add_column("Active Tasks", style="green")
            agent_table.add_column("Roles", style="yellow")

            for agent_id, activity in progress["agent_activity"].items():
                roles = ", ".join(activity.get("roles", []))
                agent_table.add_row(agent_id, str(activity["active_tasks"]), roles)

            console.print(agent_table)

        # Resource utilization
        if status["resource_pools"]:
            resource_table = Table(title="Resource Utilization")
            resource_table.add_column("Resource Type", style="cyan")
            resource_table.add_column("Total", style="green")
            resource_table.add_column("Available", style="yellow")
            resource_table.add_column("Utilization", style="red")

            for resource_type, pool_info in status["resource_pools"].items():
                resource_table.add_row(
                    resource_type.upper(),
                    f"{pool_info['total_capacity']:.1f}",
                    f"{pool_info['available_capacity']:.1f}",
                    f"{pool_info['utilization_percentage']:.1f}%",
                )

            console.print(resource_table)


@app.command()
def progress(
    project_id: str = typer.Argument(..., help="Project ID"),
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Watch progress continuously"
    ),
):
    """Monitor project progress in real-time."""
    try:
        if watch:
            asyncio.run(_watch_project_progress(project_id))
        else:
            asyncio.run(_show_project_progress(project_id))
    except Exception as e:
        error_handler(e)


async def _show_project_progress(project_id: str):
    """Show current project progress."""
    from ...core.collaboration import get_large_project_coordinator

    coordinator = await get_large_project_coordinator()
    progress = await coordinator.monitor_project_progress(project_id)

    # Create a progress bar visualization
    completion = progress["completion_percentage"]
    total_width = 50
    filled_width = int(completion / 100 * total_width)
    bar = "█" * filled_width + "░" * (total_width - filled_width)

    console.print("\n[bold]Project Progress:[/bold]")
    console.print(f"[cyan]{bar}[/cyan] {completion:.1f}%")
    console.print(f"Tasks: {progress['completed_tasks']}/{progress['total_tasks']}")
    console.print(f"Active: {progress['active_tasks']}")
    console.print(f"Critical Path: {progress['critical_path_completion']:.1f}%")


async def _watch_project_progress(project_id: str):
    """Watch project progress continuously."""
    from ...core.collaboration import get_large_project_coordinator
    from ...core.constants import Intervals

    coordinator = await get_large_project_coordinator()

    console.print(f"[bold green]Watching project {project_id} progress...[/bold green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    try:
        while True:
            console.clear()
            await _show_project_progress(project_id)
            await asyncio.sleep(Intervals.AUTONOMOUS_DASHBOARD_DEFAULT)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped watching project progress[/yellow]")


@app.command()
def resolve_conflict(
    project_id: str = typer.Argument(..., help="Project ID"),
    conflict_type: str = typer.Argument(..., help="Type of conflict"),
    agents: list[str] = typer.Argument(..., help="Involved agent IDs"),
    context_file: str | None = typer.Option(
        None, help="JSON file with conflict context"
    ),
):
    """Handle conflicts in large project coordination."""
    try:
        asyncio.run(
            _resolve_project_conflict(project_id, conflict_type, agents, context_file)
        )
    except Exception as e:
        error_handler(e)


async def _resolve_project_conflict(
    project_id: str, conflict_type: str, agents: list[str], context_file: str | None
):
    """Async implementation of conflict resolution."""
    from ...core.collaboration import get_large_project_coordinator

    coordinator = await get_large_project_coordinator()

    # Load context from file if provided
    context = {}
    if context_file:
        try:
            with open(context_file) as f:
                context = json.load(f)
        except Exception as e:
            warning_message(f"Could not load context file: {e}")

    info_message(f"Resolving conflict in project {project_id}")
    info_message(f"Conflict Type: {conflict_type}")
    info_message(f"Involved Agents: {', '.join(agents)}")

    result = await coordinator.handle_conflict_resolution(
        project_id=project_id,
        conflict_type=conflict_type,
        involved_agents=agents,
        context=context,
    )

    success_message("Conflict resolution initiated!")
    info_message(f"Conflict ID: {result['conflict_id']}")
    info_message(f"Resolution Strategy: {result['resolution_strategy']}")
    info_message(f"Action Taken: {result['action']}")


@app.command()
def list_projects():
    """List all active large projects."""
    try:
        asyncio.run(_list_projects())
    except Exception as e:
        error_handler(e)


async def _list_projects():
    """Async implementation of project listing."""
    from ...core.collaboration import get_large_project_coordinator

    coordinator = await get_large_project_coordinator()

    if not coordinator.active_projects:
        info_message("No active large projects found")
        return

    projects_table = Table(title="Active Large Projects")
    projects_table.add_column("Project ID", style="cyan")
    projects_table.add_column("Name", style="green")
    projects_table.add_column("Scale", style="yellow")
    projects_table.add_column("Phase", style="magenta")
    projects_table.add_column("Lead Agent", style="blue")
    projects_table.add_column("Agents", style="red")

    for project_id, workspace in coordinator.active_projects.items():
        projects_table.add_row(
            project_id,
            workspace.name,
            workspace.scale.value.upper(),
            workspace.phase.value.upper(),
            workspace.lead_agent,
            str(len(workspace.participating_agents)),
        )

    console.print(projects_table)


if __name__ == "__main__":
    app()
