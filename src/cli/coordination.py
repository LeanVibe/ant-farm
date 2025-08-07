"""CLI commands for agent coordination system."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..core.agent_coordination import (
    coordination_system,
    CollaborationType,
    TaskPhase,
)
from ..core.message_broker import message_broker

app = typer.Typer(help="Agent coordination and collaboration commands")
console = Console()


@app.command()
def start_collaboration(
    title: str = typer.Argument(..., help="Title of the collaboration"),
    description: str = typer.Argument(..., help="Description of what needs to be done"),
    collaboration_type: str = typer.Option(
        "sequential",
        help="Type of collaboration (sequential, parallel, pipeline, consensus, competitive, delegation)",
    ),
    coordinator: str = typer.Option(
        "architect-01", help="Name of the coordinating agent"
    ),
    capabilities: str = typer.Option(
        "", help="Required capabilities (comma-separated)"
    ),
    deadline_hours: int = typer.Option(2, help="Deadline in hours from now"),
    priority: int = typer.Option(5, help="Priority level (1=highest, 9=lowest)"),
):
    """Start a new agent collaboration."""

    async def _start():
        try:
            # Parse capabilities
            required_capabilities = [
                cap.strip() for cap in capabilities.split(",") if cap.strip()
            ]

            # Calculate deadline
            deadline = datetime.now() + timedelta(hours=deadline_hours)

            # Initialize coordination system
            await coordination_system.initialize()

            # Start collaboration
            collaboration_id = await coordination_system.start_collaboration(
                title=title,
                description=description,
                collaboration_type=CollaborationType(collaboration_type),
                coordinator_agent=coordinator,
                required_capabilities=required_capabilities
                if required_capabilities
                else None,
                deadline=deadline,
                priority=priority,
                metadata={"initiated_by": "cli", "command": "start_collaboration"},
            )

            console.print(
                Panel(
                    f"✅ Collaboration started successfully!\n\n"
                    f"[bold]Collaboration ID:[/bold] {collaboration_id}\n"
                    f"[bold]Title:[/bold] {title}\n"
                    f"[bold]Type:[/bold] {collaboration_type}\n"
                    f"[bold]Coordinator:[/bold] {coordinator}\n"
                    f"[bold]Deadline:[/bold] {deadline.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"[bold]Priority:[/bold] {priority}",
                    title="Collaboration Started",
                    border_style="green",
                )
            )

        except Exception as e:
            console.print(f"[red]❌ Error starting collaboration: {str(e)}[/red]")

    asyncio.run(_start())


@app.command()
def list_collaborations():
    """List all active collaborations."""

    async def _list():
        try:
            # Initialize coordination system
            await coordination_system.initialize()

            active_collaborations = coordination_system.active_collaborations

            if not active_collaborations:
                console.print("[yellow]No active collaborations found.[/yellow]")
                return

            table = Table(title="Active Collaborations")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Title", style="magenta")
            table.add_column("Type", style="green")
            table.add_column("Phase", style="blue")
            table.add_column("Coordinator", style="yellow")
            table.add_column("Participants", style="red")
            table.add_column("Progress", style="bright_green")

            for collab_id, context in active_collaborations.items():
                progress = (
                    len(context.results) / len(context.sub_tasks)
                    if context.sub_tasks
                    else 0
                )
                progress_str = f"{progress:.1%}"

                table.add_row(
                    collab_id[:8] + "...",
                    context.title,
                    context.collaboration_type.value,
                    context.phase.value,
                    context.coordinator_agent,
                    str(len(context.participating_agents)),
                    progress_str,
                )

            console.print(table)

        except Exception as e:
            console.print(f"[red]❌ Error listing collaborations: {str(e)}[/red]")

    asyncio.run(_list())


@app.command()
def get_status(
    collaboration_id: str = typer.Argument(..., help="Collaboration ID to check"),
):
    """Get detailed status of a collaboration."""

    async def _status():
        try:
            # Initialize coordination system
            await coordination_system.initialize()

            status = await coordination_system.get_collaboration_status(
                collaboration_id
            )

            if "error" in status:
                console.print(f"[red]❌ {status['error']}[/red]")
                return

            # Create status display
            status_panel = Panel(
                f"[bold]Title:[/bold] {status['title']}\n"
                f"[bold]Phase:[/bold] {status['phase']}\n"
                f"[bold]Type:[/bold] {status['collaboration_type']}\n"
                f"[bold]Coordinator:[/bold] {status['coordinator']}\n"
                f"[bold]Progress:[/bold] {status['completed_tasks']}/{status['total_tasks']} ({status['progress']:.1%})\n"
                f"[bold]Priority:[/bold] {status['priority']}\n"
                f"[bold]Created:[/bold] {datetime.fromtimestamp(status['created_at']).strftime('%Y-%m-%d %H:%M:%S')}",
                title=f"Collaboration Status - {collaboration_id[:8]}...",
                border_style="blue",
            )

            console.print(status_panel)

            # Show participating agents
            if status["participating_agents"]:
                agents_table = Table(title="Participating Agents")
                agents_table.add_column("Agent Name", style="cyan")

                for agent in status["participating_agents"]:
                    agents_table.add_row(agent)

                console.print(agents_table)

        except Exception as e:
            console.print(f"[red]❌ Error getting status: {str(e)}[/red]")

    asyncio.run(_status())


@app.command()
def simulate_completion(
    collaboration_id: str = typer.Argument(..., help="Collaboration ID to simulate"),
    agent_name: str = typer.Option("test-agent", help="Name of agent completing task"),
):
    """Simulate task completion for testing (development only)."""

    async def _simulate():
        try:
            # Initialize message broker
            await message_broker.initialize()

            # Get collaboration context
            if collaboration_id not in coordination_system.active_collaborations:
                console.print(
                    f"[red]❌ Collaboration {collaboration_id} not found[/red]"
                )
                return

            context = coordination_system.active_collaborations[collaboration_id]
            incomplete_tasks = [
                task_id
                for task_id in context.sub_tasks.keys()
                if task_id not in context.results
            ]

            if not incomplete_tasks:
                console.print("[yellow]No incomplete tasks found[/yellow]")
                return

            # Complete first incomplete task
            task_id = incomplete_tasks[0]
            task = context.sub_tasks[task_id]

            await message_broker.send_message(
                from_agent=agent_name,
                to_agent="coordination_system",
                topic="sub_task_completed",
                payload={
                    "collaboration_id": collaboration_id,
                    "task_id": task_id,
                    "result": {
                        "success": True,
                        "output": f"Simulated completion of {task_id}",
                        "agent": agent_name,
                        "simulated": True,
                    },
                },
            )

            console.print(
                Panel(
                    f"✅ Simulated task completion\n\n"
                    f"[bold]Collaboration:[/bold] {collaboration_id[:8]}...\n"
                    f"[bold]Task:[/bold] {task_id}\n"
                    f"[bold]Agent:[/bold] {agent_name}\n"
                    f"[bold]Description:[/bold] {task.get('description', 'N/A')}",
                    title="Task Completion Simulated",
                    border_style="green",
                )
            )

        except Exception as e:
            console.print(f"[red]❌ Error simulating completion: {str(e)}[/red]")

    asyncio.run(_simulate())


@app.command()
def demo():
    """Run a demonstration of the coordination system."""

    async def _demo():
        try:
            console.print(
                Panel(
                    "🚀 Starting Agent Coordination Demo\n\n"
                    "This demo will:\n"
                    "1. Initialize the coordination system\n"
                    "2. Start a sample collaboration\n"
                    "3. Show the collaboration status\n"
                    "4. Simulate task completions\n"
                    "5. Show final results",
                    title="Agent Coordination Demo",
                    border_style="bright_blue",
                )
            )

            # Initialize systems
            console.print("[yellow]Initializing coordination system...[/yellow]")
            await coordination_system.initialize()
            await message_broker.initialize()

            # Start demo collaboration
            console.print("[yellow]Starting demo collaboration...[/yellow]")
            collaboration_id = await coordination_system.start_collaboration(
                title="Demo: Full Stack Development",
                description="Build a complete web application with frontend, backend, and database",
                collaboration_type=CollaborationType.SEQUENTIAL,
                coordinator_agent="architect-demo",
                required_capabilities=[
                    "system_design",
                    "code_generation",
                    "testing",
                    "deployment",
                ],
                deadline=datetime.now() + timedelta(hours=1),
                priority=3,
                metadata={"demo": True, "purpose": "demonstration"},
            )

            console.print(
                f"[green]✅ Demo collaboration started: {collaboration_id[:8]}...[/green]"
            )

            # Show initial status
            await asyncio.sleep(1)
            status = await coordination_system.get_collaboration_status(
                collaboration_id
            )
            console.print(
                f"[blue]📊 Initial status: {status['completed_tasks']}/{status['total_tasks']} tasks[/blue]"
            )

            # Simulate some task completions
            context = coordination_system.active_collaborations.get(collaboration_id)
            if context and context.sub_tasks:
                console.print("[yellow]Simulating task completions...[/yellow]")

                for i, (task_id, task) in enumerate(
                    list(context.sub_tasks.items())[:2]
                ):
                    await asyncio.sleep(1)

                    agent_name = task.get("assigned_agent", "demo-agent")
                    await message_broker.send_message(
                        from_agent=agent_name,
                        to_agent="coordination_system",
                        topic="sub_task_completed",
                        payload={
                            "collaboration_id": collaboration_id,
                            "task_id": task_id,
                            "result": {
                                "success": True,
                                "output": f"Demo completion of {task.get('description', task_id)}",
                                "agent": agent_name,
                                "demo": True,
                            },
                        },
                    )

                    console.print(
                        f"[green]✅ Completed task {i + 1}: {task.get('description', task_id)[:50]}...[/green]"
                    )

            # Show final status
            await asyncio.sleep(1)
            final_status = await coordination_system.get_collaboration_status(
                collaboration_id
            )

            if "error" not in final_status:
                console.print(
                    Panel(
                        f"🎉 Demo completed!\n\n"
                        f"[bold]Final Progress:[/bold] {final_status['completed_tasks']}/{final_status['total_tasks']} tasks\n"
                        f"[bold]Progress Percentage:[/bold] {final_status['progress']:.1%}\n"
                        f"[bold]Participating Agents:[/bold] {len(final_status['participating_agents'])}\n"
                        f"[bold]Phase:[/bold] {final_status['phase']}",
                        title="Demo Results",
                        border_style="green",
                    )
                )
            else:
                console.print(
                    "[yellow]Demo collaboration may have completed automatically[/yellow]"
                )

        except Exception as e:
            console.print(f"[red]❌ Demo error: {str(e)}[/red]")

    asyncio.run(_demo())


if __name__ == "__main__":
    app()
