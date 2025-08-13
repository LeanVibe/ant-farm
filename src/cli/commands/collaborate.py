"""
CLI commands for enhanced AI pair programming and collaboration.
"""

import asyncio
import json
from datetime import UTC, datetime

import typer
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from ..utils import error_handler, info_message, success_message, warning_message

app = typer.Typer()
console = Console()


@app.command()
def start_session(
    participants: list[str] = typer.Argument(
        ..., help="Agent IDs participating in the session"
    ),
    mode: str = typer.Option("driver_navigator", help="Collaboration mode"),
    task: str = typer.Argument(..., help="Task description"),
    project_context_file: str | None = typer.Option(
        None, help="JSON file with project context"
    ),
):
    """Start an enhanced AI pair programming session."""
    try:
        asyncio.run(
            _start_collaboration_session(participants, mode, task, project_context_file)
        )
    except Exception as e:
        error_handler(e)


async def _start_collaboration_session(
    participants: list[str], mode: str, task: str, project_context_file: str | None
):
    """Async implementation of starting collaboration session."""
    from ...core.collaboration import CollaborationMode, get_enhanced_pair_programming

    # Load project context if provided
    project_context = {}
    if project_context_file:
        try:
            with open(project_context_file) as f:
                project_context = json.load(f)
        except Exception as e:
            warning_message(f"Could not load project context: {e}")

    # Validate mode
    try:
        collaboration_mode = CollaborationMode(mode)
    except ValueError:
        valid_modes = [m.value for m in CollaborationMode]
        raise ValueError(
            f"Invalid mode '{mode}'. Valid modes: {', '.join(valid_modes)}"
        )

    enhanced_system = await get_enhanced_pair_programming()

    info_message("Starting enhanced collaboration session...")
    info_message(f"Participants: {', '.join(participants)}")
    info_message(f"Mode: {mode}")
    info_message(f"Task: {task}")

    session_id = await enhanced_system.start_enhanced_session(
        participants=participants,
        mode=collaboration_mode,
        project_context=project_context,
        task_description=task,
    )

    success_message("Enhanced collaboration session started!")
    info_message(f"Session ID: {session_id}")


@app.command()
def share_context(
    session_id: str = typer.Argument(..., help="Session ID"),
    source_agent: str = typer.Argument(..., help="Source agent ID"),
    context_type: str = typer.Argument(..., help="Type of context to share"),
    content_file: str = typer.Argument(..., help="JSON file with context content"),
    tags: list[str] = typer.Option([], help="Tags for the context"),
):
    """Share context between agents in a collaboration session."""
    try:
        asyncio.run(
            _share_collaboration_context(
                session_id, source_agent, context_type, content_file, tags
            )
        )
    except Exception as e:
        error_handler(e)


async def _share_collaboration_context(
    session_id: str,
    source_agent: str,
    context_type: str,
    content_file: str,
    tags: list[str],
):
    """Async implementation of sharing context."""
    from ...core.collaboration import ContextShareType, get_enhanced_pair_programming

    # Load content from file
    try:
        with open(content_file) as f:
            content = json.load(f)
    except Exception as e:
        raise ValueError(f"Could not load content file: {e}")

    # Validate context type
    try:
        context_type_enum = ContextShareType(context_type)
    except ValueError:
        valid_types = [t.value for t in ContextShareType]
        raise ValueError(
            f"Invalid context type '{context_type}'. Valid types: {', '.join(valid_types)}"
        )

    enhanced_system = await get_enhanced_pair_programming()

    info_message(f"Sharing context in session {session_id}")
    info_message(f"Source: {source_agent}")
    info_message(f"Type: {context_type}")

    success = await enhanced_system.share_context(
        session_id=session_id,
        source_agent=source_agent,
        context_type=context_type_enum,
        content=content,
        tags=set(tags),
    )

    if success:
        success_message("Context shared successfully!")
    else:
        warning_message("Failed to share context - session not found")


@app.command()
def get_context(
    session_id: str = typer.Argument(..., help="Session ID"),
    agent_id: str = typer.Argument(..., help="Requesting agent ID"),
    query: str = typer.Argument(..., help="Query for relevant context"),
    context_types: list[str] = typer.Option([], help="Filter by context types"),
    output_file: str | None = typer.Option(None, help="Save results to JSON file"),
):
    """Get relevant context for an agent."""
    try:
        asyncio.run(
            _get_relevant_context(
                session_id, agent_id, query, context_types, output_file
            )
        )
    except Exception as e:
        error_handler(e)


async def _get_relevant_context(
    session_id: str,
    agent_id: str,
    query: str,
    context_types: list[str],
    output_file: str | None,
):
    """Async implementation of getting relevant context."""
    from ...core.collaboration import ContextShareType, get_enhanced_pair_programming

    # Convert context types if provided
    parsed_context_types = None
    if context_types:
        try:
            parsed_context_types = [ContextShareType(ct) for ct in context_types]
        except ValueError as e:
            raise ValueError(f"Invalid context type: {e}")

    enhanced_system = await get_enhanced_pair_programming()

    info_message(f"Getting relevant context for {agent_id}")
    info_message(f"Query: {query}")

    contexts = await enhanced_system.get_relevant_context(
        session_id=session_id,
        requesting_agent=agent_id,
        query=query,
        context_types=parsed_context_types,
    )

    if not contexts:
        warning_message("No relevant context found")
        return

    # Display results
    context_table = Table(title=f"Relevant Context for: {query}")
    context_table.add_column("Type", style="cyan")
    context_table.add_column("Source", style="green")
    context_table.add_column("Relevance", style="yellow")
    context_table.add_column("Tags", style="magenta")
    context_table.add_column("Content Preview", style="white")

    results_data = []
    for context in contexts:
        content_preview = (
            str(context.content)[:50] + "..."
            if len(str(context.content)) > 50
            else str(context.content)
        )

        context_table.add_row(
            context.context_type.value,
            context.source_agent,
            f"{context.relevance_score:.2f}",
            ", ".join(context.tags),
            content_preview,
        )

        results_data.append(
            {
                "context_type": context.context_type.value,
                "source_agent": context.source_agent,
                "relevance_score": context.relevance_score,
                "tags": list(context.tags),
                "content": context.content,
                "timestamp": context.timestamp,
            }
        )

    console.print(context_table)

    # Save to file if requested
    if output_file:
        try:
            with open(output_file, "w") as f:
                json.dump(results_data, f, indent=2)
            success_message(f"Results saved to {output_file}")
        except Exception as e:
            warning_message(f"Could not save to file: {e}")


@app.command()
def suggest_patterns(
    session_id: str = typer.Argument(..., help="Session ID"),
    code_file: str = typer.Argument(..., help="File containing current code"),
    context: str = typer.Option("", help="Additional context description"),
):
    """Get code pattern suggestions for current work."""
    try:
        asyncio.run(_suggest_code_patterns(session_id, code_file, context))
    except Exception as e:
        error_handler(e)


async def _suggest_code_patterns(session_id: str, code_file: str, context: str):
    """Async implementation of getting code suggestions."""
    from ...core.collaboration import get_enhanced_pair_programming

    # Read code from file
    try:
        with open(code_file) as f:
            current_code = f.read()
    except Exception as e:
        raise ValueError(f"Could not read code file: {e}")

    enhanced_system = await get_enhanced_pair_programming()

    info_message(f"Getting code pattern suggestions for session {session_id}")

    suggestions = await enhanced_system.suggest_code_patterns(
        session_id=session_id, current_code=current_code, context=context
    )

    if not suggestions:
        warning_message("No pattern suggestions found")
        return

    # Display suggestions
    suggestions_table = Table(title="Code Pattern Suggestions")
    suggestions_table.add_column("Type", style="cyan")
    suggestions_table.add_column("Relevance", style="yellow")
    suggestions_table.add_column("Source", style="green")
    suggestions_table.add_column("Suggestion", style="white")
    suggestions_table.add_column("Tags", style="magenta")

    for suggestion in suggestions:
        suggestions_table.add_row(
            suggestion["type"],
            f"{suggestion['relevance']:.2f}",
            suggestion["source"],
            suggestion["content"],
            ", ".join(suggestion.get("tags", [])),
        )

    console.print(suggestions_table)


@app.command()
def switch_driver(
    session_id: str = typer.Argument(..., help="Session ID"),
    new_driver: str = typer.Argument(..., help="New driver agent ID"),
):
    """Switch the active driver in a collaboration session."""
    try:
        asyncio.run(_switch_collaboration_driver(session_id, new_driver))
    except Exception as e:
        error_handler(e)


async def _switch_collaboration_driver(session_id: str, new_driver: str):
    """Async implementation of switching driver."""
    from ...core.collaboration import get_enhanced_pair_programming

    enhanced_system = await get_enhanced_pair_programming()

    info_message(f"Switching driver to {new_driver} in session {session_id}")

    success = await enhanced_system.switch_driver(session_id, new_driver)

    if success:
        success_message(f"Driver switched to {new_driver}")
    else:
        warning_message("Failed to switch driver - session not found or invalid driver")


@app.command()
def metrics(
    session_id: str = typer.Argument(..., help="Session ID"),
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Watch metrics continuously"
    ),
):
    """Get collaboration metrics for a session."""
    try:
        if watch:
            asyncio.run(_watch_collaboration_metrics(session_id))
        else:
            asyncio.run(_get_collaboration_metrics(session_id))
    except Exception as e:
        error_handler(e)


async def _get_collaboration_metrics(session_id: str):
    """Get current collaboration metrics."""
    from ...core.collaboration import get_enhanced_pair_programming

    enhanced_system = await get_enhanced_pair_programming()
    metrics = await enhanced_system.get_collaboration_metrics(session_id)

    if "session_id" not in metrics:
        warning_message("Session not found or no metrics available")
        return

    # Display metrics
    metrics_table = Table(title=f"Collaboration Metrics - Session {session_id}")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")

    metrics_table.add_row("Participants", ", ".join(metrics["participants"]))
    metrics_table.add_row("Mode", metrics["mode"])
    metrics_table.add_row("Duration (minutes)", f"{metrics['duration_minutes']:.1f}")
    metrics_table.add_row("Context Exchanges", str(metrics["context_exchanges"]))
    metrics_table.add_row("Pattern Matches", str(metrics["pattern_matches"]))
    metrics_table.add_row("Suggestions Applied", str(metrics["suggestions_applied"]))
    metrics_table.add_row("Active Files", str(len(metrics["active_files"])))
    metrics_table.add_row("Total Edits", str(metrics["edit_count"]))

    console.print(metrics_table)


async def _watch_collaboration_metrics(session_id: str):
    """Watch collaboration metrics continuously."""
    from ...core.collaboration import get_enhanced_pair_programming
    from ...core.constants import Intervals

    enhanced_system = await get_enhanced_pair_programming()

    console.print(
        f"[bold green]Watching collaboration metrics for session {session_id}...[/bold green]"
    )
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    try:
        while True:
            console.clear()
            await _get_collaboration_metrics(session_id)
            await asyncio.sleep(Intervals.AUTONOMOUS_DASHBOARD_DEFAULT)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped watching collaboration metrics[/yellow]")


@app.command()
def live_session(
    session_id: str = typer.Argument(..., help="Session ID"),
    agent_id: str = typer.Argument(..., help="Your agent ID"),
):
    """Join a live collaboration session with real-time updates."""
    try:
        asyncio.run(_join_live_session(session_id, agent_id))
    except Exception as e:
        error_handler(e)


async def _join_live_session(session_id: str, agent_id: str):
    """Join and display live collaboration session."""
    from ...core.collaboration import get_enhanced_pair_programming
    from ...core.constants import Intervals

    enhanced_system = await get_enhanced_pair_programming()

    console.print(
        f"[bold green]Joining live collaboration session {session_id}...[/bold green]"
    )
    console.print(f"[dim]Your agent ID: {agent_id}[/dim]")
    console.print("[dim]Press Ctrl+C to leave[/dim]\n")

    # Create layout for live session
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )

    layout["main"].split_row(Layout(name="left"), Layout(name="right"))

    try:
        with Live(layout, refresh_per_second=2):
            while True:
                # Get current session metrics
                metrics = await enhanced_system.get_collaboration_metrics(session_id)

                if "session_id" not in metrics:
                    layout["header"].update(Panel("[red]Session not found[/red]"))
                    break

                # Update header
                layout["header"].update(
                    Panel(
                        f"Live Session: {session_id} | Mode: {metrics['mode']} | Duration: {metrics['duration_minutes']:.1f}m"
                    )
                )

                # Update left panel - participants and activity
                participants_text = "\n".join(
                    [f"• {p}" for p in metrics["participants"]]
                )
                layout["left"].update(
                    Panel(
                        f"Participants:\n{participants_text}\n\nActive Files: {len(metrics['active_files'])}",
                        title="Session Info",
                    )
                )

                # Update right panel - metrics
                metrics_text = f"""Context Exchanges: {metrics["context_exchanges"]}
Pattern Matches: {metrics["pattern_matches"]}
Suggestions Applied: {metrics["suggestions_applied"]}
Total Edits: {metrics["edit_count"]}"""
                layout["right"].update(Panel(metrics_text, title="Live Metrics"))

                # Update footer
                layout["footer"].update(
                    Panel(f"Last update: {datetime.now(UTC).strftime('%H:%M:%S')}")
                )

                await asyncio.sleep(Intervals.AUTONOMOUS_DASHBOARD_DEFAULT)

    except KeyboardInterrupt:
        console.print("\n[yellow]Left live collaboration session[/yellow]")


@app.command()
def end_session(session_id: str = typer.Argument(..., help="Session ID")):
    """End an enhanced collaboration session."""
    try:
        asyncio.run(_end_collaboration_session(session_id))
    except Exception as e:
        error_handler(e)


async def _end_collaboration_session(session_id: str):
    """End collaboration session and show summary."""
    from ...core.collaboration import get_enhanced_pair_programming

    enhanced_system = await get_enhanced_pair_programming()

    info_message(f"Ending collaboration session {session_id}")

    result = await enhanced_system.end_session(session_id)

    if result.success:
        success_message("Collaboration session ended successfully!")

        # Display summary
        console.print(Panel(result.collaboration_summary, title="Session Summary"))

        if result.metrics:
            metrics_table = Table(title="Final Metrics")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")

            for key, value in result.metrics.items():
                if isinstance(value, float):
                    metrics_table.add_row(key.replace("_", " ").title(), f"{value:.2f}")
                else:
                    metrics_table.add_row(key.replace("_", " ").title(), str(value))

            console.print(metrics_table)
    else:
        warning_message(f"Failed to end session: {result.error_message}")


if __name__ == "__main__":
    app()
