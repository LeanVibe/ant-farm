"""
LeanVibe Hive CLI - Main entry point for the hive command
"""

import asyncio

import typer
from rich.console import Console

from .commands import agent, collaborate, context, monitor, project, system, task
from .utils import error_handler

# Try to import coordination - may not be available
try:
    from .coordination import app as coordination_app

    COORDINATION_AVAILABLE = True
except ImportError:
    COORDINATION_AVAILABLE = False

app = typer.Typer(
    name="hive",
    help="LeanVibe Agent Hive - Autonomous Multi-Agent System CLI",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

console = Console()

# Add command groups
app.add_typer(system.app, name="system", help="System management commands")
app.add_typer(agent.app, name="agent", help="Agent management commands")
app.add_typer(task.app, name="task", help="Task management commands")
app.add_typer(monitor.app, name="monitor", help="Real-time monitoring commands")
app.add_typer(context.app, name="context", help="Context engine commands")
app.add_typer(project.app, name="project", help="Large project coordination commands")
app.add_typer(
    collaborate.app, name="collaborate", help="Enhanced AI pair programming commands"
)

# Add coordination if available
if COORDINATION_AVAILABLE:
    app.add_typer(
        coordination_app,
        name="coordination",
        help="Agent coordination and collaboration",
    )
else:
    console.print(
        "[yellow]Warning: Coordination commands not available (missing dependencies)[/yellow]"
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
):
    """
    LeanVibe Agent Hive CLI

    A powerful command-line interface for managing the autonomous agent system.
    """
    if version:
        console.print("LeanVibe Hive CLI v2.0.0", style="bold green")
        raise typer.Exit()

    if debug:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    # If no command was provided, show help
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command()
def init():
    """Initialize the hive system (database, migrations, setup)"""

    async def _init():
        from ..core.config import get_settings

        get_settings()

        console.print("🚀 Initializing LeanVibe Hive...", style="bold blue")

        try:
            # Initialize database
            console.print("📊 Setting up database...", style="yellow")

            # Run database migrations
            import subprocess

            result = subprocess.run(
                ["alembic", "upgrade", "head"], capture_output=True, text=True, cwd="."
            )

            if result.returncode == 0:
                console.print("✅ Database migrations applied", style="green")
            else:
                console.print(
                    f"❌ Database migration failed: {result.stderr}", style="red"
                )
                return

            # Start Docker services if not running
            console.print("🐳 Checking Docker services...", style="yellow")
            result = subprocess.run(
                ["docker", "compose", "up", "-d", "postgres", "redis"],
                capture_output=True,
                text=True,
                cwd=".",
            )

            if result.returncode == 0:
                console.print("✅ Docker services started", style="green")
            else:
                console.print(
                    "⚠️  Docker services may already be running", style="yellow"
                )

            console.print("🎉 Hive initialization complete!", style="bold green")
            console.print("\n💡 [dim]Next steps:[/dim]")
            console.print("  • Run 'hive system start' to start the API server")
            console.print("  • Run 'hive system status' to check system health")
            console.print("  • Run 'hive agent spawn meta' to start your first agent")

        except Exception as e:
            console.print(f"❌ Initialization failed: {e}", style="red")

    asyncio.run(_init())


def cli_main():
    """Entry point for the CLI"""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!", style="yellow")
    except Exception as e:
        error_handler(e)


if __name__ == "__main__":
    cli_main()
