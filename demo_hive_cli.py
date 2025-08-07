#!/usr/bin/env python3
"""
LeanVibe Hive CLI Demo Script

This script demonstrates the new modern CLI interface for managing
the LeanVibe Agent Hive system. The CLI provides kubectl/gh-style
commands for power users.
"""

import asyncio
import subprocess
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def run_command(cmd: str, description: str = None) -> str:
    """Run a command and display the result"""
    if description:
        console.print(f"\n🔧 [bold cyan]{description}[/bold cyan]")

    console.print(f"[dim]$ {cmd}[/dim]")

    try:
        result = subprocess.run(
            cmd.split(), capture_output=True, text=True, cwd=Path(__file__).parent
        )

        if result.stdout:
            console.print(result.stdout)
        if result.stderr and result.returncode != 0:
            console.print(f"[red]{result.stderr}[/red]")

        return result.stdout
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return ""


def demo_section(title: str, description: str):
    """Create a demo section header"""
    console.print(f"\n")
    console.print(
        Panel(
            f"[bold]{title}[/bold]\n{description}", border_style="blue", padding=(1, 2)
        )
    )


def main():
    """Run the complete CLI demo"""
    console.print(
        Panel(
            "[bold green]🏠 LeanVibe Hive CLI Demonstration[/bold green]\n\n"
            "This demo showcases the new modern CLI interface inspired by kubectl and gh.\n"
            "The CLI provides powerful, discoverable commands for managing the agent hive.",
            title="Welcome to Hive CLI",
            border_style="green",
            padding=(1, 2),
        )
    )

    # Phase 1: Basic CLI discovery
    demo_section(
        "Phase 1: CLI Discovery", "Explore the CLI structure and available commands"
    )

    run_command("python -m src.cli.main --help", "Show main CLI help")
    run_command("python -m src.cli.main --version", "Check CLI version")

    # Phase 2: System management
    demo_section(
        "Phase 2: System Management", "Check system health and manage core services"
    )

    run_command("python -m src.cli.main system --help", "System commands help")
    run_command("python -m src.cli.main system status", "Check system health")

    # Phase 3: Agent management
    demo_section("Phase 3: Agent Management", "Manage AI agents in the hive")

    run_command("python -m src.cli.main agent --help", "Agent commands help")
    run_command("python -m src.cli.main agent list", "List all agents")

    # Phase 4: Task management
    demo_section("Phase 4: Task Management", "Submit and monitor tasks")

    run_command("python -m src.cli.main task --help", "Task commands help")
    run_command("python -m src.cli.main task list", "List tasks")

    # Phase 5: Context engine
    demo_section("Phase 5: Context Engine", "Manage agent memory and context")

    run_command("python -m src.cli.main context --help", "Context commands help")

    # Phase 6: Makefile integration
    demo_section("Phase 6: Makefile Integration", "Use convenient make shortcuts")

    run_command("make status", "Quick status check via Makefile")

    # Conclusion
    console.print(f"\n")
    console.print(
        Panel(
            "[bold green]✅ CLI Demo Complete![/bold green]\n\n"
            "[bold]Key Features Demonstrated:[/bold]\n"
            "• kubectl/gh-style subcommands (system, agent, task, context)\n"
            "• Rich formatted output with tables and colors\n"
            "• Comprehensive help system at every level\n"
            "• Health checks with detailed service status\n"
            "• Makefile integration for quick access\n"
            "• Self-documenting interface\n\n"
            "[bold]Next Steps:[/bold]\n"
            "• Start API server: hive system start\n"
            "• Spawn your first agent: hive agent spawn meta\n"
            "• Submit a self-improvement task: hive task self-improvement\n"
            "• Monitor progress: hive task list",
            title="Demo Summary",
            border_style="green",
            padding=(1, 2),
        )
    )


if __name__ == "__main__":
    main()
