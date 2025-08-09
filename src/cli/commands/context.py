"""
Context engine commands for the Hive CLI
"""

import asyncio
from pathlib import Path

import httpx
import typer
from rich.console import Console

from ..utils import (
    error_handler,
    info_message,
    success_message,
)

app = typer.Typer(help="Context engine and memory management commands")
console = Console()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query for semantic search"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results"),
):
    """Perform semantic search on the context engine"""
    asyncio.run(_search_context(query, limit))


async def _search_context(query: str, limit: int):
    """Internal async context search"""
    try:
        info_message(f"Searching for: '{query}'")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:9001/api/v1/context/meta-agent/search",
                params={"query": query, "limit": limit},
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    results = data.get("data", {}).get("results", [])

                    if results:
                        console.print(
                            f"\n🔍 [bold cyan]Found {len(results)} results:[/bold cyan]"
                        )
                        for i, result in enumerate(results, 1):
                            console.print(f"\n[bold]{i}. [/bold]", end="")
                            console.print(
                                f"[cyan]Score: {result.get('similarity_score', 0):.3f}[/cyan]"
                            )
                            console.print(
                                f"[dim]Category: {result.get('category', 'unknown')}[/dim]"
                            )
                            content = result.get("content", "")[:200]
                            console.print(
                                f"{content}{'...' if len(result.get('content', '')) > 200 else ''}"
                            )
                    else:
                        info_message("No results found")
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
def add(
    file: str = typer.Argument(..., help="File to add to the context engine"),
    agent_id: str = typer.Option(
        "meta-agent", "--agent", "-a", help="Agent to associate the context with"
    ),
):
    """Add a document to the context engine"""
    asyncio.run(_add_context(file, agent_id))


async def _add_context(file_path: str, agent_id: str):
    """Internal async context addition"""
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            error_handler(Exception(f"File not found: {file_path}"))
            return

        info_message(f"Adding {file_path} to context engine for agent {agent_id}")

        # Read file content
        content = file_path_obj.read_text(encoding="utf-8")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"http://localhost:9001/api/v1/context/{agent_id}/add",
                params={
                    "content": content,
                    "content_type": "file",
                    "category": file_path_obj.suffix.lstrip(".") or "text",
                    "importance_score": 0.7,
                    "topic": file_path_obj.stem,
                },
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    result = data.get("data", {})
                    success_message(f"Successfully added {file_path} to context engine")
                    console.print(f"Context ID: {result.get('context_id')}")
                    console.print(
                        f"Content length: {result.get('content_length')} characters"
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
def stats(
    agent_id: str = typer.Option(
        "meta-agent", "--agent", "-a", help="Agent to show stats for"
    ),
):
    """Show context engine statistics for an agent"""
    asyncio.run(_show_context_stats(agent_id))


async def _show_context_stats(agent_id: str):
    """Internal async context stats"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"http://localhost:9001/api/v1/context/{agent_id}"
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    stats = data.get("data", {})

                    console.print(
                        f"\n📊 [bold cyan]Context Stats for {agent_id}[/bold cyan]"
                    )
                    console.print(f"Total Contexts: {stats.get('total_contexts', 0)}")
                    console.print(
                        f"Storage Size: {stats.get('storage_size_mb', 0):.2f} MB"
                    )
                    console.print(
                        f"Oldest Context: {stats.get('oldest_context_age_days', 0):.1f} days"
                    )

                    contexts_by_importance = stats.get("contexts_by_importance", {})
                    if contexts_by_importance:
                        console.print("\n📈 [bold]By Importance:[/bold]")
                        for importance, count in contexts_by_importance.items():
                            console.print(f"  {importance}: {count}")

                    contexts_by_category = stats.get("contexts_by_category", {})
                    if contexts_by_category:
                        console.print("\n📂 [bold]By Category:[/bold]")
                        for category, count in contexts_by_category.items():
                            console.print(f"  {category}: {count}")

                    if stats.get("most_accessed_context_id"):
                        console.print(
                            f"\n🔥 Most Accessed: {stats['most_accessed_context_id']}"
                        )

                else:
                    error_handler(Exception(data.get("error", "Unknown API error")))
            elif response.status_code == 404:
                error_handler(Exception(f"Agent '{agent_id}' not found"))
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
def consolidate(
    agent_id: str = typer.Argument(..., help="Agent to consolidate memory for"),
):
    """Trigger memory consolidation for an agent"""
    asyncio.run(_consolidate_memory(agent_id))


async def _consolidate_memory(agent_id: str):
    """Internal async memory consolidation"""
    try:
        async with httpx.AsyncClient(
            timeout=60.0
        ) as client:  # Longer timeout for consolidation
            response = await client.post(
                f"http://localhost:9001/api/v1/context/{agent_id}/consolidate"
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    stats = data.get("data", {}).get("consolidation_stats", {})

                    success_message(f"Memory consolidation completed for {agent_id}")

                    if stats:
                        console.print("\n📊 [bold]Consolidation Results:[/bold]")
                        for key, value in stats.items():
                            console.print(f"  {key.replace('_', ' ').title()}: {value}")

                else:
                    error_handler(Exception(data.get("error", "Unknown API error")))
            elif response.status_code == 404:
                error_handler(Exception(f"Agent '{agent_id}' not found"))
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
