#!/usr/bin/env python3
"""
Context population script for the Agent Hive system.

This script scans the src directory, reads all .py files, and uses the
ContextEngine to store their contents for the initial knowledge base.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Tuple

import structlog

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.context_engine import get_context_engine
from src.core.config import get_settings

logger = structlog.get_logger()


class ContextPopulator:
    """Populates the context engine with initial codebase knowledge."""

    def __init__(self, context_engine, base_path: Path):
        self.context_engine = context_engine
        self.base_path = base_path
        self.agent_id = "system-bootstrap"  # Special agent ID for bootstrap

    async def populate_codebase_context(self) -> dict:
        """Scan codebase and populate context engine."""
        stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "contexts_created": 0,
            "total_lines": 0,
            "errors": [],
        }

        logger.info(
            "Starting codebase context population", base_path=str(self.base_path)
        )

        # Get all Python files
        python_files = self._find_python_files()
        logger.info("Found Python files", count=len(python_files))

        for file_path in python_files:
            try:
                await self._process_file(file_path, stats)
                stats["files_processed"] += 1
            except Exception as e:
                logger.error(
                    "Failed to process file", file=str(file_path), error=str(e)
                )
                stats["errors"].append(f"{file_path}: {str(e)}")
                stats["files_skipped"] += 1

        logger.info("Context population completed", stats=stats)
        return stats

    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the src directory."""
        python_files = []

        # Scan src directory
        src_path = self.base_path / "src"
        if src_path.exists():
            for py_file in src_path.rglob("*.py"):
                # Skip __pycache__ and .pyc files
                if "__pycache__" not in str(py_file):
                    python_files.append(py_file)

        # Also scan important root files
        for root_file in ["bootstrap.py", "autonomous_bootstrap.py"]:
            root_path = self.base_path / root_file
            if root_path.exists():
                python_files.append(root_path)

        return sorted(python_files)

    async def _process_file(self, file_path: Path, stats: dict):
        """Process a single Python file and add to context."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                logger.debug("Skipping empty file", file=str(file_path))
                return

            # Calculate relative path from project root
            rel_path = file_path.relative_to(self.base_path)

            # Determine file category and importance
            category, importance = self._categorize_file(rel_path, content)

            # Extract topic from file path and content
            topic = self._extract_topic(rel_path, content)

            # Create metadata
            metadata = {
                "file_path": str(rel_path),
                "absolute_path": str(file_path),
                "file_size": len(content),
                "line_count": len(content.split("\n")),
                "file_type": "python_source",
                "populated_at": "bootstrap",
            }

            # Store in context engine
            context_id = await self.context_engine.store_context(
                agent_id=self.agent_id,
                content=content,
                content_type="code",
                importance_score=importance,
                category=category,
                topic=topic,
                metadata=metadata,
            )

            stats["contexts_created"] += 1
            stats["total_lines"] += metadata["line_count"]

            logger.debug(
                "Processed file",
                file=str(rel_path),
                context_id=context_id,
                category=category,
                importance=importance,
                lines=metadata["line_count"],
            )

        except Exception as e:
            logger.error("Error processing file", file=str(file_path), error=str(e))
            raise

    def _categorize_file(self, rel_path: Path, content: str) -> Tuple[str, float]:
        """Categorize file and assign importance score."""
        path_str = str(rel_path).lower()

        # Core system files - highest importance
        if any(
            core in path_str
            for core in [
                "core/",
                "agents/",
                "orchestrator",
                "task_queue",
                "context_engine",
            ]
        ):
            return "core", 0.9

        # API and web interface - high importance
        elif any(api in path_str for api in ["api/", "web/", "cli/"]):
            return "interface", 0.8

        # Models and database - high importance
        elif "models" in path_str or "database" in path_str:
            return "data", 0.8

        # Tests - medium importance
        elif "test" in path_str:
            return "test", 0.6

        # Configuration and scripts - medium importance
        elif any(config in path_str for config in ["config", "script", "bootstrap"]):
            return "config", 0.7

        # Documentation and examples - lower importance
        elif any(doc in path_str for doc in ["doc", "example", "demo"]):
            return "documentation", 0.5

        # Everything else - default importance
        else:
            return "general", 0.6

    def _extract_topic(self, rel_path: Path, content: str) -> str:
        """Extract topic from file path and content."""
        # Start with file name
        topic_parts = [rel_path.stem]

        # Add directory context
        if len(rel_path.parts) > 1:
            topic_parts.append(rel_path.parts[-2])  # Parent directory

        # Look for class/function definitions for additional context
        lines = content.split("\n")
        for line in lines[:20]:  # First 20 lines
            line = line.strip()
            if line.startswith("class "):
                class_name = line.split("class ")[1].split("(")[0].split(":")[0].strip()
                topic_parts.append(class_name)
                break
            elif line.startswith("def ") and "__" not in line:
                func_name = line.split("def ")[1].split("(")[0].strip()
                topic_parts.append(func_name)
                break

        return " ".join(topic_parts[:3])  # Limit to 3 parts


async def main():
    """Main entry point for context population."""
    # Initialize settings
    settings = get_settings()

    # Get project root directory
    project_root = Path(__file__).parent.parent

    try:
        # Initialize context engine
        logger.info("Initializing context engine")
        context_engine = await get_context_engine(settings.database_url)

        # Create populator
        populator = ContextPopulator(context_engine, project_root)

        # Populate context
        stats = await populator.populate_codebase_context()

        # Print summary
        print("\\n" + "=" * 60)
        print("CONTEXT POPULATION COMPLETED")
        print("=" * 60)
        print(f"Files processed: {stats['files_processed']}")
        print(f"Files skipped: {stats['files_skipped']}")
        print(f"Contexts created: {stats['contexts_created']}")
        print(f"Total lines processed: {stats['total_lines']}")

        if stats["errors"]:
            print(f"\\nErrors encountered: {len(stats['errors'])}")
            for error in stats["errors"][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(stats["errors"]) > 5:
                print(f"  ... and {len(stats['errors']) - 5} more")

        print("\\nInitial knowledge base ready for MetaAgent!")

    except Exception as e:
        logger.error("Context population failed", error=str(e))
        print(f"\\nERROR: Context population failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
