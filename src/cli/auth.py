"""
CLI Authentication Module for LeanVibe Agent Hive

This module provides authentication functionality specifically for CLI operations,
handling both authenticated and anonymous access patterns.
"""

import os
from typing import Optional

from ..core.auth import get_cli_user, get_current_user
from ..core.security import User
from ..core.config import get_settings

# Global CLI user instance
_cli_user: Optional[User] = None
_auth_token: Optional[str] = None


def get_cli_auth_token() -> Optional[str]:
    """
    Get authentication token from environment or config.

    Returns:
        str: Authentication token if available, None otherwise
    """
    # Check environment variable first
    token = os.getenv("HIVE_CLI_TOKEN")
    if token:
        return token

    # Check config file (simplified - would read from ~/.hive/config in real implementation)
    try:
        settings = get_settings()
        # In a real implementation, this would read from a config file
        # For now, we'll just return None to fall back to CLI user
        return None
    except Exception:
        return None


def get_authenticated_cli_user() -> User:
    """
    Get authenticated CLI user, creating anonymous user if needed.

    This function provides a consistent interface for CLI operations
    that may or may not require authentication.

    Returns:
        User: Authenticated or anonymous CLI user
    """
    global _cli_user, _auth_token

    # If we already have a CLI user, return it
    if _cli_user is not None:
        return _cli_user

    # Try to get auth token
    token = get_cli_auth_token()

    if token:
        # Try to authenticate with token
        try:
            # In a real implementation, this would validate the token
            # For now, we'll create a user with appropriate permissions
            from ..core.security import User

            _cli_user = User(
                id="cli-authenticated",
                username="cli-user",
                email="cli@localhost",
                hashed_password="cli-authenticated-no-password",
                is_active=True,
                permissions=[
                    "agent_read",
                    "agent_spawn",
                    "task_read",
                    "task_create",
                    "system_read",
                    "context_read",
                    "message_read",
                ],
            )
            _auth_token = token
            return _cli_user
        except Exception:
            # Fall back to anonymous user if token validation fails
            pass

    # Create anonymous CLI user (this matches the API behavior)
    import asyncio

    # For CLI usage, we need to handle the async function properly
    # In a real implementation, this would be handled differently
    # For now, we'll create a mock user that matches the API behavior
    from ..core.security import User

    _cli_user = User(
        id="cli-anonymous",
        username="cli-user",
        email="cli@localhost",
        hashed_password="cli-anonymous-no-password",
        is_active=True,
        permissions=[
            "agent_read",
            "agent_spawn",
            "task_read",
            "task_create",
            "system_read",
            "context_read",
            "message_read",
        ],
    )
    return _cli_user


def clear_cli_auth_cache() -> None:
    """Clear cached CLI authentication to force refresh."""
    global _cli_user, _auth_token
    _cli_user = None
    _auth_token = None


def is_cli_authenticated() -> bool:
    """
    Check if CLI is currently authenticated with a real user account.

    Returns:
        bool: True if authenticated with real account, False if anonymous
    """
    if _cli_user is None:
        return False

    return _cli_user.id != "cli-anonymous"


def get_current_auth_token() -> Optional[str]:
    """
    Get the current authentication token being used.

    Returns:
        str: Current auth token, or None if using anonymous access
    """
    return _auth_token


# Convenience functions for common CLI auth patterns
def require_cli_auth():
    """
    Decorator to require CLI authentication for sensitive operations.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            user = get_authenticated_cli_user()
            if not is_cli_authenticated():
                from rich.console import Console

                console = Console()
                console.print(
                    "[bold red]❌ Authentication required for this operation[/bold red]"
                )
                console.print(
                    "Please set HIVE_CLI_TOKEN environment variable or log in."
                )
                console.print("Example: export HIVE_CLI_TOKEN=your_token_here")
                raise SystemExit(1)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def optional_cli_auth():
    """
    Decorator for operations that can work with or without authentication.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Ensure we have a user (anonymous or authenticated)
            get_authenticated_cli_user()
            return func(*args, **kwargs)

        return wrapper

    return decorator
