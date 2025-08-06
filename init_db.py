#!/usr/bin/env python3
"""Initialize database tables for LeanVibe Agent Hive 2.0."""

from src.core.models import get_database_manager
from src.core.config import settings

def init_database():
    """Initialize database tables."""
    print("Initializing database tables...")
    
    db_manager = get_database_manager(settings.database_url)
    db_manager.create_tables()
    
    print("✓ Database tables created successfully!")

if __name__ == "__main__":
    init_database()