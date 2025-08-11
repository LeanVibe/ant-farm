#!/usr/bin/env python3
"""Startup script for API server with proper environment loading."""

import os
import sys
from pathlib import Path

# Set environment variables first
os.environ.setdefault("REDIS_URL", "redis://localhost:6381")
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://bogdan@localhost:5432/leanvibe_hive"
)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn

    print(f"Starting API server with Redis URL: {os.environ.get('REDIS_URL')}")

    uvicorn.run(
        "src.api.main:app", host="127.0.0.1", port=9001, reload=True, log_level="info"
    )
