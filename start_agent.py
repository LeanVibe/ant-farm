#!/usr/bin/env python3
"""Entry point for starting agents with proper module imports."""

import sys
import asyncio
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from agents.runner import main

if __name__ == "__main__":
    asyncio.run(main())
