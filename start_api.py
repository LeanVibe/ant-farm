#!/usr/bin/env python3
"""Start the API server with proper imports."""

import uvicorn
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
