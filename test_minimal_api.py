#!/usr/bin/env python3
"""Minimal API test to isolate middleware issues."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Create minimal app
app = FastAPI(title="Test API")

@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "healthy"})

@app.get("/test")
async def test_endpoint():
    return JSONResponse(content={"message": "test"})

if __name__ == "__main__":
    import uvicorn
    print("Starting minimal API test...")
    uvicorn.run(app, host="127.0.0.1", port=9003, log_level="info")