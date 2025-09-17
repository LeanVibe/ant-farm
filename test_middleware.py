#!/usr/bin/env python3
"""Test middleware implementation."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

# Test basic middleware
class TestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        return response

# Create a minimal app
app = FastAPI()

# Add basic middleware
app.add_middleware(TestMiddleware)

print("Basic middleware test passed")

# Now try importing our middleware
try:
    from src.api.middleware import SecurityHeadersMiddleware, RequestTimeoutMiddleware
    print("Middleware imports successful")

    # Try adding them
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestTimeoutMiddleware, timeout_seconds=30)
    print("Middleware addition successful")

except Exception as e:
    print(f"Middleware test failed: {e}")
    import traceback
    traceback.print_exc()

# Try importing the problematic ones
try:
    from src.core.auth import SecurityMiddleware
    from src.core.analytics import PerformanceMiddleware
    print("Core middleware imports successful")

    # Try adding them
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(PerformanceMiddleware)
    print("Core middleware addition successful")

except Exception as e:
    print(f"Core middleware test failed: {e}")
    import traceback
    traceback.print_exc()