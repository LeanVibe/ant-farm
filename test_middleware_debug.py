#!/usr/bin/env python3
"""Debug script to isolate middleware stack building error."""

import sys
import os
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from fastapi import FastAPI
    from starlette.middleware.base import BaseHTTPMiddleware

    print("FastAPI and Starlette imports successful")

    # Test basic middleware creation
    class TestMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            return await call_next(request)

    print("TestMiddleware class created successfully")

    # Test FastAPI app creation
    app = FastAPI()
    print("FastAPI app created successfully")

    # Test adding middleware
    app.add_middleware(TestMiddleware)
    print("TestMiddleware added successfully")

    # Now test our actual middleware
    try:
        from src.core.auth import SecurityMiddleware
        print("SecurityMiddleware import successful")

        app2 = FastAPI()
        app2.add_middleware(SecurityMiddleware)
        print("SecurityMiddleware added successfully")

    except Exception as e:
        print(f"SecurityMiddleware error: {e}")
        import traceback
        traceback.print_exc()

    try:
        from src.core.analytics import PerformanceMiddleware
        print("PerformanceMiddleware import successful")

        app3 = FastAPI()
        app3.add_middleware(PerformanceMiddleware)
        print("PerformanceMiddleware added successfully")

    except Exception as e:
        print(f"PerformanceMiddleware error: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"General error: {e}")
    import traceback
    traceback.print_exc()