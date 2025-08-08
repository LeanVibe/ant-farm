"""Request validation middleware for API security and data integrity."""

import json
import time
import uuid
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .error_handlers import (
    APIError,
    rate_limiter,
    validate_json_depth,
    validate_request_size,
)

logger = structlog.get_logger()


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request validation."""

    def __init__(
        self,
        app,
        max_request_size_mb: int = 10,
        max_json_depth: int = 10,
        rate_limit_requests: int = 100,
        rate_limit_window: int = 60,
        enable_request_logging: bool = True,
    ):
        super().__init__(app)
        self.max_request_size_mb = max_request_size_mb
        self.max_json_depth = max_json_depth
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window = rate_limit_window
        self.enable_request_logging = enable_request_logging

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with validation and monitoring."""

        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Add request ID to state for error handling
        request.state.request_id = request_id

        # Get client identifier for rate limiting
        client_ip = self._get_client_ip(request)
        client_id = f"{client_ip}:{request.url.path}"

        try:
            # Validate request size
            validate_request_size(request, self.max_request_size_mb)

            # Check rate limiting
            if rate_limiter.is_rate_limited(
                client_id, self.rate_limit_requests, self.rate_limit_window
            ):
                remaining = rate_limiter.get_remaining_requests(
                    client_id, self.rate_limit_requests, self.rate_limit_window
                )

                logger.warning(
                    "Rate limit exceeded",
                    client_ip=client_ip,
                    path=request.url.path,
                    remaining_requests=remaining,
                    request_id=request_id,
                )

                raise APIError(
                    status_code=429,
                    message="Rate limit exceeded. Please try again later.",
                    code="RATE_LIMIT_EXCEEDED",
                    headers={
                        "Retry-After": str(self.rate_limit_window),
                        "X-RateLimit-Remaining": str(remaining),
                    },
                )

            # Validate JSON depth for POST/PUT requests
            if request.method in ("POST", "PUT", "PATCH"):
                content_type = request.headers.get("content-type", "")
                if "application/json" in content_type:
                    try:
                        body = await request.body()
                        if body:
                            data = json.loads(body)
                            validate_json_depth(data, self.max_json_depth)

                            # Re-create request with body for downstream processing
                            from starlette.requests import Request as StarletteRequest

                            scope = dict(request.scope)

                            async def receive():
                                return {"type": "http.request", "body": body}

                            request = StarletteRequest(scope, receive)
                            request.state.request_id = request_id

                    except json.JSONDecodeError:
                        raise APIError(
                            status_code=400,
                            message="Invalid JSON format",
                            code="INVALID_JSON",
                        )

            # Log request if enabled
            if self.enable_request_logging:
                logger.info(
                    "API request received",
                    method=request.method,
                    path=request.url.path,
                    client_ip=client_ip,
                    user_agent=request.headers.get("user-agent"),
                    request_id=request_id,
                )

            # Process request
            response = await call_next(request)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{processing_time:.3f}s"

            # Add rate limit headers
            remaining = rate_limiter.get_remaining_requests(
                client_id, self.rate_limit_requests, self.rate_limit_window
            )
            response.headers["X-RateLimit-Limit"] = str(self.rate_limit_requests)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(
                int(time.time() + self.rate_limit_window)
            )

            # Log successful response
            if self.enable_request_logging:
                logger.info(
                    "API request completed",
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    processing_time=processing_time,
                    request_id=request_id,
                )

            return response

        except APIError:
            # Re-raise our custom API errors
            raise
        except Exception as e:
            # Log unexpected errors
            processing_time = time.time() - start_time
            logger.error(
                "Unexpected error in request validation middleware",
                error=str(e),
                method=request.method,
                path=request.url.path,
                processing_time=processing_time,
                request_id=request_id,
            )
            raise

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (reverse proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct connection
        if hasattr(request.client, "host"):
            return request.client.host

        return "unknown"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""

        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        # Only add HSTS in production
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        # Content Security Policy (basic)
        csp = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
        response.headers["Content-Security-Policy"] = csp

        return response


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request timeouts."""

    def __init__(self, app, timeout_seconds: int = 30):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with timeout."""

        import asyncio

        try:
            # Process request with timeout
            response = await asyncio.wait_for(
                call_next(request), timeout=self.timeout_seconds
            )
            return response

        except asyncio.TimeoutError:
            logger.warning(
                "Request timeout",
                path=request.url.path,
                method=request.method,
                timeout=self.timeout_seconds,
                request_id=getattr(request.state, "request_id", "unknown"),
            )

            raise APIError(
                status_code=504,
                message=f"Request timeout after {self.timeout_seconds} seconds",
                code="REQUEST_TIMEOUT",
            )
