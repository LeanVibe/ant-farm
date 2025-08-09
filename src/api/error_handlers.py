"""Comprehensive error handling for FastAPI application."""

import time
import traceback
import uuid
from typing import Any

import structlog
from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = structlog.get_logger()


class ErrorDetail(BaseModel):
    """Detailed error information."""

    message: str
    field: str | None = None
    code: str | None = None
    context: dict[str, Any] | None = None


class APIErrorResponse(BaseModel):
    """Standardized API error response."""

    success: bool = False
    error: str
    details: list[ErrorDetail] | None = None
    timestamp: float
    request_id: str
    path: str | None = None
    method: str | None = None


class APIError(HTTPException):
    """Base API error with enhanced details."""

    def __init__(
        self,
        status_code: int,
        message: str,
        details: list[ErrorDetail] | None = None,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.message = message
        self.details = details or []
        self.code = code
        super().__init__(
            status_code=status_code,
            detail=message,
            headers=headers,
        )


class ValidationAPIError(APIError):
    """Validation error with field-specific details."""

    def __init__(self, errors: list[dict[str, Any]]):
        details = []
        for error in errors:
            field_path = " -> ".join(str(loc) for loc in error.get("loc", []))
            details.append(
                ErrorDetail(
                    message=error.get("msg", "Validation error"),
                    field=field_path,
                    code=error.get("type"),
                    context={"input": error.get("input")},
                )
            )

        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="Request validation failed",
            details=details,
            code="VALIDATION_ERROR",
        )


class BusinessLogicError(APIError):
    """Business logic validation error."""

    def __init__(self, message: str, code: str = "BUSINESS_ERROR"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            message=message,
            code=code,
        )


class ResourceNotFoundError(APIError):
    """Resource not found error."""

    def __init__(self, resource_type: str, identifier: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            message=f"{resource_type} '{identifier}' not found",
            code="RESOURCE_NOT_FOUND",
        )


class ConflictError(APIError):
    """Resource conflict error."""

    def __init__(self, message: str, code: str = "CONFLICT"):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            message=message,
            code=code,
        )


class InternalServerError(APIError):
    """Internal server error with safe public messaging."""

    def __init__(
        self, message: str = "An internal error occurred", code: str = "INTERNAL_ERROR"
    ):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=message,
            code=code,
        )


def create_error_response(
    request: Request,
    status_code: int,
    message: str,
    details: list[ErrorDetail] | None = None,
    error_code: str | None = None,
) -> JSONResponse:
    """Create standardized error response."""

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    error_response = APIErrorResponse(
        error=message,
        details=details,
        timestamp=time.time(),
        request_id=request_id,
        path=str(request.url.path),
        method=request.method,
    )

    # Log error details
    logger.error(
        "API error occurred",
        status_code=status_code,
        error_message=message,
        error_code=error_code,
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        details=details,
    )

    return JSONResponse(
        status_code=status_code,
        content=error_response.dict(),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""

    validation_error = ValidationAPIError(exc.errors())

    return create_error_response(
        request=request,
        status_code=validation_error.status_code,
        message=validation_error.message,
        details=validation_error.details,
        error_code=validation_error.code,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions with enhanced error details."""

    # Check if it's our custom API error
    if isinstance(exc, APIError):
        return create_error_response(
            request=request,
            status_code=exc.status_code,
            message=exc.message,
            details=exc.details,
            error_code=exc.code,
        )

    # Handle standard HTTP exceptions
    error_messages = {
        400: "Bad request - invalid input data",
        401: "Authentication required",
        403: "Access forbidden - insufficient permissions",
        404: "Resource not found",
        405: "Method not allowed",
        409: "Conflict - resource already exists or invalid state",
        422: "Unprocessable entity - validation failed",
        429: "Too many requests - rate limit exceeded",
        500: "Internal server error",
        502: "Bad gateway - upstream service error",
        503: "Service unavailable - system temporarily down",
        504: "Gateway timeout - request took too long",
    }

    message = error_messages.get(exc.status_code, str(exc.detail))

    return create_error_response(
        request=request,
        status_code=exc.status_code,
        message=message,
        error_code=f"HTTP_{exc.status_code}",
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions safely."""

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # Log full exception details for debugging
    logger.error(
        "Unhandled exception occurred",
        exception_type=type(exc).__name__,
        exception_message=str(exc),
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        traceback=traceback.format_exc(),
    )

    # Return safe generic error message
    return create_error_response(
        request=request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message="An unexpected error occurred. Please try again later.",
        error_code="UNEXPECTED_ERROR",
    )


# Rate limiting helper
class RateLimitTracker:
    """Simple in-memory rate limiting tracker."""

    def __init__(self):
        self.requests: dict[str, list[float]] = {}

    def is_rate_limited(
        self, client_id: str, limit: int, window_seconds: int = 60
    ) -> bool:
        """Check if client has exceeded rate limit."""
        now = time.time()
        window_start = now - window_seconds

        # Get client's request times
        client_requests = self.requests.get(client_id, [])

        # Remove old requests outside the window
        client_requests = [
            req_time for req_time in client_requests if req_time > window_start
        ]

        # Check if limit exceeded
        if len(client_requests) >= limit:
            return True

        # Add current request and update
        client_requests.append(now)
        self.requests[client_id] = client_requests

        return False

    def get_remaining_requests(
        self, client_id: str, limit: int, window_seconds: int = 60
    ) -> int:
        """Get number of remaining requests for client."""
        now = time.time()
        window_start = now - window_seconds

        client_requests = self.requests.get(client_id, [])
        current_count = len([req for req in client_requests if req > window_start])

        return max(0, limit - current_count)


# Global rate limiter instance
rate_limiter = RateLimitTracker()


def validate_request_size(request: Request, max_size_mb: int = 10) -> None:
    """Validate request content size."""
    content_length = request.headers.get("content-length")
    if content_length:
        size_mb = int(content_length) / (1024 * 1024)
        if size_mb > max_size_mb:
            raise APIError(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                message=f"Request size {size_mb:.1f}MB exceeds limit of {max_size_mb}MB",
                code="REQUEST_TOO_LARGE",
            )


def validate_json_depth(data: Any, max_depth: int = 10, current_depth: int = 0) -> None:
    """Validate JSON nesting depth to prevent DoS attacks."""
    if current_depth > max_depth:
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            message=f"JSON nesting depth exceeds maximum of {max_depth}",
            code="JSON_TOO_DEEP",
        )

    if isinstance(data, dict):
        for value in data.values():
            validate_json_depth(value, max_depth, current_depth + 1)
    elif isinstance(data, list):
        for item in data:
            validate_json_depth(item, max_depth, current_depth + 1)
