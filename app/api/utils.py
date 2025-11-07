from typing import Any

import requests
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    """Middleware that limits request body size to 10 MB."""

    async def dispatch(self, request: Request, call_next):
        max_body_size = 10 * 1024 * 1024
        content_length = request.headers.get("content-length")

        if content_length and int(content_length) > max_body_size:
            return JSONResponse(content={"detail": "Payload too large"}, status_code=413)

        return await call_next(request)

def _ollama_alive(url: str, timeout: float = 10.0) -> bool:
    """Check if Ollama service is alive."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return True
    except Exception:
        return False

def to_dict(row: Any) -> dict[str, Any]:
    """Convert Pydantic/BaseModel/dict to a plain dict."""
    if hasattr(row, "model_dump"):
        return row.model_dump()
    if hasattr(row, "dict"):
        return row.dict()
    if isinstance(row, dict):
        return row
    raise TypeError(f"Unsupported type for to_dict: {type(row)}")
