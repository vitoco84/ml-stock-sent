import requests
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        max_body_size = 10 * 1024 * 1024  # 10 MB
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_body_size:
            return JSONResponse(content={"detail": "Payload too large"}, status_code=413)
        return await call_next(request)

def _ollama_alive(url: str, timeout: float = 1.0) -> bool:
    try:
        requests.get(url, timeout=timeout)
        return True
    except Exception:
        return False

def to_dict(row):
    if hasattr(row, "model_dump"): return row.model_dump()
    if hasattr(row, "dict"): return row.dict()
    if isinstance(row, dict): return row
    raise TypeError(f"Unsupported type: {type(row)}")
