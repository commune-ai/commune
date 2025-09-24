
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
import json
import commune as c
import time
import asyncio

class Middleware(BaseHTTPMiddleware):
    def __init__(self, app, 
                max_bytes: int = 1000000, 
                max_requests: int = 1000,
                tempo = 60,
                **kwargs
                ):
        super().__init__(app)
        self.tempo = tempo
        self.max_bytes = max_bytes
        self.max_requests = max_requests
        self.request_count = 0
        self.last_reset = time.time()
        
    async def dispatch(self, request: Request, call_next):
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_reset > self.tempo:  # Reset counter every minute
            self.request_count = 0
            self.last_reset = current_time
        # Size validation
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > self.max_bytes:
            return JSONResponse(status_code=413, content={"error": "Request too large"})
        response = await call_next(request)
        return response
