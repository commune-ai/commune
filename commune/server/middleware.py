
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
                auth_module = 'server.auth',
                tx_collector_module = 'server.txcollector',
                ):
        super().__init__(app)
        self.max_bytes = max_bytes
        self.max_requests = max_requests
        self.auth = c.module(auth_module)()
        self.tx_collector = c.module(tx_collector_module)()
        self.request_count = 0
        self.last_reset = time.time()
        
    async def dispatch(self, request: Request, call_next):
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_reset > 60:  # Reset counter every minute
            self.request_count = 0
            self.last_reset = current_time
            
        self.request_count += 1
        if self.request_count > self.max_requests:
            return JSONResponse(
                status_code=429, 
                content={"error": "Too many requests", "retry_after": 60 - (current_time - self.last_reset)}
            )
            
        # Size validation
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > self.max_bytes:
            return JSONResponse(status_code=413, content={"error": "Request too large"})
        
        # Verify authentication
        try:
            # Clone the request headers for verification
            headers = dict(request.headers)
            
            # For POST requests with JSON body, verify the payload matches the signature
            if request.method == "POST" and 'token' in headers:
                body = await request.body()
                if len(body) > self.max_bytes:
                    return JSONResponse(status_code=413, content={"error": "Request too large"})
                
                # Parse the body to verify it against the token
                try:
                    body_json = json.loads(body)
                    print(headers, body_json)
                    # Verify the token matches the request data
                    verified = self.auth.verify_headers(headers, data=body_json)
                    
                    # Store request metadata for transaction tracking
                    request_info = {
                        'client': verified,
                        'timestamp': time.time(),
                        'path': str(request.url.path),
                        'method': request.method,
                        'ip': request.client.host if request.client else None
                    }
                    
                    # Asynchronously log the transaction
                    asyncio.create_task(self.tx_collector.record_transaction(request_info))
                    
                except json.JSONDecodeError:
                    return JSONResponse(status_code=400, content={"error": "Invalid JSON in request body"})
                except Exception as e:
                    return JSONResponse(status_code=401, content={"error": f"Authentication failed: {str(e)}"})
        except Exception as e:
            # If verification fails but no token was provided, let the request through
            # The endpoint will handle authentication if needed
            pass
            
        # Continue processing the request
        response = await call_next(request)
        return response
