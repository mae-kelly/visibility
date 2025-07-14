from fastapi import HTTPException, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
import structlog
from typing import Optional
import time

logger = structlog.get_logger()

class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, secret_key: str = "development-secret-key"):
        super().__init__(app)
        self.secret_key = secret_key
        self.excluded_paths = {
            "/api/health",
            "/api/docs",
            "/api/redoc",
            "/api/openapi.json",
            "/favicon.ico"
        }
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Skip auth for development
        if request.headers.get("x-development-mode") == "true":
            request.state.user = {"sub": "dev-user", "name": "Development User"}
            return await call_next(request)
        
        # Extract token
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return self._unauthorized_response("Missing or invalid authorization header")
        
        token = auth_header.split(" ")[1]
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            request.state.user = payload
        except jwt.ExpiredSignatureError:
            return self._unauthorized_response("Token has expired")
        except jwt.InvalidTokenError:
            return self._unauthorized_response("Invalid token")
        
        return await call_next(request)
    
    def _unauthorized_response(self, detail: str):
        return Response(
            content=f'{{"detail": "{detail}"}}',
            status_code=401,
            headers={"content-type": "application/json"}
        )

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = None):
    """Dependency to get current user (for development)"""
    return {"sub": "dev-user", "name": "Development User", "roles": ["admin"]}
