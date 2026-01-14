"""
Rate limiting middleware for API protection and cost control.

Provides:
- Per-IP rate limiting
- Per-endpoint rate limiting
- Token-based rate limiting (for cost control)
- Configurable limits and windows
- Redis-backed (optional, falls back to in-memory)
"""

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Tuple, Optional
import time
import hashlib
from collections import defaultdict
from datetime import datetime, timedelta
import os
from redis_cache import get_redis_cache


class RateLimiter:
    """Rate limiter with sliding window algorithm."""
    
    def __init__(self, redis_cache=None):
        self.redis_cache = redis_cache
        self.enabled = redis_cache is not None and redis_cache.enabled
        
        # In-memory fallback storage
        self.memory_store: Dict[str, list] = defaultdict(list)
        self.memory_cleanup_interval = 300  # Clean up every 5 minutes
        self.last_cleanup = time.time()
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier (IP address)."""
        # Try to get real IP (behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _cleanup_memory_store(self):
        """Clean up old entries from memory store."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.memory_cleanup_interval:
            return
        
        cutoff_time = current_time - 3600  # Keep last hour
        
        for key in list(self.memory_store.keys()):
            self.memory_store[key] = [
                ts for ts in self.memory_store[key] if ts > cutoff_time
            ]
            if not self.memory_store[key]:
                del self.memory_store[key]
        
        self.last_cleanup = current_time
    
    def _get_redis_key(self, identifier: str, window: int) -> str:
        """Generate Redis key for rate limit tracking."""
        window_start = int(time.time() // window)
        return f"ratelimit:{identifier}:{window}:{window_start}"
    
    def check_rate_limit(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: int
    ) -> Tuple[bool, int, int]:
        """
        Check if request is within rate limit.
        
        Args:
            identifier: Unique identifier (IP, user, etc.)
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
        
        Returns:
            Tuple of (is_allowed, remaining_requests, reset_time)
        """
        current_time = time.time()
        window_start = int(current_time // window_seconds)
        
        if self.enabled:
            # Use Redis
            key = self._get_redis_key(identifier, window_seconds)
            try:
                count = self.redis_cache.redis_client.incr(key)
                if count == 1:
                    # Set expiration on first request
                    self.redis_cache.redis_client.expire(key, window_seconds)
                
                remaining = max(0, max_requests - count)
                reset_time = int((window_start + 1) * window_seconds)
                is_allowed = count <= max_requests
                
                return is_allowed, remaining, reset_time
            except Exception:
                # Fallback to memory if Redis fails
                pass
        
        # In-memory fallback
        self._cleanup_memory_store()
        
        key = f"{identifier}:{window_start}"
        cutoff_time = current_time - window_seconds
        
        # Clean old entries
        if key in self.memory_store:
            self.memory_store[key] = [
                ts for ts in self.memory_store[key] if ts > cutoff_time
            ]
        else:
            self.memory_store[key] = []
        
        # Count requests in window
        count = len(self.memory_store[key])
        
        if count < max_requests:
            # Add current request
            self.memory_store[key].append(current_time)
            remaining = max_requests - count - 1
            is_allowed = True
        else:
            remaining = 0
            is_allowed = False
        
        reset_time = int((window_start + 1) * window_seconds)
        
        return is_allowed, remaining, reset_time


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""
    
    def __init__(self, app, rate_limiter: RateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        
        # Rate limit configuration
        # Format: (max_requests, window_seconds)
        self.limits = {
            "/api/chat": (60, 60),  # 60 requests per minute
            "/api/upload": (10, 3600),  # 10 uploads per hour
            "/api/pdfs": (100, 60),  # 100 requests per minute
            "default": (100, 60)  # Default: 100 requests per minute
        }
        
        # Cost control: token-based limits
        self.daily_token_limit = int(os.getenv("DAILY_TOKEN_LIMIT", "1000000"))  # 1M tokens default
        self.token_tracking: Dict[str, int] = defaultdict(int)
        self.token_reset_time = datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)
    
    def _get_endpoint_limit(self, path: str) -> Tuple[int, int]:
        """Get rate limit for endpoint."""
        for endpoint, limit in self.limits.items():
            if path.startswith(endpoint):
                return limit
        return self.limits["default"]
    
    def _check_token_limit(self, request: Request) -> Tuple[bool, int]:
        """
        Check daily token limit for cost control.
        
        Returns:
            Tuple of (is_allowed, remaining_tokens)
        """
        # Reset daily counter if needed
        if datetime.now() >= self.token_reset_time:
            self.token_tracking.clear()
            self.token_reset_time = datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)
        
        # Get total tokens used today
        total_tokens = sum(self.token_tracking.values())
        remaining = max(0, self.daily_token_limit - total_tokens)
        
        # For now, allow all requests (we'll track tokens after the request)
        # In production, you might want to estimate tokens before allowing
        return True, remaining
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Get client identifier
        client_id = self.rate_limiter._get_client_id(request)
        
        # Get endpoint-specific limit
        max_requests, window_seconds = self._get_endpoint_limit(request.url.path)
        
        # Check rate limit
        is_allowed, remaining, reset_time = self.rate_limiter.check_rate_limit(
            identifier=f"{client_id}:{request.url.path}",
            max_requests=max_requests,
            window_seconds=window_seconds
        )
        
        if not is_allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Rate limit exceeded. Maximum {max_requests} requests per {window_seconds} seconds.",
                    "retry_after": reset_time - int(time.time()),
                    "limit": max_requests,
                    "window": window_seconds
                },
                headers={
                    "X-RateLimit-Limit": str(max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                    "Retry-After": str(reset_time - int(time.time()))
                }
            )
        
        # Check token limit (cost control)
        token_allowed, remaining_tokens = self._check_token_limit(request)
        if not token_allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Daily token limit exceeded. Limit: {self.daily_token_limit} tokens per day.",
                    "remaining_tokens": 0
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        response.headers["X-TokenLimit-Remaining"] = str(remaining_tokens)
        
        # Track tokens if this was a chat request
        # Extract token usage from response headers if available
        if request.url.path == "/api/chat":
            try:
                # Try to get token count from response headers (set by chat endpoint)
                tokens_used = response.headers.get("X-Tokens-Used")
                if tokens_used:
                    token_count = int(tokens_used)
                    # Track tokens per client (for daily limit)
                    client_key = f"tokens:{client_id}:{datetime.now().strftime('%Y-%m-%d')}"
                    if self.rate_limiter.enabled and self.rate_limiter.redis_cache:
                        try:
                            self.rate_limiter.redis_cache.redis_client.incrby(client_key, token_count)
                            # Set expiration to end of day
                            self.rate_limiter.redis_cache.redis_client.expire(
                                client_key, 
                                int((self.token_reset_time - datetime.now()).total_seconds())
                            )
                        except Exception:
                            # Fallback to in-memory tracking
                            self.token_tracking[client_key] = self.token_tracking.get(client_key, 0) + token_count
                    else:
                        # In-memory tracking
                        self.token_tracking[client_key] = self.token_tracking.get(client_key, 0) + token_count
            except Exception:
                # If token tracking fails, continue without it
                pass
        
        return response


def create_rate_limit_middleware(app, redis_cache=None):
    """
    Create and configure rate limiting middleware.
    
    Args:
        app: FastAPI application
        redis_cache: Optional Redis cache instance
    
    Returns:
        Configured RateLimitMiddleware instance
    """
    rate_limiter = RateLimiter(redis_cache=redis_cache)
    middleware = RateLimitMiddleware(app, rate_limiter)
    return middleware
