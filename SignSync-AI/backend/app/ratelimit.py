import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


class RateLimiter:
    """Tiny in-memory sliding-window limiter (per client IP). Use Redis for multi-worker setups."""

    def __init__(self, max_calls: int, period_seconds: int = 60):
        self.max_calls = max_calls
        self.period = period_seconds
        self.hits: dict[str, deque] = defaultdict(deque)

    def __call__(self, request: Request) -> None:
        key = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = self.hits[key]
        while window and now - window[0] > self.period:
            window.popleft()
        if len(window) >= self.max_calls:
            raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")
        window.append(now)
