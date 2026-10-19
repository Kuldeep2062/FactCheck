"""Bounded provider execution: shared concurrency cap, per-request timeout, one selective retry."""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class ProviderError(Exception):
    """kind: timeout | rate_limit | transient | auth | bad_request | invalid_response"""

    def __init__(self, kind: str, message: str, retryable: bool, retry_after_s: float | None = None):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.retry_after_s = retry_after_s
        self.attempts: list[dict] = []


@dataclass
class AttemptRecord:
    attempt: int
    outcome: str
    latency_ms: int
    error_kind: str | None = None


class RequestSpacer:
    """Spaces request starts to stay under a requests-per-minute quota."""

    def __init__(self, requests_per_minute: float | None, clock=time.monotonic, sleep=asyncio.sleep):
        self.interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()
        self._clock = clock
        self._sleep = sleep

    async def wait(self) -> None:
        if not self.interval:
            return
        async with self._lock:
            now = self._clock()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            await self._sleep(start - now)


class TokenBudget:
    """Token bucket for a tokens-per-minute quota; waits before a request instead of earning a 429."""

    def __init__(self, tokens_per_minute: float | None, clock=time.monotonic, sleep=asyncio.sleep):
        self.capacity = float(tokens_per_minute or 0)
        self.rate = self.capacity / 60.0
        self._tokens = self.capacity
        self._stamp: float | None = None
        self._lock = asyncio.Lock()
        self._clock = clock
        self._sleep = sleep
        self.waited_s = 0.0

    async def acquire(self, tokens: float) -> None:
        if not self.capacity or tokens <= 0:
            return
        need = min(tokens, self.capacity)
        async with self._lock:  # FIFO: one waiter at a time keeps large requests from starving
            while True:
                now = self._clock()
                if self._stamp is not None:
                    self._tokens = min(self.capacity, self._tokens + (now - self._stamp) * self.rate)
                self._stamp = now
                if self._tokens >= need:
                    self._tokens -= need
                    return
                wait = (need - self._tokens) / self.rate
                self.waited_s += wait
                await self._sleep(wait)


@dataclass
class ProviderGate:
    max_concurrent: int = 3
    timeout_s: float = 15.0
    max_attempts: int = 2
    retry_delay_s: float = 2.0
    retry_delay_cap_s: float = 10.0
    requests_per_minute: float | None = None
    tokens_per_minute: float | None = None
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    active: int = field(default=0, init=False)
    peak_active: int = field(default=0, init=False)
    total_attempts: int = field(default=0, init=False)

    def __post_init__(self):
        self._sem = asyncio.Semaphore(self.max_concurrent)
        self._spacer = RequestSpacer(self.requests_per_minute, sleep=self.sleep)
        self.budget = TokenBudget(self.tokens_per_minute, sleep=self.sleep)

    async def call(self, fn: Callable[[int], Awaitable[T]], estimated_tokens: float = 0
                   ) -> tuple[T, list[AttemptRecord]]:
        records: list[AttemptRecord] = []
        for attempt in range(1, self.max_attempts + 1):
            # The slot is held only while a request is in flight; backoff sleeps happen outside it.
            async with self._sem:
                # Quota waits happen before the request starts, so they never count toward the timeout.
                await self.budget.acquire(estimated_tokens)
                await self._spacer.wait()
                self.active += 1
                self.peak_active = max(self.peak_active, self.active)
                self.total_attempts += 1
                started = time.monotonic()
                try:
                    result = await asyncio.wait_for(fn(attempt), timeout=self.timeout_s)
                    records.append(AttemptRecord(attempt, "ok", _ms(started)))
                    return result, records
                except asyncio.TimeoutError:
                    err = ProviderError("timeout", f"provider request exceeded {self.timeout_s}s", retryable=True)
                except ProviderError as exc:
                    err = exc
                finally:
                    self.active -= 1
            records.append(AttemptRecord(attempt, "error", _ms(started), err.kind))
            if not err.retryable or attempt == self.max_attempts:
                err.attempts = [r.__dict__ for r in records]
                raise err
            delay = err.retry_after_s if err.retry_after_s is not None else self.retry_delay_s
            await self.sleep(min(max(delay, 0.0), self.retry_delay_cap_s))
        raise AssertionError("unreachable")


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
