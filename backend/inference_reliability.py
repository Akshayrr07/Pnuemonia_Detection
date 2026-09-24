"""Small, process-local primitives for reliable hosted inference.

The API has no distributed queue or state store.  The guard below therefore
uses only a lock and a monotonic clock, which is appropriate for a single
local/container process and avoids turning the health endpoint into a
backend-dependent operation.
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Union


class InferenceCapacityError(RuntimeError):
    """Raised when this process cannot accept more inference immediately."""


class InferenceLease:
    """A capacity slot that can only be released once."""

    def __init__(self, guard: "ConcurrencyRateGuard"):
        self._guard = guard
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        with self._guard._lock:
            if self._released:
                return
            self._released = True
            if self._guard._active > 0:
                self._guard._active -= 1

    def __enter__(self) -> "InferenceLease":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    def __bool__(self) -> bool:
        return not self._released


class ConcurrencyRateGuard:
    """Bound active work and optionally bound accepted requests per second.

    ``acquire`` is deliberately non-blocking.  The route can therefore return
    a controlled 503 rather than piling requests onto an unbounded executor
    queue.  The lease must remain held until the worker has actually stopped;
    an HTTP deadline cannot cancel a synchronous provider call.
    """

    def __init__(
        self,
        max_concurrency: int = 4,
        max_requests_per_second: Union[int, float] = 0,
    ) -> None:
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_requests_per_second = max(0.0, float(max_requests_per_second))
        if (
            self.max_requests_per_second > 0
            and not self.max_requests_per_second.is_integer()
        ):
            raise ValueError("max_requests_per_second must be a whole number")
        self._max_request_tokens = int(self.max_requests_per_second)
        self._active = 0
        self._request_times: Deque[float] = deque()
        self._lock = threading.Lock()

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    def acquire(self) -> InferenceLease:
        """Reserve one capacity slot and one rate-limit token, or raise."""

        now = time.monotonic()
        with self._lock:
            self._discard_old_rate_tokens(now)

            if self._active >= self.max_concurrency:
                raise InferenceCapacityError("Inference capacity is busy")

            if (
                self.max_requests_per_second > 0
                and len(self._request_times) >= self._max_request_tokens
            ):
                raise InferenceCapacityError("Inference rate limit exceeded")

            self._request_times.append(now)
            self._active += 1

        return InferenceLease(self)

    def release(self) -> None:
        """Release a slot when callers use the guard without a lease.

        Normal route code should use the lease returned by :meth:`acquire`.
        This method is kept as a small convenience for direct callers and
        tests, and is safe to call more than once.
        """

        with self._lock:
            if self._active:
                self._active -= 1

    def _discard_old_rate_tokens(self, now: float) -> None:
        if not self.max_requests_per_second:
            self._request_times.clear()
            return

        cutoff = now - 1.0
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()


@dataclass(frozen=True)
class InferenceReliabilityConfig:
    """Environment-backed limits for one API process."""

    deadline_seconds: float = 65.0
    max_concurrency: int = 4
    worker_count: int = 4
    max_requests_per_second: float = 0.0

    @classmethod
    def from_env(cls) -> "InferenceReliabilityConfig":
        deadline = _read_float_env(
            (
                "PREDICTION_DEADLINE_SECONDS",
                "INFERENCE_DEADLINE_SECONDS",
                "TOTAL_PREDICTION_TIMEOUT_SECONDS",
                "TOTAL_INFERENCE_TIMEOUT_SECONDS",
                "PREDICTION_TIMEOUT_SECONDS",
                "INFERENCE_TIMEOUT_SECONDS",
            ),
            default=65.0,
            minimum=0.001,
        )
        max_concurrency = _read_int_env(
            (
                "INFERENCE_MAX_CONCURRENCY",
                "MAX_CONCURRENT_PREDICTIONS",
                "PREDICTION_MAX_CONCURRENCY",
            ),
            default=4,
            minimum=1,
        )
        workers = _read_int_env(
            (
                "INFERENCE_WORKERS",
                "INFERENCE_THREAD_POOL_SIZE",
                "INFERENCE_MAX_WORKERS",
            ),
            default=max_concurrency,
            minimum=1,
        )
        # There is no reason to create more pool threads than admitted work.
        workers = min(workers, max_concurrency)
        rate_limit = _read_float_env(
            (
                "INFERENCE_RATE_LIMIT_PER_SECOND",
                "PREDICTION_RATE_LIMIT_PER_SECOND",
                "MAX_PREDICTIONS_PER_SECOND",
            ),
            default=0.0,
            minimum=0.0,
        )
        if rate_limit > 0 and not rate_limit.is_integer():
            rate_limit = float(int(rate_limit))
        return cls(
            deadline_seconds=deadline,
            max_concurrency=max_concurrency,
            worker_count=workers,
            max_requests_per_second=rate_limit,
        )


def _read_int_env(names: tuple[str, ...], *, default: int, minimum: int) -> int:
    for name in names:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            continue
        try:
            value = int(raw)
        except ValueError:
            return default
        return max(minimum, value)
    return default


def _read_float_env(
    names: tuple[str, ...],
    *,
    default: float,
    minimum: float,
) -> float:
    for name in names:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            continue
        try:
            value = float(raw)
        except ValueError:
            return default
        if not math.isfinite(value):
            return default
        return max(minimum, value)
    return default


__all__ = [
    "ConcurrencyRateGuard",
    "InferenceCapacityError",
    "InferenceLease",
    "InferenceReliabilityConfig",
]
