"""Reliability patterns for the LLM call: bounded timeout, retry with
exponential backoff on transient failures, and graceful degradation when the
vector index is unreachable.

These wrap *why* a call failed into typed exceptions so app/main.py can map
each one to the right HTTP response instead of leaking a 500 + stack trace.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger("rag.reliability")

T = TypeVar("T")


class LLMTimeoutError(Exception):
    """Raised when the LLM call did not complete within the configured timeout."""


class LLMTransientError(Exception):
    """Raised for retryable upstream failures (connection reset, 5xx, rate limit)."""


class RetrievalUnavailableError(Exception):
    """Raised when the vector index / retriever cannot be queried."""


async def call_with_timeout(coro: Awaitable[T], seconds: float) -> T:
    """Bounds an LLM call to `seconds`; raises LLMTimeoutError on expiry so the
    API layer can return a clean HTTP 504 instead of hanging the request."""
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError as e:
        raise LLMTimeoutError(f"LLM call exceeded {seconds}s timeout") from e


def with_retry(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Retries up to 3 attempts on LLMTransientError, waiting 1s -> 2s -> 4s,
    as required by the RAID reliability spec."""
    return retry(
        retry=retry_if_exception_type(LLMTransientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )(func)


def safe_retrieve(retrieve_fn: Callable[..., T], *args, **kwargs) -> T:
    """Wraps a retriever call so any failure (index missing, corrupted store,
    FAISS error) surfaces as a single typed exception the API can turn into a
    `degraded: true` response instead of a 500."""
    try:
        return retrieve_fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 - intentionally broad: any retrieval
        # backend failure should degrade gracefully, not crash the request.
        logger.exception("retrieval backend failure")
        raise RetrievalUnavailableError(str(e)) from e
