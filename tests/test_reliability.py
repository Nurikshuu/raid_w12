"""Unit tests for the reliability patterns: bounded timeout, retry with
exponential backoff, and graceful degradation on retrieval failure. Each test
sends a mocked failure through the handler and asserts the correct response,
per the RAID spec (no live fault injection needed)."""
import asyncio

import pytest

from app.reliability import (
    LLMTimeoutError,
    LLMTransientError,
    RetrievalUnavailableError,
    call_with_timeout,
    safe_retrieve,
    with_retry,
)


def test_call_with_timeout_raises_on_slow_call():
    async def slow_call():
        await asyncio.sleep(1.0)
        return "too late"

    with pytest.raises(LLMTimeoutError):
        asyncio.run(call_with_timeout(slow_call(), seconds=0.05))


def test_call_with_timeout_returns_result_when_fast_enough():
    async def fast_call():
        return "ok"

    result = asyncio.run(call_with_timeout(fast_call(), seconds=5.0))
    assert result == "ok"


def test_retry_recovers_after_two_transient_failures():
    attempts = {"count": 0}

    @with_retry
    async def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise LLMTransientError("simulated transient failure")
        return "recovered"

    result = asyncio.run(flaky())
    assert result == "recovered"
    assert attempts["count"] == 3


def test_retry_gives_up_after_3_attempts():
    attempts = {"count": 0}

    @with_retry
    async def always_fails():
        attempts["count"] += 1
        raise LLMTransientError("simulated permanent-for-this-test failure")

    with pytest.raises(LLMTransientError):
        asyncio.run(always_fails())
    assert attempts["count"] == 3


def test_retry_does_not_catch_non_transient_errors():
    @with_retry
    async def raises_value_error():
        raise ValueError("not a reliability concern")

    with pytest.raises(ValueError):
        asyncio.run(raises_value_error())


def test_safe_retrieve_wraps_backend_failure():
    def broken_retriever(*args, **kwargs):
        raise RuntimeError("FAISS index corrupted")

    with pytest.raises(RetrievalUnavailableError):
        safe_retrieve(broken_retriever, "some question")


def test_safe_retrieve_passes_through_on_success():
    def working_retriever(question):
        return ([], False, {"embedding_ms": 1.0, "search_ms": 1.0})

    result = safe_retrieve(working_retriever, "some question")
    assert result == ([], False, {"embedding_ms": 1.0, "search_ms": 1.0})
