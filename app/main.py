"""FastAPI RAG service: /health, /ready, /ask, /metrics.

Run with: uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import hashlib
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app import generation
from app.config import settings
from app.observability import RequestRecord, log_request, logger, metrics
from app.reliability import (
    LLMTimeoutError,
    LLMTransientError,
    RetrievalUnavailableError,
    call_with_timeout,
    safe_retrieve,
)
from app.retrieval import TTLCache, retriever
from app.schemas import (
    AnswerResponse,
    AskRequest,
    HealthResponse,
    MetricsSnapshot,
    ReadyResponse,
    SourceRef,
)

response_cache = TTLCache(max_size=256, ttl_seconds=settings.response_cache_ttl_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("loading index and embedding model...")
    retriever.load()
    logger.info("ready.")
    yield


app = FastAPI(title="Kazakh Labor & Tax Code RAG", lifespan=lifespan)


def _response_cache_key(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready():
    loaded = retriever.is_loaded
    return ReadyResponse(ready=loaded, index_loaded=loaded, model_loaded=loaded, cache_loaded=True)


@app.get("/metrics", response_model=MetricsSnapshot)
def get_metrics():
    return MetricsSnapshot(**metrics.snapshot())


@app.post("/ask", response_model=AnswerResponse)
async def ask(payload: AskRequest, request: Request):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    t_start = time.time()
    question = payload.question

    cache_key = _response_cache_key(question)
    cached = response_cache.get(cache_key)
    if cached is not None:
        total_ms = (time.time() - t_start) * 1000
        resp = AnswerResponse(
            **cached,
            request_id=request_id,
            cache_hit=True,
            latency_ms={"embedding_ms": 0.0, "retrieval_ms": 0.0, "generation_ms": 0.0, "total_ms": total_ms},
        )
        _finish(request_id, question, resp, error=False)
        return resp

    try:
        chunks, retrieval_cache_hit, retrieval_timing = safe_retrieve(retriever.retrieve, question, payload.top_k)
    except RetrievalUnavailableError:
        total_ms = (time.time() - t_start) * 1000
        resp = AnswerResponse(
            answer="Не удалось выполнить поиск по документам прямо сейчас. Попробуйте позже.",
            sources=[],
            confidence=0.0,
            used_context=False,
            request_id=request_id,
            degraded=True,
            cache_hit=False,
            prompt_version=generation.PROMPT_VERSION,
            latency_ms={"embedding_ms": 0.0, "retrieval_ms": 0.0, "generation_ms": 0.0, "total_ms": total_ms},
        )
        _finish(request_id, question, resp, error=True)
        return resp

    t_gen_start = time.time()
    try:
        gen_payload = await call_with_timeout(
            generation.answer(question, chunks), settings.llm_timeout_seconds
        )
    except LLMTimeoutError:
        _finish_error(request_id, question, error=True)
        raise HTTPException(
            status_code=504,
            detail={"error": "llm_timeout", "request_id": request_id},
        )
    except LLMTransientError:
        _finish_error(request_id, question, error=True)
        raise HTTPException(
            status_code=503,
            detail={"error": "llm_unavailable", "request_id": request_id},
        )
    generation_ms = (time.time() - t_gen_start) * 1000

    total_ms = (time.time() - t_start) * 1000
    sources = _resolve_sources(chunks, gen_payload)

    cache_hit = retrieval_cache_hit
    cacheable_fields = {
        "answer": gen_payload["answer"],
        "sources": [s.model_dump() for s in sources],
        "confidence": gen_payload["confidence"],
        "used_context": gen_payload["used_context"],
        "prompt_version": generation.PROMPT_VERSION,
        "degraded": False,
    }
    response_cache.set(cache_key, cacheable_fields)

    resp = AnswerResponse(
        **cacheable_fields,
        request_id=request_id,
        cache_hit=cache_hit,
        latency_ms={
            "embedding_ms": round(retrieval_timing["embedding_ms"], 2),
            "retrieval_ms": round(retrieval_timing["search_ms"], 2),
            "generation_ms": round(generation_ms, 2),
            "total_ms": round(total_ms, 2),
        },
    )
    _finish(request_id, question, resp, error=False, chunks=chunks, tokens=gen_payload.get("_token_usage", 0))
    return resp


def _resolve_sources(chunks, gen_payload: dict) -> list[SourceRef]:
    """Maps the LLM-cited (law, article_number) pairs back to the retrieved
    chunks that back them. If the model didn't cite anything specific but did
    claim to use the context, fall back to all retrieved chunks rather than
    silently dropping provenance."""
    if not gen_payload.get("used_context"):
        return []

    cited = {(s.get("law"), s.get("article_number")) for s in gen_payload.get("sources", [])}
    matched = [c for c in chunks if (c.law, c.article_number) in cited]
    relevant_chunks = matched or chunks
    return [
        SourceRef(law=c.law, article_number=c.article_number, section_title=c.section_title,
                  chunk_id=c.chunk_id, score=c.score)
        for c in relevant_chunks
    ]


def _finish(request_id, question, resp: AnswerResponse, error: bool, chunks=None, tokens: int = 0):
    top1_score = chunks[0].score if chunks else None
    metrics.record(RequestRecord(
        total_latency_ms=resp.latency_ms["total_ms"],
        error=error,
        cache_hit=resp.cache_hit,
        tokens=tokens,
        top1_retrieval_score=top1_score,
    ))
    log_request(
        logger,
        request_id=request_id,
        question=question,
        retrieved_chunk_ids=[c.chunk_id for c in (chunks or [])],
        prompt_version=generation.PROMPT_VERSION,
        model_name=settings.llm_model,
        latency_ms_by_stage=resp.latency_ms,
        token_usage=tokens,
        answer_length=len(resp.answer),
        cache_hit_bool=resp.cache_hit,
        error_bool=error,
    )


def _finish_error(request_id, question, error: bool):
    metrics.record(RequestRecord(
        total_latency_ms=0.0, error=error, cache_hit=False, tokens=0, top1_retrieval_score=None,
    ))
    log_request(
        logger,
        request_id=request_id,
        question=question,
        retrieved_chunk_ids=[],
        prompt_version=generation.PROMPT_VERSION,
        model_name=settings.llm_model,
        latency_ms_by_stage={},
        token_usage=0,
        answer_length=0,
        cache_hit_bool=False,
        error_bool=error,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "unexpected server error"},
    )
