"""Pydantic request/response models for the RAG API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=20)
    filters: dict | None = None


class SourceRef(BaseModel):
    law: str
    article_number: str
    section_title: str
    chunk_id: str
    score: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    confidence: float = Field(ge=0.0, le=1.0)
    used_context: bool
    request_id: str
    degraded: bool = False
    cache_hit: bool = False
    prompt_version: str
    latency_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    ready: bool
    index_loaded: bool
    model_loaded: bool
    cache_loaded: bool


class MetricsSnapshot(BaseModel):
    request_count: int
    p50_latency_ms: float
    p95_latency_ms: float
    error_rate: float
    cache_hit_rate: float
    total_tokens: int
    mean_top1_retrieval_score: float
