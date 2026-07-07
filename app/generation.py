"""LLM call wrapped in a grounded, JSON-only prompt.

Backend is any OpenAI-compatible chat completions endpoint - configured via
LLM_BASE_URL / LLM_API_KEY / LLM_MODEL in .env. Works unmodified against
Ollama's local /v1 endpoint, the Alem AI endpoint used in this deployment, or
OpenAI itself.
"""
from __future__ import annotations

import json
import logging
import re

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError

from app.config import settings
from app.reliability import LLMTransientError, with_retry
from app.retrieval import RetrievedChunk

logger = logging.getLogger("rag.generation")

PROMPT_VERSION = "rag_v1"

RELEVANCE_SCORE_FLOOR = 0.25  # cosine similarity below this = context is noise, refuse without calling the LLM

SYSTEM_PROMPT = """\
Ты — ассистент, который отвечает на вопросы сотрудников казахстанской финтех-компании,
используя ТОЛЬКО предоставленные фрагменты Трудового кодекса РК и Налогового кодекса РК.

Правила:
1. Отвечай ТОЛЬКО на основе текста в блоке КОНТЕКСТ ниже. Не используй никакие другие знания
   о законодательстве, даже если они тебе известны.
2. Если контекст не содержит ответа на заданный вопрос, верни:
   {"answer": "I don't know from the provided context", "sources": [], "confidence": 0.0, "used_context": false}
3. Никогда не выдумывай номера статей, кодексы, проценты или суммы, которых нет в контексте.
4. Каждый факт в ответе должен быть подкреплён источником из контекста.
5. Верни ответ СТРОГО в виде одного JSON-объекта (без markdown, без пояснений вне JSON) с полями:
   - "answer": строка с ответом на русском языке
   - "sources": список объектов {"law": "...", "article_number": "..."}, использованных для ответа
   - "confidence": число от 0 до 1 - насколько ответ полностью подтверждён контекстом
   - "used_context": true/false
"""

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    return _client


def _build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = []
    for c in chunks:
        context_blocks.append(
            f"[{c.law} | Статья {c.article_number} | {c.section_title}]\n{c.text}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    return f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {question}"


def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(json)?", "", raw).rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _refusal_payload() -> dict:
    return {
        "answer": "I don't know from the provided context",
        "sources": [],
        "confidence": 0.0,
        "used_context": False,
    }


@with_retry
async def _call_llm(question: str, chunks: list[RetrievedChunk]) -> dict:
    client = _get_client()
    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(question, chunks)},
            ],
            temperature=0.0,
            max_tokens=800,
        )
    except (APIConnectionError, APITimeoutError) as e:
        raise LLMTransientError(str(e)) from e
    except APIStatusError as e:
        if e.status_code >= 500 or e.status_code == 429:
            raise LLMTransientError(str(e)) from e
        raise

    raw = response.choices[0].message.content or ""
    usage = response.usage
    tokens = usage.total_tokens if usage else 0

    try:
        payload = _parse_json_response(raw)
    except (json.JSONDecodeError, AttributeError):
        logger.warning("LLM returned non-JSON output, falling back to raw text")
        payload = {
            "answer": raw.strip() or "I don't know from the provided context",
            "sources": [],
            "confidence": 0.3,
            "used_context": bool(chunks),
        }

    payload["_token_usage"] = tokens
    return payload


async def answer(question: str, retrieved_chunks: list[RetrievedChunk]) -> dict:
    """Returns a dict matching AnswerResponse's generation-relevant fields
    (answer, sources, confidence, used_context) plus a private _token_usage
    key consumed by the caller for observability."""
    if not retrieved_chunks or max((c.score for c in retrieved_chunks), default=0.0) < RELEVANCE_SCORE_FLOOR:
        payload = _refusal_payload()
        payload["_token_usage"] = 0
        return payload

    return await _call_llm(question, retrieved_chunks)
