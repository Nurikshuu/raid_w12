"""Endpoint-level tests using FastAPI's TestClient against the real, already
-built index (see Makefile: `make index` must run before `make test`)."""
from unittest.mock import AsyncMock, patch


def test_health_always_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_true_once_index_loaded(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["index_loaded"] is True
    assert body["model_loaded"] is True


def test_ask_empty_question_returns_422_not_500(client):
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ask_too_short_question_returns_422(client):
    resp = client.post("/ask", json={"question": "ok"})
    assert resp.status_code == 422


def test_ask_too_long_question_returns_422(client):
    resp = client.post("/ask", json={"question": "a" * 501})
    assert resp.status_code == 422


def test_ask_missing_question_field_returns_422(client):
    resp = client.post("/ask", json={})
    assert resp.status_code == 422


def _mock_generation_answer(payload):
    async def _mock(question, chunks):
        return payload
    return _mock


def test_ask_valid_question_returns_grounded_answer(client):
    mock_payload = {
        "answer": "Нормальная продолжительность рабочего времени не должна превышать 40 часов в неделю.",
        "sources": [{"law": "labor_code", "article_number": "68"}],
        "confidence": 0.9,
        "used_context": True,
        "_token_usage": 42,
    }
    with patch("app.main.generation.answer", new=AsyncMock(return_value=mock_payload)):
        resp = client.post("/ask", json={"question": "Сколько часов в неделю рабочее время?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "40" in body["answer"]
    assert len(body["sources"]) > 0
    assert body["used_context"] is True
    assert "request_id" in body
    assert set(body["latency_ms"].keys()) == {"embedding_ms", "retrieval_ms", "generation_ms", "total_ms"}


def test_ask_response_cache_hit_on_repeated_question(client):
    mock_payload = {
        "answer": "24 календарных дня.",
        "sources": [{"law": "labor_code", "article_number": "88"}],
        "confidence": 0.9,
        "used_context": True,
        "_token_usage": 30,
    }
    question = "Какова продолжительность ежегодного отпуска (уникальный тест-вопрос)?"
    mock = AsyncMock(return_value=mock_payload)
    with patch("app.main.generation.answer", new=mock):
        first = client.post("/ask", json={"question": question})
        second = client.post("/ask", json={"question": question})

    assert first.status_code == 200 and second.status_code == 200
    assert second.json()["cache_hit"] is True
    # generation.answer must only be called once - the second request was served from cache
    assert mock.await_count == 1


def test_metrics_endpoint_shape(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("p50_latency_ms", "p95_latency_ms", "error_rate", "cache_hit_rate", "total_tokens"):
        assert key in body
