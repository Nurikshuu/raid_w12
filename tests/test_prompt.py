"""Prompt regression suite (rag_v1): for each question we know exactly what
the model must return, or that it must refuse. Calls the LLM backend
configured in .env (LLM_BASE_URL/LLM_API_KEY/LLM_MODEL) - this is the same
backend the API serves with, so a prompt change that breaks grounding shows
up here before it reaches a user.
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import generation  # noqa: E402
from app.retrieval import RetrievedChunk  # noqa: E402

requires_llm = pytest.mark.skipif(
    not os.getenv("LLM_API_KEY") and not (ROOT / ".env").exists(),
    reason="no LLM backend configured (.env missing) - see .env.example",
)


def _chunk(law, article_number, section_title, text, score=0.8):
    return RetrievedChunk(
        chunk_id=f"{law}:{article_number}:0",
        law=law,
        article_number=article_number,
        section_title=section_title,
        text=text,
        score=score,
    )


LABOR_68 = _chunk(
    "labor_code", "68", "Глава 6. РАБОЧЕЕ ВРЕМЯ / Статья 68. Нормальная продолжительность рабочего времени",
    "Статья 68. Нормальная продолжительность рабочего времени 1. Нормальная продолжительность рабочего "
    "времени не должна превышать 40 часов в неделю.",
)

LABOR_88 = _chunk(
    "labor_code", "88", "Глава 7. ВРЕМЯ ОТДЫХА / Статья 88. Продолжительность основного оплачиваемого ежегодного трудового отпуска",
    "Статья 88. Продолжительность основного оплачиваемого ежегодного трудового отпуска Основной оплачиваемый "
    "ежегодный трудовой отпуск работникам предоставляется продолжительностью двадцать четыре календарных дня.",
)

TAX_503 = _chunk(
    "tax_code", "503", "Глава 51 / Статья 503. Ставки налога на добавленную стоимость",
    "Статья 503. Ставки налога на добавленную стоимость 1. Ставка налога на добавленную стоимость составляет "
    "16 процентов и применяется к размеру облагаемого оборота и облагаемого импорта.",
)

TAX_723 = _chunk(
    "tax_code", "723", "Глава 78 / Статья 723. Условия применения специального налогового режима на основе упрощенной декларации",
    "Статья 723. Условия применения специального налогового режима на основе упрощенной декларации 1. "
    "Предельный доход не превышает 600 000-кратный размер месячного расчетного показателя.",
)


def test_no_chunks_refuses_without_calling_llm():
    """Deterministic: an empty retrieval result must short-circuit to the
    canned refusal without ever hitting the network."""
    result = asyncio.run(generation.answer("Сколько дней в неделе?", []))
    assert result["answer"] == "I don't know from the provided context"
    assert result["used_context"] is False
    assert result["confidence"] == 0.0


def test_low_relevance_chunks_refuse_without_calling_llm():
    """A chunk present but with a score below RELEVANCE_SCORE_FLOOR must also
    short-circuit - this is what happens for genuinely off-topic questions
    once retrieval returns only noise."""
    noise_chunk = _chunk("labor_code", "1", "irrelevant", "не относящийся к вопросу текст", score=0.05)
    result = asyncio.run(generation.answer("Какая завтра погода в Астане?", [noise_chunk]))
    assert result["answer"] == "I don't know from the provided context"
    assert result["used_context"] is False


@requires_llm
def test_normal_work_week_hours():
    result = asyncio.run(generation.answer(
        "Сколько часов в неделю составляет нормальная продолжительность рабочего времени?",
        [LABOR_68],
    ))
    assert "40" in result["answer"]
    assert result["used_context"] is True


@requires_llm
def test_annual_leave_duration():
    result = asyncio.run(generation.answer(
        "Сколько календарных дней длится основной оплачиваемый ежегодный трудовой отпуск?",
        [LABOR_88],
    ))
    # the source article spells the number out ("двадцать четыре"), so the
    # model may faithfully echo either the digit or the word form
    answer_lower = result["answer"].lower()
    assert "24" in answer_lower or "двадцать четыре" in answer_lower
    assert result["used_context"] is True


@requires_llm
def test_vat_rate():
    result = asyncio.run(generation.answer(
        "Какая ставка НДС в Казахстане?",
        [TAX_503],
    ))
    assert "16" in result["answer"]
    assert result["used_context"] is True


@requires_llm
def test_simplified_regime_threshold():
    result = asyncio.run(generation.answer(
        "Какой предельный доход для упрощенной декларации?",
        [TAX_723],
    ))
    assert "600" in result["answer"]
    assert result["used_context"] is True


@requires_llm
def test_refuses_when_context_does_not_answer_the_question():
    """The retrieved chunk is real but irrelevant to the asked question - the
    model must refuse rather than hallucinate from an unrelated article."""
    result = asyncio.run(generation.answer(
        "Сколько длится отпуск по уходу за ребенком?",
        [TAX_503],  # VAT article, unrelated to childcare leave
    ))
    assert result["used_context"] is False
    assert "don't know" in result["answer"].lower() or "не зна" in result["answer"].lower()
