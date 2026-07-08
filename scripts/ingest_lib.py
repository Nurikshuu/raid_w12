"""Shared parsing + chunking logic used by notebooks/01_ingestion.ipynb and scripts/build_index.py.

Parses the two source laws (Kazakh Labor Code, Kazakh Tax Code) out of the raw
adilet.zan.kz HTML dumps into per-article records, then chunks them with two
independent strategies so the notebook can compare chunk counts.

adilet.zan.kz marks every article title as ``<p><b>Статья N. Title</b></p>``
followed by the article body as sibling ``<p>`` tags, with chapter/section
headings as ``<h3>`` tags. We walk the tag tree directly instead of
regex-splitting flattened text, which gives clean titles and paragraph-joined
bodies for free.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

DOCUMENT_VERSION = "adilet.zan.kz snapshot 2026-07-08"

ARTICLE_TITLE_RE = re.compile(r"^Статья\s+(\d+(?:-\d+)?)\.\s*(.*)$", re.DOTALL)

# Tax code is ~800 articles across every tax type. For a focused fintech
# "ask the docs" corpus we keep the chapters a founding ML engineer would
# actually get questions about: general rules, corporate & individual income
# tax, VAT, and the special regimes small businesses use.
TAX_CHAPTER_WHITELIST = {
    "General": (1, 61),            # Раздел 1-2: general provisions, taxpayer rights, tax obligation
    "CIT": (224, 359),             # Раздел 5: Корпоративный подоходный налог
    "PIT": (360, 434),             # Раздел 6: Индивидуальный подоходный налог
    "VAT": (435, 518),             # Раздел 7: НДС
    "SpecialRegimes": (683, 731),  # Раздел 16: Специальные налоговые режимы
}


@dataclass
class Article:
    source_file: str
    law: str
    article_number: str
    title: str
    chapter: str
    text: str
    document_version: str = DOCUMENT_VERSION


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    law: str
    article_number: str
    section_title: str
    document_version: str
    text: str
    char_start: int
    char_end: int
    strategy: str
    page: int | None = None


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _walk_articles(soup: BeautifulSoup, law: str, source_file: str) -> list[Article]:
    """Walk every <h3>/<p> tag in document order, grouping paragraphs under
    the article title that precedes them and tracking the nearest <h3> as
    the current chapter."""
    articles: list[Article] = []
    current_chapter = ""
    current_article: Article | None = None
    body_parts: list[str] = []

    def flush():
        nonlocal current_article
        if current_article is not None:
            body = _clean(" ".join(body_parts))
            current_article.text = f"Статья {current_article.article_number}. {current_article.title} {body}".strip()
            articles.append(current_article)

    for tag in soup.find_all(["h3", "p"]):
        if tag.name == "h3":
            heading = _clean(tag.get_text(" "))
            if heading:
                current_chapter = heading
            continue

        text = _clean(tag.get_text(" "))
        if not text:
            continue

        bold = tag.find("b")
        bold_text = _clean(bold.get_text(" ")) if bold else ""
        m = ARTICLE_TITLE_RE.match(bold_text) if bold_text else None

        if m:
            flush()
            art_num, title = m.group(1), _clean(m.group(2))
            current_article = Article(
                source_file=source_file,
                law=law,
                article_number=art_num,
                title=title,
                chapter=current_chapter,
                text="",
            )
            body_parts = []
            # a title paragraph occasionally also carries the first body
            # sentence after the bold tag; keep whatever text trails the bold run
            remainder = text[len(bold_text):].strip()
            if remainder:
                body_parts.append(remainder)
        elif current_article is not None:
            body_parts.append(text)

    flush()
    return articles


def parse_labor_code(html_path: Path) -> list[Article]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    return _walk_articles(soup, law="labor_code", source_file="labor_code_raw.html")


def parse_tax_code(html_path: Path) -> list[Article]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    all_articles = _walk_articles(soup, law="tax_code", source_file="tax_code_raw.html")

    def article_int(a: Article) -> int:
        return int(a.article_number.split("-")[0])

    kept = []
    for a in all_articles:
        n = article_int(a)
        for _, (lo, hi) in TAX_CHAPTER_WHITELIST.items():
            if lo <= n <= hi:
                kept.append(a)
                break
    return kept


def recursive_char_chunks(articles: list[Article], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Strategy A: RecursiveCharacterTextSplitter applied per-article so chunks
    never cross an article boundary (metadata stays attached to one article)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(chunk_overlap, chunk_size // 2),
        separators=["\n\n", ". ", "; ", " ", ""],
    )
    chunks: list[Chunk] = []
    for art in articles:
        pieces = splitter.split_text(art.text)
        cursor = 0
        for j, piece in enumerate(pieces):
            start = art.text.find(piece, max(0, cursor - chunk_overlap))
            if start == -1:
                start = cursor
            end = start + len(piece)
            cursor = end
            chunks.append(
                Chunk(
                    chunk_id=f"{art.law}:{art.article_number}:rc{chunk_size}:{j}",
                    source_file=art.source_file,
                    law=art.law,
                    article_number=art.article_number,
                    section_title=f"{art.chapter} / Статья {art.article_number}. {art.title}",
                    document_version=art.document_version,
                    text=piece,
                    char_start=start,
                    char_end=end,
                    strategy=f"recursive_char_{chunk_size}",
                )
            )
    return chunks


def article_based_chunks(articles: list[Article], max_chars: int = 1600) -> list[Chunk]:
    """Strategy B: one chunk per article (self-contained legal unit). Articles
    longer than max_chars are split on numbered sub-clauses ("1. ", "2. ", ...)
    which is how Kazakh legal text itself is structured, instead of a blind
    character cut."""
    subclause_re = re.compile(r"(?<=\D)(?=\d+(?:-\d+)?\.\s)")
    chunks: list[Chunk] = []
    for art in articles:
        section_title = f"{art.chapter} / Статья {art.article_number}. {art.title}"
        if len(art.text) <= max_chars:
            chunks.append(
                Chunk(
                    chunk_id=f"{art.law}:{art.article_number}:art:0",
                    source_file=art.source_file,
                    law=art.law,
                    article_number=art.article_number,
                    section_title=section_title,
                    document_version=art.document_version,
                    text=art.text,
                    char_start=0,
                    char_end=len(art.text),
                    strategy="article_based",
                )
            )
            continue

        parts = subclause_re.split(art.text)
        buf, buf_start, idx, cursor = "", 0, 0, 0
        for part in parts:
            if len(buf) + len(part) > max_chars and buf:
                chunks.append(
                    Chunk(
                        chunk_id=f"{art.law}:{art.article_number}:art:{idx}",
                        source_file=art.source_file,
                        law=art.law,
                        article_number=art.article_number,
                        section_title=section_title,
                        document_version=art.document_version,
                        text=buf.strip(),
                        char_start=buf_start,
                        char_end=buf_start + len(buf),
                        strategy="article_based",
                    )
                )
                idx += 1
                buf_start = cursor
                buf = ""
            buf += part
            cursor += len(part)
        if buf.strip():
            chunks.append(
                Chunk(
                    chunk_id=f"{art.law}:{art.article_number}:art:{idx}",
                    source_file=art.source_file,
                    law=art.law,
                    article_number=art.article_number,
                    section_title=section_title,
                    document_version=art.document_version,
                    text=buf.strip(),
                    char_start=buf_start,
                    char_end=buf_start + len(buf),
                    strategy="article_based",
                )
            )
    return chunks
